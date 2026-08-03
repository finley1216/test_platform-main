# -*- coding: utf-8 -*-
"""
跨鏡頭路徑枚舉 + 對數似然比（LLR）計分（統一入口）
==================================================
合併：repo_paths / config / track_path_legacy（硬規則＋載入）/
      track_path LLR / calibrate / evaluate / render_viz

結構修正（2026-07-15，附依據）：
  1. emb 硬門檻 → 0.80（鑑別交給 LLR_emb）
  2. 共存超節點：時間重疊≥0.5s 且（OVERLAP 或 H×dH<80）→ union-find
  3. 節點證據 = w·ln(P(sim|GT)/P(sim|nonGT))；transit dt 先驗 σ=1.0（PRIOR-WEAK）
  4. MIN_TRANSIT hop1→0（相鄰視野邊界相接，無辯護下界）；hop2 維持 6s
  5. 分段軌跡／排名：單路徑與多段假設進同一排名池（計分公式不變，只改誰跟誰比）

用法：
  python3 track_path.py run ...
  python3 track_path.py calibrate ...
  python3 track_path.py evaluate ...
  python3 track_path.py viz ...
"""

from __future__ import annotations

import argparse
import ast
import glob
import itertools
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from scipy.special import logsumexp


# ============================================================
# TOP SETTINGS：路徑常數（原 repo_paths.py）
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT.parent / "output"
BOTSORT_ROOT = REPO_ROOT / "BoT-SORT"
CLIP_REID_ROOT = REPO_ROOT / "CLIP-ReID"
SEGMENT_ROOT = REPO_ROOT / "backend" / "segment"

DEFAULT_PERSON_QUERY = REPO_ROOT / "p9.jpg"
DEFAULT_VEHICLE_QUERY_0528 = REPO_ROOT / "wc.png"
DEFAULT_VEHICLE_QUERY_0507 = REPO_ROOT / "BSH-5613.jpg"

QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"
V1_OUTPUT_ROOT = OUTPUT_ROOT / "v1.0"
ARCHIVE_PATH_ENUM_LLR = OUTPUT_ROOT / "path_enum_llr"  # freeze archive only


def embed_cache_dir(data_root: Path, video_id: str) -> Path:
    """與 crop / mapping 同根的 embedding 快取目錄。"""
    slug = video_id.lower().replace("-", "")
    return data_root / "embed_cache" / slug


def person_embed_cache_path(data_root: Path, video_id: str) -> Path:
    return embed_cache_dir(data_root, video_id) / "person_clipreid_embeddings_cache.pkl"


def vehicle_embed_cache_path(data_root: Path, video_id: str) -> Path:
    return embed_cache_dir(data_root, video_id) / "vehicle_clipreid_embeddings_cache.pkl"

# ============================================================
# TOP SETTINGS：LLR emb 硬門檻（原 config.py）
# ============================================================

# LLR 管線硬門檻覆寫
# ====================================================
# 日期：2026-07-15
#
# 修正一依據（GT 更正後，人員 0507，CALIB_SOURCE=GT_20260507）：
#   - emb|same ≈ Normal(μ=0.917, σ=0.023)
#   - emb|diff ≈ Normal(μ=0.874, σ=0.029)（見 calibration_gt0507_report）
#   - 舊 EMB_EDGE_MIN=0.91 ≈ μ_same − 0.3σ，構造性拒絕約 38% 真轉移
#     （瓶頸例：22_22→07_112 emb=0.859；07_112→01_50 emb=0.897）
#   - 改為 0.80 ≈ μ_diff − 2.5σ：僅作粗理智檢查，外觀鑑別交給 LLR_emb
#
# 執行時以 apply_llr_emb_gates() 覆寫本模組 EMB_EDGE_MIN / EMB_HIST_MIN。
#
# ---
# dt 軟計分（2026-07-15）：
#   tau／通行時間無本場景實測；transit 邊 LLR_dt 可經 --dt-scoring off 停用。
#   硬規則（時間順序／MIN_TRANSIT／DT_MAX）不受影響。

# 原始預設（僅供對照／還原）
ORIGINAL_EMB_EDGE_MIN = 0.91
ORIGINAL_EMB_HIST_MIN = 0.90

# 修正一
LLR_EMB_EDGE_MIN = 0.80
LLR_EMB_HIST_MIN = 0.80

RATIONALE = (
    "2026-07-15：emb|same N(0.917,0.023)；EMB_EDGE_MIN=0.91 為 μ−0.3σ，"
    "構造性拒真轉移；改 0.80≈μ_diff−2.5σ，鑑別交給 LLR_emb。"
)

DT_SCORING_RATIONALE = (
    "2026-07-15：tau 無本場景實測來源；transit 邊 LLR_dt 軟證據自即日起可停用"
    "（--dt-scoring off）。硬規則不動；handoff 本來不算 LLR_dt。"
)


def apply_llr_emb_gates(enabled: bool = True) -> dict:
    """覆寫或還原本模組的 emb 硬門檻。回傳生效值。"""
    global EMB_EDGE_MIN, EMB_HIST_MIN
    if enabled:
        EMB_EDGE_MIN = float(LLR_EMB_EDGE_MIN)
        EMB_HIST_MIN = float(LLR_EMB_HIST_MIN)
    else:
        EMB_EDGE_MIN = float(ORIGINAL_EMB_EDGE_MIN)
        EMB_HIST_MIN = float(ORIGINAL_EMB_HIST_MIN)
    return {
        "enabled": bool(enabled),
        "EMB_EDGE_MIN": float(EMB_EDGE_MIN),
        "EMB_HIST_MIN": float(EMB_HIST_MIN),
        "rationale": RATIONALE if enabled else "restored path_enum_scoring defaults",
    }


# ============================================================
# LEGACY CORE：硬規則 + 載入 + Top-1 拼圖（原 track_path_legacy.py）
# ============================================================

# ============================================================
# 一、參數（全部有物理意義，用現場實測填，不要用調參填）
# ============================================================

# 鏡頭拓撲：相鄰鏡頭對（雙向）。依資料夾自動選人員／車輛圖。
# 人員：cross_camera_chain_test/run_cross_camera_chain.py DEFAULT_ADJACENCY
# 車輛：run_layered_cluster_v3_vehicle_clean.py VEHICLE_ADJACENCY
# 人員鏡頭相鄰。2026-07-15 依使用者提供之場地配置補登
# K8-09↔K8-10、K8-10↔K8-12、K8-12↔K8-30（線形走廊；適用所有後續資料集）。
PERSON_ADJACENT = {
    ("K8-01", "K8-05"),
    ("K8-01", "K8-08"),
    ("K8-01", "K8-09"),
    ("K8-05", "K8-07"),
    ("K8-05", "K8-08"),
    ("K8-05", "K8-09"),
    ("K8-05", "K8-22"),
    ("K8-05", "K8-23"),
    ("K8-07", "K8-08"),
    ("K8-07", "K8-09"),
    ("K8-08", "K8-09"),
    ("K8-09", "K8-10"),  # 2026-07-15 場地配置
    ("K8-10", "K8-12"),  # 2026-07-15 場地配置
    ("K8-12", "K8-30"),  # 2026-07-15 場地配置
    ("K8-22", "K8-23"),
    ("K8-20", "K8-21"),
}
VEHICLE_ADJACENT_RAW = {
    "5": ["7", "8", "9", "1", "23"],
    "7": ["5", "8", "9"],
    "8": ["5", "7", "9", "1"],
    "1": ["5", "8", "9"],
    "9": ["5", "7", "8", "1", "10"],
    "10": ["9", "12"],
    "12": ["10", "30"],
    "23": ["5", "22"],
    "22": ["23", "21", "20"],
    "21": ["22", "20", "19"],  # 走廊續接 19（無 H，靠相鄰+外觀）
    "20": ["21", "22", "19"],
    "19": ["20", "21", "28"],
    "28": ["19", "16", "15"],
    "16": ["28", "15"],
    "15": ["28", "16", "30"],
    "30": ["15", "12"],
}

def _cam(n: str) -> str:
    return f"K8-{int(n):02d}"

def _pairs_from_adj(raw: dict) -> set:
    out = set()
    for a, nbs in raw.items():
        for b in nbs:
            out.add(tuple(sorted((_cam(a), _cam(b)))))
    return out

PERSON_ADJACENT = {tuple(sorted(p)) for p in PERSON_ADJACENT}
VEHICLE_ADJACENT = _pairs_from_adj(VEHICLE_ADJACENT_RAW)

# 人員視野重疊容許。2026-07-15 依場地配置補登 K8-09↔K8-10（tol=3s，無 H）：
# 實際小面積重疊，同物件會共存；適用所有後續資料集。
PERSON_OVERLAP_PAIRS = {
    ("K8-22", "K8-23"): 20.0,
    ("K8-09", "K8-10"): 3.0,  # 2026-07-15 場地配置；無 Homography
}
VEHICLE_OVERLAP_PAIRS = {
    ("K8-05", "K8-07"): 20.0,
    ("K8-05", "K8-08"): 20.0,
    ("K8-07", "K8-08"): 20.0,
    ("K8-22", "K8-23"): 20.0,
    ("K8-20", "K8-21"): 20.0,
    ("K8-15", "K8-16"): 20.0,
    ("K8-21", "K8-19"): 10.0,
    ("K8-19", "K8-28"): 10.0,
    ("K8-22", "K8-20"): 10.0,
}
PERSON_OVERLAP_PAIRS = {tuple(sorted(k)): v for k, v in PERSON_OVERLAP_PAIRS.items()}
VEHICLE_OVERLAP_PAIRS = {tuple(sorted(k)): v for k, v in VEHICLE_OVERLAP_PAIRS.items()}

# 車輛走廊行進順序（遠鏡頭可能較晚被偵測，但實體上較早出現）
VEHICLE_CORRIDOR = [
    "K8-23", "K8-22", "K8-20", "K8-21", "K8-19", "K8-28",
    "K8-16", "K8-15", "K8-30", "K8-12", "K8-10", "K8-09",
]
VEHICLE_CORRIDOR_RANK = {c: i for i, c in enumerate(VEHICLE_CORRIDOR)}

# 執行時由 configure_for_input() 依資料夾覆寫
ADJACENT = set(PERSON_ADJACENT)
OVERLAP_PAIRS = dict(PERSON_OVERLAP_PAIRS)
MODE = "person"

# Homography（Homography/）同物件腳底投影
H_DIST_GATE = 150.0     # cross-cam 邊合法性門檻（非 supernode 合併門檻）
H_TIME_WINDOW = 15.0    # 對齊兩 track 腳底點的最大時間差（秒）
H_MATRICES = {}         # (cam_a, cam_b) -> 3x3 ndarray（a 投影到 b）
# HOMOGRAPHY_DIR 見下方路徑常數區塊

# 2026-07-16 實驗採納：exp1_h_projection_distance（N=13）
# H 合併門檻 = μ+3σ = 94.427...，四捨五入到 5px → 95px。
SUPER_DH_MAX = 95.0
# 同批樣本重擬合幾何計分 dH|same ~ HalfNormal(σ=37.557)，n=13。
DH_SAME_SIGMA = 37.557
DH_SAME_N = 13
# 2026-07-20：名單制共存合併（OVERLAP_PAIRS 且無 H）加 emb 底線。
# emb|same μ−3σ = 0.917 − 3×0.023 = 0.848（calibration_gt0507，全系統 3σ 原則）。
# 幾何制（H 投影 <95px）刻意不驗外觀——位置排他性。
COEXISTENCE_OVERLAP_EMB_MIN = 0.848

# 每個鏡頭對的最短通行時間（秒）：拿碼表現場走出來的下界。
# 沒填的鏡頭對用 DEFAULT_MIN_TRANSIT。同鏡頭再入預設 0。
MIN_TRANSIT = {
    # ("K8-22", "K8-23"): 2.0,
}
# 2026-07-15：hop1 DEFAULT 改 0.0。
# 依據：相鄰鏡頭視野可能邊界相接，通行下界無法辯護；舊值 2.0s 為無實測佔位，
# 已誤殺真轉移（例：0528 的 09_3→{08_17,01_8} 聯集 dt=0.72s）。
DEFAULT_MIN_TRANSIT_HOP1 = 0.0     # 相鄰鏡頭：無辯護下界 → 0
# 2026-07-16：exp2 消融證明 hop2 下界冗餘；系統不再含任何手寫秒數下界。
DEFAULT_MIN_TRANSIT_HOP2 = 0.0
# 車輛在 configure_for_input 改為接近 0（車速快、相鄰幾乎同時）

# 預期通行時間 TAU（秒）：正常步行的典型值，超過的部分開始扣分
TAU = {
    # ("K8-22", "K8-23"): 5.0,
}
DEFAULT_TAU_HOP0 = 3.0
DEFAULT_TAU_HOP1 = 8.0
DEFAULT_TAU_HOP2 = 20.0

TOL      = 2.0      # 一般鏡頭對容許的時間重疊（秒）
# OVERLAP_PAIRS 由 configure_for_input() 設定
# 2026-07-16：exp3 採納，DT_MAX = max(GT true transition dt)×1.5。
# max=84.25s → 126.375s，取 130s；敏感度穩定區間 120–180。
DT_MAX   = 130.0

# ---- 外觀一致性（「原本就很像」+「歷史都很像」）----
SIM_MIN       = 0.90   # 對 query 低於此者不進候選（原本就要夠像）
EMB_EDGE_MIN  = 0.91   # 相鄰兩 track 平均 embedding 餘弦相似度下界
EMB_HIST_MIN  = 0.90   # 新 track 對路徑歷史平均 embedding 下界


CAMERA_RE = re.compile(r"(K8-\d+)")
ASE_ROOT = REPO_ROOT  # 相容舊變數名（本 repo 根）
DEFAULT_MAPPING = OUTPUT_ROOT / "人員追蹤_20260507_crop_time_mapping.json"
# test_platform：mapping/crop 在上一層 output/；H 矩陣在 Homography/


def _resolve_homography_dir() -> Path:
    """優先 Homography/；相容舊名 上傳/（本 repo 或上一層 ASE）。"""
    cands = [
        REPO_ROOT / "Homography",
        REPO_ROOT / "上傳",
        REPO_ROOT.parent / "Homography",
        REPO_ROOT.parent / "上傳",
    ]
    for p in cands:
        if p.is_dir():
            return p
    return cands[0]


HOMOGRAPHY_DIR = _resolve_homography_dir()
UPLOAD_DIR = HOMOGRAPHY_DIR  # 舊變數名相容

# ============================================================
# 二、資料載入  ★★ 這一段要接你的實際檔案格式 ★★
# ============================================================

@dataclass
class Track:
    tid: str            # 全域唯一 ID，例如 "K8-22_37"
    cam: str            # 鏡頭 ID
    t_start: float      # 首次出現的 wall-clock 秒（用逐 frame 時間戳，勿用 frame/fps）
    t_end: float        # 最後出現的 wall-clock 秒
    sim: float = 0.0    # 對 query 的相似度
    emb: object = None  # L2-normalized 平均 embedding（np.ndarray）；外觀門檻／計分用
    foots: list = field(default_factory=list)  # [(t_sec, x, y), ...] 腳底／車底中心
    meta: dict = field(default_factory=dict)

def _infer_dataset_tag(input_dir: str) -> str:
    name = Path(input_dir).name
    if "人員追蹤_" in name or "車輛追蹤_" in name:
        return name
    return "人員追蹤_20260507"


def _load_h_matrices():
    """從 Homography/*/H_K8-XXtoK8-YY_method0.npy 載入 Homography。"""
    global H_MATRICES
    H_MATRICES = {}
    if not HOMOGRAPHY_DIR.is_dir():
        return
    for fp in HOMOGRAPHY_DIR.glob("*/H_K8-*toK8-*_method0.npy"):
        m = re.search(r"H_(K8-\d+)to(K8-\d+)_method0\.npy$", fp.name)
        if not m:
            continue
        H_MATRICES[(m.group(1), m.group(2))] = np.load(fp)


def configure_for_input(input_dir: str) -> str:
    """依資料夾切換人員／車輛拓撲、通行時間與 Homography。"""
    global ADJACENT, OVERLAP_PAIRS, MODE
    global DEFAULT_MIN_TRANSIT_HOP1, DEFAULT_MIN_TRANSIT_HOP2
    global DEFAULT_TAU_HOP1, DEFAULT_TAU_HOP2
    name = Path(input_dir).name
    if "車輛追蹤_" in name:
        ADJACENT = set(VEHICLE_ADJACENT)
        OVERLAP_PAIRS = dict(VEHICLE_OVERLAP_PAIRS)
        MODE = "vehicle"
        # 車速快、相鄰幾乎同時出現：最短通行放寬；交給 H／overlap 判斷同物件
        DEFAULT_MIN_TRANSIT_HOP1 = 0.0
        DEFAULT_MIN_TRANSIT_HOP2 = 0.0
        DEFAULT_TAU_HOP1 = 5.0
        DEFAULT_TAU_HOP2 = 12.0
        _load_h_matrices()
        return "vehicle"
    ADJACENT = set(PERSON_ADJACENT)
    OVERLAP_PAIRS = dict(PERSON_OVERLAP_PAIRS)
    MODE = "person"
    DEFAULT_MIN_TRANSIT_HOP1 = 0.0
    DEFAULT_MIN_TRANSIT_HOP2 = 0.0
    DEFAULT_TAU_HOP1 = 8.0
    DEFAULT_TAU_HOP2 = 20.0
    _load_h_matrices()
    return "person"


# 拓撲敏感度消融（預設空＝不移除任何邊）。
# 2026-07-15：僅供診斷；最終採用與否待場地圖確認。不改 PERSON_ADJACENT 預設。
PERSON_ADJACENT_EXCLUDE: set = set()  # frozenset of sorted pairs to remove at runtime


def apply_person_adjacent_exclusions(exclude: set | None = None) -> dict:
    """
    在 configure_for_input() 之後呼叫：從現行 ADJACENT 移除指定鏡頭對。
    exclude=None 時使用模組級 PERSON_ADJACENT_EXCLUDE。
    回傳生效摘要。預設不移除任何邊。
    """
    global ADJACENT, PERSON_ADJACENT_EXCLUDE
    if exclude is not None:
        PERSON_ADJACENT_EXCLUDE = {tuple(sorted(p)) for p in exclude}
    else:
        PERSON_ADJACENT_EXCLUDE = {tuple(sorted(p)) for p in PERSON_ADJACENT_EXCLUDE}
    before = set(ADJACENT)
    ADJACENT = set(before) - PERSON_ADJACENT_EXCLUDE
    removed = sorted(before - ADJACENT)
    return {
        "excluded_requested": sorted(PERSON_ADJACENT_EXCLUDE),
        "removed_from_ADJACENT": [list(p) for p in removed],
        "n_adjacent_before": len(before),
        "n_adjacent_after": len(ADJACENT),
    }


def reset_person_adjacent_exclusions() -> None:
    """還原排除旗標，並把 ADJACENT 重置為 PERSON 全量（需已是 person 模式）。"""
    global ADJACENT, PERSON_ADJACENT_EXCLUDE
    PERSON_ADJACENT_EXCLUDE = set()
    if MODE == "person":
        ADJACENT = set(PERSON_ADJACENT)


def _parse_box(box):
    if box is None:
        return None
    if isinstance(box, str):
        box = ast.literal_eval(box)
    if not box or len(box) < 4:
        return None
    return [float(x) for x in box[:4]]


def _foot_from_box(box) -> tuple:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _project_point(H: np.ndarray, x: float, y: float):
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return None
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def h_min_dist(u: Track, v: Track) -> float | None:
    """用 Homography 投影腳底，回傳最小像素距離；無 H 則 None。"""
    H = H_MATRICES.get((u.cam, v.cam))
    if H is None or not u.foots or not v.foots:
        return None
    best = 1e18
    for tu, xu, yu in u.foots:
        proj = _project_point(H, xu, yu)
        if proj is None:
            continue
        for tv, xv, yv in v.foots:
            if abs(tv - tu) > H_TIME_WINDOW:
                continue
            d = ((proj[0] - xv) ** 2 + (proj[1] - yv) ** 2) ** 0.5
            if d < best:
                best = d
    return None if best >= 1e17 else float(best)


def same_object_h(u: Track, v: Track) -> tuple:
    """(ok, min_dist_or_None)。雙向試 H；任一方向 < H_DIST_GATE 即同物件。"""
    d1 = h_min_dist(u, v)
    d2 = h_min_dist(v, u)
    cands = [d for d in (d1, d2) if d is not None]
    if not cands:
        return False, None
    d = min(cands)
    return d < H_DIST_GATE, d


def corridor_rank(cam: str) -> int:
    return VEHICLE_CORRIDOR_RANK.get(cam, 10_000)


def corridor_prefers(u: Track, v: Track) -> bool:
    """車輛走廊：實體行進 u→v 是否比時間戳更可信（允許下游較早被偵測）。"""
    if MODE != "vehicle":
        return False
    ru, rv = corridor_rank(u.cam), corridor_rank(v.cam)
    if ru >= rv or ru >= 10_000:
        return False
    # 時間窗需有重疊或小 gap
    gap = v.t_start - u.t_end
    key = tuple(sorted((u.cam, v.cam)))
    tol = OVERLAP_PAIRS.get(key, max(TOL, 10.0))
    return gap >= -tol


def _load_crop_timestamp_index(mapping_json: Path):
    """(video_id, crop_name) -> {ts, box}"""
    payload = json.loads(mapping_json.read_text(encoding="utf-8"))
    idx = {}
    for seg in payload.get("segments", []):
        video_id = seg["video_id"]
        for c in seg.get("crops", []):
            name = Path(c["crop_path"]).name
            idx[(video_id, name)] = {
                "ts": c["absolute_timestamp"],
                "box": _parse_box(c.get("box")),
            }
    return idx, payload.get("base_date")


def _iso_to_day_seconds(ts: str, base_date: str | None) -> float:
    """把 absolute_timestamp 轉成當日 wall-clock 秒（支援變動幀率；不用 frame/fps）。"""
    t = datetime.fromisoformat(ts)
    if base_date:
        # base_date 可能是 "2026-05-07" 或完整 ISO
        try:
            day0 = datetime.fromisoformat(base_date)
        except ValueError:
            day0 = datetime.fromisoformat(base_date + "T00:00:00")
        if day0.tzinfo and not t.tzinfo:
            t = t.replace(tzinfo=day0.tzinfo)
        return (t - day0).total_seconds()
    day0 = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return (t - day0).total_seconds()


def _crop_names_for_track(input_dir: Path, cam: str, track: dict) -> list | None:
    """
    只接受 filter_results 的 kept（對應 *_filtered_merged.png）。
    若無 filter 檔或 kept 為空，回傳 None → 不進候選池。
    """
    tid = int(track["track_id"])
    filt = (
        input_dir
        / "filter_results"
        / f"{input_dir.name}_{cam}_crop08_botsort09"
        / f"track_{tid}_filter_result.json"
    )
    if not filt.is_file():
        return None
    payload = json.loads(filt.read_text(encoding="utf-8"))
    kept = payload.get("kept") or []
    if not kept:
        return None
    return [Path(x).name for x in kept]


def _load_emb_cache(input_dir: Path, cam: str) -> dict:
    p = (
        input_dir
        / "filter_results"
        / f"{input_dir.name}_{cam}_crop08_botsort09"
        / "clipreid_embeddings_cache.pkl"
    )
    if not p.is_file():
        return {}
    with p.open("rb") as f:
        return pickle.load(f)


def _cache_lookup(cache: dict, crop_name: str):
    name = Path(crop_name).name
    for k, v in cache.items():
        if Path(k).name == name:
            return np.asarray(v, dtype=np.float64)
    return None


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-12)


def emb_sim(u: Track, v: Track) -> float:
    if u.emb is None or v.emb is None:
        return 0.0
    return float(np.dot(u.emb, v.emb))


def hist_emb_sim(hist_embs: list, v: Track) -> float:
    """新節點對路徑歷史平均 embedding 的相似度。"""
    if v.emb is None or not hist_embs:
        return 0.0
    hist = _l2_normalize(np.mean(np.stack(hist_embs, axis=0), axis=0))
    return float(np.dot(hist, v.emb))


def load_tracks(input_dir: str) -> list:
    """
    讀 query_filter_merge/{人員追蹤_YYYYMMDD}/ 下各鏡頭 *_merged.json。

    實際格式：
      matched_tracks[].track_id / similarity / crop_paths / start_time / end_time
      逐 frame 時間戳在 ../output/{dataset}_crop_time_mapping.json
        → crops[].absolute_timestamp（勿用 frame / fps）
      若有 filter_results/.../track_*_filter_result.json 的 kept，只用 kept crops
      平均 embedding 來自同目錄 clipreid_embeddings_cache.pkl
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise SystemExit(f"找不到資料夾：{input_dir}")

    dataset = _infer_dataset_tag(input_dir)
    mapping_json = OUTPUT_ROOT / f"{dataset}_crop_time_mapping.json"
    if not mapping_json.is_file():
        mapping_json = DEFAULT_MAPPING
    if not mapping_json.is_file():
        raise SystemExit(
            f"找不到 mapping：{mapping_json}\n"
            f"請先執行：python3 backend/scripts/export_category_crop_time_mapping.py "
            f"--category {dataset} --base_date YYYY-MM-DD --labels person"
        )

    ts_index, base_date = _load_crop_timestamp_index(mapping_json)

    tracks = []
    skipped_sim = 0
    skipped_nofilt = 0
    merged_files = sorted(root.glob("*_merged.json"))
    if not merged_files:
        raise SystemExit(f"在 {input_dir} 找不到 *_merged.json")

    cache_by_cam = {}
    for fp in merged_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        video_id = data.get("video_id") or ""
        m = CAMERA_RE.search(video_id) or CAMERA_RE.search(fp.name)
        if not m:
            continue
        cam = m.group(1)
        if cam not in cache_by_cam:
            cache_by_cam[cam] = _load_emb_cache(root, cam)
        cache = cache_by_cam[cam]

        for tr in data.get("matched_tracks", []):
            sim = float(tr.get("similarity", 0.0))
            if sim < SIM_MIN:
                skipped_sim += 1
                continue
            tid = int(tr["track_id"])
            names = _crop_names_for_track(root, cam, tr)
            if names is None:
                skipped_nofilt += 1
                continue
            stamps = []
            embs = []
            foots = []
            for name in names:
                key = (video_id, name)
                rec = ts_index.get(key)
                if rec is None:
                    continue
                stamps.append(rec["ts"])
                e = _cache_lookup(cache, name)
                if e is not None:
                    embs.append(e)
                if rec.get("box") is not None:
                    fx, fy = _foot_from_box(rec["box"])
                    foots.append((_iso_to_day_seconds(rec["ts"], base_date), fx, fy))
            if not stamps:
                raise SystemExit(
                    f"{cam} track_{tid} 的 crops 在 mapping 找不到 absolute_timestamp"
                    f"（video_id={video_id}, n_names={len(names)}）"
                )
            if not embs:
                print(f"  警告：{cam} track_{tid} 無 embedding，跳過"
                      f"（可能缺 filter kept / cache）")
                continue
            secs = [_iso_to_day_seconds(s, base_date) for s in stamps]
            tracks.append(Track(
                tid=f"{cam}_{tid}",
                cam=cam,
                t_start=float(min(secs)),
                t_end=float(max(secs)),
                sim=sim,
                emb=_l2_normalize(np.mean(np.stack(embs, axis=0), axis=0)),
                foots=sorted(foots),
                meta={
                    "video_id": video_id,
                    "track_label": tr.get("track_label"),
                    "source_track_ids": tr.get("source_track_ids"),
                    "n_crops_used": len(names),
                    "n_embs": len(embs),
                    "iso_start": min(stamps),
                    "iso_end": max(stamps),
                },
            ))

    if not tracks:
        raise SystemExit(
            f"在 {input_dir} 沒有讀到任何 track（SIM_MIN={SIM_MIN}，"
            f"因 sim 過低跳過 {skipped_sim} 條，因無 filter kept 跳過 {skipped_nofilt} 條）"
        )
    print(f"  （SIM_MIN={SIM_MIN} 跳過 {skipped_sim} 條低 sim；"
          f"無 filter kept 跳過 {skipped_nofilt} 條＝只用 *_filtered_merged 層）")
    return sorted(tracks, key=lambda t: t.t_start)

# ============================================================
# 三、硬規則：邊存不存在
# ============================================================

def hop_count(cam_u: str, cam_v: str):
    if cam_u == cam_v:
        return 0
    if tuple(sorted((cam_u, cam_v))) in ADJACENT:
        return 1
    # 跳一支：存在共同鄰居
    nb_u = {b if a == cam_u else a for a, b in ADJACENT if cam_u in (a, b)}
    nb_v = {b if a == cam_v else a for a, b in ADJACENT if cam_v in (a, b)}
    if nb_u & nb_v:
        return 2
    return None  # 拓撲到不了

def min_transit(cam_u, cam_v, hop, h_ok=False):
    key = tuple(sorted((cam_u, cam_v)))
    if h_ok or key in OVERLAP_PAIRS:
        return 0.0          # H 同物件／視野重疊：無通行時間下界
    if key in MIN_TRANSIT:
        return MIN_TRANSIT[key]
    return {0: 0.0, 1: DEFAULT_MIN_TRANSIT_HOP1, 2: DEFAULT_MIN_TRANSIT_HOP2}[hop]

def tau(cam_u, cam_v, hop):
    key = tuple(sorted((cam_u, cam_v)))
    if key in TAU:
        return TAU[key]
    return {0: DEFAULT_TAU_HOP0, 1: DEFAULT_TAU_HOP1, 2: DEFAULT_TAU_HOP2}[hop]

def edge_check(u: Track, v: Track):
    """回傳 (ok, reason, dt, hop, emb, h_dist)——reason 留給診斷用"""
    dt_raw = v.t_start - u.t_end
    key = tuple(sorted((u.cam, v.cam)))
    tol = OVERLAP_PAIRS.get(key, TOL)

    h_ok, h_dist = same_object_h(u, v)
    # 車輛走廊：H 同物件或相鄰走廊，允許「下游較早被偵測」的方向
    if dt_raw < -tol:
        if not (h_ok or corridor_prefers(u, v)):
            return False, f"時間順序（重疊 {-dt_raw:.1f}s 超過容許 {tol:.1f}s）", dt_raw, None, 0.0, h_dist
    dt = max(dt_raw, 0.0)  # 重疊交接視為無縫接手

    hop = hop_count(u.cam, v.cam)
    # 有 H 同物件：只要求相鄰（或已在 ADJACENT）；不靠 hop2 硬湊遠鏡頭
    if hop is None:
        if h_ok and tuple(sorted((u.cam, v.cam))) in ADJACENT:
            hop = 1
        else:
            return False, "拓撲不可達", dt, hop, 0.0, h_dist

    mt = min_transit(u.cam, v.cam, hop, h_ok=h_ok)
    if dt < mt:
        return False, f"瞬移（dt={dt:.1f}s < 最短通行 {mt:.1f}s）", dt, hop, 0.0, h_dist
    if dt > DT_MAX:
        return False, f"斷太久（dt={dt:.1f}s > DT_MAX）", dt, hop, 0.0, h_dist

    emb = emb_sim(u, v)
    # H 已確認同物件時，外觀門檻略放寬（視角差大）
    emb_need = EMB_EDGE_MIN - 0.02 if h_ok else EMB_EDGE_MIN
    if emb < emb_need:
        return False, f"外觀不像（emb={emb:.3f} < {emb_need}）", dt, hop, emb, h_dist

    # 車輛：無 H 的相鄰邊仍可靠 emb；有 H 則標記於 reason 空字串外的回傳
    return True, "", dt, hop, emb, h_dist


ASE_PARENT = REPO_ROOT.parent


def _resolve_crop_path(p: str | Path) -> Path | None:
    p = Path(p)
    cands = [
        p,
        Path(str(p).replace("/home/M133040024/ASE", str(ASE_PARENT))),
        OUTPUT_ROOT / Path(*Path(str(p).replace("\\", "/")).parts[-2:]),
    ]
    for cand in cands:
        if cand.is_file():
            return cand
    return None


def _crop_paths_for_track(merge_dir: Path, cam: str, tid: int) -> tuple[dict, list[Path]]:
    fp = next(merge_dir.glob(f"*_{cam}_*_merged.json"))
    data = json.loads(fp.read_text(encoding="utf-8"))
    tr = next(t for t in data["matched_tracks"] if int(t["track_id"]) == tid)
    crop_dir = OUTPUT_ROOT / f"{merge_dir.name}_{cam}"
    by_name: dict[str, Path] = {}
    for cp in tr.get("crop_paths", []):
        rp = _resolve_crop_path(cp)
        if rp:
            by_name[rp.name] = rp
        else:
            by_name[Path(cp).name] = crop_dir / Path(cp).name

    filt = (
        merge_dir
        / "filter_results"
        / f"{merge_dir.name}_{cam}_crop08_botsort09"
        / f"track_{tid}_filter_result.json"
    )
    names: list[str] = []
    if filt.is_file():
        kept = json.loads(filt.read_text(encoding="utf-8")).get("kept") or []
        names = [Path(x).name for x in kept]
    if not names:
        names = [Path(p).name for p in tr.get("crop_paths", [])]

    crops: list[Path] = []
    for name in names:
        if name in by_name and by_name[name].is_file():
            crops.append(by_name[name])
        elif (crop_dir / name).is_file():
            crops.append(crop_dir / name)
    return tr, crops


def _pick_rep_crop(crops: list[Path]) -> Path | None:
    if not crops:
        return None
    return crops[len(crops) // 2]


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (236, 236, 236))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def _font(size: int):
    for name in (
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ):
        p = Path(name)
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def render_top1_collage(
    merge_dir: Path,
    top: dict,
    out_png: Path,
    title_prefix: str = "path_enum Top-1",
) -> Path:
    tids = top["tids"]
    edges = top.get("edges") or []
    n = len(tids)
    is_vehicle = "車輛" in merge_dir.name
    if is_vehicle:
        cell_w, cell_h, arrow_w = 150, 210, 54
        tw, th = 134, 100
    else:
        cell_w, cell_h, arrow_w = 180, 320, 78
        tw, th = 160, 220

    margin, title_h, foot_h = 16, 42, 48
    width = margin * 2 + n * cell_w + max(0, n - 1) * arrow_w
    height = title_h + cell_h + foot_h + margin
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_t = _font(18)
    font_s = _font(13)
    font_xs = _font(11)

    title = (
        f"{title_prefix}  score={top['score']:.2f}  "
        f"(SIM>={SIM_MIN} emb>={EMB_EDGE_MIN})  [{merge_dir.name}]"
    )
    draw.text((margin, 10), title, fill=(20, 20, 20), font=font_t)

    y0 = title_h
    for i, tid in enumerate(tids):
        cam, tid_s = tid.rsplit("_", 1)
        tid_i = int(tid_s)
        tr, crops = _crop_paths_for_track(merge_dir, cam, tid_i)
        rep = _pick_rep_crop(crops)
        x = margin + i * (cell_w + arrow_w)

        draw.rectangle([x, y0, x + cell_w - 1, y0 + cell_h - 1], outline=(40, 40, 40), width=2)
        draw.text((x + 8, y0 + 6), tid, fill=(0, 0, 0), font=font_s)

        if rep is not None:
            timg = _thumb(rep, (tw, th))
            img.paste(timg, (x + (cell_w - tw) // 2, y0 + 28))
        else:
            draw.text((x + 20, y0 + 100), "(no crop)", fill=(160, 0, 0), font=font_s)

        sim = float(tr.get("similarity", 0.0))
        n_crops = len(crops) if crops else int(tr.get("n_crops", 0))
        draw.text((x + 8, y0 + cell_h - 42), f"sim={sim:.3f}", fill=(30, 30, 30), font=font_xs)
        draw.text((x + 8, y0 + cell_h - 24), f"n={n_crops}", fill=(80, 80, 80), font=font_xs)

        if i < len(edges):
            e = edges[i]
            ax0 = x + cell_w
            ax1 = ax0 + arrow_w
            mid_y = y0 + cell_h // 2
            draw.line([(ax0 + 6, mid_y), (ax1 - 10, mid_y)], fill=(0, 0, 0), width=2)
            draw.polygon(
                [(ax1 - 10, mid_y - 6), (ax1 - 2, mid_y), (ax1 - 10, mid_y + 6)],
                fill=(0, 0, 0),
            )
            hop = e.get("hop")
            dt = e.get("dt", 0.0)
            emb = e.get("emb", 0.0)
            sc = e.get("score", 0.0)
            sc_color = (0, 128, 0) if sc >= 0 else (180, 0, 0)
            draw.text((ax0 + 4, mid_y - 38), f"hop={hop}", fill=(40, 40, 40), font=font_xs)
            draw.text((ax0 + 4, mid_y - 22), f"dt={dt:.1f}s", fill=(40, 40, 40), font=font_xs)
            draw.text((ax0 + 4, mid_y + 8), f"emb={emb:.3f}", fill=(40, 40, 40), font=font_xs)
            draw.text((ax0 + 4, mid_y + 24), f"{sc:+.2f}", fill=sc_color, font=font_s)

    draw.text((margin, title_h + cell_h + 12), "  ->  ".join(tids), fill=(30, 30, 30), font=font_xs)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def write_suspect_coexistence_txt(path: Path, suspects: list[dict]) -> Path | None:
    if not suspects:
        return None
    lines = [
        "# suspect_coexistence — 名單制 OVERLAP 候選，emb 低於 COEXISTENCE_OVERLAP_EMB_MIN",
        f"# threshold={COEXISTENCE_OVERLAP_EMB_MIN} (emb|same μ−3σ = 0.917−3×0.023)",
        "",
    ]
    for s in suspects:
        lines.append(
            f"{s['a']} <-> {s['b']}  emb={s['emb_ab']:.6f}  "
            f"overlap={s['overlap_sec']:.3f}s  cams={s.get('cams','')}  ({s.get('note','')})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def supernode_member_pairs(super_report: dict, by_tid: dict[str, Track]) -> list[dict]:
    rows = []
    for sn in super_report.get("supernodes") or []:
        members = sn.get("members") or []
        if len(members) < 2:
            continue
        ts = [by_tid[t] for t in members if t in by_tid]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                u, v = ts[i], ts[j]
                rows.append(
                    {
                        "supernode": sn.get("sid"),
                        "members": members,
                        "a": u.tid,
                        "b": v.tid,
                        "emb_ab": float(emb_sim(u, v)),
                        "path": classify_coexistence_merge_path(u, v),
                        "h_dist": _h_projection_dist(u, v),
                        "overlap_sec": float(max(_time_overlap_sec(u, v), 0.0)),
                    }
                )
    return rows


def build_supernode_comparison_table(
    before_report: dict,
    after_report: dict,
    by_tid: dict[str, Track],
) -> list[dict]:
    before_pairs = {
        tuple(sorted((r["a"], r["b"]))): r
        for r in supernode_member_pairs(before_report, by_tid)
    }
    after_pairs = {
        tuple(sorted((r["a"], r["b"]))): r
        for r in supernode_member_pairs(after_report, by_tid)
    }
    keys = sorted(set(before_pairs) | set(after_pairs))
    rows = []
    for key in keys:
        b = before_pairs.get(key)
        a = after_pairs.get(key)
        ref = a or b
        affected = False
        if b and not a:
            affected = True
            status = "removed"
        elif a and not b:
            affected = True
            status = "added"
        elif b and a and (
            b.get("supernode") != a.get("supernode")
            or set(b.get("members") or []) != set(a.get("members") or [])
        ):
            affected = True
            status = "changed"
        else:
            status = "same"
        rows.append(
            {
                "status": status,
                "a": ref["a"],
                "b": ref["b"],
                "emb_ab": ref["emb_ab"],
                "path": ref["path"],
                "before_supernode": b.get("supernode") if b else "—",
                "after_supernode": a.get("supernode") if a else "—",
                "before_members": b.get("members") if b else [],
                "after_members": a.get("members") if a else [],
                "affected_this_round": affected,
            }
        )
    return rows


def render_supernodes_collage(
    merge_dir: Path,
    super_report: dict,
    by_tid: dict[str, Track],
    out_png: Path,
    *,
    title: str,
) -> Path | None:
    multi = [s for s in (super_report.get("supernodes") or []) if len(s.get("members") or []) > 1]
    if not multi:
        return None
    tw, th = 120, 160
    cell_w, cell_h = 280, th + 72
    margin, title_h, gap = 16, 36, 12
    width = margin * 2 + cell_w
    height = margin * 2 + title_h + len(multi) * (cell_h + gap) - gap
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_t = _font(16)
    font_s = _font(11)
    draw.text((margin, 8), title, fill=(20, 20, 20), font=font_t)
    y = title_h
    for sn in multi:
        members = sn.get("members") or []
        ts = [by_tid[t] for t in members if t in by_tid]
        x = margin
        nmem = max(len(ts), 1)
        sub_w = (cell_w - 8 * (nmem - 1)) // nmem if nmem else cell_w
        for idx, t in enumerate(ts):
            cam, tid_s = t.tid.rsplit("_", 1)
            try:
                _, crops = _crop_paths_for_track(merge_dir, cam, int(tid_s))
                crop = _pick_rep_crop(crops)
            except Exception:
                crop = None
            thumb = _thumb(crop, (sub_w - 8, th)) if crop else Image.new("RGB", (sub_w - 8, th), (220, 220, 220))
            img.paste(thumb, (x + 4, y + 4))
            draw.text(
                (x + 4, y + th + 6),
                f"{t.tid}\n{t.cam} {t.t_start:.1f}-{t.t_end:.1f}s",
                fill=(30, 30, 30),
                font=font_s,
            )
            x += sub_w + 8
        if len(ts) >= 2:
            emb_lines = []
            path_lines = []
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    u, v = ts[i], ts[j]
                    emb_lines.append(f"{u.tid}<->{v.tid} emb={emb_sim(u,v):.3f}")
                    path_lines.append(classify_coexistence_merge_path(u, v))
            draw.text(
                (margin + 4, y + th + 34),
                " | ".join(emb_lines) + f"  path={','.join(sorted(set(path_lines)))}",
                fill=(60, 60, 60),
                font=font_s,
            )
        y += cell_h + gap
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png



# ============================================================
# LLR / 分段排名（原 track_path.py）
# ============================================================

PDF_FLOOR = 1e-12
SHRINK_K = 10.0
HANDOFF_DT_MAX = 2.0
SUPER_OVERLAP_MIN = 0.5
PRIOR_DT_SIGMA = 1.0  # PRIOR-WEAK（原 0.5）

NODE_EVIDENCE_NOTE = (
    "節點證據 = w·ln(P(sim|GT)/P(sim|nonGT))，w=n/(n+10)；"
    "需 calibration 含 sim_gt / sim_nongt"
)
PRIOR_WEAK_NOTE = (
    f"PRIOR-WEAK：dt 退先驗時 LogNormal(σ={PRIOR_DT_SIGMA})，tau 仍佔位；收縮 w=1"
)
SEGMENT_RANK_NOTE = (
    "2026-07-15：分段假設參與排名（總分=各段分數和）；"
    "同一 Softmax 池；假設內部共存矛盾（非 OVERLAP/非 ADJACENT 同時出現）作廢；"
    "計分公式不動"
)
# 自高分極大路徑取前 K 條當作多段假設的 seed（其餘仍以單段假設進排名池）
DEFAULT_SEGMENT_SEED_TOP_K = 400
DEFAULT_MAX_HYP_SEGMENTS = 8


# ---------------------------------------------------------------------------
# Calibration / LLR densities
# ---------------------------------------------------------------------------

def load_calibration(path: Path) -> dict:
    with path.open("rb") as f:
        calib = pickle.load(f)
    # 2026-07-16：exp1 採納（N=13），固定幾何分數 dH|same 的統計出身。
    dh_same = dict(calib.get("dh_same") or {})
    dh_same.update(
        {
            "family": "halfnorm",
            "sigma": float(DH_SAME_SIGMA),
            "n": int(DH_SAME_N),
            "source": "exp1_h_projection_distance_2026-07-16",
        }
    )
    dh_same["shrink_w"] = float(shrink_weight(dh_same.get("n")))
    calib["dh_same"] = dh_same
    return calib


def _pdf(dist: dict, x: float) -> float:
    fam = dist["family"]
    if fam == "norm":
        p = float(stats.norm.pdf(x, loc=dist["mu"], scale=dist["sigma"]))
    elif fam == "lognorm":
        xx = max(float(x), 1e-6)
        p = float(stats.lognorm.pdf(xx, dist["sigma"], loc=0.0, scale=math.exp(dist["mu"])))
    elif fam == "halfnorm":
        p = float(stats.halfnorm.pdf(max(float(x), 0.0), loc=0.0, scale=dist["sigma"]))
    elif fam == "uniform":
        lo, hi = float(dist["low"]), float(dist["high"])
        if hi <= lo:
            p = 0.0
        elif lo <= x <= hi:
            p = 1.0 / (hi - lo)
        else:
            p = 0.0
    else:
        raise ValueError(f"unknown family: {fam}")
    return max(p, PDF_FLOOR)


def llr(dist_same: dict, dist_diff: dict, x: float) -> float:
    return math.log(_pdf(dist_same, x)) - math.log(_pdf(dist_diff, x))


def shrink_weight(n: int | None, *, force_full: bool = False) -> float:
    if force_full:
        return 1.0
    nn = max(int(n or 0), 0)
    return float(nn) / float(nn + SHRINK_K)


def node_evidence(sim: float, calib: dict | None) -> dict:
    """回傳 {raw, w, score, enabled}。無 sim_gt/sim_nongt 時 score=0。"""
    if not calib or "sim_gt" not in calib or "sim_nongt" not in calib:
        return {"raw": 0.0, "w": 0.0, "score": 0.0, "enabled": False, "sim": float(sim)}
    raw = llr(calib["sim_gt"], calib["sim_nongt"], float(sim))
    # 收縮用 GT 樣本數
    w = shrink_weight(calib["sim_gt"].get("n"))
    return {"raw": raw, "w": w, "score": w * raw, "enabled": True, "sim": float(sim)}


def _dt_same_dist(calib: dict, cam_u: str, cam_v: str) -> dict:
    key = "|".join(sorted((cam_u, cam_v)))
    d = calib["dt_same_by_pair"].get(key)
    if d is not None:
        # 先驗檔若仍寫 sigma=0.5，執行時改為 PRIOR-WEAK sigma
        if d.get("prior") and float(d.get("sigma", PRIOR_DT_SIGMA)) < PRIOR_DT_SIGMA - 1e-9:
            d = dict(d)
            d["sigma"] = PRIOR_DT_SIGMA
            d["prior_weak"] = True
        return d
    hop = hop_count(cam_u, cam_v)
    if hop is None:
        hop = 1
    tau0 = float(tau(cam_u, cam_v, hop))
    return {
        "family": "lognorm",
        "mu": math.log(max(tau0, 1e-3)),
        "sigma": PRIOR_DT_SIGMA,
        "prior": True,
        "prior_weak": True,
        "n": 0,
        "tau": tau0,
    }


def _is_tau_placeholder(tau0: float | None) -> bool:
    if tau0 is None:
        return True
    return abs(float(tau0) - float(DEFAULT_TAU_HOP1)) < 1e-9


def _dt_weight_and_note(dist_same: dict) -> tuple[float, bool, str | None]:
    n = int(dist_same.get("n") or 0)
    is_prior = bool(dist_same.get("prior"))
    tau0 = dist_same.get("tau")
    if not is_prior and n >= 20:
        return 1.0, False, None
    note = "PRIOR-WEAK"
    if _is_tau_placeholder(tau0 if tau0 is not None else math.exp(float(dist_same.get("mu", 0.0)))):
        note = f"PRIOR-WEAK (tau placeholder=8.0, sigma={PRIOR_DT_SIGMA})"
    return 1.0, True, note


def is_handoff_edge(u: Track, v: Track, dt: float, h_dist: float | None) -> bool:
    if float(dt) > HANDOFF_DT_MAX:
        return False
    key = tuple(sorted((u.cam, v.cam)))
    if key in OVERLAP_PAIRS:
        return True
    return h_dist is not None


def edge_llr(
    u: Track,
    v: Track,
    dt: float,
    emb: float,
    h_dist: float | None,
    calib: dict,
    *,
    dt_scoring: bool = True,
    transition_prior: bool = False,
) -> dict:
    """
    dt_scoring=False：transit 邊 LLR_dt 記 0、顯示 "removed"（硬規則不受影響）。
    transition_prior=True：每邊加 ln(p_edge)（見 calib['transition_prior']）。
    """
    dt_dist = _dt_same_dist(calib, u.cam, v.cam)
    raw_dt = llr(dt_dist, calib["dt_diff"], dt)
    raw_emb = llr(calib["emb_same"], calib["emb_diff"], emb)
    raw_dh = None
    if h_dist is not None:
        raw_dh = llr(calib["dh_same"], calib["dh_diff"], h_dist)

    handoff = is_handoff_edge(u, v, dt, h_dist)
    dt_model = "handoff" if handoff else "transit"
    w_emb = shrink_weight(calib["emb_same"].get("n"))
    w_dh = shrink_weight(calib["dh_same"].get("n")) if raw_dh is not None else None
    w_dt, dt_prior_physical, dt_note = _dt_weight_and_note(dt_dist)

    trans_llr = 0.0
    if transition_prior:
        tp = calib.get("transition_prior") or {}
        p_edge = tp.get("p_edge")
        if p_edge is not None and float(p_edge) > 0.0:
            trans_llr = math.log(float(p_edge))

    if handoff:
        eff_dt = 0.0
        llr_dt_display = None
        raw_total_for_edge = raw_emb + (raw_dh if raw_dh is not None else 0.0)
        eff_emb = w_emb * raw_emb
        eff_dh = (w_dh * raw_dh) if (raw_dh is not None and w_dh is not None) else None
        eff_total = eff_emb + (eff_dh if eff_dh is not None else 0.0) + trans_llr
        dt_note_out = None
    else:
        if dt_scoring:
            eff_dt = w_dt * raw_dt
            llr_dt_display = eff_dt
            dt_note_out = dt_note
            raw_total_for_edge = raw_dt + raw_emb + (raw_dh if raw_dh is not None else 0.0)
        else:
            # 軟證據停用；仍保留 raw 供對照。依據：tau 無實測來源（2026-07-15）
            eff_dt = 0.0
            llr_dt_display = "removed"
            dt_note_out = "dt-scoring=off (2026-07-15: tau 無實測，軟 dt 停用)"
            raw_total_for_edge = raw_emb + (raw_dh if raw_dh is not None else 0.0)
        eff_emb = w_emb * raw_emb
        eff_dh = (w_dh * raw_dh) if (raw_dh is not None and w_dh is not None) else None
        eff_total = eff_dt + eff_emb + (eff_dh if eff_dh is not None else 0.0) + trans_llr

    denom = raw_total_for_edge if abs(raw_total_for_edge) > 1e-12 else 1.0
    # w 不含 transition 常數項，避免除法扭曲
    core = eff_total - trans_llr
    w_edge = core / denom if abs(denom) > 1e-12 else 1.0
    return {
        "from": u.tid,
        "to": v.tid,
        "dt": dt,
        "emb": emb,
        "h_dist": h_dist,
        "dt_model": dt_model,
        "LLR_dt": llr_dt_display,
        "LLR_dt_raw": raw_dt,
        "w_dt": None if handoff or not dt_scoring else w_dt,
        "LLR_emb": eff_emb,
        "LLR_emb_raw": raw_emb,
        "w_emb": w_emb,
        "LLR_dH": eff_dh,
        "LLR_dH_raw": raw_dh,
        "w_dH": w_dh,
        "LLR_transition": trans_llr if transition_prior else None,
        "raw_LLR": raw_total_for_edge,
        "w": w_edge,
        "effective_LLR": eff_total,
        "score": eff_total,
        "dt_prior_physical": bool(dt_prior_physical) and not handoff and dt_scoring,
        "dt_note": dt_note_out,
        "dt_scoring": bool(dt_scoring),
        "transition_prior": bool(transition_prior),
    }


# ---------------------------------------------------------------------------
# 修正二：共存超節點
# ---------------------------------------------------------------------------

@dataclass
class SuperNode:
    sid: str
    members: list  # Track, sorted by t_start
    emb: np.ndarray = field(default=None)
    sim: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    cams: list = field(default_factory=list)

    @property
    def tids(self) -> list[str]:
        return [t.tid for t in self.members]

    @property
    def label(self) -> str:
        if len(self.members) == 1:
            return self.members[0].tid
        return "{" + ",".join(self.tids) + "}"


def _time_overlap_sec(a: Track, b: Track) -> float:
    return min(a.t_end, b.t_end) - max(a.t_start, b.t_start)


def _coexistence_time_ok(u: Track, v: Track) -> tuple[bool, float, str]:
    """
    一般：重疊 ≥ 0.5s。
    退化：任一方時長 ≈0 且時間點落在對方 [t_start,t_end] 內
    （如 K8-22_19 單幀）→ 視為時間共存，否則 OVERLAP/H 對永遠併不進去。
    """
    ov = _time_overlap_sec(u, v)
    if ov >= SUPER_OVERLAP_MIN:
        return True, ov, f"overlap={ov:.2f}s"
    if ov < -1e-9:
        return False, ov, f"overlap={ov:.2f}s"
    # ov ∈ [0, 0.5)：檢查單幀／點時刻落在對方區間
    for short, long in ((u, v), (v, u)):
        dur = float(short.t_end - short.t_start)
        if dur <= 1e-6:
            t0 = float(short.t_start)
            if long.t_start - 1e-9 <= t0 <= long.t_end + 1e-9:
                return True, max(ov, 0.0), f"point_in_span t={t0:.2f} host={long.tid}"
    return False, ov, f"overlap={ov:.2f}s"


def _h_projection_dist(u: Track, v: Track) -> float | None:
    """回傳 H 投影最小距離；無 H 矩陣或無 foot 則 None。"""
    d1 = h_min_dist(u, v)
    d2 = h_min_dist(v, u)
    cands = [d for d in (d1, d2) if d is not None]
    return float(min(cands)) if cands else None


def classify_coexistence_merge_path(u: Track, v: Track) -> str:
    """回報判定路徑：幾何制 / 名單制 / 無。"""
    key = tuple(sorted((u.cam, v.cam)))
    d = _h_projection_dist(u, v)
    if d is not None and float(d) < SUPER_DH_MAX:
        return "幾何制"
    if key in OVERLAP_PAIRS and d is None:
        return "名單制"
    return "—"


def coexistence_merge(
    u: Track,
    v: Track,
    *,
    overlap_emb_min: float | None = COEXISTENCE_OVERLAP_EMB_MIN,
) -> tuple[bool, str, dict | None]:
    """
    同鏡跳過；時間共存且（幾何制 H×dH<95 或 名單制 OVERLAP+emb 底線）。
    第三項：名單制 emb 不足時的 suspect 紀錄（供 suspect_coexistence.txt）。
    """
    if u.cam == v.cam:
        return False, "same_cam_skip", None
    tok, ov, tnote = _coexistence_time_ok(u, v)
    if not tok:
        return False, f"no_coexist_time ({tnote})", None
    key = tuple(sorted((u.cam, v.cam)))
    d = _h_projection_dist(u, v)
    # 幾何制：位置排他性，刻意不驗外觀（2026-07-20 註記維持）。
    if d is not None and float(d) < SUPER_DH_MAX:
        return True, f"H dH={d:.1f}px {tnote}", None
    # 名單制：OVERLAP_PAIRS 且無 H 可用 → emb 底線。
    if key in OVERLAP_PAIRS and d is None:
        emb = emb_sim(u, v)
        if overlap_emb_min is not None and emb + 1e-12 < float(overlap_emb_min):
            suspect = {
                "a": u.tid,
                "b": v.tid,
                "emb_ab": float(emb),
                "overlap_sec": float(max(ov, 0.0)),
                "cams": f"{u.cam}↔{v.cam}",
                "threshold": float(overlap_emb_min),
                "note": tnote,
            }
            return (
                False,
                f"OVERLAP_emb_low emb={emb:.3f}<{overlap_emb_min:.3f} ({tnote})",
                suspect,
            )
        return True, f"OVERLAP_PAIRS {tnote}", None
    return False, f"no_overlap_or_H ({tnote})", None


def build_supernodes(
    tracks: list,
    *,
    overlap_emb_min: float | None = COEXISTENCE_OVERLAP_EMB_MIN,
) -> tuple[list[SuperNode], dict]:
    n = len(tracks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merge_log = []
    suspect_coexistence = []
    for i, j in itertools.combinations(range(n), 2):
        ok, reason, suspect = coexistence_merge(
            tracks[i], tracks[j], overlap_emb_min=overlap_emb_min
        )
        if suspect is not None:
            suspect_coexistence.append(suspect)
        if ok:
            union(i, j)
            merge_log.append(
                {
                    "a": tracks[i].tid,
                    "b": tracks[j].tid,
                    "reason": reason,
                    "path": classify_coexistence_merge_path(tracks[i], tracks[j]),
                }
            )

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(tracks[i])

    supers: list[SuperNode] = []
    for k, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(t.t_start for t in kv[1]))):
        members = sorted(members, key=lambda t: (t.t_start, t.t_end, t.tid))
        embs = np.stack([t.emb for t in members], axis=0)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        sn = SuperNode(
            sid=f"SN{k}",
            members=members,
            emb=mean_emb,
            sim=float(np.mean([t.sim for t in members])),
            t_start=float(min(t.t_start for t in members)),
            t_end=float(max(t.t_end for t in members)),
            cams=sorted({t.cam for t in members}),
        )
        supers.append(sn)

    report = {
        "n_tracks": n,
        "n_supernodes": len(supers),
        "n_merged_pairs": len(merge_log),
        "merge_log": merge_log,
        "suspect_coexistence": suspect_coexistence,
        "coexistence_overlap_emb_min": overlap_emb_min,
        "supernodes": [
            {
                "sid": s.sid,
                "members": s.tids,
                "cams": s.cams,
                "t_start": s.t_start,
                "t_end": s.t_end,
                "sim_mean": s.sim,
                "multi": len(s.members) > 1,
            }
            for s in supers
        ],
        "multi_only": [
            s.tids for s in supers if len(s.members) > 1
        ],
    }
    return supers, report


def verify_mislabel_not_in_gt_super(
    super_report: dict, gt_tids: set[str], mislabel: str = "K8-08_43"
) -> dict:
    bad = []
    for sn in super_report["supernodes"]:
        mem = set(sn["members"])
        if mislabel in mem and (mem & gt_tids):
            bad.append(sn)
    return {
        "mislabel": mislabel,
        "merged_with_gt": bool(bad),
        "offending_supernodes": bad,
        "note": (
            "23↔08 無 OVERLAP、無 H，規則上不應與 23_8 合併；"
            "若併入其他 GT 超節點亦屬失敗。"
        ),
    }


def _best_member_edge(sa: SuperNode, sb: SuperNode):
    """
    超節點間合法邊：
      - dt 一律用聯集端點：dt = max(sb.t_start − sa.t_end, 0)
        （MIN_TRANSIT / DT_MAX / 時間順序容許 / handoff 皆用此 dt）
      - emb 取成員對最大值；hop／H 依該成員對鏡頭
    回傳 (u, v, dt_union, hop, emb, h_dist) 或 (None, rejects)。
    """
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            if v.t_end < u.t_start - DT_MAX and sb.t_end < sa.t_start - DT_MAX:
                continue
            key = tuple(sorted((u.cam, v.cam)))
            tol = OVERLAP_PAIRS.get(key, TOL)
            h_ok, h_dist = same_object_h(u, v)

            # 時間順序：以聯集 dt_raw 為準
            if dt_raw < -tol:
                if not (h_ok or corridor_prefers(u, v)):
                    rejects.append(
                        (
                            u.tid,
                            v.tid,
                            f"時間順序（聯集重疊 {-dt_raw:.1f}s 超過容許 {tol:.1f}s）",
                        )
                    )
                    continue

            hop = hop_count(u.cam, v.cam)
            if hop is None:
                if h_ok and tuple(sorted((u.cam, v.cam))) in ADJACENT:
                    hop = 1
                else:
                    rejects.append((u.tid, v.tid, "拓撲不可達"))
                    continue

            mt = min_transit(u.cam, v.cam, hop, h_ok=h_ok)
            if dt < mt:
                rejects.append(
                    (u.tid, v.tid, f"瞬移（聯集dt={dt:.1f}s < 最短通行 {mt:.1f}s）")
                )
                continue
            if dt > DT_MAX:
                rejects.append(
                    (u.tid, v.tid, f"斷太久（聯集dt={dt:.1f}s > DT_MAX）")
                )
                continue

            emb = emb_sim(u, v)
            emb_need = EMB_EDGE_MIN - 0.02 if h_ok else EMB_EDGE_MIN
            if emb < emb_need:
                rejects.append(
                    (u.tid, v.tid, f"外觀不像（emb={emb:.3f} < {emb_need}）")
                )
                continue

            cand = (u, v, dt, hop, emb, h_dist)
            if best is None or emb > best[4]:
                best = cand
    return best, rejects


# emb=0.80 時合法邊可 >200，全量 DFS 不可行；改 beam。
# 邊數 ≤ 此值時仍全枚舉（含前綴，與舊行為一致）。
FULL_ENUM_EDGE_CAP = 80
DEFAULT_BEAM_WIDTH = 64
DEFAULT_BEAM_MAX_LEAVES = 5000


def _build_nodes(tracks: list, use_supernode: bool):
    if use_supernode:
        return build_supernodes(tracks)
    nodes = [
        SuperNode(
            sid=t.tid,
            members=[t],
            emb=t.emb,
            sim=t.sim,
            t_start=t.t_start,
            t_end=t.t_end,
            cams=[t.cam],
        )
        for t in tracks
    ]
    super_report = {
        "n_tracks": len(tracks),
        "n_supernodes": len(nodes),
        "n_merged_pairs": 0,
        "merge_log": [],
        "supernodes": [
            {
                "sid": s.sid,
                "members": s.tids,
                "cams": s.cams,
                "t_start": s.t_start,
                "t_end": s.t_end,
                "sim_mean": s.sim,
                "multi": False,
            }
            for s in nodes
        ],
        "multi_only": [],
        "disabled": True,
    }
    return nodes, super_report


def _build_succ(nodes: list) -> tuple[list, list, int]:
    n = len(nodes)
    succ = [[] for _ in range(n)]
    rejected_edges = []
    n_legal_edges = 0
    for i, j in itertools.permutations(range(n), 2):
        sa, sb = nodes[i], nodes[j]
        if sb.t_end < sa.t_start - DT_MAX:
            continue
        best, rejects = _best_member_edge(sa, sb)
        if best is not None:
            u, v, dt, hop, emb, h_dist = best
            succ[i].append((j, u, v, dt, hop, emb, h_dist))
            n_legal_edges += 1
        else:
            for r in rejects[:3]:
                rejected_edges.append(r)
    return succ, rejected_edges, n_legal_edges


def _hist_ok(hist_embs, nodes, j, h_dist) -> tuple[bool, float, float]:
    class _EmbProxy:
        pass

    proxy = _EmbProxy()
    proxy.emb = nodes[j].emb
    hsim = hist_emb_sim(hist_embs, proxy)
    emb_need = (
        EMB_HIST_MIN - 0.02
        if (h_dist is not None and h_dist < H_DIST_GATE)
        else EMB_HIST_MIN
    )
    return hsim >= emb_need, hsim, emb_need


def _make_edge_rec(
    nodes,
    idx,
    j,
    u,
    v,
    dt,
    hop,
    emb,
    h_dist,
    hsim,
    calib,
    *,
    dt_scoring: bool = True,
    transition_prior: bool = False,
) -> dict:
    e = edge_llr(
        u,
        v,
        dt,
        emb,
        h_dist,
        calib,
        dt_scoring=dt_scoring,
        transition_prior=transition_prior,
    )
    e["hop"] = hop
    e["hist_emb"] = hsim
    e["from_super"] = nodes[idx].label
    e["to_super"] = nodes[j].label
    e["from_members"] = nodes[idx].tids
    e["to_members"] = nodes[j].tids
    e["dt_source"] = "super_union"
    return e


def enumerate_paths_llr(
    tracks: list,
    calib: dict,
    *,
    use_supernode: bool = True,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
    dt_scoring: bool = True,
    transition_prior: bool = False,
):
    """可選超節點圖上枚舉；邊多時改 beam（只留極大 leaf）。"""
    nodes, super_report = _build_nodes(tracks, use_supernode)
    succ, rejected_edges, n_legal_edges = _build_succ(nodes)
    n = len(nodes)
    use_beam = (not force_full) and (n_legal_edges > FULL_ENUM_EDGE_CAP)
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_scoring": bool(dt_scoring),
        "transition_prior": bool(transition_prior),
        "note": (
            f"合法邊={n_legal_edges} > {FULL_ENUM_EDGE_CAP}：beam 近似 Softmax／Top-k"
            if use_beam
            else "全量 DFS（含前綴）"
        ),
    }

    def _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim):
        return _make_edge_rec(
            nodes,
            idx,
            j,
            u,
            v,
            dt,
            hop,
            emb,
            h_dist,
            hsim,
            calib,
            dt_scoring=dt_scoring,
            transition_prior=transition_prior,
        )

    all_paths = []

    if not use_beam:
        def dfs(idx, path_idx, edges_info, hist_embs):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                if j in path_idx:
                    continue
                ok_h, hsim, emb_need = _hist_ok(hist_embs, nodes, j, h_dist)
                if not ok_h:
                    rejected_edges.append(
                        (
                            nodes[idx].label,
                            nodes[j].label,
                            f"歷史不像（hist_emb={hsim:.3f} < {emb_need}）",
                        )
                    )
                    continue
                e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                edges_info.append(e)
                path_idx.append(j)
                hist_embs.append(nodes[j].emb)
                dfs(j, path_idx, edges_info, hist_embs)
                hist_embs.pop()
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [], [nodes[s].emb])
        return all_paths, rejected_edges, n_legal_edges, nodes, super_report

    # -------- beam：每層保留 score 最高的 beam_width 個前綴；葉節點進結果 --------
    leaves = []
    for s in range(n):
        beam = [(0.0, [s], [], [nodes[s].emb])]
        while beam:
            nxt = []
            for sc, path_idx, edges_info, hist_embs in beam:
                idx = path_idx[-1]
                extended = False
                for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                    if j in path_idx:
                        continue
                    ok_h, hsim, _ = _hist_ok(hist_embs, nodes, j, h_dist)
                    if not ok_h:
                        continue
                    e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                    extended = True
                    nxt.append(
                        (
                            sc + e["score"],
                            path_idx + [j],
                            edges_info + [e],
                            hist_embs + [nodes[j].emb],
                        )
                    )
                if not extended:
                    leaves.append((path_idx, edges_info))
            if not nxt:
                break
            nxt.sort(key=lambda x: -x[0])
            beam = nxt[:beam_width]
            if len(leaves) >= beam_max_leaves:
                break
        if len(leaves) >= beam_max_leaves:
            break

    seen = set()
    for path_idx, edges_info in leaves:
        key = tuple(path_idx)
        if key in seen:
            continue
        seen.add(key)
        all_paths.append((path_idx, edges_info))

    super_report["enumeration"]["n_beam_leaves"] = len(all_paths)
    return all_paths, rejected_edges, n_legal_edges, nodes, super_report


def expand_path_tids(nodes: list[SuperNode], path_idx: list[int]) -> list[str]:
    """輸出展開：各超節點成員依 t_start 排序串接。"""
    out = []
    for i in path_idx:
        out.extend(nodes[i].tids)
    return out


def path_score_llr(nodes_on_path: list[SuperNode], edges_info: list, calib: dict) -> tuple[float, list]:
    node_evs = []
    total = 0.0
    for sn in nodes_on_path:
        ne = node_evidence(sn.sim, calib)
        node_evs.append({"super": sn.label, "members": sn.tids, **ne})
        total += ne["score"]
    total += sum(e["score"] for e in edges_info)
    return total, node_evs


def maximal_paths(scored: list) -> list:
    tid_seqs = {tuple(p["tids"]) for p in scored}
    is_prefix = set()
    for q in tid_seqs:
        for k in range(1, len(q)):
            is_prefix.add(q[:k])
    return [p for p in scored if tuple(p["tids"]) not in is_prefix]


def attach_softmax(maximal: list) -> list:
    if not maximal:
        return maximal
    scores = np.asarray([p["score"] for p in maximal], dtype=np.float64)
    log_z = logsumexp(scores)
    for p, s in zip(maximal, scores):
        p["path_probability"] = float(math.exp(s - log_z))
    return maximal


def _time_overlap_sec(a: Track, b: Track) -> float:
    return max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))


def tracks_physical_coexist_contradiction(a: Track, b: Track) -> bool:
    """
    同假設內部裁決：跨鏡時間共存，且鏡頭對既非 OVERLAP 亦非 ADJACENT → 矛盾。
    （假設外的採用節點與假設內節點共存不構成問題。）
    """
    if a.cam == b.cam:
        return False
    ov = _time_overlap_sec(a, b)
    if ov <= 0.0:
        tok, _, _ = _coexistence_time_ok(a, b)
        if not tok:
            return False
    key = tuple(sorted((a.cam, b.cam)))
    if key in OVERLAP_PAIRS or key in ADJACENT:
        return False
    return True


def hypothesis_internal_contradictions(
    tids: list[str],
    by_tid: dict[str, Track],
) -> list[dict]:
    """回傳假設內部所有矛盾對（空＝通過）。"""
    tracks = [by_tid[t] for t in tids if t in by_tid]
    bad = []
    for u, v in itertools.combinations(tracks, 2):
        if tracks_physical_coexist_contradiction(u, v):
            bad.append(
                {
                    "a": u.tid,
                    "b": v.tid,
                    "cams": f"{u.cam}↔{v.cam}",
                    "overlap_sec": float(_time_overlap_sec(u, v)),
                }
            )
    return bad


def _path_as_segment(path: dict, segment_i: int, gap: float | None) -> dict:
    return {
        "segment": segment_i,
        "path": " -> ".join(path.get("super_labels") or path["tids"]),
        "super_labels": list(path.get("super_labels") or []),
        "tids": list(path["tids"]),
        "super_ids": list(path.get("super_ids") or []),
        "score": float(path["score"]),
        "t_start": float(path["t_start"]),
        "t_end": float(path["t_end"]),
        "gap_after_prev_sec": gap,
        "edges": path.get("edges"),
        "node_evidence": path.get("node_evidence"),
    }


def _hypothesis_from_segments(
    segments: list[dict],
    *,
    source: str,
    seed_rank: int | None = None,
) -> dict:
    tids: list[str] = []
    super_labels: list[str] = []
    super_ids: list[str] = []
    for seg in segments:
        for t in seg["tids"]:
            if t not in tids:
                tids.append(t)
        for lab in seg.get("super_labels") or []:
            if lab not in super_labels:
                super_labels.append(lab)
        for sid in seg.get("super_ids") or []:
            if sid not in super_ids:
                super_ids.append(sid)
    total = float(sum(s["score"] for s in segments))
    n_seg = len(segments)
    return {
        "hypothesis_type": "single" if n_seg == 1 else "segmented",
        "n_segments": n_seg,
        "score": total,
        "tids": tids,
        "super_labels": super_labels,
        "super_ids": super_ids,
        "t_start": float(segments[0]["t_start"]),
        "t_end": float(segments[-1]["t_end"]),
        "segments": segments,
        "path": " || ".join(s["path"] for s in segments),
        "edges": segments[0].get("edges") if n_seg == 1 else None,
        "node_evidence": (
            segments[0].get("node_evidence") if n_seg == 1 else None
        ),
        "source": source,
        "seed_rank": seed_rank,
    }


def grow_segmented_hypothesis(
    seed_path: dict,
    all_nodes: list,
    calib: dict,
    *,
    dt_scoring: bool,
    transition_prior: bool,
    max_segments: int = DEFAULT_MAX_HYP_SEGMENTS,
    pool_cache: dict | None = None,
) -> list[dict]:
    """
    以 seed 為 seg1，在未使用且 t_start>前段結束的超節點中遞迴取最佳延續段。
    回傳 segments（至少含 seed）；若無法延續則僅 1 段。
    """
    cache = pool_cache if pool_cache is not None else {}
    seg1 = _path_as_segment(seed_path, 1, None)
    segments = [seg1]
    used_sids = set(seed_path.get("super_ids") or [])
    if not used_sids:
        used_tids = set(seed_path["tids"])
        used_sids = {n.sid for n in all_nodes if used_tids & set(n.tids)}
    remaining = [n for n in all_nodes if n.sid not in used_sids]
    prev_end = float(seed_path["t_end"])

    for seg_i in range(2, max_segments + 1):
        pool = [n for n in remaining if n.t_start > prev_end]
        if len(pool) < 1:
            break
        key = frozenset(n.sid for n in pool)
        if key in cache:
            maximal = cache[key]
        else:
            maximal, _n_legal, _meta = _score_paths_on_nodes(
                pool,
                calib,
                dt_scoring=dt_scoring,
                transition_prior=transition_prior,
            )
            cache[key] = maximal
        if not maximal:
            break
        top = maximal[0]
        gap = float(top["t_start"] - prev_end)
        segments.append(_path_as_segment(top, seg_i, gap))
        used_now = set(top.get("super_ids") or [])
        remaining = [n for n in remaining if n.sid not in used_now]
        prev_end = float(top["t_end"])

    return segments


def build_ranked_hypotheses(
    single_maximal: list,
    all_nodes: list,
    tracks: list,
    calib: dict,
    *,
    dt_scoring: bool,
    transition_prior: bool,
    seed_top_k: int = DEFAULT_SEGMENT_SEED_TOP_K,
    max_segments: int = DEFAULT_MAX_HYP_SEGMENTS,
) -> tuple[list, dict]:
    """
    單路徑假設 ∪ 高分路徑延伸的多段假設 → 同一 Softmax 排名池。
    假設內部共存矛盾者作廢。計分公式不變。
    """
    by_tid = {t.tid: t for t in tracks}
    rejected = []
    pool = []
    seen_keys = set()
    pool_cache: dict = {}

    def _try_add(hyp: dict) -> None:
        key = tuple(
            tuple(seg["tids"]) for seg in hyp["segments"]
        )
        if key in seen_keys:
            return
        bad = hypothesis_internal_contradictions(hyp["tids"], by_tid)
        if bad:
            rejected.append(
                {
                    "path": hyp["path"],
                    "n_segments": hyp["n_segments"],
                    "score": hyp["score"],
                    "contradictions": bad,
                }
            )
            return
        seen_keys.add(key)
        hyp["internal_contradictions"] = []
        pool.append(hyp)

    # 1) 所有極大單路徑 → 單段假設
    for rank, p in enumerate(single_maximal, 1):
        hyp = _hypothesis_from_segments(
            [_path_as_segment(p, 1, None)],
            source="single_maximal",
            seed_rank=rank,
        )
        _try_add(hyp)

    # 2) 高分 seed → 嘗試多段
    seeds = single_maximal[: max(int(seed_top_k), 0)]
    n_grown = 0
    for rank, seed in enumerate(seeds, 1):
        # 無後續候選則跳過（省枚舉）
        used_sids = set(seed.get("super_ids") or [])
        if not used_sids:
            used_tids = set(seed["tids"])
            used_sids = {n.sid for n in all_nodes if used_tids & set(n.tids)}
        has_later = any(
            n.sid not in used_sids and n.t_start > float(seed["t_end"])
            for n in all_nodes
        )
        if not has_later:
            continue
        segs = grow_segmented_hypothesis(
            seed,
            all_nodes,
            calib,
            dt_scoring=dt_scoring,
            transition_prior=transition_prior,
            max_segments=max_segments,
            pool_cache=pool_cache,
        )
        if len(segs) < 2:
            continue
        hyp = _hypothesis_from_segments(
            segs, source="seed_grown", seed_rank=rank
        )
        before = len(pool)
        _try_add(hyp)
        if len(pool) > before:
            n_grown += 1

    pool.sort(key=lambda h: -h["score"])
    pool = attach_softmax(pool)
    meta = {
        "n_single_maximal": len(single_maximal),
        "n_hypotheses_ranked": len(pool),
        "n_segmented_added": n_grown,
        "n_rejected_contradiction": len(rejected),
        "seed_top_k": seed_top_k,
        "rejected_sample": rejected[:20],
        "note": SEGMENT_RANK_NOTE,
    }
    return pool, meta


def path_time_span(path: dict, nodes: list | None = None) -> tuple[float, float]:
    """路徑觀測區間：成員 t_start 最小、t_end 最大。"""
    if path.get("edges") is not None and path.get("super_ids") and nodes:
        by_sid = {n.sid: n for n in nodes}
        chosen = [by_sid[s] for s in path["super_ids"] if s in by_sid]
        if chosen:
            return (
                float(min(n.t_start for n in chosen)),
                float(max(n.t_end for n in chosen)),
            )
    # fallback：從 tids 無法還原時間；用 edges / 外層附帶
    if "t_start" in path and "t_end" in path:
        return float(path["t_start"]), float(path["t_end"])
    raise ValueError("path_time_span：路徑缺時間資訊")


def _score_paths_on_nodes(
    nodes: list,
    calib: dict,
    *,
    dt_scoring: bool,
    transition_prior: bool,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = DEFAULT_BEAM_MAX_LEAVES,
) -> tuple[list, int, dict]:
    """在既有超節點子集上枚舉＋計分（不重建超節點）。"""
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}

    # 暫以 tracks=各超節點代表？不行——enumerate 需要再建圖。
    # 直接對 nodes 子列表建 succ / beam。
    n = len(nodes)
    succ = [[] for _ in range(n)]
    n_legal = 0
    for i, j in itertools.permutations(range(n), 2):
        sa, sb = nodes[i], nodes[j]
        if sb.t_end < sa.t_start - DT_MAX:
            continue
        best, _ = _best_member_edge(sa, sb)
        if best is not None:
            u, v, dt, hop, emb, h_dist = best
            succ[i].append((j, u, v, dt, hop, emb, h_dist))
            n_legal += 1

    use_beam = n_legal > FULL_ENUM_EDGE_CAP
    enum_meta = {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "n_nodes": n,
    }

    all_paths = []

    def _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim):
        return _make_edge_rec(
            nodes, idx, j, u, v, dt, hop, emb, h_dist, hsim, calib,
            dt_scoring=dt_scoring, transition_prior=transition_prior,
        )

    if not use_beam:
        def dfs(idx, path_idx, edges_info, hist_embs):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                if j in path_idx:
                    continue
                ok_h, hsim, _ = _hist_ok(hist_embs, nodes, j, h_dist)
                if not ok_h:
                    continue
                e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                edges_info.append(e)
                path_idx.append(j)
                hist_embs.append(nodes[j].emb)
                dfs(j, path_idx, edges_info, hist_embs)
                hist_embs.pop()
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [], [nodes[s].emb])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [], [nodes[s].emb])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info, hist_embs in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                        if j in path_idx:
                            continue
                        ok_h, hsim, _ = _hist_ok(hist_embs, nodes, j, h_dist)
                        if not ok_h:
                            continue
                        e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                        extended = True
                        nxt.append(
                            (
                                sc + e["score"],
                                path_idx + [j],
                                edges_info + [e],
                                hist_embs + [nodes[j].emb],
                            )
                        )
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[:beam_width]
                if len(leaves) >= beam_max_leaves:
                    break
            if len(leaves) >= beam_max_leaves:
                break
        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))
        enum_meta["n_beam_leaves"] = len(all_paths)

    scored = []
    for path_idx, edges_info in all_paths:
        sn_path = [nodes[i] for i in path_idx]
        score, node_evs = path_score_llr(sn_path, edges_info, calib)
        scored.append(
            {
                "tids": expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": node_evs,
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = attach_softmax(maximal_paths(scored))
    return maximal, n_legal, enum_meta


def extract_path_segments(
    all_nodes: list,
    calib: dict,
    *,
    first_path: dict | None = None,
    dt_scoring: bool = False,
    transition_prior: bool = False,
    max_segments: int = 20,
) -> list[dict]:
    """
    分段軌跡（2026-07-15）：
      Segment 1 = 全圖 Top-1（若傳 first_path 則直接採用，不再重枚舉）。
      其後：在未使用的超節點中，取 t_start > 前段結束 者，重跑同一套計分，
      Softmax 僅在該段候選極大路徑內；標注與前段的觀測空窗秒數。
      DT_MAX 不變。遞迴至無合法段落。
    """
    remaining = list(all_nodes)
    segments = []
    prev_end = None

    for seg_i in range(1, max_segments + 1):
        if not remaining:
            break

        if seg_i == 1 and first_path is not None:
            top = dict(first_path)
            if "t_start" not in top or "t_end" not in top:
                sids = set(top.get("super_ids") or [])
                chosen = [n for n in all_nodes if n.sid in sids]
                if not chosen and top.get("super_labels"):
                    by_lab = {n.label: n for n in all_nodes}
                    chosen = [by_lab[l] for l in top["super_labels"] if l in by_lab]
                if not chosen:
                    # 最後手段：用 tids 對齊成員
                    tid_set = set(top["tids"])
                    chosen = [n for n in all_nodes if tid_set & set(n.tids)]
                top["t_start"] = float(min(n.t_start for n in chosen))
                top["t_end"] = float(max(n.t_end for n in chosen))
                top["super_ids"] = [n.sid for n in chosen]
            gap = None
            enum_meta = {"mode": "from_global_top1"}
            n_legal = None
            n_cand = None
            top3 = None
        else:
            pool = (
                remaining
                if prev_end is None
                else [n for n in remaining if n.t_start > prev_end]
            )
            if not pool:
                break
            maximal, n_legal, enum_meta = _score_paths_on_nodes(
                pool,
                calib,
                dt_scoring=dt_scoring,
                transition_prior=transition_prior,
            )
            if not maximal:
                break
            top = maximal[0]
            gap = None if prev_end is None else float(top["t_start"] - prev_end)
            n_cand = len(maximal)
            top3 = [
                {
                    "rank": j,
                    "score": p["score"],
                    "path_probability": p.get("path_probability"),
                    "path": " -> ".join(p.get("super_labels") or p["tids"]),
                    "tids": p["tids"],
                }
                for j, p in enumerate(maximal[:3], 1)
            ]

        segments.append(
            {
                "segment": seg_i,
                "path": " -> ".join(top.get("super_labels") or top["tids"]),
                "super_labels": top.get("super_labels"),
                "tids": top["tids"],
                "super_ids": top.get("super_ids"),
                "score": top["score"],
                "path_probability": top.get("path_probability"),
                "t_start": top["t_start"],
                "t_end": top["t_end"],
                "gap_after_prev_sec": gap,
                "n_candidates_maximal": n_cand,
                "n_legal_edges_in_pool": n_legal,
                "enumeration": enum_meta,
                "edges": top.get("edges"),
                "node_evidence": top.get("node_evidence"),
                "top3": top3,
            }
        )
        used_sids = set(top.get("super_ids") or [])
        if not used_sids:
            # 用 tids 去除
            used_tids = set(top["tids"])
            remaining = [n for n in remaining if not (used_tids & set(n.tids))]
        else:
            remaining = [n for n in remaining if n.sid not in used_sids]
        prev_end = float(top["t_end"])

    return segments


def score_labeled_path(nodes, labels, calib, *, dt_scoring: bool, transition_prior: bool):
    """精確逐邊+節點計分（含 hist gate）。失敗回傳 ok=False。"""
    by_member = {}
    for sn in nodes:
        for tid in sn.tids:
            by_member[tid] = sn
    chain = []
    seen = set()
    for lab in labels:
        if lab.startswith("{"):
            mems = [x.strip() for x in lab[1:-1].split(",") if x.strip()]
            sn = by_member[mems[0]]
        else:
            sn = by_member[lab]
        if sn.sid in seen:
            continue
        seen.add(sn.sid)
        chain.append(sn)
    edges = []
    hist = [chain[0].emb]
    for i in range(len(chain) - 1):
        sa, sb = chain[i], chain[i + 1]
        best, _ = _best_member_edge(sa, sb)
        if best is None:
            return {"ok": False, "reason": f"no edge {sa.label}->{sb.label}"}
        u, v, dt, hop, emb, h_dist = best
        ok_h, hsim, need = _hist_ok(hist, chain, i + 1, h_dist)
        if not ok_h:
            return {
                "ok": False,
                "reason": f"hist fail {sa.label}->{sb.label} hist={hsim:.3f}<{need}",
            }
        e = edge_llr(
            u,
            v,
            dt,
            emb,
            h_dist,
            calib,
            dt_scoring=dt_scoring,
            transition_prior=transition_prior,
        )
        e["hop"] = hop
        e["hist_emb"] = hsim
        e["from_super"] = sa.label
        e["to_super"] = sb.label
        e["from_members"] = sa.tids
        e["to_members"] = sb.tids
        e["via"] = f"{u.tid}->{v.tid}"
        edges.append(e)
        hist.append(sb.emb)
    score, node_ev = path_score_llr(chain, edges, calib)
    tids = []
    for sn in chain:
        tids.extend(sn.tids)
    return {
        "ok": True,
        "score": score,
        "tids": tids,
        "super_labels": [sn.label for sn in chain],
        "edges": edges,
        "node_evidence": node_ev,
    }


def best_disjoint_alternative(maximal: list) -> dict | None:
    if len(maximal) < 2:
        return None
    top = maximal[0]
    top_set = set(top["tids"])
    for p in maximal[1:]:
        if set(p["tids"]).isdisjoint(top_set):
            return p
    return None


def render_collage_if_available(merge_dir: Path, top: dict, out_png: Path) -> Path | None:
    try:
        return render_top1_collage(merge_dir, top, out_png, title_prefix="path_enum LLR Top-1")
    except Exception as exc:
        print(f"警告：拼圖失敗（{exc}），略過 PNG")
        return None


def _fmt_llr_dt(e: dict) -> str:
    if e.get("LLR_dt") == "removed":
        return "removed"
    if e.get("dt_model") == "handoff" or e.get("LLR_dt") is None:
        return "n/a"
    return f"{e['LLR_dt']:+.3f}"


def write_txt_report(
    out_txt: Path,
    merge_dir: Path,
    tracks: list,
    scored: list,
    maximal: list,
    n_legal_edges: int,
    alt: dict | None,
    *,
    super_report: dict | None = None,
    gate_info: dict | None = None,
    segments: list | None = None,
) -> None:
    lines = []
    lines.append(f"輸入：{merge_dir}")
    lines.append(f"SIM_MIN={SIM_MIN}  MODE={MODE}  H矩陣={len(H_MATRICES)}")
    lines.append(
        f"MIN_TRANSIT hop1={DEFAULT_MIN_TRANSIT_HOP1}  hop2={DEFAULT_MIN_TRANSIT_HOP2}"
    )
    if gate_info:
        lines.append(
            f"EMB gates: EDGE={gate_info['EMB_EDGE_MIN']} HIST={gate_info['EMB_HIST_MIN']}  "
            f"({gate_info.get('rationale','')[:80]})"
        )
    lines.append(NODE_EVIDENCE_NOTE)
    lines.append(PRIOR_WEAK_NOTE)
    lines.append(SEGMENT_RANK_NOTE)
    lines.append(f"候選 track={len(tracks)}  合法邊={n_legal_edges}")
    lines.append(f"合法路徑（含前綴）={len(scored)}  排名假設={len(maximal)}")
    if super_report:
        lines.append(
            f"超節點：{super_report['n_supernodes']}  "
            f"合併對數={super_report['n_merged_pairs']}  "
            f"多成員={super_report.get('multi_only')}"
        )
    lines.append("")
    lines.append(f"===== Top {min(3, len(maximal))} 假設（單路徑＋分段，同 Softmax）=====")
    for rank, p in enumerate(maximal[:3], 1):
        n_seg = int(p.get("n_segments") or 1)
        lines.append(
            f"\n#{rank}  score={p['score']:.4f}  P={p.get('path_probability', 0):.6f}  "
            f"段數={n_seg}  type={p.get('hypothesis_type', 'single')}"
        )
        segs = p.get("segments") or []
        if segs:
            for seg in segs:
                gap = seg.get("gap_after_prev_sec")
                gap_s = f"  空窗={gap:.1f}s" if gap is not None else ""
                lines.append(
                    f"  [seg{seg['segment']}] score={seg['score']:.4f}{gap_s}  "
                    f"{seg['path']}"
                )
                lines.append(
                    f"    span=[{seg['t_start']:.1f}, {seg['t_end']:.1f}]"
                )
        else:
            lines.append(
                "  " + " -> ".join(p.get("super_labels") or p["tids"])
            )
        lines.append("  expanded union: " + " -> ".join(p["tids"]))
        if n_seg == 1:
            for ne in p.get("node_evidence") or []:
                if ne.get("enabled"):
                    lines.append(
                        f"      NODE {ne['super']} sim={ne['sim']:.3f}  "
                        f"LLR_raw={ne['raw']:+.3f} w={ne['w']:.3f}  "
                        f"score={ne['score']:+.3f}"
                    )
            for e in p.get("edges") or []:
                hd = e.get("h_dist")
                hd_s = f"{hd:.1f}" if hd is not None else "—"
                ldh = e.get("LLR_dH")
                ldh_s = f"{ldh:+.3f}" if ldh is not None else "—"
                prior_tag = ""
                if e.get("dt_prior_physical"):
                    prior_tag = f"  [{e.get('dt_note') or 'PRIOR-WEAK'}]"
                lines.append(
                    f"      {e.get('from_super', e['from'])} -> "
                    f"{e.get('to_super', e['to'])}   "
                    f"hop={e['hop']}  dt={e['dt']:.1f}s  "
                    f"dt_model={e.get('dt_model')}  "
                    f"emb={e['emb']:.3f}  d_H={hd_s}  "
                    f"LLR_dt={_fmt_llr_dt(e)}  LLR_emb={e['LLR_emb']:+.3f}  "
                    f"LLR_dH={ldh_s}  edge={e['score']:+.3f}{prior_tag}"
                )

    if maximal:
        top = maximal[0]
        lines.append("")
        lines.append(
            f"Top-1 path_probability={top.get('path_probability', 0):.6f}  "
            f"score={top['score']:.4f}  n_segments={top.get('n_segments', 1)}"
        )
        if alt is not None:
            ratio = top["path_probability"] / max(alt["path_probability"], 1e-300)
            lines.append(
                f"最佳不共用 track 替代假設：P={alt['path_probability']:.6f}  "
                f"score={alt['score']:.4f}  ratio(Top1/alt)={ratio:.3f}"
            )
            lines.append("  alt: " + (alt.get("path") or " -> ".join(alt["tids"])))

    if segments:
        lines.append("")
        lines.append(f"===== Top-1 假設各段（共 {len(segments)} 段）=====")
        for seg in segments:
            gap = seg.get("gap_after_prev_sec")
            gap_s = f"  觀測空窗={gap:.1f}s" if gap is not None else ""
            lines.append(
                f"\n[seg{seg['segment']}] score={seg['score']:.4f}{gap_s}"
            )
            lines.append(f"  {seg['path']}")
            lines.append(
                f"  span=[{seg['t_start']:.1f}, {seg['t_end']:.1f}]"
            )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_json(
    merge_dir: Path,
    scored: list,
    maximal: list,
    collage: Path | None,
    alt: dict | None,
    n_legal_edges: int,
    n_tracks: int,
    *,
    super_report: dict | None = None,
    gate_info: dict | None = None,
    options: dict | None = None,
) -> dict:
    top = maximal[0] if maximal else None
    ratio = None
    if top and alt is not None:
        ratio = top["path_probability"] / max(alt["path_probability"], 1e-300)
    segments = (options or {}).get("segments") or []
    # JSON 精簡：段落不重複塞完整 edges
    segments_out = []
    for seg in segments:
        segments_out.append(
            {
                k: seg.get(k)
                for k in (
                    "segment",
                    "path",
                    "super_labels",
                    "tids",
                    "score",
                    "path_probability",
                    "t_start",
                    "t_end",
                    "gap_after_prev_sec",
                    "n_candidates_maximal",
                    "n_legal_edges_in_pool",
                    "enumeration",
                    "top3",
                )
            }
        )

    def _hyp_brief(p: dict, rank: int) -> dict:
        segs_brief = []
        for seg in p.get("segments") or []:
            segs_brief.append(
                {
                    "segment": seg.get("segment"),
                    "path": seg.get("path"),
                    "tids": seg.get("tids"),
                    "super_labels": seg.get("super_labels"),
                    "score": seg.get("score"),
                    "t_start": seg.get("t_start"),
                    "t_end": seg.get("t_end"),
                    "gap_after_prev_sec": seg.get("gap_after_prev_sec"),
                }
            )
        return {
            "rank": rank,
            "score": p["score"],
            "path_probability": p.get("path_probability"),
            "hypothesis_type": p.get("hypothesis_type"),
            "n_segments": p.get("n_segments"),
            "tids": p["tids"],
            "super_labels": p.get("super_labels"),
            "path": p.get("path") or " -> ".join(p.get("super_labels") or p["tids"]),
            "segments": segs_brief,
            "source": p.get("source"),
            "seed_rank": p.get("seed_rank"),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": MODE,
        "scoring": "LLR",
        "options": {
            k: v
            for k, v in (options or {}).items()
            if k not in ("segments",)
        },
        "gate_info": gate_info,
        "supernodes": super_report,
        "node_evidence_note": NODE_EVIDENCE_NOTE,
        "prior_weak_note": PRIOR_WEAK_NOTE,
        "segment_rank_note": SEGMENT_RANK_NOTE,
        "input_dir": str(merge_dir.resolve()),
        "sim_min": SIM_MIN,
        "n_tracks": n_tracks,
        "n_legal_edges": n_legal_edges,
        "n_paths_all": len(scored),
        "n_hypotheses_ranked": len(maximal),
        "n_paths_maximal": len(maximal),  # 兼容舊欄位＝排名假設數
        "collage": str(collage.resolve()) if collage else None,
        "segments": segments_out,
        "top1": _hyp_brief(top, 1) if top else None,
        "top3_hypotheses": [
            _hyp_brief(p, i) for i, p in enumerate(maximal[:3], 1)
        ],
        "best_disjoint_alternative": {
            "score": alt["score"],
            "path_probability": alt.get("path_probability"),
            "tids": alt["tids"],
            "n_segments": alt.get("n_segments"),
            "path": alt.get("path") or " -> ".join(alt["tids"]),
            "probability_ratio_top1_over_alt": ratio,
        }
        if alt
        else None,
        "top10_paths": [
            _hyp_brief(p, i) for i, p in enumerate(maximal[:10], 1)
        ],
    }


def run_llr(
    merge_dir: Path,
    calib: dict,
    *,
    use_emb_gate_fix: bool = True,
    use_supernode: bool = True,
    use_node_evidence: bool = True,
    dt_scoring: bool = True,
    transition_prior: bool = False,
):
    gate_info = apply_llr_emb_gates(enabled=use_emb_gate_fix)
    tracks = load_tracks(str(merge_dir))
    print(f"讀入 {len(tracks)} 條 track")
    print(
        f"options: emb_gate={use_emb_gate_fix}({gate_info['EMB_EDGE_MIN']})  "
        f"supernode={use_supernode}  node_evidence={use_node_evidence}  "
        f"dt_scoring={dt_scoring}  transition_prior={transition_prior}"
    )

    calib_use = calib
    if not use_node_evidence:
        calib_use = dict(calib)
        calib_use.pop("sim_gt", None)
        calib_use.pop("sim_nongt", None)

    all_paths, rejected, n_legal_edges, nodes, super_report = enumerate_paths_llr(
        tracks,
        calib_use,
        use_supernode=use_supernode,
        dt_scoring=dt_scoring,
        transition_prior=transition_prior,
    )
    print(
        f"超節點={super_report['n_supernodes']}  多成員={super_report.get('multi_only')}"
    )

    scored = []
    for path_idx, edges_info in all_paths:
        sn_path = [nodes[i] for i in path_idx]
        score, node_evs = path_score_llr(sn_path, edges_info, calib_use)
        scored.append(
            {
                "tids": expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": node_evs,
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    single_maximal = maximal_paths(scored)
    # 單路徑 Softmax 僅供診斷；真正排名改為假設池
    single_maximal_diag = attach_softmax([dict(p) for p in single_maximal])
    print(f"合法邊={n_legal_edges}  路徑={len(scored)}  極大單路徑={len(single_maximal)}")

    ranked, rank_meta = build_ranked_hypotheses(
        single_maximal,
        nodes,
        tracks,
        calib_use,
        dt_scoring=dt_scoring,
        transition_prior=transition_prior,
    )
    print(
        f"排名假設={rank_meta['n_hypotheses_ranked']}  "
        f"多段新增={rank_meta['n_segmented_added']}  "
        f"共存矛盾作廢={rank_meta['n_rejected_contradiction']}"
    )
    if ranked:
        top = ranked[0]
        print(
            f"Top-1 P={top['path_probability']:.6f}  "
            f"score={top['score']:.4f}  "
            f"n_seg={top['n_segments']}  "
            + top.get("path", "")
        )
        for seg in top.get("segments") or []:
            gap = seg.get("gap_after_prev_sec")
            gap_s = f"  gap={gap:.1f}s" if gap is not None else ""
            print(
                f"  seg{seg['segment']}: score={seg['score']:.4f}{gap_s}  {seg['path']}"
            )

    # 兼容：maximal = 排名後的假設（含單／多段）
    maximal = ranked
    # options.segments = Top-1 假設的各段（供拼圖／舊協定）
    segments = list((maximal[0].get("segments") if maximal else None) or [])

    options = {
        "use_emb_gate_fix": use_emb_gate_fix,
        "use_supernode": use_supernode,
        "use_node_evidence": use_node_evidence,
        "dt_scoring": bool(dt_scoring),
        "transition_prior": bool(transition_prior),
        "prior_dt_sigma": PRIOR_DT_SIGMA,
        "dt_scoring_note": (
            None
            if dt_scoring
            else "2026-07-15：tau 無實測來源，transit LLR_dt 軟證據停用"
        ),
        "min_transit_hop1": float(DEFAULT_MIN_TRANSIT_HOP1),
        "min_transit_hop2": float(DEFAULT_MIN_TRANSIT_HOP2),
        "segments": segments,
        "ranking_protocol": SEGMENT_RANK_NOTE,
        "ranking_meta": rank_meta,
        "single_maximal_top1": (
            {
                "score": single_maximal_diag[0]["score"],
                "path": " -> ".join(
                    single_maximal_diag[0].get("super_labels")
                    or single_maximal_diag[0]["tids"]
                ),
                "path_probability": single_maximal_diag[0].get("path_probability"),
                "tids": single_maximal_diag[0]["tids"],
            }
            if single_maximal_diag
            else None
        ),
    }
    return tracks, scored, maximal, n_legal_edges, nodes, super_report, gate_info, options



def parse_run_args(argv=None):
    p = argparse.ArgumentParser(description="path enum with LLR scoring + structural fixes")
    p.add_argument(
        "input_dir",
        nargs="?",
        default=str(QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507"),
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-emb-gate-fix", action="store_true", help="關閉修正一（維持 0.91/0.90）")
    p.add_argument("--no-supernode", action="store_true", help="關閉修正二")
    p.add_argument("--no-node-evidence", action="store_true", help="關閉修正三節點證據")
    p.add_argument(
        "--dt-scoring",
        choices=["on", "off"],
        default="on",
        help="transit LLR_dt：on=計分 / off=0+removed（硬規則不動）",
    )
    p.add_argument(
        "--transition-prior",
        choices=["on", "off"],
        default="off",
        help="每邊加 ln(p_edge)；p_edge 來自 GT 校準",
    )
    return p.parse_args(argv)


def cmd_run(argv=None):
    args = parse_run_args(argv)
    merge_dir = Path(args.input_dir).resolve()
    if not merge_dir.is_dir():
        raise SystemExit(f"找不到資料夾：{merge_dir}")

    out_dir = (args.out_dir or (OUTPUT_ROOT / "v1.0")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_path = (args.calibration or (out_dir / "calibration_gt0507.pkl")).resolve()
    if not calib_path.is_file():
        raise SystemExit(f"找不到 calibration：{calib_path}")

    global SIM_MIN
    SIM_MIN = float(args.sim_min)
    mode = configure_for_input(str(merge_dir))
    print(f"模式：{mode}  SIM_MIN={SIM_MIN}")
    print(f"calibration：{calib_path}")
    calib = load_calibration(calib_path)

    (
        tracks,
        scored,
        maximal,
        n_legal_edges,
        nodes,
        super_report,
        gate_info,
        options,
    ) = run_llr(
        merge_dir,
        calib,
        use_emb_gate_fix=not args.no_emb_gate_fix,
        use_supernode=not args.no_supernode,
        use_node_evidence=not args.no_node_evidence,
        dt_scoring=(args.dt_scoring == "on"),
        transition_prior=(args.transition_prior == "on"),
    )
    alt = best_disjoint_alternative(maximal)

    name = merge_dir.name
    if "20260507" in name:
        short = "0507"
    elif "20260528" in name:
        short = "0528"
    else:
        short = name
    out_txt = out_dir / f"{short}_out.txt"
    out_json = out_dir / f"{short}_top1.json"
    out_png = out_dir / f"{short}_top1_collage.png"
    out_super = out_dir / f"{short}_supernodes.json"
    out_suspect = out_dir / f"{short}_suspect_coexistence.txt"

    write_txt_report(
        out_txt,
        merge_dir,
        tracks,
        scored,
        maximal,
        n_legal_edges,
        alt,
        super_report=super_report,
        gate_info=gate_info,
        segments=options.get("segments"),
    )
    collage = None
    if maximal:
        collage = render_collage_if_available(merge_dir, maximal[0], out_png)
    summary = build_summary_json(
        merge_dir,
        scored,
        maximal,
        collage,
        alt,
        n_legal_edges,
        len(tracks),
        super_report=super_report,
        gate_info=gate_info,
        options=options,
    )
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_super.write_text(json.dumps(super_report, ensure_ascii=False, indent=2), encoding="utf-8")
    suspect_path = write_suspect_coexistence_txt(
        out_suspect, super_report.get("suspect_coexistence") or []
    )

    print(f"文字報告：{out_txt}")
    print(f"JSON：{out_json}")
    print(f"超節點：{out_super}")
    if suspect_path:
        print(f"共存 suspect：{suspect_path}")
    if collage:
        print(f"拼圖：{collage}")
    return summary



# ============================================================
# CALIBRATE（原 calibrate.py）
# ============================================================

DH_DIFF_UNIFORM_MAX = 800.0
# PRIOR-WEAK（2026-07-15）：原 0.5 → 1.0；tau 仍佔位。見 path_enum_llr.PRIOR_WEAK_NOTE
PRIOR_SIGMA = 1.0
MIN_SAMPLES_FIT = 20
# PDF_FLOOR already defined above
# SHRINK_K already defined above (float 10.0)
CALIB_SOURCE = "GT_20260507"


def _calib_time_overlap(a: Track, b: Track) -> bool:
    return not (a.t_end < b.t_start or b.t_end < a.t_start)


def _fit_lognormal(samples: np.ndarray) -> dict | None:
    samples = samples[samples > 0]
    if len(samples) < 2:
        return None
    shape, loc, scale = stats.lognorm.fit(samples, floc=0)
    return {
        "family": "lognorm",
        "mu": float(np.log(scale)),
        "sigma": float(shape),
        "n": int(len(samples)),
    }


def _fit_normal(samples: np.ndarray) -> dict | None:
    if len(samples) < 2:
        return None
    mu, sigma = stats.norm.fit(samples)
    return {
        "family": "norm",
        "mu": float(mu),
        "sigma": max(float(sigma), 1e-4),
        "n": int(len(samples)),
    }


def _fit_halfnormal(samples: np.ndarray) -> dict | None:
    samples = samples[samples >= 0]
    if len(samples) < 2:
        return None
    _, scale = stats.halfnorm.fit(samples, floc=0)
    return {
        "family": "halfnorm",
        "sigma": max(float(scale), 1e-4),
        "n": int(len(samples)),
    }


def _tau_for_pair(cam_a: str, cam_b: str) -> float:
    hop = hop_count(cam_a, cam_b)
    if hop is None:
        hop = 1
    return float(tau(cam_a, cam_b, hop))


def _norm_fit_stats(arr: np.ndarray) -> dict:
    if len(arr) < 1:
        return {"n": 0, "mu": None, "sigma": None}
    if len(arr) < 2:
        return {"n": int(len(arr)), "mu": float(arr[0]), "sigma": None}
    mu, sigma = stats.norm.fit(arr)
    return {"n": int(len(arr)), "mu": float(mu), "sigma": float(max(sigma, 1e-4))}


def _emb_same_cross_cam(tracks: list) -> np.ndarray:
    vals: list[float] = []
    for i, u in enumerate(tracks):
        for v in tracks[i + 1 :]:
            if u.cam == v.cam:
                continue
            vals.append(emb_sim(u, v))
    return np.asarray(vals, dtype=np.float64)


def collect_gt_samples(
    tracks: list,
    gt_tids: list[str],
    *,
    removed_mislabel: list[str] | None = None,
) -> dict:
    by_tid = {t.tid: t for t in tracks}
    gt_set = set(gt_tids)
    missing = [t for t in gt_tids if t not in by_tid]
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    non_gt = [t for t in tracks if t.tid not in gt_set]

    emb_same: list[float] = []
    emb_diff: list[float] = []
    dh_same: list[float] = []
    dt_same_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    time_overlaps = []

    # 剔除誤標前後 emb|same 對照（含 08_43 的舊 GT vs 現行 GT）
    removed = list(removed_mislabel or [])
    old_gt_tids = list(dict.fromkeys(list(gt_tids) + removed))
    old_gt_tracks = [by_tid[t] for t in old_gt_tids if t in by_tid]
    emb_same_before = _emb_same_cross_cam(old_gt_tracks)
    emb_same_after = _emb_same_cross_cam(gt_tracks)
    emb_same_compare = {
        "before_remove_08_43": _norm_fit_stats(emb_same_before),
        "after_remove_08_43": _norm_fit_stats(emb_same_after),
        "removed": removed,
    }

    # emb|same：GT×GT 跨鏡（已不含 08_43）
    emb_same = emb_same_after.tolist()

    # emb|diff：GT × 非GT（全部配對，含同鏡）→ 11×15
    for u in gt_tracks:
        for v in non_gt:
            emb_diff.append(emb_sim(u, v))

    # sim|GT / sim|nonGT：對 query 的 track.sim（節點證據用）
    sim_gt = np.asarray([float(t.sim) for t in gt_tracks], dtype=np.float64)
    sim_nongt = np.asarray([float(t.sim) for t in non_gt], dtype=np.float64)

    # 時間重疊如實記錄；dH|same：有 H 的重疊交接
    for i, u in enumerate(gt_tracks):
        for v in gt_tracks[i + 1 :]:
            if not _calib_time_overlap(u, v):
                continue
            ov = min(u.t_end, v.t_end) - max(u.t_start, v.t_start)
            time_overlaps.append({"a": u.tid, "b": v.tid, "overlap_sec": ov})
            ok_h, d = same_object_h(u, v)
            if ok_h and d is not None:
                dh_same.append(float(d))

    # dt|same：t_start 排序相鄰且 edge_check 合法
    ordered = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))
    dt_edges = []
    for u, v in zip(ordered, ordered[1:]):
        ok, reason, dt, hop, emb, h_dist = edge_check(u, v)
        if not ok:
            continue
        key = tuple(sorted((u.cam, v.cam)))
        dt_same_by_pair[key].append(float(dt))
        dt_edges.append(
            {
                "from": u.tid,
                "to": v.tid,
                "pair": f"{key[0]}|{key[1]}",
                "dt": float(dt),
                "hop": hop,
                "emb": float(emb),
            }
        )

    return {
        "emb_same": np.asarray(emb_same, dtype=np.float64),
        "emb_diff": np.asarray(emb_diff, dtype=np.float64),
        "dh_same": np.asarray(dh_same, dtype=np.float64),
        "sim_gt": sim_gt,
        "sim_nongt": sim_nongt,
        "dt_same_by_pair": {k: np.asarray(v, dtype=np.float64) for k, v in dt_same_by_pair.items()},
        "emb_gate_for_dt": None,
        "missing_gt": missing,
        "n_gt": len(gt_tracks),
        "n_non_gt": len(non_gt),
        "time_overlaps": time_overlaps,
        "dt_edges": dt_edges,
        "emb_same_compare": emb_same_compare,
        "counts": {
            "n_emb_same": len(emb_same),
            "n_emb_diff": len(emb_diff),
            "n_dh_same": len(dh_same),
            "n_sim_gt": int(len(sim_gt)),
            "n_sim_nongt": int(len(sim_nongt)),
            "n_dt_pairs_total": int(sum(len(v) for v in dt_same_by_pair.values())),
            "n_gt": len(gt_tracks),
            "n_non_gt": len(non_gt),
            "n_time_overlaps": len(time_overlaps),
        },
    }


def fit_calibration(samples: dict) -> dict:
    n_emb_same = int(len(samples["emb_same"]))
    n_emb_diff = int(len(samples["emb_diff"]))
    n_dh_same = int(len(samples["dh_same"]))

    emb_same = _fit_normal(samples["emb_same"])
    emb_diff = _fit_normal(samples["emb_diff"])
    if emb_same is None:
        emb_same = {
            "family": "norm",
            "mu": 0.95,
            "sigma": 0.03,
            "n": n_emb_same,
            "prior": True,
            "reason": "n<2 cannot fit Normal",
        }
    if emb_diff is None:
        emb_diff = {
            "family": "norm",
            "mu": 0.70,
            "sigma": 0.10,
            "n": n_emb_diff,
            "prior": True,
            "reason": "n<2 cannot fit Normal",
        }

    dh_same = _fit_halfnormal(samples["dh_same"])
    if dh_same is None:
        dh_same = {
            "family": "halfnorm",
            "sigma": 40.0,
            "n": n_dh_same,
            "prior": True,
            "reason": "n<2 cannot fit HalfNormal",
        }

    # 收縮權重如實寫入（供報告；LLR 執行時用 n 重算 w=n/(n+10)）
    for dist in (emb_same, emb_diff, dh_same):
        n = int(dist.get("n") or 0)
        dist["shrink_w"] = float(n) / float(n + SHRINK_K)

    sim_gt = _fit_normal(samples["sim_gt"])
    sim_nongt = _fit_normal(samples["sim_nongt"])
    if sim_gt is None:
        sim_gt = {
            "family": "norm",
            "mu": float(np.mean(samples["sim_gt"])) if len(samples["sim_gt"]) else 0.9,
            "sigma": 0.05,
            "n": int(len(samples["sim_gt"])),
            "prior": len(samples["sim_gt"]) < 2,
        }
    if sim_nongt is None:
        sim_nongt = {
            "family": "norm",
            "mu": float(np.mean(samples["sim_nongt"])) if len(samples["sim_nongt"]) else 0.8,
            "sigma": 0.05,
            "n": int(len(samples["sim_nongt"])),
            "prior": len(samples["sim_nongt"]) < 2,
        }
    for dist in (sim_gt, sim_nongt):
        n = int(dist.get("n") or 0)
        dist["shrink_w"] = float(n) / float(n + SHRINK_K)

    dt_by_pair = {}
    prior_pairs = []
    for key in sorted(ADJACENT):
        if key in OVERLAP_PAIRS:
            continue
        arr = samples["dt_same_by_pair"].get(key, np.asarray([], dtype=np.float64))
        if len(arr) >= MIN_SAMPLES_FIT:
            fit = _fit_lognormal(arr)
            if fit is not None:
                fit["shrink_w"] = 1.0
                dt_by_pair[key] = fit
                continue
        tau0 = _tau_for_pair(key[0], key[1])
        dt_by_pair[key] = {
            "family": "lognorm",
            "mu": float(np.log(max(tau0, 1e-3))),
            "sigma": PRIOR_SIGMA,
            "n": int(len(arr)),
            "prior": True,
            "prior_physical": True,
            "prior_weak": True,
            "tau": tau0,
            "shrink_w": 1.0,
            "note": "PRIOR-WEAK",
        }
        prior_pairs.append(
            {
                "pair": key,
                "n_samples": int(len(arr)),
                "tau": tau0,
                "sigma": PRIOR_SIGMA,
                "note": "PRIOR-WEAK",
            }
        )

    for key, arr in samples["dt_same_by_pair"].items():
        if key in dt_by_pair:
            continue
        if len(arr) >= MIN_SAMPLES_FIT:
            fit = _fit_lognormal(arr)
            if fit is not None:
                fit["shrink_w"] = 1.0
                dt_by_pair[key] = fit
                continue
        # hop2 等非 ADJACENT 鏡頭對：樣本不足仍記先驗 + 實際 n
        tau0 = _tau_for_pair(key[0], key[1])
        dt_by_pair[key] = {
            "family": "lognorm",
            "mu": float(np.log(max(tau0, 1e-3))),
            "sigma": PRIOR_SIGMA,
            "n": int(len(arr)),
            "prior": True,
            "prior_physical": True,
            "prior_weak": True,
            "tau": tau0,
            "shrink_w": 1.0,
            "note": "PRIOR-WEAK",
        }
        prior_pairs.append(
            {
                "pair": key,
                "n_samples": int(len(arr)),
                "tau": tau0,
                "sigma": PRIOR_SIGMA,
                "note": "PRIOR-WEAK",
            }
        )

    return {
        "emb_same": emb_same,
        "emb_diff": emb_diff,
        "dh_same": dh_same,
        "dh_diff": {"family": "uniform", "low": 0.0, "high": DH_DIFF_UNIFORM_MAX, "n": None},
        "dt_diff": {"family": "uniform", "low": 0.0, "high": float(DT_MAX), "n": None},
        "sim_gt": sim_gt,
        "sim_nongt": sim_nongt,
        "dt_same_by_pair": {f"{a}|{b}": v for (a, b), v in dt_by_pair.items()},
        "prior_pairs": prior_pairs,
        "meta": {
            "CALIB_SOURCE": CALIB_SOURCE,
            "min_samples_fit": MIN_SAMPLES_FIT,
            "prior_sigma": PRIOR_SIGMA,
            "prior_weak_note": "transit dt prior LogNormal sigma=1.0 (was 0.5); tau placeholder",
            "pdf_floor": PDF_FLOOR,
            "shrink_k": SHRINK_K,
            "counts": samples["counts"],
            "missing_gt": samples["missing_gt"],
            "time_overlaps": samples["time_overlaps"],
            "dt_edges": samples["dt_edges"],
            "emb_same_compare": samples["emb_same_compare"],
            "warning": "IN-SAMPLE：校準與評估同一資料集 0507，僅供診斷",
        },
    }


def compute_transition_prior(tracks: list, gt_tids: list[str]) -> dict:
    """
    p_edge = GT真轉移邊數 / 全部合法邊數。
    GT真轉移 = 純 GT 超節點按 t_start 排序後相鄰且 _best_member_edge 合法。
    全部合法邊 = 全候選超節點圖上合法邊數（同聯集 dt 語意）。
    """
    import itertools

    apply_llr_emb_gates(True)
    supers, srep = build_supernodes(tracks)
    n_legal = 0
    for i, j in itertools.permutations(range(len(supers)), 2):
        best, _ = _best_member_edge(supers[i], supers[j])
        if best is not None:
            n_legal += 1

    gt_set = set(gt_tids)
    gt_supers = [
        s
        for s in supers
        if set(s.tids).issubset(gt_set) and len(s.tids) > 0
    ]
    gt_supers.sort(key=lambda s: (s.t_start, s.t_end, s.sid))
    gt_trans = []
    for u, v in zip(gt_supers, gt_supers[1:]):
        best, _ = _best_member_edge(u, v)
        if best is not None:
            uu, vv, dt, hop, emb, h_dist = best
            gt_trans.append(
                {
                    "from": u.label,
                    "to": v.label,
                    "via": f"{uu.tid}->{vv.tid}",
                    "dt": float(dt),
                    "hop": hop,
                    "emb": float(emb),
                }
            )
    n_gt = len(gt_trans)
    p_edge = float(n_gt) / float(n_legal) if n_legal > 0 else 0.0
    return {
        "p_edge": p_edge,
        "n_gt_true_transitions": n_gt,
        "n_legal_edges": n_legal,
        "formula": "p_edge = n_gt_true_transitions / n_legal_edges",
        "gt_transitions": gt_trans,
        "n_gt_supers": len(gt_supers),
        "multi_only": srep.get("multi_only"),
        "ln_p_edge": float(math.log(p_edge)) if p_edge > 0 else None,
        "note": (
            "GT真轉移=純GT超節點時間相鄰且合法；"
            "合法邊=全圖超節點邊（聯集dt）。2026-07-15"
        ),
    }


def save_emb_histogram(emb_same: np.ndarray, emb_diff: np.ndarray, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 41)
    if len(emb_same):
        ax.hist(
            emb_same,
            bins=bins,
            alpha=0.55,
            label=f"same GT×GT cross-cam (n={len(emb_same)})",
            color="tab:green",
            density=True,
        )
    if len(emb_diff):
        ax.hist(
            emb_diff,
            bins=bins,
            alpha=0.55,
            label=f"diff GT×nonGT (n={len(emb_diff)})",
            color="tab:red",
            density=True,
        )
    ax.set_xlabel("embedding cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"emb | same vs diff  [{CALIB_SOURCE}]")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def write_report(calib: dict, samples: dict, hist_png: Path, out_txt: Path) -> None:
    lines = []
    lines.append("=== calibration from GT ===")
    lines.append(f"CALIB_SOURCE={CALIB_SOURCE}")
    lines.append("WARNING: IN-SAMPLE（校準與評估同一資料集），結論僅供診斷")
    lines.append(f"counts: {calib['meta']['counts']}")
    lines.append(f"missing_gt: {calib['meta']['missing_gt']}")
    lines.append("")
    cmp_ = calib["meta"].get("emb_same_compare") or {}
    lines.append("--- emb|same (μ,σ) 剔除 K8-08_43 前後對照 ---")
    b = cmp_.get("before_remove_08_43") or {}
    a = cmp_.get("after_remove_08_43") or {}
    lines.append(
        f"  before (含 08_43): n={b.get('n')} mu={b.get('mu')} sigma={b.get('sigma')}"
    )
    lines.append(
        f"  after  (剔除後):   n={a.get('n')} mu={a.get('mu')} sigma={a.get('sigma')}"
    )
    lines.append("")
    lines.append("--- emb|same (Normal) ---")
    lines.append(str(calib["emb_same"]))
    lines.append("--- emb|diff (Normal) ---")
    lines.append(str(calib["emb_diff"]))
    lines.append("--- sim|GT / sim|nonGT (query sim, Normal) ---")
    lines.append(str(calib.get("sim_gt")))
    lines.append(str(calib.get("sim_nongt")))
    lines.append("--- dH|same (HalfNormal) ---")
    lines.append(str(calib["dh_same"]))
    lines.append("--- dH|diff / dt|diff (fixed Uniform) ---")
    lines.append(str(calib["dh_diff"]))
    lines.append(str(calib["dt_diff"]))
    lines.append("")
    lines.append("--- dt|same by camera pair ---")
    for k, v in sorted(calib["dt_same_by_pair"].items()):
        tag = " PRIOR" if v.get("prior") else ""
        lines.append(
            f"  {k}: n={v.get('n', 0)} mu={v.get('mu'):.4f} "
            f"sigma={v.get('sigma'):.4f} w={v.get('shrink_w', float('nan')):.3f}{tag}"
        )
    lines.append("")
    lines.append(f"prior pairs: {len(calib['prior_pairs'])}")
    for p in calib["prior_pairs"]:
        lines.append(f"  {p['pair']}: n={p['n_samples']} tau={p['tau']:.2f}")
    lines.append("")
    tp = calib.get("transition_prior") or {}
    lines.append("--- transition prior p_edge ---")
    lines.append(f"  formula: {tp.get('formula')}")
    lines.append(
        f"  n_gt_true_transitions={tp.get('n_gt_true_transitions')}  "
        f"n_legal_edges={tp.get('n_legal_edges')}  "
        f"p_edge={tp.get('p_edge')}  ln(p)={tp.get('ln_p_edge')}"
    )
    for e in tp.get("gt_transitions") or []:
        lines.append(
            f"  GT transfer {e['from']} -> {e['to']} via {e['via']}  "
            f"dt={e['dt']:.2f} hop={e['hop']} emb={e['emb']:.3f}"
        )
    lines.append("")
    lines.append(f"GT time overlaps reported: {len(samples['time_overlaps'])}")
    for o in samples["time_overlaps"]:
        lines.append(f"  {o['a']} ↔ {o['b']}  overlap={o['overlap_sec']:.2f}s")
    lines.append("")
    lines.append(f"histogram: {hist_png}")
    lines.append(f"shrink_k={SHRINK_K}  (w=n/(n+K) recorded per distribution)")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")



def cmd_calibrate(argv=None):
    p = argparse.ArgumentParser(description="Calibrate LLR distributions from human GT")
    p.add_argument(
        "--gt",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
    )
    p.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "v1.0",
    )
    args = p.parse_args(argv)

    merge_dir = args.merge_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    global SIM_MIN
    SIM_MIN = float(args.sim_min)
    configure_for_input(str(merge_dir))
    tracks = load_tracks(str(merge_dir))
    print(f"載入 {len(tracks)} 條 track；GT={len(gt['person_tids'])}")

    samples = collect_gt_samples(
        tracks,
        gt["person_tids"],
        removed_mislabel=list(gt.get("removed_mislabel") or ["K8-08_43"]),
    )
    print(
        f"emb_same={samples['counts']['n_emb_same']}  "
        f"emb_diff={samples['counts']['n_emb_diff']}  "
        f"dh_same={samples['counts']['n_dh_same']}  "
        f"sim_gt={samples['counts']['n_sim_gt']}  "
        f"sim_nongt={samples['counts']['n_sim_nongt']}  "
        f"dt={samples['counts']['n_dt_pairs_total']}  "
        f"overlaps={samples['counts']['n_time_overlaps']}"
    )
    cmp_ = samples["emb_same_compare"]
    print(
        "emb|same before/after 08_43:",
        cmp_["before_remove_08_43"],
        "->",
        cmp_["after_remove_08_43"],
    )
    if samples["missing_gt"]:
        print("缺失 GT track：", samples["missing_gt"])

    calib = fit_calibration(samples)
    calib["dataset"] = merge_dir.name
    calib["input_dir"] = str(merge_dir)
    calib["gt_path"] = str(args.gt.resolve())

    # 轉移先驗（需在 emb 門檻覆寫後算；用聯集 dt 超節點邊）
    tp = compute_transition_prior(tracks, gt["person_tids"])
    calib["transition_prior"] = tp
    print(
        f"transition prior: p_edge={tp['p_edge']:.6f} = "
        f"{tp['n_gt_true_transitions']}/{tp['n_legal_edges']}  "
        f"ln(p)={tp.get('ln_p_edge')}"
    )

    pkl_path = out_dir / "calibration_gt0507.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(calib, f)

    hist_png = out_dir / "emb_same_diff_hist_gt0507.png"
    save_emb_histogram(samples["emb_same"], samples["emb_diff"], hist_png)

    report = out_dir / "calibration_gt0507_report.txt"
    write_report(calib, samples, hist_png, report)

    print(f"寫入：{pkl_path}")
    print(f"寫入：{report}")
    print(f"寫入：{hist_png}")
    print(f"CALIB_SOURCE={CALIB_SOURCE}")



# ============================================================
# EVALUATE（原 evaluate.py）
# ============================================================

N_GT = 11  # K8-08_43 已自 GT 剔除（誤標）
SPECIAL_TIDS = ("K8-07_112", "K8-22_22")


def load_gt(path: Path) -> dict:
    gt = json.loads(path.read_text(encoding="utf-8"))
    tids = list(gt["person_tids"])
    if len(tids) != N_GT:
        print(f"警告：GT 標注 {len(tids)} 條（預期 {N_GT}）")
    return gt


def precision_recall(path_tids: list[str], gt_set: set[str]) -> dict:
    n = len(path_tids)
    hit = sum(1 for t in path_tids if t in gt_set)
    return {
        "n_path": n,
        "n_hit": hit,
        "precision": (hit / n) if n else 0.0,
        "recall": hit / float(N_GT),
        "hit_tids": [t for t in path_tids if t in gt_set],
        "miss_tids": [t for t in path_tids if t not in gt_set],
    }


def eval_top_json(top_json: Path, gt_set: set[str]) -> dict:
    data = json.loads(top_json.read_text(encoding="utf-8"))
    rows = []
    for p in data.get("top10_paths") or []:
        tids = p["tids"]
        pr = precision_recall(tids, gt_set)
        rows.append(
            {
                "rank": p.get("rank"),
                "score": p.get("score"),
                "path_probability": p.get("path_probability"),
                "path": " -> ".join(tids),
                "tids": tids,
                **pr,
            }
        )
    return {
        "json": str(top_json.resolve()),
        "n_paths_all": data.get("n_paths_all"),
        "n_paths_maximal": data.get("n_paths_maximal"),
        "top10": rows,
    }


def _score_all_paths_old(tracks: list, calib: dict | None = None) -> list[dict]:
    """舊 path_score 已移除；改以 LLR 枚舉供 gt_best / rank_old 相容欄位。"""
    return _score_all_paths_llr(tracks, calib)


def _score_all_paths_llr(tracks: list, calib: dict | None) -> list[dict]:
    if calib is None:
        raise SystemExit("evaluate 需要 --calibration 以枚舉／計分路徑（舊 path_score 已併入 LLR）")
    all_paths, _, _, nodes, _ = enumerate_paths_llr(tracks, calib)
    scored = []
    for path_idx, edges_info in all_paths:
        sn_path = [nodes[i] for i in path_idx]
        score, _node_evs = path_score_llr(sn_path, edges_info, calib)
        scored.append(
            {
                "tids": expand_path_tids(nodes, path_idx),
                "score": score,
                "edges": edges_info,
            }
        )
    scored.sort(key=lambda p: -p["score"])
    return scored


def find_gt_best_path(scored_paths: list[dict], gt_set: set[str]) -> dict | None:
    """在所有枚舉路徑中找 precision=100% 且 recall 最高者；同分取較短／較早出現。"""
    best = None
    for rank, p in enumerate(scored_paths, 1):
        # rank here is within this scored list (already sorted by that method's score)
        pr = precision_recall(p["tids"], gt_set)
        if pr["precision"] < 1.0 - 1e-12:
            continue
        cand = {
            "tids": p["tids"],
            "path": " -> ".join(p["tids"]),
            "score": p["score"],
            "rank_in_scored": rank,
            **pr,
        }
        if best is None:
            best = cand
            continue
        if cand["recall"] > best["recall"]:
            best = cand
        elif abs(cand["recall"] - best["recall"]) < 1e-12 and cand["n_path"] < best["n_path"]:
            best = cand
    return best


def rank_of_path(scored_paths: list[dict], tids: list[str]) -> int | None:
    key = tuple(tids)
    for i, p in enumerate(scored_paths, 1):
        if tuple(p["tids"]) == key:
            return i
    return None


def _eval_time_overlap(a: Track, b: Track) -> bool:
    return not (a.t_end < b.t_start or b.t_end < a.t_start)


def diagnose_gt_feasibility(tracks: list, gt_tids: list[str]) -> dict:
    by_tid = {t.tid: t for t in tracks}
    missing = [t for t in gt_tids if t not in by_tid]
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    gt_tracks_sorted = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))

    overlaps = []
    for i, a in enumerate(gt_tracks_sorted):
        for b in gt_tracks_sorted[i + 1 :]:
            if _eval_time_overlap(a, b):
                overlaps.append(
                    {
                        "a": a.tid,
                        "b": b.tid,
                        "a_cam": a.cam,
                        "b_cam": b.cam,
                        "a_span": [a.t_start, a.t_end],
                        "b_span": [b.t_start, b.t_end],
                        "overlap_sec": min(a.t_end, b.t_end) - max(a.t_start, b.t_start),
                    }
                )

    # 時間相鄰（排序後 consecutive）edge_check
    consecutive = []
    for u, v in zip(gt_tracks_sorted, gt_tracks_sorted[1:]):
        ok, reason, dt, hop, emb, h_dist = edge_check(u, v)
        consecutive.append(
            {
                "from": u.tid,
                "to": v.tid,
                "ok": bool(ok),
                "reason": reason or "",
                "dt": float(dt) if dt is not None else None,
                "hop": hop,
                "emb": float(emb) if emb is not None else None,
                "h_dist": float(h_dist) if h_dist is not None else None,
                "u_span": [u.t_start, u.t_end],
                "v_span": [v.t_start, v.t_end],
            }
        )

    # 特殊三條：與所有其他 GT 的雙向 edge_check
    special = {}
    for tid in SPECIAL_TIDS:
        if tid not in by_tid:
            special[tid] = {"missing_in_candidates": True, "as_from": [], "as_to": []}
            continue
        u = by_tid[tid]
        as_from, as_to = [], []
        for v in gt_tracks:
            if v.tid == tid:
                continue
            ok, reason, dt, hop, emb, h_dist = edge_check(u, v)
            as_from.append(
                {
                    "to": v.tid,
                    "ok": bool(ok),
                    "reason": reason or "",
                    "dt": float(dt) if dt is not None else None,
                    "hop": hop,
                    "emb": float(emb) if emb is not None else None,
                    "h_dist": float(h_dist) if h_dist is not None else None,
                }
            )
            ok2, reason2, dt2, hop2, emb2, h_dist2 = edge_check(v, u)
            as_to.append(
                {
                    "from": v.tid,
                    "ok": bool(ok2),
                    "reason": reason2 or "",
                    "dt": float(dt2) if dt2 is not None else None,
                    "hop": hop2,
                    "emb": float(emb2) if emb2 is not None else None,
                    "h_dist": float(h_dist2) if h_dist2 is not None else None,
                }
            )
        special[tid] = {
            "missing_in_candidates": False,
            "span": [u.t_start, u.t_end],
            "sim": u.sim,
            "n_ok_as_from": sum(1 for x in as_from if x["ok"]),
            "n_ok_as_to": sum(1 for x in as_to if x["ok"]),
            "as_from": as_from,
            "as_to": as_to,
        }

    # GT 子圖：所有有序對的 edge_check；再用 DFS + EMB_HIST_MIN 找最長可行路徑
    gt_list = gt_tracks_sorted
    n = len(gt_list)
    succ = [[] for _ in range(n)]
    all_edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u, v = gt_list[i], gt_list[j]
            ok, reason, dt, hop, emb, h_dist = edge_check(u, v)
            rec = {
                "from": u.tid,
                "to": v.tid,
                "ok": bool(ok),
                "reason": reason or "",
                "dt": float(dt) if dt is not None else None,
                "hop": hop,
                "emb": float(emb) if emb is not None else None,
                "h_dist": float(h_dist) if h_dist is not None else None,
            }
            all_edges.append(rec)
            if ok:
                succ[i].append((j, dt, hop, emb, h_dist))

    longest = {"tids": [], "n": 0, "edges": []}

    def dfs(idx, path_idx, edges_info, hist_embs):
        nonlocal longest
        tids = [gt_list[k].tid for k in path_idx]
        if len(tids) > longest["n"]:
            longest = {
                "tids": list(tids),
                "n": len(tids),
                "edges": list(edges_info),
            }
        for j, dt, hop, emb, h_dist in succ[idx]:
            if j in path_idx:
                continue
            v = gt_list[j]
            hsim = hist_emb_sim(hist_embs, v)
            emb_need = (
                EMB_HIST_MIN - 0.02
                if (h_dist is not None and h_dist < H_DIST_GATE)
                else EMB_HIST_MIN
            )
            if hsim < emb_need:
                continue
            u = gt_list[idx]
            edges_info.append(
                {
                    "from": u.tid,
                    "to": v.tid,
                    "dt": dt,
                    "hop": hop,
                    "emb": emb,
                    "hist_emb": hsim,
                    "h_dist": h_dist,
                }
            )
            path_idx.append(j)
            hist_embs.append(v.emb)
            dfs(j, path_idx, edges_info, hist_embs)
            hist_embs.pop()
            path_idx.pop()
            edges_info.pop()

    for s in range(n):
        dfs(s, [s], [], [gt_list[s].emb])

    # 瓶頸：連續時間序中被拒的邊，以及最長路徑未覆蓋的 GT
    bottlenecks = [e for e in consecutive if not e["ok"]]
    uncovered = [t for t in gt_tids if t not in set(longest["tids"])]

    return {
        "n_gt": len(gt_tids),
        "n_gt_in_candidates": len(gt_tracks),
        "missing_from_candidates": missing,
        "gt_sorted_by_t_start": [
            {
                "tid": t.tid,
                "cam": t.cam,
                "t_start": t.t_start,
                "t_end": t.t_end,
                "sim": t.sim,
            }
            for t in gt_tracks_sorted
        ],
        "time_overlaps_among_gt": overlaps,
        "consecutive_edge_checks": consecutive,
        "special_tracks": special,
        "n_legal_gt_edges": sum(1 for e in all_edges if e["ok"]),
        "n_checked_gt_ordered_pairs": len(all_edges),
        "longest_feasible_path": longest,
        "max_gt_coverable": longest["n"],
        "uncovered_by_longest": uncovered,
        "bottleneck_consecutive_edges": bottlenecks,
        "note": (
            "時間重疊的 GT 對如實列出，不当成異常剔除。"
            "最長可行路徑使用與管線相同的 edge_check + EMB_HIST_MIN gate。"
        ),
    }


def write_diagnose_txt(diag: dict, out_txt: Path) -> None:
    lines = []
    lines.append("=== GT 軌跡可行性診斷 ===")
    lines.append(f"GT 數={diag['n_gt']}  候選中找到={diag['n_gt_in_candidates']}")
    if diag["missing_from_candidates"]:
        lines.append("缺失：" + ", ".join(diag["missing_from_candidates"]))
    lines.append("")
    lines.append("--- GT 按 t_start 排序 ---")
    for r in diag["gt_sorted_by_t_start"]:
        lines.append(
            f"  {r['tid']:12} cam={r['cam']}  "
            f"[{r['t_start']:.1f}, {r['t_end']:.1f}]  sim={r['sim']:.3f}"
        )
    lines.append("")
    lines.append(f"--- GT 之間時間重疊（共 {len(diag['time_overlaps_among_gt'])} 對）---")
    if not diag["time_overlaps_among_gt"]:
        lines.append("  （無）")
    for o in diag["time_overlaps_among_gt"]:
        lines.append(
            f"  {o['a']} ↔ {o['b']}  overlap={o['overlap_sec']:.2f}s  "
            f"cams={o['a_cam']}|{o['b_cam']}"
        )
    lines.append("")
    lines.append("--- 時間相鄰 GT 邊（edge_check）---")
    for e in diag["consecutive_edge_checks"]:
        if e["ok"]:
            h = f"{e['h_dist']:.1f}px" if e["h_dist"] is not None else "—"
            lines.append(
                f"  OK   {e['from']} -> {e['to']}  "
                f"hop={e['hop']} dt={e['dt']:.2f}s emb={e['emb']:.3f} h={h}"
            )
        else:
            lines.append(f"  REJECT {e['from']} -> {e['to']}  reason={e['reason']}")
    lines.append("")
    lines.append("--- 特殊 track（K8-07_112 / K8-22_22；08_43 已自 GT 剔除）---")
    for tid in SPECIAL_TIDS:
        sp = diag["special_tracks"].get(tid, {})
        lines.append(f"## {tid}")
        if sp.get("missing_in_candidates"):
            lines.append("  （不在候選池）")
            continue
        lines.append(
            f"  span=[{sp['span'][0]:.1f},{sp['span'][1]:.1f}] sim={sp['sim']:.3f}  "
            f"ok_as_from={sp['n_ok_as_from']} ok_as_to={sp['n_ok_as_to']}"
        )
        lines.append("  作為 from（到其他 GT）：")
        for x in sp["as_from"]:
            tag = "OK" if x["ok"] else "REJECT"
            extra = (
                f"hop={x['hop']} dt={x['dt']:.2f} emb={x['emb']:.3f}"
                if x["ok"]
                else x["reason"]
            )
            lines.append(f"    {tag:6} -> {x['to']:12}  {extra}")
        lines.append("  作為 to（從其他 GT 來）：")
        for x in sp["as_to"]:
            tag = "OK" if x["ok"] else "REJECT"
            extra = (
                f"hop={x['hop']} dt={x['dt']:.2f} emb={x['emb']:.3f}"
                if x["ok"]
                else x["reason"]
            )
            lines.append(f"    {tag:6} {x['from']:12} ->  {extra}")
    lines.append("")
    lines.append("=== 結論 ===")
    lines.append(
        f"現有硬規則（edge_check + EMB_HIST_MIN）下，"
        f"一條路徑最多能涵蓋 {diag['max_gt_coverable']} / {diag['n_gt']} 條 GT。"
    )
    lines.append("最長可行路徑：" + " -> ".join(diag["longest_feasible_path"]["tids"]))
    if diag["uncovered_by_longest"]:
        lines.append("未覆蓋：" + ", ".join(diag["uncovered_by_longest"]))
    lines.append("時間相鄰瓶頸邊：")
    if not diag["bottleneck_consecutive_edges"]:
        lines.append("  （無——連續序全部合法；瓶頸可能在非相鄰跳接或 hist gate）")
    for e in diag["bottleneck_consecutive_edges"]:
        lines.append(f"  {e['from']} -> {e['to']}: {e['reason']}")
    lines.append("")
    lines.append(diag["note"])
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")



def cmd_evaluate(argv=None):
    p = argparse.ArgumentParser(description="Evaluate paths vs GT + diagnose GT feasibility")
    p.add_argument(
        "--gt",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
    )
    p.add_argument(
        "--old-json",
        type=Path,
        default=OUTPUT_ROOT / "path_enum" / "人員追蹤_20260507_top1.json",
    )
    p.add_argument(
        "--llr-json",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "0507_top1.json",
    )
    p.add_argument(
        "--llr-gt-json",
        type=Path,
        default=None,
        help="可選：GT 校準後 LLR top JSON",
    )
    p.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "calibration_gt0507.pkl",
        help="用於重算全路徑 LLR 排名的校準檔",
    )
    p.add_argument(
        "--calibration-gt",
        type=Path,
        default=None,
        help="可選：GT 校準檔，用於第三套排名",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "v1.0",
    )
    args = p.parse_args(argv)

    gt = load_gt(args.gt)
    gt_set = set(gt["person_tids"])
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Top-10 評估（來自已存 JSON）---
    results = {
        "gt": gt,
        "old": eval_top_json(args.old_json, gt_set) if args.old_json.is_file() else None,
        "llr_prior": eval_top_json(args.llr_json, gt_set) if args.llr_json.is_file() else None,
        "llr_gt": None,
    }
    if args.llr_gt_json and args.llr_gt_json.is_file():
        results["llr_gt"] = eval_top_json(args.llr_gt_json, gt_set)

    # --- 全路徑掃描：GT 最佳 + 各計分排名 ---
    merge_dir = args.merge_dir.resolve()
    global SIM_MIN
    SIM_MIN = float(args.sim_min)
    configure_for_input(str(merge_dir))
    tracks = load_tracks(str(merge_dir))
    all_tids = {t.tid for t in tracks}
    non_gt = sorted(all_tids - gt_set)
    results["candidate_pool"] = {
        "n_tracks": len(tracks),
        "n_gt_in_pool": sum(1 for t in gt_set if t in all_tids),
        "n_non_gt": len(non_gt),
        "non_gt_tids": non_gt,
    }

    calib = None
    scored_llr = None
    if args.calibration.is_file():
        print(f"載入校準並以 LLR 枚舉全路徑：{args.calibration}")
        calib = load_calibration(args.calibration)
        apply_llr_emb_gates(True)
        scored_llr = _score_all_paths_llr(tracks, calib)
    else:
        print(f"找不到校準檔，跳過 GT-best／LLR 排名：{args.calibration}")

    # 舊 path_score 已移除；rank_old 與 score_old 改記 LLR 結果（相容欄位名）
    scored_old = scored_llr or []
    print("以 LLR 路徑集合尋找 GT 最佳路徑…")
    gt_best = find_gt_best_path(scored_old, gt_set) if scored_old else None
    results["gt_best_path"] = None
    if gt_best:
        results["gt_best_path"] = {
            "tids": gt_best["tids"],
            "path": gt_best["path"],
            "precision": gt_best["precision"],
            "recall": gt_best["recall"],
            "n_hit": gt_best["n_hit"],
            "n_path": gt_best["n_path"],
            "rank_old": rank_of_path(scored_old, gt_best["tids"]),
            "rank_llr_prior": rank_of_path(scored_old, gt_best["tids"]),
            "rank_llr_gt": None,
            "score_old": next(
                (p["score"] for p in scored_old if p["tids"] == gt_best["tids"]), None
            ),
            "score_llr_prior": next(
                (p["score"] for p in scored_old if p["tids"] == gt_best["tids"]), None
            ),
            "note": "rank_old/score_old = LLR（舊 path_score 已併入）",
        }
        print(
            f"GT 最佳路徑 recall={gt_best['recall']:.3f}  "
            f"len={gt_best['n_path']}  llr_rank={results['gt_best_path']['rank_llr_prior']}"
        )
        print("  " + gt_best["path"])

    if args.calibration_gt and args.calibration_gt.is_file():
        print(f"用 GT 校準重算 LLR 全路徑：{args.calibration_gt}")
        calib_gt = load_calibration(args.calibration_gt)
        scored_llr_gt = _score_all_paths_llr(tracks, calib_gt)
        if results["gt_best_path"]:
            results["gt_best_path"]["rank_llr_gt"] = rank_of_path(
                scored_llr_gt, results["gt_best_path"]["tids"]
            )
            results["gt_best_path"]["score_llr_gt"] = next(
                (
                    p["score"]
                    for p in scored_llr_gt
                    if p["tids"] == results["gt_best_path"]["tids"]
                ),
                None,
            )

    # --- 可行性診斷 ---
    print("GT 可行性診斷…")
    diag = diagnose_gt_feasibility(tracks, gt["person_tids"])
    results["feasibility"] = diag
    diag_txt = out_dir / "gt_feasibility_20260507.txt"
    write_diagnose_txt(diag, diag_txt)
    print(f"可行性報告：{diag_txt}")
    print(
        f"結論：最多可涵蓋 {diag['max_gt_coverable']}/{diag['n_gt']} 條 GT"
    )

    out_json = out_dir / "evaluate_20260507.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"評估 JSON：{out_json}")

    # 簡短 Top-10 表
    for tag in ("old", "llr_prior", "llr_gt"):
        block = results.get(tag)
        if not block:
            continue
        print(f"\n===== {tag} Top-10 precision/recall =====")
        for r in block["top10"]:
            pp = r.get("path_probability")
            pp_s = f"P={pp:.4f}" if pp is not None else ""
            print(
                f"  #{r['rank']} prec={r['precision']:.2f} rec={r['recall']:.2f} "
                f"hit={r['n_hit']}/{r['n_path']} {pp_s}  {r['path']}"
            )

    return results



# ============================================================
# VIZ（原 render_viz.py）
# ============================================================

OUT = V1_OUTPUT_ROOT

def _viz_font(size: int, bold: bool = False):
    cands = [
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in cands:
        p = Path(name)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()



def _viz_thumb(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None or not Path(path).is_file():
        return Image.new("RGB", size, (230, 230, 230))
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (236, 236, 236))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas



def _pick_three(crops: list[Path]) -> list[Path | None]:
    if not crops:
        return [None, None, None]
    if len(crops) == 1:
        return [crops[0], crops[0], crops[0]]
    if len(crops) == 2:
        return [crops[0], crops[0], crops[1]]
    return [crops[0], crops[len(crops) // 2], crops[-1]]



def parse_super_columns(path: dict) -> list[dict]:
    """回傳 columns: [{label, members:[tid,...]}]，優先用 edges 的 members。"""
    edges = path.get("edges") or []
    labels = path.get("super_labels")
    tids = path.get("tids") or []

    if edges and edges[0].get("from_members") is not None:
        cols = [{"label": edges[0].get("from_super") or edges[0]["from"], "members": list(edges[0]["from_members"])}]
        for e in edges:
            cols.append(
                {
                    "label": e.get("to_super") or e["to"],
                    "members": list(e.get("to_members") or [e["to"]]),
                }
            )
        return cols

    if labels:
        cols = []
        for lab in labels:
            if lab.startswith("{") and lab.endswith("}"):
                members = [x.strip() for x in lab[1:-1].split(",") if x.strip()]
            else:
                members = [lab]
            cols.append({"label": lab, "members": members})
        return cols

    return [{"label": t, "members": [t]} for t in tids]



def track_meta(by_tid: dict, tid: str) -> dict:
    t = by_tid.get(tid)
    if t is None:
        cam, tid_s = tid.rsplit("_", 1)
        return {
            "tid": tid,
            "cam": cam,
            "t_start": None,
            "t_end": None,
            "sim": None,
        }
    return {
        "tid": tid,
        "cam": t.cam,
        "t_start": float(t.t_start),
        "t_end": float(t.t_end),
        "sim": float(t.sim),
    }



def load_crops(merge_dir: Path, tid: str) -> list[Path]:
    cam, tid_s = tid.rsplit("_", 1)
    try:
        _, crops = _crop_paths_for_track(merge_dir, cam, int(tid_s))
        return crops
    except Exception:
        return []



def render_top1_sequence(
    top1: dict,
    by_tid: dict,
    merge_dir: Path,
    gt_set: set[str],
    out_png: Path,
    segments: list | None = None,
    title: str | None = None,
) -> tuple[Path, list[str]]:
    """
    橫軸=絕對時間；每一路徑節點佔一列（依路徑順序由上而下）；
    超節點成員在該列內上下堆疊；空檔在橫軸標 dt/hop；
    GT=綠、非GT=紅。
    """
    columns = parse_super_columns(top1)
    edges = top1.get("edges") or []

    # 附加 segment（若有 seg2+）
    blocks: list[tuple[str, list[dict], list[dict], float | None]] = [
        ("seg1", columns, edges, None)
    ]
    for seg in segments or []:
        if int(seg.get("segment") or 1) <= 1:
            continue
        fake = {
            "super_labels": seg.get("super_labels"),
            "tids": seg.get("tids"),
            "edges": seg.get("edges") or [],
        }
        cols = parse_super_columns(fake)
        blocks.append(
            (
                f"seg{seg['segment']}",
                cols,
                fake.get("edges") or [],
                float(seg.get("gap_after_prev_sec") or 0.0),
            )
        )

    all_members: list[str] = []
    for _, cols, _, _ in blocks:
        for c in cols:
            all_members.extend(c["members"])
    metas = [track_meta(by_tid, t) for t in all_members if t in by_tid]
    t_min = min(m["t_start"] for m in metas)
    t_max = max(m["t_end"] for m in metas)
    if t_max <= t_min:
        t_max = t_min + 1.0

    thumb_w, thumb_h = 64, 86
    mem_h = thumb_h + 36
    lane_pad = 18
    title_h = 70
    axis_h = 50
    margin_l, margin_r, margin_b = 160, 30, 50

    # 每欄列高 = 成員數 * mem_h
    lane_heights = []
    for _, cols, _, gap in blocks:
        if gap is not None:
            lane_heights.append(36)  # 空窗列
        for c in cols:
            lane_heights.append(max(1, len(c["members"])) * mem_h + lane_pad)

    body_h = sum(lane_heights) + 20
    px_per_sec = 2.4
    plot_w = max(1400, int((t_max - t_min) * px_per_sec) + 180)
    width = margin_l + plot_w + margin_r
    height = title_h + body_h + axis_h + margin_b

    img = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    font_t = _viz_font(16)
    font_s = _viz_font(11)
    font_xs = _viz_font(9)
    font_b = _viz_font(11, bold=True)

    score = top1.get("score")
    pprob = top1.get("path_probability")
    n_seg = int(top1.get("n_segments") or (1 + sum(1 for s in (segments or []) if int(s.get("segment") or 1) > 1)))
    default_title = (
        f"Top-1 sequence  n_seg={n_seg}  score={score:.3f}  P={pprob:.4f}"
        if score is not None
        else "Top-1 sequence"
    )
    draw.text(
        (16, 8),
        title or default_title,
        fill=(20, 20, 20),
        font=font_t,
    )
    ly = 36
    draw.rectangle([16, ly, 30, ly + 14], outline=(34, 139, 34), width=3)
    draw.text((34, ly), "GT", fill=(34, 139, 34), font=font_xs)
    draw.rectangle([70, ly, 84, ly + 14], outline=(200, 40, 40), width=3)
    draw.text((88, ly), "non-GT（07_1/07_93/09_96/07_139/09_167 等）", fill=(200, 40, 40), font=font_xs)
    draw.text((420, ly), "列=路徑順序；橫軸=絕對時間；超節點成員同列堆疊", fill=(80, 80, 80), font=font_xs)

    def x_of(t: float) -> int:
        return margin_l + int((t - t_min) / (t_max - t_min) * plot_w)

    axis_y = title_h + body_h + 4
    draw.line([(margin_l, axis_y), (margin_l + plot_w, axis_y)], fill=(50, 50, 50), width=2)
    for i in range(0, int(t_max) + 1, 50):
        if i < t_min - 1:
            continue
        xx = x_of(float(i))
        if xx < margin_l or xx > margin_l + plot_w:
            continue
        draw.line([(xx, axis_y - 5), (xx, axis_y + 5)], fill=(50, 50, 50))
        draw.text((xx - 10, axis_y + 8), f"{i}s", fill=(70, 70, 70), font=font_xs)

    crop_log: list[str] = []
    y = title_h
    prev_box_x1 = None

    for bi, (bname, cols, bedges, gap) in enumerate(blocks):
        if gap is not None:
            # 空窗列
            draw.rectangle(
                [margin_l, y, margin_l + plot_w, y + 28],
                fill=(255, 245, 230),
                outline=(200, 120, 40),
            )
            draw.text(
                (margin_l + 8, y + 6),
                f"觀測空窗 {gap:.1f}s  →  {bname}",
                fill=(160, 80, 0),
                font=font_b,
            )
            y += 36

        for ci, col in enumerate(cols):
            members = col["members"]
            spans = []
            for tid in members:
                m = track_meta(by_tid, tid)
                if m["t_start"] is None:
                    continue
                spans.append((float(m["t_start"]), float(m["t_end"]), tid, m))
            spans.sort(key=lambda z: (z[0], z[2]))
            n = max(1, len(spans))
            lane_h = n * mem_h + lane_pad
            # 左側標籤
            draw.text((8, y + 8), col["label"][:28], fill=(30, 30, 30), font=font_xs)
            if bi == 0 and ci < len(edges):
                # path order index
                draw.text((8, y + 22), f"#{ci+1}", fill=(100, 100, 100), font=font_xs)

            if not spans:
                y += lane_h
                continue

            t0 = min(s[0] for s in spans)
            t1 = max(s[1] for s in spans)
            x0 = x_of(t0)
            x1 = max(x_of(max(t1, t0 + 0.3)), x0 + thumb_w + 160)

            if all(tid in gt_set for tid in members):
                bc = (34, 139, 34)
            else:
                bc = (200, 40, 40)

            # 外框（超節點合併）
            pad = 3
            draw.rectangle(
                [x0 - pad, y + 2, x1 + pad, y + n * mem_h + 4],
                outline=bc,
                width=3 if len(members) > 1 else 2,
            )
            if len(members) > 1:
                draw.text((x0, y - 1), "超節點合併", fill=bc, font=font_xs)

            for mi, (ts, te, tid, m) in enumerate(spans):
                yy = y + mi * mem_h + 4
                bx0 = x_of(ts)
                bx1 = max(x_of(max(te, ts + 0.25)), bx0 + 4)
                # 時間橫條（細）
                bar_y0, bar_y1 = yy + 2, yy + 14
                draw.rectangle([bx0, bar_y0, bx1, bar_y1], fill=bc, outline=bc)
                # crop
                crops = load_crops(merge_dir, tid)
                mid = _pick_three(crops)[1]
                thumb = _viz_thumb(mid, (thumb_w, thumb_h))
                cx = min(max(bx0, x0), max(x0, x1 - thumb_w))
                img.paste(thumb, (cx, yy + 16))
                draw.rectangle(
                    [cx, yy + 16, cx + thumb_w, yy + 16 + thumb_h],
                    outline=bc,
                    width=2,
                )
                sim_s = f"{m['sim']:.3f}" if m["sim"] is not None else "?"
                draw.text(
                    (cx + thumb_w + 4, yy + 18),
                    f"{tid}  {m['cam']}",
                    fill=(20, 20, 20),
                    font=font_xs,
                )
                draw.text(
                    (cx + thumb_w + 4, yy + 32),
                    f"sim={sim_s}  [{ts:.1f},{te:.1f}]",
                    fill=(60, 60, 60),
                    font=font_xs,
                )
                if mid is not None:
                    crop_log.append(f"{tid}\t{mid}")
                else:
                    crop_log.append(f"{tid}\t(no crop)")

            # 與前一欄的邊：標在列左側（路徑序），避免絕對時間與路徑序不一致時軸上錯位
            if ci > 0 and ci - 1 < len(bedges):
                e = bedges[ci - 1]
                dt = e.get("dt")
                hop = e.get("hop")
                label = f"↑ dt={dt:.1f}s hop={hop}" if dt is not None else f"↑ hop={hop}"
                draw.text((8, y + lane_h - 16), label, fill=(40, 40, 140), font=font_xs)
                # 若時間上嚴格先後，另在橫軸畫區間
                if prev_box_x1 is not None and x0 > prev_box_x1 + 4:
                    ay = axis_y - 18
                    draw.line([(prev_box_x1, ay), (x0 - pad, ay)], fill=(80, 80, 160), width=2)
                    draw.text(
                        ((prev_box_x1 + x0) // 2 - 28, ay - 14),
                        f"dt={dt:.1f}s",
                        fill=(40, 40, 140),
                        font=font_xs,
                    )

            prev_box_x1 = x1 + pad
            draw.line(
                [(margin_l, y + n * mem_h + lane_pad - 4), (margin_l + plot_w, y + n * mem_h + lane_pad - 4)],
                fill=(230, 230, 230),
            )
            y += lane_h

        prev_box_x1 = None  # 下一段重新起算橫軸標註

    draw.text(
        (16, height - 32),
        "紅=非GT／綠=GT。僅供人工檢視，不改演算法。",
        fill=(70, 70, 70),
        font=font_xs,
    )
    draw.text(
        (16, height - 18),
        f"時間範圍 [{t_min:.1f}, {t_max:.1f}]s",
        fill=(90, 90, 90),
        font=font_xs,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png, crop_log





def _labels_from_seg(seg: dict) -> list[str]:
    labs = seg.get("super_labels")
    if labs:
        return list(labs)
    path = seg.get("path") or ""
    parts = []
    buf = ""
    depth = 0
    for ch in path:
        if ch == "{":
            depth += 1
            buf += ch
        elif ch == "}":
            depth -= 1
            buf += ch
        elif ch == ">" and depth == 0:
            parts.append(buf.strip().rstrip("-").strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


def rebuild_display_path(nodes, labels, calib) -> dict:
    r = score_labeled_path(
        nodes, labels, calib, dt_scoring=False, transition_prior=False
    )
    if not r.get("ok"):
        raise RuntimeError(f"viz rebuild failed: {r.get('reason')} labels={labels}")
    return {
        "score": r["score"],
        "tids": r["tids"],
        "super_labels": r["super_labels"],
        "edges": r["edges"],
        "node_evidence": r["node_evidence"],
        "n_segments": 1,
    }


def render_dataset(tag: str, short: str, n_gt_label: str) -> Path:
    merge = QUERY_FILTER_OUTPUT_ROOT / tag
    top_json = OUT / f"{short}_top1.json"
    gt_path = OUT / (
        "ground_truth_20260507.json" if short == "0507" else "ground_truth_20260528.json"
    )
    calib_path = OUT / "calibration_gt0507.pkl"

    global SIM_MIN
    SIM_MIN = 0.85
    configure_for_input(str(merge))
    apply_llr_emb_gates(True)
    tracks = load_tracks(str(merge))
    by_tid = {t.tid: t for t in tracks}
    nodes, _ = build_supernodes(tracks)
    calib = pickle.loads(calib_path.read_bytes())
    gt_set = set(json.loads(gt_path.read_text(encoding="utf-8"))["person_tids"])

    data = json.loads(top_json.read_text(encoding="utf-8"))
    top = data["top1"]
    segs_src = top.get("segments") or data.get("segments") or []
    segs_src = sorted(segs_src, key=lambda s: int(s.get("segment") or 1))

    rebuilt = []
    for seg in segs_src:
        labs = _labels_from_seg(seg)
        disp = rebuild_display_path(nodes, labs, calib)
        rebuilt.append(
            {
                "segment": int(seg.get("segment") or 1),
                "path": " -> ".join(disp["super_labels"]),
                "super_labels": disp["super_labels"],
                "tids": disp["tids"],
                "score": float(seg.get("score") or disp["score"]),
                "t_start": seg.get("t_start"),
                "t_end": seg.get("t_end"),
                "gap_after_prev_sec": seg.get("gap_after_prev_sec"),
                "edges": disp["edges"],
            }
        )

    seg1 = rebuilt[0]
    top1_viz = {
        "score": top.get("score"),
        "path_probability": top.get("path_probability"),
        "n_segments": top.get("n_segments") or len(rebuilt),
        "super_labels": seg1["super_labels"],
        "tids": seg1["tids"],
        "edges": seg1["edges"],
    }
    extra = [s for s in rebuilt if int(s["segment"]) > 1]
    n_seg = int(top1_viz["n_segments"])
    title = (
        f"{tag}  v1.0 Top-1  n_seg={n_seg}  "
        f"score={top1_viz['score']:.3f}  P={top1_viz['path_probability']:.4f}  "
        f"({n_gt_label})"
    )
    viz_dir = OUT / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_png = viz_dir / f"{short}_top1_sequence.png"
    png, crop_log = render_top1_sequence(
        top1_viz, by_tid, merge, gt_set, out_png, segments=extra, title=title
    )
    crop_txt = viz_dir / f"{short}_top1_crop_list.txt"
    crop_txt.write_text("tid\tcrop_path\n" + "\n".join(crop_log) + "\n", encoding="utf-8")
    print(f"寫入 {png}")
    return png



def cmd_viz(argv=None):
    _ = argv  # reserved for future CLI flags
    OUT.mkdir(parents=True, exist_ok=True)
    render_dataset("人員追蹤_20260507", "0507", "0507 in-sample GT=11")
    render_dataset("人員追蹤_20260528", "0528", "0528 OOS GT=16")
    print(f"視覺化目錄：{OUT / 'viz'}")



# ============================================================
# CLI：subcommands run / calibrate / evaluate / viz
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="track_path：跨鏡頭路徑枚舉 LLR 計分 + calibrate/evaluate/viz"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # run — mirrors former track_path.main
    pr = sub.add_parser("run", help="LLR 路徑枚舉計分（原 track_path 主程式）")
    pr.add_argument(
        "input_dir",
        nargs="?",
        default=str(QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507"),
    )
    pr.add_argument("--sim-min", type=float, default=0.85)
    pr.add_argument("--calibration", type=Path, default=None)
    pr.add_argument("--out-dir", type=Path, default=None)
    pr.add_argument("--no-emb-gate-fix", action="store_true", help="關閉修正一（維持 0.91/0.90）")
    pr.add_argument("--no-supernode", action="store_true", help="關閉修正二")
    pr.add_argument("--no-node-evidence", action="store_true", help="關閉修正三節點證據")
    pr.add_argument(
        "--dt-scoring",
        choices=["on", "off"],
        default="on",
        help="transit LLR_dt：on=計分 / off=0+removed（硬規則不動）",
    )
    pr.add_argument(
        "--transition-prior",
        choices=["on", "off"],
        default="off",
        help="每邊加 ln(p_edge)；p_edge 來自 GT 校準",
    )
    pr.set_defaults(_handler=cmd_run_from_ns)

    # calibrate
    pc = sub.add_parser("calibrate", help="以人工 GT 校準 LLR 分布")
    pc.add_argument(
        "--gt",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
    )
    pc.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
    )
    pc.add_argument("--sim-min", type=float, default=0.85)
    pc.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "v1.0",
    )
    pc.set_defaults(_handler=cmd_calibrate)

    # evaluate
    pe = sub.add_parser("evaluate", help="路徑評估 + GT 可行性診斷")
    pe.add_argument(
        "--gt",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
    )
    pe.add_argument(
        "--old-json",
        type=Path,
        default=OUTPUT_ROOT / "path_enum" / "人員追蹤_20260507_top1.json",
    )
    pe.add_argument(
        "--llr-json",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "0507_top1.json",
    )
    pe.add_argument(
        "--llr-gt-json",
        type=Path,
        default=None,
        help="可選：GT 校準後 LLR top JSON",
    )
    pe.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
    )
    pe.add_argument(
        "--calibration",
        type=Path,
        default=OUTPUT_ROOT / "v1.0" / "calibration_gt0507.pkl",
        help="用於重算全路徑 LLR 排名的校準檔",
    )
    pe.add_argument(
        "--calibration-gt",
        type=Path,
        default=None,
        help="可選：GT 校準檔，用於第三套排名",
    )
    pe.add_argument("--sim-min", type=float, default=0.85)
    pe.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "v1.0",
    )
    pe.set_defaults(_handler=cmd_evaluate)

    # viz
    pv = sub.add_parser("viz", help="v1.0 Top-1 時間軸視覺化")
    pv.set_defaults(_handler=cmd_viz)

    return p


def cmd_run_from_ns(args=None, argv=None):
    """Adapter: subparser Namespace → cmd_run via argv reconstruction, or direct ns."""
    if argv is not None:
        return cmd_run(argv)
    # Rebuild argv from namespace for parse_run_args compatibility
    if args is None:
        return cmd_run(None)
    argv2 = []
    if getattr(args, "input_dir", None):
        argv2.append(str(args.input_dir))
    argv2 += ["--sim-min", str(args.sim_min)]
    if args.calibration is not None:
        argv2 += ["--calibration", str(args.calibration)]
    if args.out_dir is not None:
        argv2 += ["--out-dir", str(args.out_dir)]
    if args.no_emb_gate_fix:
        argv2.append("--no-emb-gate-fix")
    if args.no_supernode:
        argv2.append("--no-supernode")
    if args.no_node_evidence:
        argv2.append("--no-node-evidence")
    argv2 += ["--dt-scoring", args.dt_scoring]
    argv2 += ["--transition-prior", args.transition_prior]
    return cmd_run(argv2)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 相容：未給 subcommand 時預設 run（smoke tests）
    known = {"run", "calibrate", "evaluate", "viz", "-h", "--help"}
    if not argv or argv[0] not in known:
        argv = ["run"] + argv
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args._handler
    if handler is cmd_run_from_ns:
        return handler(args=args)
    if handler is cmd_viz:
        return handler(None)
    # calibrate / evaluate: re-parse with their own argparse using remaining?
    # Their cmd_* builds its own ArgumentParser — pass reconstructed argv without subcommand.
    # Easier: call with namespace by adapting.
    # For calibrate/evaluate, strip subcommand and re-invoke their internal parsers.
    sub = args.command
    # Rebuild argv for nested parsers (they have their own ArgumentParser)
    rest = []
    raw = list(sys.argv[1:] if False else argv)
    # raw already starts with subcommand
    if raw and raw[0] == sub:
        rest = raw[1:]
    else:
        rest = raw
    return handler(rest)


if __name__ == "__main__":
    main()
