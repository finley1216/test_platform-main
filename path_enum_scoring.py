# -*- coding: utf-8 -*-
"""
跨鏡頭全路徑枚舉 + 物理計分 + Top-1 路徑圖
==========================================
接在 query_filter_botsort_merge_filter_dataset.py 之後：

  輸入：../output/query_filter_merge/{dataset}/
        *_merged.json + filter_results/.../kept
  輸出：../output/path_enum/
        *_top1_collage.png（路徑拼圖）
        *_top1.json（路徑摘要）
        *_out.txt（文字報告）

設計原則：
  1. 對 query 不夠像的 track 直接不進候選池（SIM_MIN）
  2. 硬規則決定邊存不存在（時間順序、不可瞬移、拓撲可達、
     相鄰 track 外觀夠像、與路徑歷史外觀夠像）
  3. 軟規則決定邊／節點分數（相鄰鏡頭 + 時間吻合 + 外觀／query 像 → 高分）
  4. 每條路徑的分數可分解到每條邊，方便診斷
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import re
import ast
import pickle
import itertools
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 一、參數（全部有物理意義，用現場實測填，不要用調參填）
# ============================================================

# 鏡頭拓撲：相鄰鏡頭對（雙向）。依資料夾自動選人員／車輛圖。
# 人員：cross_camera_chain_test/run_cross_camera_chain.py DEFAULT_ADJACENCY
# 車輛：run_layered_cluster_v3_vehicle_clean.py VEHICLE_ADJACENCY
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

PERSON_OVERLAP_PAIRS = {
    ("K8-22", "K8-23"): 20.0,
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
H_DIST_GATE = 150.0     # 與既有 cross-cam 匹配一致
H_TIME_WINDOW = 15.0    # 對齊兩 track 腳底點的最大時間差（秒）
W_H = 5.0               # H 判定同物件時的邊分加成
H_MATRICES = {}         # (cam_a, cam_b) -> 3x3 ndarray（a 投影到 b）
# HOMOGRAPHY_DIR 見下方路徑常數區塊

# 每個鏡頭對的最短通行時間（秒）：拿碼表現場走出來的下界。
# 沒填的鏡頭對用 DEFAULT_MIN_TRANSIT。同鏡頭再入預設 0。
MIN_TRANSIT = {
    # ("K8-22", "K8-23"): 2.0,
}
DEFAULT_MIN_TRANSIT_HOP1 = 2.0     # 相鄰鏡頭最短通行（人員）
DEFAULT_MIN_TRANSIT_HOP2 = 6.0     # 跳一支鏡頭最短通行（人員）
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
DT_MAX   = 120.0    # 斷開超過這個秒數不強行連（先寬鬆，診斷後再收）
CAP      = 30.0     # 單一節點的時長獎勵上限（秒），防超長 track 買通分數
LAM      = 0.02     # 遲到懲罰係數：score -= LAM * delay^2
BASE     = {0: 1.0, 1: 1.0, 2: 0.4}   # hop 型態基礎分：同鏡頭/相鄰/跳一支

# ---- 外觀一致性（「原本就很像」+「歷史都很像」）----
SIM_MIN       = 0.90   # 對 query 低於此者不進候選（原本就要夠像）
EMB_EDGE_MIN  = 0.91   # 相鄰兩 track 平均 embedding 餘弦相似度下界
EMB_HIST_MIN  = 0.90   # 新 track 對路徑歷史平均 embedding 下界
W_EMB         = 8.0    # 邊分：+ W_EMB * emb_uv
W_QUERY       = 20.0   # 節點分：+ W_QUERY * sim_query（時長獎勵另計）

TOP_K_PRINT = 10    # 報告印前幾名

CAMERA_RE = re.compile(r"(K8-\d+)")
REPO_ROOT = Path(__file__).resolve().parent
# test_platform：mapping/crop 在上一層 output/；H 矩陣在 Homography/
try:
    from repo_paths import OUTPUT_ROOT as _OUTPUT_ROOT
except ImportError:
    _OUTPUT_ROOT = REPO_ROOT.parent / "output"
OUTPUT_ROOT = Path(_OUTPUT_ROOT)
ASE_ROOT = REPO_ROOT  # 相容舊變數名（本 repo 根）
DEFAULT_MAPPING = OUTPUT_ROOT / "人員追蹤_20260507_crop_time_mapping.json"


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
    DEFAULT_MIN_TRANSIT_HOP1 = 2.0
    DEFAULT_MIN_TRANSIT_HOP2 = 6.0
    DEFAULT_TAU_HOP1 = 8.0
    DEFAULT_TAU_HOP2 = 20.0
    _load_h_matrices()
    return "person"


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

# ============================================================
# 四、軟規則：邊分數 + 路徑分數
# ============================================================

def edge_score(u: Track, v: Track, dt: float, hop: int, emb: float, h_dist=None) -> float:
    delay = max(0.0, dt - tau(u.cam, v.cam, hop))
    # H 同物件：基礎分當 hop1，並加 W_H
    h_ok = h_dist is not None and h_dist < H_DIST_GATE
    base_hop = 1 if h_ok else hop
    sc = BASE[base_hop] - LAM * delay ** 2 + W_EMB * emb
    if h_ok:
        sc += W_H * (1.0 - min(h_dist, H_DIST_GATE) / H_DIST_GATE)
    # 車輛走廊：鼓勵 23→22→20→21→19→28，懲罰逆向亂跳
    if MODE == "vehicle":
        ru, rv = corridor_rank(u.cam), corridor_rank(v.cam)
        if ru < 10_000 and rv < 10_000:
            if rv == ru + 1:
                sc += 12.0          # 走廊下一步
            elif rv > ru:
                sc += 4.0           # 同向但跳站
            elif rv < ru:
                sc -= 20.0          # 逆向（例如 20→22、22→23 當主方向應是 23→22）
    return sc

def node_reward(t: Track) -> float:
    # 時長獎勵 × query 相似度，並另加 query 像度分，避免長但不像的 track 買通
    dur = min(t.t_end - t.t_start, CAP)
    return dur * t.sim + W_QUERY * t.sim

def path_score(path, edges_info):
    return sum(node_reward(t) for t in path) + sum(e["score"] for e in edges_info)

# ============================================================
# 五、全路徑枚舉（DFS）
# ============================================================

def enumerate_paths(tracks):
    n = len(tracks)
    succ = [[] for _ in range(n)]
    rejected_edges = []
    for i, j in itertools.permutations(range(n), 2):
        u, v = tracks[i], tracks[j]
        # 不再強制 v.t_start >= u.t_start：重疊／走廊允許「後鏡頭先被偵測」
        # 僅略過完全無關的反向（v 整段遠早於 u）
        if v.t_end < u.t_start - DT_MAX:
            continue
        ok, reason, dt, hop, emb, h_dist = edge_check(u, v)
        if ok:
            succ[i].append((j, dt, hop, emb, h_dist))
        elif reason:
            rejected_edges.append((u.tid, v.tid, reason))

    all_paths = []

    def dfs(idx, path_idx, edges_info, hist_embs):
        all_paths.append((list(path_idx), list(edges_info)))
        for j, dt, hop, emb, h_dist in succ[idx]:
            if j in path_idx:
                continue
            v = tracks[j]
            # 歷史外觀：新節點必須與路徑上已走 track 的平均 embedding 夠像
            hsim = hist_emb_sim(hist_embs, v)
            emb_need = EMB_HIST_MIN - 0.02 if (h_dist is not None and h_dist < H_DIST_GATE) else EMB_HIST_MIN
            if hsim < emb_need:
                rejected_edges.append(
                    (tracks[idx].tid, v.tid,
                     f"歷史不像（hist_emb={hsim:.3f} < {emb_need}）")
                )
                continue
            u = tracks[idx]
            edges_info.append({
                "from": u.tid, "to": v.tid, "dt": dt, "hop": hop,
                "emb": emb, "hist_emb": hsim,
                "h_dist": h_dist,
                "score": edge_score(u, v, dt, hop, emb, h_dist),
            })
            path_idx.append(j)
            hist_embs.append(v.emb)
            dfs(j, path_idx, edges_info, hist_embs)
            hist_embs.pop()
            path_idx.pop()
            edges_info.pop()

    for s in range(n):
        dfs(s, [s], [], [tracks[s].emb])
    return all_paths, rejected_edges

# ============================================================
# 六、診斷 + 報告
# ============================================================

def run(input_dir, ground_truth_tids=None):
    mode = configure_for_input(input_dir)
    print(f"模式：{mode}（拓撲邊數={len(ADJACENT)}，overlap={len(OVERLAP_PAIRS)}，"
          f"H矩陣={len(H_MATRICES)}）")
    if mode == "vehicle":
        print("query 相似度沿用 merge 結果（0528 為 wc.png、0507 為 BSH-5613.jpg）")
        print("同物件：Homography 腳底距離 < "
              f"{H_DIST_GATE:.0f}px；走廊方向允許 23→22 等「遠鏡頭先拍」")
    tracks = load_tracks(input_dir)
    print(f"讀入 {len(tracks)} 條 track，時間範圍 "
          f"{tracks[0].t_start:.1f}s – {max(t.t_end for t in tracks):.1f}s")

    all_paths, rejected = enumerate_paths(tracks)

    scored = []
    for path_idx, edges_info in all_paths:
        path = [tracks[i] for i in path_idx]
        scored.append({
            "tids": [t.tid for t in path],
            "score": path_score(path, edges_info),
            "edges": edges_info,
        })
    scored.sort(key=lambda p: -p["score"])

    # 前綴路徑降噪：只留不是其他路徑前綴的「極大路徑」進主報告
    # （等價於暴力 any 前綴檢查；O(|paths|·L)，避免 26 track 時 O(n²) 卡死）
    tid_seqs = {tuple(p["tids"]) for p in scored}
    is_prefix = set()
    for q in tid_seqs:
        for k in range(1, len(q)):
            is_prefix.add(q[:k])
    maximal = [p for p in scored if tuple(p["tids"]) not in is_prefix]

    print(f"\n合法路徑共 {len(scored)} 條（含前綴），極大路徑 {len(maximal)} 條")
    print(f"\n===== Top {TOP_K_PRINT} 極大路徑 =====")
    for rank, p in enumerate(maximal[:TOP_K_PRINT], 1):
        print(f"\n#{rank}  score={p['score']:.2f}   {'  ->  '.join(p['tids'])}")
        for e in p["edges"]:
            hd = e.get("h_dist")
            hd_s = f"h={hd:.1f}px" if hd is not None else "h=—"
            print(f"      {e['from']} -> {e['to']}   hop={e['hop']}  "
                  f"dt={e['dt']:.1f}s  emb={e.get('emb', 0):.3f}  "
                  f"hist={e.get('hist_emb', 0):.3f}  {hd_s}  "
                  f"edge_score={e['score']:+.2f}")

    if len(maximal) >= 2:
        gap = maximal[0]["score"] - maximal[1]["score"]
        print(f"\n信心指標：best − second = {gap:.2f}"
              + ("（差距小，這段本質有歧義）" if gap < 5 else ""))

    # ---- 兩段式診斷 ----
    if ground_truth_tids:
        gt = tuple(ground_truth_tids)
        hit = [i for i, p in enumerate(scored) if tuple(p["tids"]) == gt]
        print("\n===== 診斷 =====")
        if not hit:
            print("真路徑【不在】枚舉集合裡 → 病在枚舉層，調分數沒用。被拒的關鍵邊：")
            for a, b in zip(gt, gt[1:]):
                hits = [r for r in rejected if r[0] == a and r[1] == b]
                for r in hits:
                    print(f"   {r[0]} -> {r[1]} 被拒：{r[2]}")
                if not hits:
                    ta = next((t for t in tracks if t.tid == a), None)
                    tb = next((t for t in tracks if t.tid == b), None)
                    if ta is None or tb is None:
                        missing = a if ta is None else b
                        print(f"   {a} -> {b}：{missing} 不在候選池（gate 或上游過濾砍掉了）")
                    else:
                        ok, reason, _, _, _, _ = edge_check(ta, tb)
                        if ok:
                            print(f"   {a} -> {b}：這條邊本身合法，問題出在真路徑更前段的邊")
                        else:
                            print(f"   {a} -> {b} 被拒：{reason}")
        else:
            rank_in_all = hit[0] + 1
            print(f"真路徑在枚舉集合裡，全體排名第 {rank_in_all} → 病在計分層。")
            if rank_in_all > 1:
                print("看它輸給第一名的邊分差，逐邊比對上面的分解表，"
                      "通常是某條邊的 delay／emb／H 加成或 hop 基礎分造成。")
    return scored, maximal



# ============================================================
# 七、Top-1 路徑拼圖 + JSON 輸出
# ============================================================

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


def build_summary(merge_dir: Path, scored: list, maximal: list, collage: Path | None) -> dict:
    top = maximal[0] if maximal else None
    gap = None
    if len(maximal) >= 2:
        gap = maximal[0]["score"] - maximal[1]["score"]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": MODE,
        "input_dir": str(merge_dir.resolve()),
        "source": "query_filter_merge (*_merged.json + filter kept)",
        "sim_min": SIM_MIN,
        "emb_edge_min": EMB_EDGE_MIN,
        "emb_hist_min": EMB_HIST_MIN,
        "collage": str(collage.resolve()) if collage else None,
        "n_paths_all": len(scored),
        "n_paths_maximal": len(maximal),
        "confidence_gap_best_second": gap,
        "top1": {
            "score": top["score"],
            "tids": top["tids"],
            "path": " -> ".join(top["tids"]),
            "edges": top["edges"],
        }
        if top
        else None,
        "top10_paths": [
            {"rank": i, "score": p["score"], "tids": p["tids"]}
            for i, p in enumerate(maximal[:10], 1)
        ],
    }


def parse_args(argv=None):
    try:
        from repo_paths import QUERY_FILTER_OUTPUT_ROOT
        default_input = QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528"
    except ImportError:
        default_input = OUTPUT_ROOT / "query_filter_merge" / "人員追蹤_20260528"

    p = argparse.ArgumentParser(
        description="跨鏡頭路徑枚舉計分，輸出 Top-1 路徑圖與 JSON"
    )
    p.add_argument(
        "input_dir",
        nargs="?",
        default=str(default_input),
        help="query_filter_merge 資料夾（含 *_merged.json）",
    )
    p.add_argument("--sim-min", type=float, default=None,
                   help="覆寫 SIM_MIN（建議與上游 tracklet-sim-thresh 一致，例如 0.85）")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="輸出目錄（預設 ../output/path_enum）")
    p.add_argument("--out-png", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-txt", type=Path, default=None)
    p.add_argument("--no-collage", action="store_true", help="只印報告，不輸出拼圖/JSON")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    merge_dir = Path(args.input_dir).resolve()
    if not merge_dir.is_dir():
        raise SystemExit(f"找不到資料夾：{merge_dir}")

    global SIM_MIN
    if args.sim_min is not None:
        SIM_MIN = float(args.sim_min)
        print(f"覆寫 SIM_MIN={SIM_MIN}")

    tag = merge_dir.name
    out_dir = (args.out_dir or (OUTPUT_ROOT / "path_enum")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_png or (out_dir / f"{tag}_top1_collage.png")
    out_json = args.out_json or (out_dir / f"{tag}_top1.json")
    out_txt = args.out_txt or (out_dir / f"{tag}_out.txt")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    print(f"輸入：{merge_dir}")
    print(f"OUTPUT_ROOT：{OUTPUT_ROOT}")
    print(f"HOMOGRAPHY_DIR：{HOMOGRAPHY_DIR} exists={HOMOGRAPHY_DIR.is_dir()}")

    gt = None
    if "車輛追蹤_20260507" in str(merge_dir):
        gt = ["K8-23_13", "K8-22_18", "K8-20_97", "K8-21_16", "K8-19_20", "K8-28_6"]

    with out_txt.open("w", encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = Tee(old, f)
        try:
            scored, maximal = run(str(merge_dir), ground_truth_tids=gt)
        finally:
            sys.stdout = old

    if args.no_collage:
        print(f"文字報告：{out_txt}")
        return scored, maximal

    if not maximal:
        raise SystemExit("沒有極大路徑可畫圖")

    collage = render_top1_collage(merge_dir, maximal[0], out_png)
    summary = build_summary(merge_dir, scored, maximal, collage)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n文字報告：{out_txt}")
    print(f"路徑 JSON：{out_json}")
    print(f"路徑拼圖：{collage}")
    print(f"Top-1：{' -> '.join(maximal[0]['tids'])}  score={maximal[0]['score']:.2f}")
    return scored, maximal


if __name__ == "__main__":
    main()
