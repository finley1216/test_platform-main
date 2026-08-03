# -*- coding: utf-8 -*-
"""
M9 路徑枚舉（交付用獨立入口）
==============================
邊分 = LLR + C + S（softmax 對 LLR；不乘 shrink_w）。

僅可選依賴本 repo 的 query_filter.py（--dataset 時先跑 merge/filter）。
路徑常數寫在本檔；不 import track_path / track_path_minimal / repo_paths。

用法：
  # 已有 merge 目錄
  python3 track_path_m9.py output/query_filter_merge/人員追蹤_20260507

  # 一連串：query_filter → M9
  python3 track_path_m9.py --dataset 人員追蹤_20260507

  # 指定校準
  python3 track_path_m9.py <merge_dir> --calibration ../output/v1.0/calibration_gt0507.pkl
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
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from scipy.special import logsumexp

# query_filter：僅在 --dataset 管線使用
import query_filter as qf  # noqa: E402

# ============================================================
# 路徑常數（原 repo_paths.py，併入本檔）
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT.parent / "output"
BOTSORT_ROOT = REPO_ROOT / "BoT-SORT"
CLIP_REID_ROOT = REPO_ROOT / "CLIP-ReID"
SEGMENT_ROOT = REPO_ROOT / "backend" / "segment"
QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"
V1_OUTPUT_ROOT = OUTPUT_ROOT / "v1.0"
DEFAULT_MAPPING = OUTPUT_ROOT / "人員追蹤_20260507_crop_time_mapping.json"
ARCHIVE_PATH_ENUM_LLR = OUTPUT_ROOT / "path_enum_llr"


# ============================================================
# 基礎設施（自 M0/track_path 內嵌，供 M9 自洽；勿再 import track_path）
# ============================================================

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
VEHICLE_ADJACENT = _pairs_from_adj(VEHICLE_ADJACENT_RAW)
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
VEHICLE_CORRIDOR = [
    "K8-23", "K8-22", "K8-20", "K8-21", "K8-19", "K8-28",
    "K8-16", "K8-15", "K8-30", "K8-12", "K8-10", "K8-09",
]
VEHICLE_CORRIDOR_RANK = {c: i for i, c in enumerate(VEHICLE_CORRIDOR)}
ADJACENT = set(PERSON_ADJACENT)
OVERLAP_PAIRS = dict(PERSON_OVERLAP_PAIRS)
MODE = "person"
H_DIST_GATE = 150.0     # cross-cam 邊合法性門檻（非 supernode 合併門檻）
H_TIME_WINDOW = 15.0    # 對齊兩 track 腳底點的最大時間差（秒）
H_MATRICES = {}         # (cam_a, cam_b) -> 3x3 ndarray（a 投影到 b）
SUPER_DH_MAX = 95.0
DH_SAME_SIGMA = 37.557
DH_SAME_N = 13
COEXISTENCE_OVERLAP_EMB_MIN = 0.848
DEFAULT_MIN_TRANSIT_HOP1 = 0.0     # 相鄰鏡頭：無辯護下界 → 0
DEFAULT_MIN_TRANSIT_HOP2 = 0.0
DEFAULT_TAU_HOP1 = 8.0
DEFAULT_TAU_HOP2 = 20.0
TOL      = 2.0      # 一般鏡頭對容許的時間重疊（秒）
SIM_MIN       = 0.90   # 對 query 低於此者不進候選（原本就要夠像）
CAMERA_RE = re.compile(r"(K8-\d+)")
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
def _resolve_crop_path(p: str | Path) -> Path | None:
    p = Path(p)
    cands = [
        p,
        Path(str(p).replace("/home/M133040024/ASE", str(REPO_ROOT.parent))),
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
PDF_FLOOR = 1e-12
SHRINK_K = 10.0
HANDOFF_DT_MAX = 2.0
SUPER_OVERLAP_MIN = 0.5
DEFAULT_SEGMENT_SEED_TOP_K = 400
DEFAULT_MAX_HYP_SEGMENTS = 8
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
def is_handoff_edge(u: Track, v: Track, dt: float, h_dist: float | None) -> bool:
    if float(dt) > HANDOFF_DT_MAX:
        return False
    key = tuple(sorted((u.cam, v.cam)))
    if key in OVERLAP_PAIRS:
        return True
    return h_dist is not None
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
def expand_path_tids(nodes: list[SuperNode], path_idx: list[int]) -> list[str]:
    """輸出展開：各超節點成員依 t_start 排序串接。"""
    out = []
    for i in path_idx:
        out.extend(nodes[i].tids)
    return out
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


M9_RULES = [
    "其餘全同 M6（時間順序建邊、極大路徑、矛盾作廢、同池排名）",
    "邊分 = LLR + C + S；LLR = ln(f_same(emb)/f_diff(emb))",
    "密度取自 calibration_gt0507.pkl 的 emb_same / emb_diff",
    "不乘 shrink_w（單尺度下 w 為共同倍率，不影響排序）",
    "C、S 公式同 M6；競爭 softmax 餵各下家的 LLR；hop 不計分",
]

# ============================================================
# M9 核心（自 track_path_minimal 抽出並去除 tp. 依賴）
# ============================================================

def edge_check_minimal(u: Track, v: Track, *, dt_max: float | None = None):
    """回傳 (ok, reason, dt, hop, emb, h_dist)。"""
    dt_raw = v.t_start - u.t_end
    key = tuple(sorted((u.cam, v.cam)))
    tol = OVERLAP_PAIRS.get(key, TOL)

    h_ok, h_dist = same_object_h(u, v)
    if dt_raw < -tol:
        if not (h_ok or corridor_prefers(u, v)):
            return False, f"時間順序（重疊 {-dt_raw:.1f}s 超過容許 {tol:.1f}s）", dt_raw, None, 0.0, h_dist
    dt = max(dt_raw, 0.0)

    hop = hop_count(u.cam, v.cam)
    if hop is None:
        if h_ok and tuple(sorted((u.cam, v.cam))) in ADJACENT:
            hop = 1
        else:
            return False, "拓撲不可達", dt, hop, 0.0, h_dist

    if dt_max is not None and dt > float(dt_max):
        return False, f"斷太久（dt={dt:.1f}s > DT_MAX={dt_max}）", dt, hop, 0.0, h_dist

    emb = emb_sim(u, v)
    return True, "", dt, hop, emb, h_dist
def collect_track_edge_embs(tracks: list) -> list[float]:
    """track 層級、僅時間+hop 的合法邊 emb（供名單制中位數；不受 DT_MAX）。"""
    embs = []
    for u, v in itertools.permutations(tracks, 2):
        ok, _, _, _, emb, _ = edge_check_minimal(u, v, dt_max=None)
        if ok:
            embs.append(float(emb))
    return embs
def median_edge_emb(tracks: list) -> float:
    embs = collect_track_edge_embs(tracks)
    if not embs:
        return 0.0
    return float(np.median(np.asarray(embs, dtype=np.float64)))
def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    """L2-normalized 向量的 cosine distance = 1 − cos。"""
    return float(1.0 - np.dot(a, b))
def _pairwise_mean_dist(embs: list) -> float:
    """kept crops 兩兩 cosine distance 平均；n<2 → 0（無可觀測內部波動）。"""
    n = len(embs)
    if n < 2:
        return 0.0
    s = 0.0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += _cos_dist(embs[i], embs[j])
            c += 1
    return s / c
def attach_crop_embs(tracks: list, merge_dir: Path) -> None:
    """在 Track.meta 寫入 crop_embs / w_intra（不改 track_path.load_tracks）。"""
    cache_by_cam: dict = {}
    for t in tracks:
        cam, tid_s = t.tid.rsplit("_", 1)
        tid = int(tid_s)
        if cam not in cache_by_cam:
            cache_by_cam[cam] = _load_emb_cache(merge_dir, cam)
        cache = cache_by_cam[cam]
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
        embs = []
        for name in names:
            e = _cache_lookup(cache, name)
            if e is not None:
                embs.append(_l2_normalize(np.asarray(e, dtype=np.float64)))
        if not embs and t.emb is not None:
            embs = [np.asarray(t.emb, dtype=np.float64)]
        t.meta["crop_embs"] = embs
        t.meta["w_intra"] = _pairwise_mean_dist(embs)
def _softmax_probs(scores: list[float]) -> list[float]:
    """對 A 做 softmax；+∞ 均分、全 −∞ 均分；無額外常數。"""
    n = len(scores)
    if n == 0:
        return []
    pos_inf = [i for i, s in enumerate(scores) if s == float("inf")]
    if pos_inf:
        p = [0.0] * n
        share = 1.0 / len(pos_inf)
        for i in pos_inf:
            p[i] = share
        return p
    finite = [i for i, s in enumerate(scores) if math.isfinite(s)]
    if not finite:
        return [1.0 / n] * n
    mx = max(scores[i] for i in finite)
    exps = [0.0] * n
    for i in finite:
        exps[i] = math.exp(scores[i] - mx)
    s = sum(exps)
    return [e / s for e in exps]
def _topo_shortest_hops(cam_u: str, cam_v: str) -> int | None:
    """鏡頭鄰接圖 BFS 最短站數（邊數）。同鏡=0。"""
    if cam_u == cam_v:
        return 0
    adj: dict[str, set[str]] = {}
    for a, b in ADJACENT:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    from collections import deque

    q = deque([(cam_u, 0)])
    seen = {cam_u}
    while q:
        c, d = q.popleft()
        for nb in adj.get(c, ()):
            if nb in seen:
                continue
            if nb == cam_v:
                return d + 1
            seen.add(nb)
            q.append((nb, d + 1))
    return None
def _enumerate_from_succ_m6(nodes, succ_raw, n_legal_edges, *, beam_width, beam_max_leaves, force_full):
    n = len(nodes)
    use_beam = (not force_full) and (n_legal_edges > FULL_ENUM_EDGE_CAP)
    all_paths = []
    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, e in succ_raw[idx]:
                if j in path_idx:
                    continue
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, e in succ_raw[idx]:
                        if j in path_idx:
                            continue
                        extended = True
                        nxt.append((sc + e["score"], path_idx + [j], edges_info + [e]))
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
    return all_paths, use_beam
def emb_llr_raw(calib: dict, emb: float) -> float:
    """LLR = ln(f_same/f_diff)；不乘 shrink_w。"""
    return float(llr(calib["emb_same"], calib["emb_diff"], float(emb)))
def _default_m9_calib_path() -> Path:
    return OUTPUT_ROOT / "v1.0" / "calibration_gt0507.pkl"
def _load_m9_calib(path: Path | str | None = None) -> tuple[dict, Path]:
    p = Path(path) if path else _default_m9_calib_path()
    p = p.resolve()
    if not p.is_file():
        raise RuntimeError(f"M9 異常：找不到校準檔 {p}")
    calib = load_calibration(p)
    if "emb_same" not in calib or "emb_diff" not in calib:
        raise RuntimeError(f"M9 異常：校準檔缺 emb_same/emb_diff：{p}")
    return calib, p
def _best_member_edge_m9(sa: SuperNode, sb: SuperNode, calib: dict):
    """時間順序合法成員對中取 LLR 最大者；hop 僅記錄。"""
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = OVERLAP_PAIRS.get(key, TOL)
            h_ok, h_dist = same_object_h(u, v)
            if dt_raw < -tol:
                if not (h_ok or corridor_prefers(u, v)):
                    rejects.append((u.tid, v.tid, "時間順序"))
                    continue
            emb = float(emb_sim(u, v))
            llr = emb_llr_raw(calib, emb)
            hop = hop_count(u.cam, v.cam)
            if hop is None:
                hop = _topo_shortest_hops(u.cam, v.cam)
            cand = (u, v, dt, hop, emb, h_dist, llr)
            if best is None:
                best = cand
            else:
                bl = best[6]
                if math.isfinite(llr) and math.isfinite(bl) and (
                    llr > bl or (llr == bl and emb > best[4])
                ):
                    best = cand
                elif math.isfinite(llr) and not math.isfinite(bl) and bl == float("-inf"):
                    best = cand
                elif llr == float("inf") and bl != float("inf"):
                    best = cand
    return best, rejects
def _build_succ_m9(nodes: list, calib: dict) -> tuple[list, list, int, dict]:
    """
    時間序建邊 → LLR → 雙向 C（softmax 對 LLR）→ S；score=LLR+C+S。
    不乘 shrink_w。Σ>=1 → degenerate。
    """
    n = len(nodes)
    raw = {}
    rejected = []
    for i, j in itertools.permutations(range(n), 2):
        best, rejects = _best_member_edge_m9(nodes[i], nodes[j], calib)
        if best is None:
            for r in rejects[:2]:
                rejected.append(r)
            continue
        raw[(i, j)] = best

    succ_idx = [[] for _ in range(n)]
    pred_idx = [[] for _ in range(n)]
    for (i, j) in raw:
        succ_idx[i].append(j)
        pred_idx[j].append(i)

    P_fwd = {}
    for i in range(n):
        js = succ_idx[i]
        if not js:
            continue
        Ls = [raw[(i, j)][6] for j in js]
        ps = _softmax_probs(Ls)
        for j, p in zip(js, ps):
            P_fwd[(i, j)] = p

    P_bwd = {}
    for j in range(n):
        is_ = pred_idx[j]
        if not is_:
            continue
        Ls = [raw[(i, j)][6] for i in is_]
        ps = _softmax_probs(Ls)
        for i, p in zip(is_, ps):
            P_bwd[(i, j)] = p

    def _ln_NP(N, p):
        if N <= 0 or p <= 0.0:
            return float("-inf")
        return float(math.log(N * p))

    succ = [[] for _ in range(n)]
    n_legal = 0
    n_degenerate = 0
    degenerate_edges = []

    for (i, j), (u, v, dt, hop, emb, h_dist, llr) in raw.items():
        N_u = len(succ_idx[i])
        N_v_pred = len(pred_idx[j])
        p_fwd = P_fwd.get((i, j), 0.0)
        p_bwd = P_bwd.get((i, j), 0.0)
        C_fwd = _ln_NP(N_u, p_fwd)
        C_bwd = _ln_NP(N_v_pred, p_bwd)
        if C_fwd == float("-inf") or C_bwd == float("-inf"):
            C = float("-inf")
        else:
            C = C_fwd + C_bwd

        t_v = float(nodes[j].t_start)
        sum_p = 0.0
        skipped = []
        for k in succ_idx[i]:
            if k == j:
                continue
            if float(nodes[k].t_start) < t_v:
                pk = float(P_fwd.get((i, k), 0.0))
                sum_p += pk
                skipped.append(
                    {
                        "to_super": nodes[k].label,
                        "to_sid": nodes[k].sid,
                        "t_start": float(nodes[k].t_start),
                        "P": pk,
                        "LLR": float(raw[(i, k)][6])
                        if math.isfinite(raw[(i, k)][6])
                        else (
                            1e300
                            if raw[(i, k)][6] == float("inf")
                            else -1e300
                        ),
                        "emb": float(raw[(i, k)][4]),
                    }
                )
        skipped.sort(key=lambda x: -x["P"])

        if sum_p >= 1.0:
            n_degenerate += 1
            degenerate_edges.append(
                {
                    "from_super": nodes[i].label,
                    "to_super": nodes[j].label,
                    "sum_P_skipped": float(sum_p),
                    "n_skipped": len(skipped),
                    "reason": "ΣP_skipped>=1",
                }
            )
            continue

        S = float(math.log(1.0 - sum_p))

        if llr == float("-inf") or C == float("-inf"):
            score = float("-inf")
        elif llr == float("inf"):
            score = float("inf") if C != float("-inf") else float("-inf")
        else:
            score = float(llr + C + S)

        e = {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[i].label,
            "to_super": nodes[j].label,
            "from_members": nodes[i].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt),
            "hop": hop,
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(is_handoff_edge(u, v, dt, h_dist)),
            "score": score,
            "LLR": float(llr),
            "C": float(C),
            "C_fwd": float(C_fwd),
            "C_bwd": float(C_bwd),
            "S": float(S),
            "sum_P_skipped": float(sum_p),
            "n_skipped": int(len(skipped)),
            "skipped": skipped,
            "N_u": int(N_u),
            "N_v_pred": int(N_v_pred),
            "P_fwd": float(p_fwd),
            "P_bwd": float(p_bwd),
            "z_emb": 0.0,
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m9_LLR_C_plus_S",
            "dt_source": "super_union",
            "hist_emb": None,
            "degenerate": False,
            "shrink_applied": False,
        }
        for key in ("score", "LLR", "C", "C_fwd", "C_bwd", "S"):
            val = e[key]
            if val == float("inf"):
                e[key] = 1e300
                e[f"{key}_inf"] = True
            elif val == float("-inf"):
                e[key] = -1e300
                e[f"{key}_inf"] = True

        succ[i].append((j, e))
        n_legal += 1

    emb_same = calib["emb_same"]
    emb_diff = calib["emb_diff"]
    meta = {
        "scoring": "m9",
        "constants": [],
        "n_legal_edges": n_legal,
        "n_degenerate": n_degenerate,
        "degenerate_sample": degenerate_edges[:20],
        "note": (
            "建邊僅時間順序；LLR+C+S；softmax 對 LLR；"
            "不乘 shrink_w；hop 不計分"
        ),
        "calib_emb_same": {
            "mu": float(emb_same["mu"]),
            "sigma": float(emb_same["sigma"]),
            "n": emb_same.get("n"),
            "shrink_w": emb_same.get("shrink_w"),
        },
        "calib_emb_diff": {
            "mu": float(emb_diff["mu"]),
            "sigma": float(emb_diff["sigma"]),
            "n": emb_diff.get("n"),
            "shrink_w": emb_diff.get("shrink_w"),
        },
        "note_no_shrink": (
            "LLR 不乘 shrink_w：單尺度／單證據下 w 為共同倍率，不影響排序。"
        ),
    }
    return succ, rejected, n_legal, meta
def enumerate_paths_m9(
    tracks: list,
    merge_dir: Path,
    *,
    calib: dict | None = None,
    calib_path: Path | str | None = None,
    use_supernode: bool = True,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    if calib is None:
        calib, calib_path = _load_m9_calib(calib_path)
    else:
        calib_path = Path(calib_path) if calib_path else _default_m9_calib_path()

    attach_crop_embs(tracks, merge_dir)
    coexist_median = median_edge_emb(tracks)
    if use_supernode:
        nodes, super_report = build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = _build_nodes(tracks, False)
    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median

    succ_raw, rejected_edges, n_legal_edges, m9_meta = _build_succ_m9(nodes, calib)
    m9_meta["calibration_path"] = str(Path(calib_path).resolve())
    stats = {
        "emb": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "dt": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "sim": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "hop_pen": 0.0,
        "scoring": "m9",
        "node_score": False,
        "dt_max": None,
        "m9": m9_meta,
        "m9_calib": calib,
        "constants": [],
    }
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal_edges,
        beam_width=beam_width,
        beam_max_leaves=beam_max_leaves,
        force_full=force_full,
    )
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": None,
        "n_degenerate": m9_meta.get("n_degenerate"),
        "note": (
            f"M9 合法邊={n_legal_edges} > {FULL_ENUM_EDGE_CAP}：beam"
            if use_beam
            else "M9 全量 DFS（含前綴）"
        ),
    }
    if use_beam:
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)
    super_report["score_stats"] = stats
    super_report["z_stats"] = stats["emb"]
    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ_raw,
    )
def _score_paths_on_nodes_m9(
    nodes: list, tracks: list, calib: dict
) -> tuple[list, int, dict]:
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}
    succ_raw, _, n_legal, m9_meta = _build_succ_m9(nodes, calib)
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal,
        beam_width=DEFAULT_BEAM_WIDTH,
        beam_max_leaves=DEFAULT_BEAM_MAX_LEAVES,
        force_full=False,
    )
    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = attach_softmax(maximal_paths(scored))
    return maximal, n_legal, {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "m9": m9_meta,
    }
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
        "min_A": p.get("min_A"),
        "min_logit": p.get("min_logit"),
    }
def _dataset_short(name: str) -> str:
    if "20260507" in name:
        return "0507"
    if "20260528" in name:
        return "0528"
    return name


# ============================================================
# M9 排名／執行／CLI
# ============================================================

def grow_segmented_hypothesis_m9(
    seed_path: dict,
    all_nodes: list,
    tracks: list,
    calib: dict,
    *,
    max_segments: int = DEFAULT_MAX_HYP_SEGMENTS,
    pool_cache: dict | None = None,
) -> list[dict]:
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
            maximal, _, _ = _score_paths_on_nodes_m9(pool, tracks, calib)
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


def build_ranked_hypotheses_m9(
    single_maximal: list,
    all_nodes: list,
    tracks: list,
    calib: dict,
    *,
    seed_top_k: int = DEFAULT_SEGMENT_SEED_TOP_K,
    max_segments: int = DEFAULT_MAX_HYP_SEGMENTS,
) -> tuple[list, dict]:
    by_tid = {t.tid: t for t in tracks}
    rejected = []
    pool = []
    seen_keys = set()
    pool_cache: dict = {}

    def _try_add(hyp: dict) -> None:
        key = tuple(tuple(seg["tids"]) for seg in hyp["segments"])
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

    for rank, p in enumerate(single_maximal, 1):
        hyp = _hypothesis_from_segments(
            [_path_as_segment(p, 1, None)],
            source="single_maximal",
            seed_rank=rank,
        )
        _try_add(hyp)

    seeds = single_maximal[: max(int(seed_top_k), 0)]
    n_grown = 0
    for rank, seed in enumerate(seeds, 1):
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
        segs = grow_segmented_hypothesis_m9(
            seed,
            all_nodes,
            tracks,
            calib,
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
        "note": "M9：極大路徑+分段同池；矛盾作廢；softmax",
    }
    return pool, meta


def run_m9(
    merge_dir: Path,
    *,
    sim_min: float = 0.85,
    calibration_path: Path | str | None = None,
    tag: str = "m9",
) -> dict:
    global SIM_MIN
    t0 = time.perf_counter()
    merge_dir = Path(merge_dir).resolve()
    SIM_MIN = float(sim_min)
    mode = configure_for_input(str(merge_dir))
    tracks = load_tracks(str(merge_dir))
    t_load = time.perf_counter()

    (
        all_paths,
        _rejected,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        _succ,
    ) = enumerate_paths_m9(
        tracks,
        merge_dir,
        calib_path=calibration_path,
    )
    t_enum = time.perf_counter()

    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    single_maximal = maximal_paths(scored)
    single_maximal_diag = attach_softmax([dict(p) for p in single_maximal])

    calib = stats.get("m9_calib")
    if calib is None:
        calib, _ = _load_m9_calib(calibration_path)
    ranked, rank_meta = build_ranked_hypotheses_m9(
        single_maximal, nodes, tracks, calib
    )
    t_end = time.perf_counter()

    timing = {
        "load_sec": t_load - t0,
        "enumerate_sec": t_enum - t_load,
        "rank_sec": t_end - t_enum,
        "total_sec": t_end - t0,
    }
    options = {
        "variant": tag,
        "scoring": "m9",
        "node_score": False,
        "dt_max": None,
        "no_calibration": False,
        "coexistence_emb_median": super_report.get("minimal_coexistence_emb_median"),
        "score_stats": {k: v for k, v in stats.items() if k != "m9_calib"},
        "enumeration": super_report.get("enumeration"),
        "ranking_meta": rank_meta,
        "rules": M9_RULES,
        "timing": timing,
        "constants": [],
        "segments": list((ranked[0].get("segments") if ranked else None) or []),
        "single_maximal_top1": (
            {
                "score": single_maximal_diag[0]["score"],
                "path_probability": single_maximal_diag[0].get("path_probability"),
                "tids": single_maximal_diag[0]["tids"],
                "super_labels": single_maximal_diag[0].get("super_labels"),
            }
            if single_maximal_diag
            else None
        ),
    }
    return {
        "tag": tag,
        "mode": mode,
        "cfg": type("Cfg", (), {"scoring": "m9", "sim_min": sim_min})(),
        "tracks": tracks,
        "nodes": nodes,
        "scored": scored,
        "ranked": ranked,
        "n_legal_edges": n_legal_edges,
        "super_report": super_report,
        "options": options,
        "timing": timing,
    }


def save_m9_summary(result: dict, merge_dir: Path, out_dir: Path, stem: str) -> dict:
    ranked = result["ranked"]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "variant": result["tag"],
        "mode": result["mode"],
        "scoring": "m9",
        "input_dir": str(merge_dir),
        "sim_min": float(result["cfg"].sim_min),
        "n_tracks": len(result["tracks"]),
        "n_legal_edges": result["n_legal_edges"],
        "n_paths_all": len(result["scored"]),
        "n_hypotheses_ranked": len(ranked),
        "options": {k: v for k, v in result["options"].items() if k != "segments"},
        "supernodes": {
            k: result["super_report"].get(k)
            for k in (
                "n_tracks",
                "n_supernodes",
                "n_merged_pairs",
                "multi_only",
                "minimal_coexistence_emb_median",
                "enumeration",
            )
        },
        "timing": result["timing"],
        "top1": _hyp_brief(ranked[0], 1) if ranked else None,
        "top3_hypotheses": [_hyp_brief(h, i) for i, h in enumerate(ranked[:3], 1)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{stem}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON：{out_json}")
    return summary


def run_query_filter_pipeline(dataset: str, qf_argv: list[str] | None = None) -> Path:
    """呼叫 query_filter 產出 merge 目錄，回傳該目錄路徑。"""
    argv = ["--dataset", dataset]
    if qf_argv:
        argv.extend(qf_argv)
    # 暫時覆寫 sys.argv 以重用 query_filter.parse_args / main 流程
    old = sys.argv
    try:
        sys.argv = ["query_filter.py", *argv]
        args = qf.parse_args()
        dataset_key = qf.resolve_dataset_key(args.dataset)
        data_dir = (args.data_dir or qf.OUTPUT_ROOT).resolve()
        output_dir = (args.output_dir or (qf.QUERY_FILTER_OUTPUT_ROOT / dataset_key)).resolve()
        if args.skip_step1 and args.skip_step2:
            raise SystemExit("不能同時 --skip-step1 與 --skip-step2")
        if not qf.STEP1.is_file():
            raise SystemExit(f"找不到 Step1 腳本：{qf.STEP1}")
        if not qf.STEP2.is_file():
            raise SystemExit(f"找不到 Step2 腳本：{qf.STEP2}")
        mapping_json = (
            args.mapping_json or (data_dir / f"{dataset_key}_crop_time_mapping.json")
        ).resolve()
        if not mapping_json.is_file():
            raise SystemExit(f"找不到 mapping：{mapping_json}")
        print(f"[query_filter] 資料集：{dataset_key}")
        print(f"[query_filter] 輸出：{output_dir}")
        if not args.skip_step1:
            qf.run_cmd(
                "Step 1/2：query 篩選 + BoT-SORT + merge",
                qf.build_step1_cmd(args, dataset_key, data_dir, output_dir),
                cwd=qf.BOTSORT_ROOT,
            )
        if not args.skip_step2:
            qf.run_cmd(
                "Step 2/2：combined intra-filter",
                qf.build_step2_cmd(args, dataset_key, data_dir, output_dir),
                cwd=qf.BOTSORT_ROOT,
            )
        return output_dir
    finally:
        sys.argv = old


def main(argv=None):
    p = argparse.ArgumentParser(
        description="M9 路徑枚舉（可選串接 query_filter）"
    )
    p.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="merge 目錄（與 --dataset 二選一）",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="先跑 query_filter 再 M9（例如 人員追蹤_20260507）",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="M9 校準 pkl（預設 v1.0/calibration_gt0507.pkl）",
    )
    p.add_argument("--tag", type=str, default="m9")
    p.add_argument(
        "--qf-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="傳給 query_filter 的額外參數（放在 --qf-args 之後）",
    )
    args = p.parse_args(argv)

    if args.dataset:
        qf_extra = list(args.qf_args or [])
        if qf_extra and qf_extra[0] == "--":
            qf_extra = qf_extra[1:]
        merge_dir = run_query_filter_pipeline(args.dataset, qf_extra)
    elif args.input_dir:
        merge_dir = Path(args.input_dir).resolve()
    else:
        merge_dir = (QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507").resolve()

    if not merge_dir.is_dir():
        raise SystemExit(f"找不到 merge 目錄：{merge_dir}")

    out_dir = (args.out_dir or (V1_OUTPUT_ROOT / "m9_comparison")).resolve()
    calib = args.calibration.resolve() if args.calibration else None

    print(f"[m9] input={merge_dir}  sim_min={args.sim_min}  tag={args.tag}")
    result = run_m9(
        merge_dir,
        sim_min=float(args.sim_min),
        calibration_path=calib,
        tag=args.tag,
    )
    ranked = result["ranked"]
    if ranked:
        top = ranked[0]
        print(
            f"Top-1 P={top['path_probability']:.6f}  score={top['score']:.4f}  "
            f"n_seg={top['n_segments']}  {top.get('path', '')}"
        )
    enum = result["super_report"].get("enumeration") or {}
    print(
        f"合法邊={result['n_legal_edges']}  mode={enum.get('mode')}  "
        f"耗時={result['timing']['total_sec']:.2f}s"
    )
    stem = f"{_dataset_short(merge_dir.name)}_{args.tag}_top1"
    return save_m9_summary(result, merge_dir, out_dir, stem)


if __name__ == "__main__":
    main()
