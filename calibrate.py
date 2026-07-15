# -*- coding: utf-8 -*-
"""
自動標註 + 分布校準（給 path_enum_llr.py 用）
==============================================
輸入（優先）：BoT-SORT 全量輸出（所有 track，不經 query 篩選）
  python3 calibrate.py --tracking-output <dir1> <dir2> ... --out calibration.pkl

每個 --tracking-output 目錄需提供：
  - tracking_rows_*.json（或 per-cam 子目錄內的同名檔）
  - 對應的 person_clipreid_embeddings_cache.pkl（同目錄或 embed_cache）

若缺逐 crop embedding 或缺 track JSON，腳本會列出可用資料後退出，
不自行用 query_filter_merge 等替代來源。

正樣本／負樣本規則（不變）：
  - OVERLAP_PAIRS：時間重疊且 H 投影距離 < 80px → emb、d_H
  - 相鄰非重疊：0<dt≤DT_MAX 且 emb≥gate → dt（按鏡頭對）
  - 同鏡同時段、中心距 > W/4 → emb|diff
  - 跨鏡同時段、dH>300 → emb|diff
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

DH_POS_MAX = 80.0
DH_NEG_MIN = 300.0
DH_DIFF_UNIFORM_MAX = 800.0
PRIOR_SIGMA = 0.5
MIN_SAMPLES_FIT = 20
PDF_FLOOR = 1e-12
CAMERA_RE = re.compile(r"(K8-\d{2})", re.IGNORECASE)
CAM_DIR_RE = re.compile(r"(?:k|K8-?)(\d{2})", re.IGNORECASE)


def _time_overlap(a: pes.Track, b: pes.Track) -> bool:
    return not (a.t_end < b.t_start or b.t_end < a.t_start)


def _pair_emb(u: pes.Track, v: pes.Track) -> float:
    return pes.emb_sim(u, v)


def _pair_dh(u: pes.Track, v: pes.Track) -> float | None:
    ok, d = pes.same_object_h(u, v)
    return d


def _cache_lookup(cache: dict, crop_name: str):
    name = Path(crop_name).name
    for k, v in cache.items():
        if Path(k).name == name:
            return np.asarray(v, dtype=np.float64)
    return None


def _iso_to_day_seconds(ts: str, base_date: str | None) -> float:
    return pes._iso_to_day_seconds(ts, base_date)


def inventory_tracking_dir(root: Path) -> dict:
    """盤點目錄內可用／缺失的全量追蹤與 embedding 檔。"""
    root = root.resolve()
    tracking_jsons = sorted(root.rglob("tracking_rows*.json"))
    emb_caches = sorted(root.rglob("*clipreid_embeddings_cache.pkl"))
    collages = sorted(root.rglob("tracking_collage*.png"))
    # 常見同名資料在 ASE/output/embed_cache
    embed_cache_alt = []
    tag = root.name
    alt = OUTPUT_ROOT / "embed_cache"
    if alt.is_dir():
        embed_cache_alt = sorted(alt.glob(f"*{tag}*/*embeddings_cache.pkl"))
        # also match 人員追蹤_20260507_k801 style children of parent list
        if not embed_cache_alt:
            embed_cache_alt = sorted(alt.glob("**/person_clipreid_embeddings_cache.pkl"))
    return {
        "dir": str(root),
        "n_tracking_rows_json": len(tracking_jsons),
        "tracking_rows_json": [str(p) for p in tracking_jsons[:20]],
        "n_emb_cache_in_dir": len(emb_caches),
        "emb_cache_in_dir": [str(p) for p in emb_caches[:20]],
        "n_collage_only": len(collages),
        "collages_sample": [str(p) for p in collages[:10]],
        "n_embed_cache_alt": len(embed_cache_alt),
        "embed_cache_alt_sample": [str(p) for p in embed_cache_alt[:10]],
    }


def report_missing_and_exit(dirs: list[Path], reason: str) -> None:
    lines = [
        "=== calibrate.py：無法用 --tracking-output 完成校準 ===",
        reason,
        "",
        "規則：全量輸出若沒有現成的逐 crop embedding（或缺 track JSON），",
        "只回報可用資料，不自行用 query_filter_merge 等替代。",
        "",
    ]
    for d in dirs:
        inv = inventory_tracking_dir(d)
        lines.append(f"--- {inv['dir']} ---")
        lines.append(f"  tracking_rows*.json : {inv['n_tracking_rows_json']}")
        for p in inv["tracking_rows_json"]:
            lines.append(f"    - {p}")
        lines.append(f"  *clipreid_embeddings_cache.pkl（目錄內）: {inv['n_emb_cache_in_dir']}")
        for p in inv["emb_cache_in_dir"]:
            lines.append(f"    - {p}")
        lines.append(f"  tracking_collage*.png（有跑過但無 JSON）: {inv['n_collage_only']}")
        for p in inv["collages_sample"]:
            lines.append(f"    - {p}")
        lines.append(
            f"  參考：ASE/output/embed_cache 可能有 per-crop emb "
            f"（掃到 {inv['n_embed_cache_alt']} 個 cache）"
        )
        for p in inv["embed_cache_alt_sample"]:
            lines.append(f"    - {p}")
        lines.append("")

    lines.extend(
        [
            "目前人員資料盤點結論：",
            "  ✓ per-crop embedding：ASE/output/embed_cache/人員追蹤_*_k*/"
            "person_clipreid_embeddings_cache.pkl",
            "  ✓ crop 時間／box：ASE/output/人員追蹤_*_crop_time_mapping.json",
            "  ✗ 全量 track 分組 JSON：人員追蹤_20260507 / 20260528 各鏡只有 "
            "tracking_collage.png，沒有 tracking_rows_*.json",
            "  格式範例僅見：BoT-SORT-K809/output/k809/tracking_rows_*.json",
            "",
            "請先重跑 BoT-SORT 並 dump tracking_rows JSON（含 crops[]），再執行：",
            "  python3 calibrate.py --tracking-output <dir1> <dir2> "
            "--out ../output/path_enum_llr/calibration.pkl",
        ]
    )
    msg = "\n".join(lines)
    print(msg)
    raise SystemExit(2)


def _infer_cam_from_path(path: Path) -> str | None:
    for part in [path.stem, path.parent.name, path.name]:
        m = CAMERA_RE.search(part)
        if m:
            return m.group(1).upper().replace("K8-", "K8-")
        m2 = CAM_DIR_RE.search(part)
        if m2:
            return f"K8-{int(m2.group(1)):02d}"
    return None


def _find_emb_cache_for_cam(tracking_root: Path, cam: str, dataset_hint: str | None) -> Path | None:
    """在 tracking 目錄或 ASE/output/embed_cache 找該鏡的 cache。"""
    cam_num = cam.split("-")[-1]
    patterns = [
        f"*_{cam}_*",
        f"*k{cam_num}*",
        f"*K8-{cam_num}*",
        f"*k{int(cam_num):02d}*",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(tracking_root.rglob(f"{pat}/**/person_clipreid_embeddings_cache.pkl"))
        candidates.extend(tracking_root.rglob(f"{pat}*embeddings_cache.pkl"))
    # direct
    candidates.extend(tracking_root.rglob("person_clipreid_embeddings_cache.pkl"))
    candidates.extend(tracking_root.rglob("*clipreid_embeddings_cache.pkl"))

    alt_root = OUTPUT_ROOT / "embed_cache"
    if alt_root.is_dir():
        for pat in patterns:
            candidates.extend(alt_root.glob(f"{pat}/person_clipreid_embeddings_cache.pkl"))
        if dataset_hint:
            candidates.extend(
                alt_root.glob(f"{dataset_hint}*k{cam_num}/person_clipreid_embeddings_cache.pkl")
            )
            candidates.extend(
                alt_root.glob(f"{dataset_hint}*k{int(cam_num):02d}/person_clipreid_embeddings_cache.pkl")
            )

    # prefer cache whose path mentions this cam number
    scored = []
    for c in candidates:
        s = str(c).lower()
        score = 0
        if f"k{cam_num}" in s or f"k8-{cam_num}" in s or f"_{cam.lower()}" in s:
            score += 10
        if "person_clipreid" in s:
            score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    for score, c in scored:
        if score >= 10 and c.is_file():
            return c
    return None


def _load_mapping_for_dataset(dataset: str):
    mapping_json = OUTPUT_ROOT / f"{dataset}_crop_time_mapping.json"
    if not mapping_json.is_file():
        mapping_json = pes.DEFAULT_MAPPING
    if not mapping_json.is_file():
        return None, None, None
    ts_index, base_date = pes._load_crop_timestamp_index(mapping_json)
    return ts_index, base_date, mapping_json


def tracks_from_tracking_rows(
    json_path: Path,
    cam: str,
    video_id: str,
    emb_cache: dict,
    ts_index: dict,
    base_date: str | None,
) -> list[pes.Track]:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    tracks = []
    for row in rows:
        tid = int(row.get("track_id") or row.get("merged_track_id") or 0)
        crops = row.get("crops") or []
        if not crops:
            continue
        stamps = []
        embs = []
        foots = []
        centers = []
        names = []
        for c in crops:
            name = Path(c.get("crop_path") or c.get("path") or "").name
            if not name:
                continue
            names.append(name)
            # prefer absolute_timestamp on crop row; else mapping
            ts = c.get("absolute_timestamp")
            box = c.get("box")
            if isinstance(box, str):
                box = pes._parse_box(box)
            rec = ts_index.get((video_id, name))
            if ts is None and rec is not None:
                ts = rec["ts"]
            if box is None and rec is not None:
                box = rec.get("box")
            if ts is None:
                # relative_seconds fallback with base midnight
                if c.get("relative_seconds") is not None and base_date:
                    try:
                        day0 = datetime.fromisoformat(base_date)
                    except ValueError:
                        day0 = datetime.fromisoformat(base_date + "T00:00:00")
                    tsec = float(c["relative_seconds"])
                    stamps.append(day0.isoformat())
                    # store via synthetic: we'll convert using relative below
                    # actually keep day seconds directly in foots via relative
                    sec = tsec
                    e = _cache_lookup(emb_cache, name)
                    if e is not None:
                        embs.append(e)
                    if box is not None and len(box) >= 4:
                        fx, fy = pes._foot_from_box(box)
                        foots.append((sec, fx, fy))
                        centers.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
                    continue
                continue
            stamps.append(ts)
            e = _cache_lookup(emb_cache, name)
            if e is not None:
                embs.append(e)
            if box is not None and len(box) >= 4:
                sec = _iso_to_day_seconds(ts, base_date)
                fx, fy = pes._foot_from_box(box)
                foots.append((sec, fx, fy))
                centers.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))

        if not stamps and not foots:
            continue
        if not embs:
            continue
        if stamps:
            secs = [_iso_to_day_seconds(s, base_date) for s in stamps]
        else:
            secs = [f[0] for f in foots]
        t_start = float(min(secs))
        t_end = float(max(secs))
        # also honour row-level first/last_timestamp if present
        for key in ("first_timestamp", "last_timestamp"):
            if row.get(key):
                try:
                    sec = _iso_to_day_seconds(row[key], base_date)
                    t_start = min(t_start, sec)
                    t_end = max(t_end, sec)
                except Exception:
                    pass
        emb = pes._l2_normalize(np.mean(np.stack(embs, axis=0), axis=0))
        center = None
        if centers:
            center = (
                float(np.mean([c[0] for c in centers])),
                float(np.mean([c[1] for c in centers])),
            )
        tracks.append(
            pes.Track(
                tid=f"{cam}_{tid}",
                cam=cam,
                t_start=t_start,
                t_end=t_end,
                sim=1.0,  # 全量輸出無 query sim；校準不依 SIM_MIN 過濾
                emb=emb,
                foots=foots,
                meta={
                    "video_id": video_id,
                    "source": "tracking_rows",
                    "n_crops": len(names),
                    "center": center,
                    "crop_names": names,
                },
            )
        )
    return tracks


def load_tracks_from_tracking_outputs(dirs: list[Path], mode_hint: str = "person") -> tuple[list, dict]:
    """
    從多個 BoT-SORT 全量輸出目錄載入所有 track。
    回傳 (tracks, stats)。
    """
    all_tracks: list[pes.Track] = []
    per_dir_stats = []
    missing = []

    for root in dirs:
        root = root.resolve()
        if not root.is_dir():
            missing.append((root, "目錄不存在"))
            continue

        jsons = sorted(root.rglob("tracking_rows*.json"))
        # prefer merged over raw/split if multiple
        preferred = [p for p in jsons if "merged" in p.name]
        if preferred:
            jsons = preferred
        if not jsons:
            missing.append((root, "找不到 tracking_rows*.json"))
            continue

        # group by camera
        by_cam: dict[str, Path] = {}
        for jp in jsons:
            cam = _infer_cam_from_path(jp)
            if cam is None:
                cam = _infer_cam_from_path(jp.parent)
            if cam is None:
                continue
            # one file per cam
            by_cam[cam] = jp

        if not by_cam:
            missing.append((root, "tracking_rows 存在但推不出鏡頭 ID"))
            continue

        # infer dataset tag for mapping
        dataset = None
        for part in [root.name, *[p.name for p in root.iterdir() if p.is_dir()]]:
            if "人員追蹤_" in part or "車輛追蹤_" in part:
                # strip _k801 suffix
                m = re.search(r"((?:人員|車輛)追蹤_\d{8})", part)
                if m:
                    dataset = m.group(1)
                    break
        if dataset is None:
            # try video_id inside first json
            sample = json.loads(next(iter(by_cam.values())).read_text(encoding="utf-8"))
            if sample:
                c0 = (sample[0].get("crops") or [{}])[0]
                # fall through
            dataset = "人員追蹤_20260507" if mode_hint == "person" else "車輛追蹤_20260507"

        pes.configure_for_input(str(QUERY_FILTER_OUTPUT_ROOT / dataset))
        ts_index, base_date, mapping_json = _load_mapping_for_dataset(dataset)
        if ts_index is None:
            missing.append((root, f"找不到 mapping：{dataset}_crop_time_mapping.json"))
            continue

        n_cam = 0
        n_tracks = 0
        emb_missing_cams = []
        for cam, jp in sorted(by_cam.items()):
            cache_path = _find_emb_cache_for_cam(root, cam, dataset)
            if cache_path is None or not cache_path.is_file():
                emb_missing_cams.append(cam)
                continue
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
            video_id = f"{dataset}_{cam}"
            tracks = tracks_from_tracking_rows(jp, cam, video_id, cache, ts_index, base_date)
            all_tracks.extend(tracks)
            n_cam += 1
            n_tracks += len(tracks)

        if emb_missing_cams:
            missing.append(
                (root, f"以下鏡頭缺逐 crop embedding cache：{emb_missing_cams}")
            )
        per_dir_stats.append(
            {
                "dir": str(root),
                "dataset": dataset,
                "n_cams": n_cam,
                "n_tracks": n_tracks,
                "mapping": str(mapping_json),
            }
        )

    if missing and not all_tracks:
        report_missing_and_exit(
            dirs,
            "原因：提供的 --tracking-output 無法載入任何全量 track。\n"
            + "\n".join(f"  - {d}: {why}" for d, why in missing),
        )
    if missing and all_tracks:
        print("警告：部分目錄載入失敗：")
        for d, why in missing:
            print(f"  - {d}: {why}")

    stats = {
        "n_tracks_total": len(all_tracks),
        "per_dir": per_dir_stats,
        "n_cams": len({t.cam for t in all_tracks}),
    }
    return all_tracks, stats


def _bbox_centers_from_tracks(tracks: list) -> tuple[dict, dict]:
    """優先用 track.meta['center']；寬度用腳底 x 範圍估。"""
    centers = {}
    cam_max_x = defaultdict(float)
    for t in tracks:
        c = (t.meta or {}).get("center")
        if c is not None:
            centers[t.tid] = c
            cam_max_x[t.cam] = max(cam_max_x[t.cam], float(c[0]) * 2.0)
        elif t.foots:
            xs = [f[1] for f in t.foots]
            ys = [f[2] for f in t.foots]
            centers[t.tid] = (float(np.mean(xs)), float(np.mean(ys)))
            cam_max_x[t.cam] = max(cam_max_x[t.cam], max(xs) if xs else 0.0)
    widths = {cam: (w if w > 1 else 1920.0) for cam, w in cam_max_x.items()}
    return centers, widths


def collect_samples(tracks: list) -> dict:
    centers, widths = _bbox_centers_from_tracks(tracks)

    emb_same: list[float] = []
    dh_same: list[float] = []
    dt_same_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    emb_diff: list[float] = []

    n_pos_h = 0
    n_neg_same_cam = 0
    n_neg_cross = 0
    pos_h_by_pair: dict[str, int] = defaultdict(int)

    for i, u in enumerate(tracks):
        for v in tracks[i + 1 :]:
            if u.cam == v.cam:
                continue
            key = tuple(sorted((u.cam, v.cam)))
            if key not in pes.OVERLAP_PAIRS:
                continue
            if not _time_overlap(u, v):
                continue
            d = _pair_dh(u, v)
            if d is None or d >= DH_POS_MAX:
                continue
            e = _pair_emb(u, v)
            emb_same.append(e)
            dh_same.append(float(d))
            n_pos_h += 1
            pos_h_by_pair[f"{key[0]}|{key[1]}"] += 1

    emb_gate = (
        float(np.percentile(emb_same, 10))
        if len(emb_same) >= 5
        else float(pes.EMB_EDGE_MIN)
    )

    for u in tracks:
        for v in tracks:
            if u.tid == v.tid or u.cam == v.cam:
                continue
            key = tuple(sorted((u.cam, v.cam)))
            if key not in pes.ADJACENT:
                continue
            if key in pes.OVERLAP_PAIRS:
                continue
            dt = v.t_start - u.t_end
            if dt <= 0 or dt > pes.DT_MAX:
                continue
            if _pair_emb(u, v) < emb_gate:
                continue
            dt_same_by_pair[key].append(float(dt))

    for i, u in enumerate(tracks):
        for v in tracks[i + 1 :]:
            if u.cam != v.cam:
                continue
            if not _time_overlap(u, v):
                continue
            cu, cv = centers.get(u.tid), centers.get(v.tid)
            if cu is None or cv is None:
                continue
            dist = math.hypot(cu[0] - cv[0], cu[1] - cv[1])
            w = widths.get(u.cam, 1920.0)
            if dist <= w / 4.0:
                continue
            emb_diff.append(_pair_emb(u, v))
            n_neg_same_cam += 1

    for i, u in enumerate(tracks):
        for v in tracks[i + 1 :]:
            if u.cam == v.cam:
                continue
            if not _time_overlap(u, v):
                continue
            d = _pair_dh(u, v)
            if d is None or d <= DH_NEG_MIN:
                continue
            emb_diff.append(_pair_emb(u, v))
            n_neg_cross += 1

    return {
        "emb_same": np.asarray(emb_same, dtype=np.float64),
        "emb_diff": np.asarray(emb_diff, dtype=np.float64),
        "dh_same": np.asarray(dh_same, dtype=np.float64),
        "dt_same_by_pair": {k: np.asarray(v, dtype=np.float64) for k, v in dt_same_by_pair.items()},
        "emb_gate_for_dt": emb_gate,
        "counts": {
            "n_tracks_total": len(tracks),
            "n_pos_h_pairs": n_pos_h,
            "n_pos_h_by_pair": dict(pos_h_by_pair),
            "n_neg_same_cam": n_neg_same_cam,
            "n_neg_cross": n_neg_cross,
            "n_dt_pairs_total": int(sum(len(v) for v in dt_same_by_pair.values())),
            "n_emb_same": int(len(emb_same)),
            "n_emb_diff": int(len(emb_diff)),
        },
    }


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
    hop = pes.hop_count(cam_a, cam_b)
    if hop is None:
        hop = 1
    return float(pes.tau(cam_a, cam_b, hop))


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

    dt_by_pair = {}
    prior_pairs = []
    for key in sorted(pes.ADJACENT):
        if key in pes.OVERLAP_PAIRS:
            continue
        arr = samples["dt_same_by_pair"].get(key, np.asarray([], dtype=np.float64))
        if len(arr) >= MIN_SAMPLES_FIT:
            fit = _fit_lognormal(arr)
            if fit is not None:
                dt_by_pair[key] = fit
                continue
        tau0 = _tau_for_pair(key[0], key[1])
        dt_by_pair[key] = {
            "family": "lognorm",
            "mu": float(np.log(max(tau0, 1e-3))),
            "sigma": PRIOR_SIGMA,
            "n": int(len(arr)),
            "prior": True,
            "tau": tau0,
            "prior_physical": True,
        }
        prior_pairs.append({"pair": key, "n_samples": int(len(arr)), "tau": tau0})

    for key, arr in samples["dt_same_by_pair"].items():
        if key in dt_by_pair:
            continue
        if len(arr) >= MIN_SAMPLES_FIT:
            fit = _fit_lognormal(arr)
            if fit is not None:
                dt_by_pair[key] = fit

    return {
        "emb_same": emb_same,
        "emb_diff": emb_diff,
        "dh_same": dh_same,
        "dh_diff": {"family": "uniform", "low": 0.0, "high": DH_DIFF_UNIFORM_MAX, "n": None},
        "dt_diff": {"family": "uniform", "low": 0.0, "high": float(pes.DT_MAX), "n": None},
        "dt_same_by_pair": {f"{a}|{b}": v for (a, b), v in dt_by_pair.items()},
        "prior_pairs": prior_pairs,
        "meta": {
            "dh_pos_max": DH_POS_MAX,
            "dh_neg_min": DH_NEG_MIN,
            "min_samples_fit": MIN_SAMPLES_FIT,
            "emb_gate_for_dt": samples["emb_gate_for_dt"],
            "counts": samples["counts"],
            "pdf_floor": PDF_FLOOR,
            "shrink_k": 10,
        },
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
            label=f"same (n={len(emb_same)})",
            color="tab:green",
            density=True,
        )
    if len(emb_diff):
        ax.hist(
            emb_diff,
            bins=bins,
            alpha=0.55,
            label=f"diff (n={len(emb_diff)})",
            color="tab:red",
            density=True,
        )
    ax.set_xlabel("embedding cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("emb | same vs emb | diff")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def write_report(calib: dict, samples: dict, out_txt: Path, hist_png: Path, load_stats: dict | None) -> None:
    lines = []
    lines.append("=== calibration report ===")
    if load_stats:
        lines.append(f"n_tracks_total: {load_stats.get('n_tracks_total')}")
        lines.append(f"n_cams: {load_stats.get('n_cams')}")
        for d in load_stats.get("per_dir") or []:
            lines.append(
                f"  dir={d['dir']}  dataset={d.get('dataset')}  "
                f"cams={d.get('n_cams')}  tracks={d.get('n_tracks')}"
            )
    lines.append(f"counts: {calib['meta']['counts']}")
    lines.append(
        f"emb_gate_for_dt (p10 of H-positives or EMB_EDGE_MIN): "
        f"{calib['meta']['emb_gate_for_dt']:.4f}"
    )
    lines.append("")
    lines.append("--- positive H pairs by camera pair ---")
    for k, n in sorted((calib["meta"]["counts"].get("n_pos_h_by_pair") or {}).items()):
        lines.append(f"  {k}: {n}")
    lines.append("")
    lines.append(
        f"emb same/diff samples: n_same={calib['meta']['counts'].get('n_emb_same')}  "
        f"n_diff={calib['meta']['counts'].get('n_emb_diff')}"
    )
    lines.append("--- emb|same (Normal) ---")
    lines.append(str(calib["emb_same"]))
    lines.append("--- emb|diff (Normal) ---")
    lines.append(str(calib["emb_diff"]))
    lines.append("--- dH|same (HalfNormal) ---")
    lines.append(str(calib["dh_same"]))
    lines.append("--- dH|diff (Uniform fixed) ---")
    lines.append(str(calib["dh_diff"]))
    lines.append("--- dt|diff (Uniform fixed) ---")
    lines.append(str(calib["dt_diff"]))
    lines.append("")
    lines.append("--- dt|same by camera pair (LogNormal; prior if n<20) ---")
    for k, v in sorted(calib["dt_same_by_pair"].items()):
        tag = " PRIOR-PHYSICAL" if v.get("prior") else ""
        lines.append(
            f"  {k}: n={v.get('n', 0)} mu={v.get('mu'):.4f} sigma={v.get('sigma'):.4f}{tag}"
        )
    lines.append("")
    lines.append(f"pairs falling back to prior: {len(calib['prior_pairs'])}")
    for p in calib["prior_pairs"]:
        lines.append(f"  {p['pair']}: n_samples={p['n_samples']} tau={p['tau']:.2f}")
    lines.append("")
    lines.append(f"histogram: {hist_png}")
    lines.append("")
    lines.append("NOTE: dt|same for adjacent non-overlap pairs uses emb >= emb_gate_for_dt")
    lines.append("as automatic same-person label (see calibrate.py docstring).")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description="Calibrate LLR distributions for path_enum_llr")
    p.add_argument(
        "--tracking-output",
        nargs="+",
        type=Path,
        default=None,
        help="BoT-SORT 全量輸出目錄（可多個日期）；需含 tracking_rows*.json + emb cache",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="calibration.pkl 路徑（也可是目錄，將寫入其中的 calibration.pkl）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="報告／直方圖輸出目錄（預設與 --out 同目錄或 ../output/path_enum_llr）",
    )
    p.add_argument(
        "--mode",
        choices=["person", "vehicle"],
        default="person",
        help="拓撲／OVERLAP 模式（依資料集自動也可，此為 fallback）",
    )
    # 舊介面僅保留盤點提示，不再靜默吃 query_filter
    p.add_argument(
        "legacy_input_dir",
        nargs="?",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = p.parse_args(argv)

    if args.tracking_output is None:
        # 友善提示：不要默默退回 query_filter
        hint_dirs = [
            OUTPUT_ROOT / "embed_cache",
            REPO_ROOT.parent / "BoT-SORT-K809" / "output",
            REPO_ROOT.parent / "dag_0507" / "botsort",
            REPO_ROOT.parent / "cost_path_experiment" / "botsort",
        ]
        existing = [d for d in hint_dirs if d.exists()]
        report_missing_and_exit(
            existing or [OUTPUT_ROOT],
            "請改用：python3 calibrate.py --tracking-output <dir1> <dir2> "
            "--out ../output/path_enum_llr/calibration.pkl\n"
            "（已停用直接吃 query_filter_merge 的舊介面）",
        )

    dirs = [Path(d).resolve() for d in args.tracking_output]

    out_target = args.out
    if out_target is None:
        out_dir = (args.out_dir or (OUTPUT_ROOT / "path_enum_llr")).resolve()
        pkl_path = out_dir / "calibration.pkl"
    else:
        out_target = Path(out_target).resolve()
        if out_target.suffix == ".pkl" or out_target.name.endswith(".pkl"):
            pkl_path = out_target
            out_dir = (args.out_dir or pkl_path.parent).resolve()
        else:
            out_dir = (args.out_dir or out_target).resolve()
            pkl_path = out_dir / "calibration.pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先做盤點：任一目錄完全沒有 tracking_rows 且沒有 emb → 依規格退出
    for d in dirs:
        inv = inventory_tracking_dir(d)
        if inv["n_tracking_rows_json"] == 0:
            report_missing_and_exit(
                dirs,
                f"目錄缺少 tracking_rows*.json（全量 track 分組）：{d}\n"
                f"雖然可能有 collage / embed_cache，但校準需要 track→crops 對應，"
                f"不能只拿散落的 crop emb。",
            )

    tracks, load_stats = load_tracks_from_tracking_outputs(dirs, mode_hint=args.mode)
    print(f"載入全量 track 總數：{len(tracks)}")

    # 用第一個 dataset 名稱設定拓撲
    if load_stats.get("per_dir"):
        ds = load_stats["per_dir"][0].get("dataset") or "人員追蹤_20260507"
        pes.configure_for_input(str(QUERY_FILTER_OUTPUT_ROOT / ds))
    print(f"模式：{pes.MODE}  H={len(pes.H_MATRICES)}")

    samples = collect_samples(tracks)
    print(
        f"正樣本 H 對={samples['counts']['n_pos_h_pairs']}  "
        f"dt 觀測={samples['counts']['n_dt_pairs_total']}  "
        f"負樣本 same-cam={samples['counts']['n_neg_same_cam']}  "
        f"cross={samples['counts']['n_neg_cross']}  "
        f"emb same/diff={samples['counts']['n_emb_same']}/{samples['counts']['n_emb_diff']}"
    )

    calib = fit_calibration(samples)
    calib["datasets"] = [d.get("dataset") for d in load_stats.get("per_dir") or []]
    calib["tracking_outputs"] = [str(d) for d in dirs]
    calib["load_stats"] = load_stats

    with pkl_path.open("wb") as f:
        pickle.dump(calib, f)

    hist_png = out_dir / "emb_same_diff_hist.png"
    save_emb_histogram(samples["emb_same"], samples["emb_diff"], hist_png)

    report = out_dir / "calibration_report.txt"
    write_report(calib, samples, report, hist_png, load_stats)

    print(f"寫入：{pkl_path}")
    print(f"寫入：{report}")
    print(f"寫入：{hist_png}")


if __name__ == "__main__":
    main()
