#!/usr/bin/env python3
"""
依 embedding + 時間接近/重疊 + 首尾邊界 IOU，鏈式合併 query_result.json 中的 tracklet。

假設同一目標若追蹤中斷，只會在中間斷成多段，因此不做全對全 Union-Find；
改為依 start_time 排序後，僅嘗試將「前段尾端」接在「後段頭端」。

合併條件（相鄰兩段 track 可併，全部 AND）：
  1. tracklet embedding cosine >= --emb-thresh
  2. 且前段 end_time 到後段 start_time：
     - 時間重疊，或
     - 間隔 <= --time-thresh 秒（預設 10）
  3. 且首尾邊界 crop IOU >= --iou-thresh

範例：
  python3 merge_query_tracks.py \\
    --query-result-json output/query_results/人員追蹤_20260528_K8-08_query_result.json

合併後 JSON 可餵給 filter_tracklet_crops（track_id 為整數，取 source_track_ids 最小值）：
  python3 filter_tracklet_crops.py \\
    --query-result-json output/query_results/人員追蹤_20260528_K8-08_query_result_merged.json \\
    --query-image CLIP-ReID-embed-test/data/p9.jpg \\
    --track-id 15
"""


from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

Label = Literal["person", "car"]

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUTPUT_ROOT = REPO_ROOT.parent / "output"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CLIP-ReID"))
sys.path.insert(0, str(HERE))


CLIP_EMBED_ROOT = OUTPUT_ROOT

from filter_tracklet_crops import (  # noqa: E402
    DEFAULT_MAPPING,
    FALLBACK_MAPPING_20260528,
    cache_lookup,
    load_bbox_lookup,
    resolve_path,
)
from query_tracklet import save_query_collage  # noqa: E402
from run_k809_botsort_clipreid import compute_global_frame, iou_matrix  # noqa: E402

DATE_RE = re.compile(r"(20\d{6})")

VEHICLE_DEFAULT_MAPPING = OUTPUT_ROOT / "車輛追蹤_crop_time_mapping.json"
VEHICLE_FALLBACK_MAPPING_20260528 = OUTPUT_ROOT / "車輛追蹤_20260528_crop_time_mapping.json"


@dataclass(frozen=True)
class LabelConfig:
    label: Label
    crop_suffix: str
    mapping_prefix: str
    cache_name: str
    default_mapping: Path
    fallback_mapping_20260528: Path

    @property
    def crop_re(self) -> re.Pattern[str]:
        return re.compile(rf"crop_s(\d+)_(\d+)_(\d+)_{self.crop_suffix}")

# 檔案標籤的設定，人員追蹤和車輛追蹤分開設定
LABEL_CONFIGS: Dict[Label, LabelConfig] = {
    "person": LabelConfig(
        label="person",
        crop_suffix="person",
        mapping_prefix="人員追蹤",
        cache_name="person_clipreid_embeddings_cache.pkl",
        default_mapping=DEFAULT_MAPPING,
        fallback_mapping_20260528=FALLBACK_MAPPING_20260528,
    ),
    "car": LabelConfig(
        label="car",
        crop_suffix="car",
        mapping_prefix="車輛追蹤",
        cache_name="vehicle_clipreid_embeddings_cache.pkl",
        default_mapping=VEHICLE_DEFAULT_MAPPING,
        fallback_mapping_20260528=VEHICLE_FALLBACK_MAPPING_20260528,
    ),}

# 解析 label
def infer_label(query_result_json: Path, video_id: str, explicit: Optional[Label]) -> Label:
    if explicit is not None:
        return explicit
    name = query_result_json.name + video_id
    if "車輛" in name or "_vehicle_query_result" in query_result_json.name:
        return "car"
    return "person"

# 解析日期 mapping 檔案
def dated_mapping_path(video_id: str, cfg: LabelConfig) -> Optional[Path]:
    m = DATE_RE.search(video_id)
    if not m:
        return None
    path = OUTPUT_ROOT / f"{cfg.mapping_prefix}_{m.group(1)}_crop_time_mapping.json"
    return path if path.is_file() else None

# 解析 video_id 的短名稱
def mapping_video_ids(video_id: str) -> List[str]:
    vids = [video_id]
    if "_" in video_id:
        short = video_id.rsplit("_", 1)[-1]
        if short not in vids:
            vids.append(short)
    return vids

# 解析 mapping 檔案
def resolve_mapping_files(
    video_id: str,
    cfg: LabelConfig,
    mapping_json: Optional[Path],
    fallback_mapping_json: Optional[Path],) -> Tuple[Path, Optional[Path]]:
    dated = dated_mapping_path(video_id, cfg)
    if dated is not None:
        print(f"    使用日期 mapping: {dated.name}")
        return dated, None

    primary = mapping_json or cfg.default_mapping
    fallback = fallback_mapping_json
    if "20260528" in video_id and cfg.fallback_mapping_20260528.is_file():
        fallback = cfg.fallback_mapping_20260528
    return primary, fallback if fallback and fallback.is_file() else None

# 解析時間戳
def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)

# 正規化向量
def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float64)
    return v / max(np.linalg.norm(v), 1e-12)

# 建立 tracklet 向量
def build_tracklet_vec(embs: Sequence[np.ndarray]) -> np.ndarray:
    return normalize_vec(np.mean(np.stack(embs, axis=0), axis=0))


def time_relation(
    start_a: str, end_a: str, start_b: str, end_b: str
) -> Tuple[float, bool]:
    s1, e1 = parse_ts(start_a), parse_ts(end_a)
    s2, e2 = parse_ts(start_b), parse_ts(end_b)
    overlap = s1 <= e2 and s2 <= e1
    if overlap:
        return 0.0, True
    if e1 <= s2:
        return (s2 - e1).total_seconds(), False
    return (s1 - e2).total_seconds(), False


def iou_box(a: Sequence[float], b: Sequence[float]) -> float:
    return float(
        iou_matrix(np.array([a], dtype=np.float64), np.array([b], dtype=np.float64))[0, 0]
    )


def crop_time_delta_sec(ca: Dict[str, Any], cb: Dict[str, Any]) -> float:
    return abs(
        (parse_ts(ca["absolute_timestamp"]) - parse_ts(cb["absolute_timestamp"])).total_seconds()
    )


def best_iou_cross_segment(
    crops_a: List[Dict[str, Any]],
    crops_b: List[Dict[str, Any]],
    *,
    crop_time_thresh: float,
) -> float:
    best = 0.0
    for ca in crops_a:
        for cb in crops_b:
            if crop_time_delta_sec(ca, cb) > crop_time_thresh:
                continue
            best = max(best, iou_box(ca["box"], cb["box"]))
    return best


def boundary_crop_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    a_last = max(a["crops"], key=lambda c: (c["absolute_timestamp"], c["global_frame"]))
    b_first = min(b["crops"], key=lambda c: (c["absolute_timestamp"], c["global_frame"]))
    if parse_ts(a_last["absolute_timestamp"]) > parse_ts(b_first["absolute_timestamp"]):
        a_last, b_first = b_first, a_last
    return iou_box(a_last["box"], b_first["box"])


def max_iou_between(
    crops_a: List[Dict[str, Any]],
    crops_b: List[Dict[str, Any]],
    *,
    frame_window: int,
) -> float:
    """Legacy：同 segment + frame 鄰近視窗（probe 腳本對照用）。"""
    best = 0.0
    for ca in crops_a:
        for cb in crops_b:
            if ca["segment_idx"] != cb["segment_idx"]:
                continue
            if ca["frame"] == cb["frame"] or abs(ca["frame"] - cb["frame"]) <= frame_window:
                best = max(best, iou_box(ca["box"], cb["box"]))
    return best


def load_embedding_cache(video_id: str, cfg: LabelConfig) -> Dict[str, np.ndarray]:
    from query_tracklet import person_cache_path  # noqa: WPS433
    from query_vehicle_tracklet import vehicle_cache_path  # noqa: WPS433

    if cfg.label == "car":
        cache_path = vehicle_cache_path(video_id)
    else:
        cache_path = person_cache_path(video_id)
    if not cache_path.is_file():
        raise FileNotFoundError(f"embedding 快取不存在: {cache_path}")
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    print(f"    載入 embedding 快取 {len(cache)} 筆: {cache_path.name}")
    return cache


def build_track_objects(
    query_payload: dict,
    bbox_lookup: Dict[str, dict],
    emb_cache: Dict[str, np.ndarray],
    cfg: LabelConfig,
) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    for entry in query_payload.get("matched_tracks", []):
        crops: List[Dict[str, Any]] = []
        embs: List[np.ndarray] = []
        for raw in entry.get("crop_paths", []):
            resolved = resolve_path(str(raw))
            name = resolved.name
            meta = bbox_lookup.get(name)
            if meta is None:
                raise KeyError(f"mapping 找不到 bbox: {name}")
            m = cfg.crop_re.search(name)
            if not m:
                raise ValueError(f"無法解析 crop 檔名: {name}")
            seg_i = int(m.group(1))
            frame = int(m.group(2))
            emb = cache_lookup(emb_cache, str(resolved))
            if emb is None:
                raise KeyError(f"embedding 快取缺少: {name}")
            crops.append(
                {
                    "crop_path": str(resolved),
                    "segment_idx": seg_i,
                    "frame": frame,
                    "global_frame": compute_global_frame(seg_i, frame),
                    "box": meta["box"],
                    "score": 1.0,
                    "absolute_timestamp": meta["absolute_timestamp"],
                }
            )
            embs.append(emb)

        crops.sort(key=lambda c: (c["global_frame"], c["crop_path"]))
        tracks.append(
            {
                "track_id": int(entry["track_id"]),
                "similarity": float(entry["similarity"]),
                "n_crops": len(crops),
                "start_time": entry["start_time"],
                "end_time": entry["end_time"],
                "crops": crops,
                "vec": build_tracklet_vec(embs),
            }
        )
    return tracks


def tail_head_gap_sec(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, bool]:
    """前段 end_time 到後段 start_time 的間隔；重疊則 gap=0。"""
    end_a = parse_ts(a["end_time"])
    start_b = parse_ts(b["start_time"])
    if start_b <= end_a:
        return 0.0, True
    return (start_b - end_a).total_seconds(), False


def merge_track_vecs(a: Dict[str, Any], b: Dict[str, Any]) -> np.ndarray:
    na = max(int(a["n_crops"]), 1)
    nb = max(int(b["n_crops"]), 1)
    return normalize_vec((a["vec"] * na + b["vec"] * nb) / (na + nb))


def append_track(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """將後段 b 接到前段 a 尾端，回傳新的 chain 狀態。"""
    all_crops: List[Dict[str, Any]] = []
    seen_paths = set()
    source_track_ids = list(a.get("source_track_ids", [a["track_id"]]))
    if b["track_id"] not in source_track_ids:
        source_track_ids.append(b["track_id"])

    for c in a["crops"] + b["crops"]:
        if c["crop_path"] in seen_paths:
            continue
        seen_paths.add(c["crop_path"])
        all_crops.append(c)
    all_crops.sort(key=lambda c: (c["global_frame"], c["crop_path"]))

    return {
        "track_id": min(source_track_ids),
        "source_track_ids": source_track_ids,
        "similarity": max(float(a["similarity"]), float(b["similarity"])),
        "n_crops": len(all_crops),
        "start_time": all_crops[0]["absolute_timestamp"],
        "end_time": all_crops[-1]["absolute_timestamp"],
        "crops": all_crops,
        "vec": merge_track_vecs(a, b),
    }


def finalize_merged_track(chain: Dict[str, Any]) -> Dict[str, Any]:
    track_ids = list(chain.get("source_track_ids", [chain["track_id"]]))
    return {
        "track_id": min(track_ids),
        "track_label": "+".join(str(tid) for tid in track_ids),
        "source_track_ids": track_ids,
        "similarity": float(chain["similarity"]),
        "n_crops": int(chain["n_crops"]),
        "start_time": chain["start_time"],
        "end_time": chain["end_time"],
        "crops": chain["crops"],
    }


def should_merge_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    emb_thresh: float,
    time_thresh: float,
    iou_thresh: float,
    crop_time_pair_thresh: float = 0.0,
) -> Tuple[bool, Dict[str, float]]:
    del crop_time_pair_thresh  # 保留參數以相容舊呼叫端；鏈式合併僅用首尾 IOU

    emb_sim = float(np.dot(a["vec"], b["vec"]))
    gap_sec, overlap = tail_head_gap_sec(a, b)
    time_ok = overlap or gap_sec <= time_thresh
    bnd_iou = boundary_crop_iou(a, b)
    iou_ok = bnd_iou >= iou_thresh
    merge = emb_sim >= emb_thresh and time_ok and iou_ok
    return merge, {
        "emb_sim": emb_sim,
        "gap_sec": gap_sec,
        "overlap": float(overlap),
        "boundary_iou": bnd_iou,
        "max_iou": bnd_iou,
    }


def merge_tracks(
    tracks: List[Dict[str, Any]],
    *,
    emb_thresh: float,
    time_thresh: float,
    iou_thresh: float,
    crop_time_pair_thresh: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    del crop_time_pair_thresh

    if not tracks:
        return [], []

    sorted_tracks = sorted(tracks, key=lambda t: t["start_time"])
    pair_logs: List[Dict[str, Any]] = []
    merged: List[Dict[str, Any]] = []

    current: Dict[str, Any] = {
        **sorted_tracks[0],
        "source_track_ids": [sorted_tracks[0]["track_id"]],
    }

    for nxt in sorted_tracks[1:]:
        ok, metrics = should_merge_pair(
            current,
            nxt,
            emb_thresh=emb_thresh,
            time_thresh=time_thresh,
            iou_thresh=iou_thresh,
        )
        if ok:
            pair_logs.append(
                {
                    "track_ids": [
                        current["source_track_ids"][-1],
                        nxt["track_id"],
                    ],
                    **metrics,
                }
            )
            current = append_track(current, nxt)
            continue

        merged.append(finalize_merged_track(current))
        current = {
            **nxt,
            "source_track_ids": [nxt["track_id"]],
        }

    merged.append(finalize_merged_track(current))
    merged.sort(key=lambda g: g["start_time"])
    return merged, pair_logs


def crop_seg_frame_key(crop: Dict[str, Any]) -> Tuple[int, int]:
    return int(crop["segment_idx"]), int(crop["frame"])


def drop_nested_singleton_tracks(
    merged: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    移除「單張 track 嵌在其他 track 時間內，且同 segment/frame 已有 crop」的結果。

    例如 track 69 已有 crop_s025_77_4，track 72 僅 crop_s025_77_1——
    merge 鏈式接軌接不起來，但不應單獨出現在 merge 成果中。
    """
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for t in merged:
        if int(t["n_crops"]) != 1:
            kept.append(t)
            continue

        tc = t["crops"][0]
        t_start = parse_ts(str(t["start_time"]))
        t_end = parse_ts(str(t["end_time"]))
        t_frame = crop_seg_frame_key(tc)
        host_id: Optional[int] = None

        for u in merged:
            if int(u["track_id"]) == int(t["track_id"]) or int(u["n_crops"]) <= 1:
                continue
            u_start = parse_ts(str(u["start_time"]))
            u_end = parse_ts(str(u["end_time"]))
            if t_start < u_start or t_end > u_end:
                continue
            u_frames = {crop_seg_frame_key(c) for c in u["crops"]}
            if t_frame in u_frames:
                host_id = int(u["track_id"])
                break

        if host_id is not None:
            removed.append(
                {
                    "track_id": int(t["track_id"]),
                    "track_label": str(t["track_label"]),
                    "similarity": float(t["similarity"]),
                    "n_crops": 1,
                    "start_time": t["start_time"],
                    "end_time": t["end_time"],
                    "crop_path": tc["crop_path"],
                    "host_track_id": host_id,
                    "reason": "nested_singleton_same_frame",
                }
            )
            continue

        kept.append(t)

    kept.sort(key=lambda g: g["start_time"])
    return kept, removed


def default_output_stem(query_json: Path) -> str:
    stem = query_json.stem
    for suffix in ("_vehicle_query_result", "_query_result"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def merged_output_suffix(label: Label) -> str:
    return "vehicle_query_result_merged" if label == "car" else "query_result_merged"


def default_output_png(query_json: Path, label: Label) -> Path:
    stem = default_output_stem(query_json)
    return query_json.with_name(f"{stem}_{merged_output_suffix(label)}.png")


def default_output_json(query_json: Path, label: Label) -> Path:
    stem = default_output_stem(query_json)
    return query_json.with_name(f"{stem}_{merged_output_suffix(label)}.json")


def save_merged_json(
    *,
    payload: dict,
    query_result_json: Path,
    merged: List[Dict[str, Any]],
    pair_logs: List[Dict[str, Any]],
    dropped_singletons: List[Dict[str, Any]],
    out_path: Path,
    emb_thresh: float,
    time_thresh: float,
    iou_thresh: float,
    crop_time_pair_thresh: float = 0.0,
) -> None:
    del crop_time_pair_thresh
    json_data = {
        "query_image": payload.get("query_image", ""),
        "video_id": payload.get("video_id", ""),
        "similarity_thresh": payload.get("similarity_thresh"),
        "merged_from": str(query_result_json.resolve()),
        "merge_params": {
            "rule": "chain_emb+tail_head_time+boundary_iou",
            "emb_thresh": emb_thresh,
            "time_thresh": time_thresh,
            "iou_thresh": iou_thresh,
        },
        "merge_pairs": [
            {
                "track_ids": log["track_ids"],
                "emb_sim": round(log["emb_sim"], 6),
                "gap_sec": round(log["gap_sec"], 3),
                "overlap": bool(log["overlap"]),
                "boundary_iou": round(log["boundary_iou"], 6),
                "max_iou": round(log["max_iou"], 6),
            }
            for log in pair_logs
        ],
        "dropped_nested_singletons": dropped_singletons,
        "matched_tracks": [
            {
                "track_id": g["track_id"],
                "track_label": g["track_label"],
                "source_track_ids": g["source_track_ids"],
                "similarity": round(g["similarity"], 6),
                "n_crops": g["n_crops"],
                "start_time": g["start_time"],
                "end_time": g["end_time"],
                "crop_paths": [c["crop_path"] for c in g["crops"]],
            }
            for g in merged
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON -> {out_path} ({len(merged)} tracks)")


def process(
    *,
    query_result_json: Path,
    output_png: Path,
    output_json: Path,
    label: Label,
    mapping_json: Optional[Path],
    fallback_mapping_json: Optional[Path],
    emb_thresh: float,
    time_thresh: float,
    iou_thresh: float,
    crop_time_pair_thresh: float,
) -> None:
    print(f"[1] 讀取 query_result: {query_result_json}")
    with query_result_json.open(encoding="utf-8") as f:
        payload = json.load(f)

    video_id = str(payload["video_id"])
    cfg = LABEL_CONFIGS[label]
    print(f"[2] video_id={video_id}  label={cfg.label}")
    emb_cache = load_embedding_cache(video_id, cfg)
    primary_mapping, fallback_mapping = resolve_mapping_files(
        video_id, cfg, mapping_json, fallback_mapping_json
    )
    print(f"    mapping: {primary_mapping.name}")
    bbox_lookup = load_bbox_lookup(
        primary_mapping,
        mapping_video_ids(video_id),
        fallback_json=fallback_mapping,
    )
    print(f"    bbox 索引 {len(bbox_lookup)} 筆")

    print("[3] 建立 track 物件 …")
    tracks = build_track_objects(payload, bbox_lookup, emb_cache, cfg)
    print(f"    原始 track 數: {len(tracks)}")

    print(
        f"[4] 鏈式合併（僅相鄰頭尾）  emb>={emb_thresh}  "
        f"且 (前段尾→後段頭 重疊/間隔<={time_thresh}s)  "
        f"且 boundary_iou>={iou_thresh} …"
    )
    merged, pair_logs = merge_tracks(
        tracks,
        emb_thresh=emb_thresh,
        time_thresh=time_thresh,
        iou_thresh=iou_thresh,
        crop_time_pair_thresh=crop_time_pair_thresh,
    )
    for log in pair_logs:
        print(
            f"    MERGE {log['track_ids'][0]} + {log['track_ids'][1]}  "
            f"emb={log['emb_sim']:.3f}  gap={log['gap_sec']:.2f}s  "
            f"overlap={bool(log['overlap'])}  "
            f"boundary={log['boundary_iou']:.3f}"
        )
    print(f"    合併後 group 數: {len(merged)}")

    print("[5] 移除嵌套單張 track（同 frame 已存在於其他 track）…")
    merged, dropped_singletons = drop_nested_singleton_tracks(merged)
    for d in dropped_singletons:
        print(
            f"    DROP track_id={d['track_id']}  host={d['host_track_id']}  "
            f"crop={Path(d['crop_path']).name}"
        )
    print(f"    最終 group 數: {len(merged)}")
    for g in merged:
        print(
            f"      group {g['track_label']} (track_id={g['track_id']})  n={g['n_crops']}  "
            f"sim={g['similarity']:.3f}  {g['start_time']} ~ {g['end_time']}"
        )

    print(f"[6] 輸出 JSON -> {output_json}")
    save_merged_json(
        payload=payload,
        query_result_json=query_result_json,
        merged=merged,
        pair_logs=pair_logs,
        dropped_singletons=dropped_singletons,
        out_path=output_json,
        emb_thresh=emb_thresh,
        time_thresh=time_thresh,
        iou_thresh=iou_thresh,
        crop_time_pair_thresh=crop_time_pair_thresh,
    )

    print(f"[7] 輸出拼圖 -> {output_png}")
    collage_entries = [{**g, "track_id": g["track_label"]} for g in merged]
    save_query_collage(matched=collage_entries, out_path=output_png)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合併 query tracklet 並輸出拼圖")
    p.add_argument("--query-result-json", type=Path, required=True)
    p.add_argument("--output-png", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--label",
        choices=["person", "car"],
        default=None,
        help="預設依檔名/video_id 自動判斷（車輛追蹤 → car）",
    )
    p.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help="預設依 video_id 日期自動選擇；20260528 則用通用 mapping + 日期 fallback",
    )
    p.add_argument("--fallback-mapping-json", type=Path, default=None)
    p.add_argument("--emb-thresh", type=float, default=0.88)
    p.add_argument("--time-thresh", type=float, default=10.0)
    p.add_argument("--iou-thresh", type=float, default=0.1)
    p.add_argument(
        "--crop-time-pair-thresh",
        type=float,
        default=2.0,
        help="已停用：保留參數以相容舊指令，鏈式合併僅使用首尾 boundary IOU",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    query_json = args.query_result_json.resolve()
    with query_json.open(encoding="utf-8") as f:
        video_id = str(json.load(f).get("video_id", ""))
    label = infer_label(query_json, video_id, args.label)
    output_png = (args.output_png or default_output_png(query_json, label)).resolve()
    output_json = (args.output_json or default_output_json(query_json, label)).resolve()
    process(
        query_result_json=query_json,
        output_png=output_png,
        output_json=output_json,
        label=label,
        mapping_json=args.mapping_json.resolve() if args.mapping_json else None,
        fallback_mapping_json=(
            args.fallback_mapping_json.resolve() if args.fallback_mapping_json else None
        ),
        emb_thresh=args.emb_thresh,
        time_thresh=args.time_thresh,
        iou_thresh=args.iou_thresh,
        crop_time_pair_thresh=args.crop_time_pair_thresh,
    )


if __name__ == "__main__":
    main()
