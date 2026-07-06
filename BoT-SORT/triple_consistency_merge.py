#!/usr/bin/env python3
"""
同鏡頭 tracklet 三重一致性合併（時間 + 空間 + 外觀，全對全 Union-Find）。

輸出 JSON / 拼圖格式與 merge_query_tracks 對齊，供 query_filter_botsort_merge_dataset 第 5 步呼叫。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from filter_tracklet_crops import load_bbox_lookup
from merge_query_tracks import (
    Label,
    LABEL_CONFIGS,
    append_track,
    build_track_objects,
    drop_nested_singleton_tracks,
    finalize_merged_track,
    load_embedding_cache,
    mapping_video_ids,
    parse_ts,
    resolve_mapping_files,
)
from query_tracklet import save_query_collage

FRAME_W = 1920
FRAME_H = 1080
FRAME_DIAG = hypot(FRAME_W, FRAME_H)

TRIPLE_RULE = "triple_consistency_time+space+appearance"


@dataclass
class UnionFind:
    parent: List[int]

    @classmethod
    def build(cls, n: int) -> "UnionFind":
        return cls(parent=list(range(n)))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class PairDecision:
    track_a: int
    track_b: int
    overlap_ratio: float
    gap_sec: float
    center_dist: float
    dist_ratio: float
    emb_sim: float
    merged: bool
    fail_reasons: List[str] = field(default_factory=list)

    @property
    def decision(self) -> str:
        return "merge" if self.merged else "reject"

    @property
    def reason(self) -> str:
        if self.merged:
            return "triple_consistency_pass"
        return "; ".join(self.fail_reasons)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "track_ids": [self.track_a, self.track_b],
            "emb_sim": round(self.emb_sim, 6),
            "overlap_ratio": round(self.overlap_ratio, 6),
            "gap_sec": round(self.gap_sec, 3),
            "center_dist": round(self.center_dist, 3),
            "dist_ratio": round(self.dist_ratio, 6),
            "decision": self.decision,
            "reason": self.reason,
        }


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    cx1, cy1 = box_center(box_a)
    cx2, cy2 = box_center(box_b)
    return hypot(cx2 - cx1, cy2 - cy1)


def overlap_ratio(
    start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime
) -> float:
    dur_a = max((end_a - start_a).total_seconds(), 0.0)
    dur_b = max((end_b - start_b).total_seconds(), 0.0)
    overlap_sec = max(
        0.0,
        (min(end_a, end_b) - max(start_a, start_b)).total_seconds(),
    )
    min_dur = min(dur_a, dur_b)
    if min_dur <= 0.0:
        return 0.0
    return overlap_sec / min_dur


def temporal_gap_sec(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    end_a = parse_ts(str(a["end_time"]))
    start_b = parse_ts(str(b["start_time"]))
    return (start_b - end_a).total_seconds()


def sorted_crops(tr: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(tr["crops"], key=lambda c: (c["global_frame"], c["crop_path"]))


def track_duration_sec(tr: Dict[str, Any]) -> float:
    return max(
        (parse_ts(str(tr["end_time"])) - parse_ts(str(tr["start_time"]))).total_seconds(),
        0.0,
    )


def is_point_fragment(tr: Dict[str, Any]) -> bool:
    return int(tr["n_crops"]) == 1 and track_duration_sec(tr) <= 0.0


def link_boxes(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    crops_a = sorted_crops(a)
    crops_b = sorted_crops(b)
    if temporal_gap_sec(a, b) < 0:
        target = parse_ts(crops_b[0]["absolute_timestamp"])
        nearest = min(
            crops_a,
            key=lambda c: abs(
                (parse_ts(c["absolute_timestamp"]) - target).total_seconds()
            ),
        )
        return nearest["box"], crops_b[0]["box"]
    return crops_a[-1]["box"], crops_b[0]["box"]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)))


def order_earlier_later(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ta = parse_ts(str(a["start_time"]))
    tb = parse_ts(str(b["start_time"]))
    if (ta, int(a["track_id"])) <= (tb, int(b["track_id"])):
        return a, b
    return b, a


def best_spatial_host_id(
    point: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    *,
    emb_thresh: float,
) -> Optional[int]:
    t = parse_ts(str(point["start_time"]))
    best_id: Optional[int] = None
    best_dist = float("inf")
    for host in candidates:
        if int(host["track_id"]) == int(point["track_id"]):
            continue
        if track_duration_sec(host) <= 0.0:
            continue
        h_start = parse_ts(str(host["start_time"]))
        h_end = parse_ts(str(host["end_time"]))
        if not (h_start <= t <= h_end):
            continue
        if cosine_sim(point["vec"], host["vec"]) < emb_thresh:
            continue
        _, near_box = link_boxes(host, point)
        d = center_distance(near_box, sorted_crops(point)[0]["box"])
        if d < best_dist:
            best_dist = d
            best_id = int(host["track_id"])
    return best_id


def evaluate_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    overlap_max: float,
    max_gap: float,
    max_dist_ratio: float,
    emb_thresh: float,
    all_tracks: Sequence[Dict[str, Any]],
) -> PairDecision:
    """A 時間較早、B 較晚；三條件全 AND。"""
    start_a = parse_ts(str(a["start_time"]))
    end_a = parse_ts(str(a["end_time"]))
    start_b = parse_ts(str(b["start_time"]))
    end_b = parse_ts(str(b["end_time"]))

    ratio = overlap_ratio(start_a, end_a, start_b, end_b)
    raw_gap = temporal_gap_sec(a, b)
    gap_sec = max(0.0, raw_gap)

    box_a, box_b = link_boxes(a, b)
    center_dist = center_distance(box_a, box_b)
    dist_ratio = center_dist / FRAME_DIAG
    emb_sim = cosine_sim(a["vec"], b["vec"])

    fail: List[str] = []
    time_ok = True
    if ratio >= overlap_max:
        time_ok = False
        fail.append(f"①時間 overlap_ratio={ratio:.3f}≥{overlap_max}（並存）")
    if gap_sec > max_gap:
        time_ok = False
        extra = f"（raw_gap={raw_gap:.1f}s）" if raw_gap != gap_sec else ""
        fail.append(f"①時間 gap={gap_sec:.1f}s>{max_gap}s{extra}")

    space_ok = dist_ratio <= max_dist_ratio
    if not space_ok:
        fail.append(
            f"②空間 dist_ratio={dist_ratio:.3f}>{max_dist_ratio} "
            f"(dist={center_dist:.0f}px)"
        )

    for point, host in ((a, b), (b, a)):
        if not is_point_fragment(point):
            continue
        pt = parse_ts(str(point["start_time"]))
        h_start = parse_ts(str(host["start_time"]))
        h_end = parse_ts(str(host["end_time"]))
        if not (h_start <= pt <= h_end):
            continue
        best_host = best_spatial_host_id(point, all_tracks, emb_thresh=emb_thresh)
        if best_host is not None and int(host["track_id"]) != best_host:
            space_ok = False
            fail.append(
                f"②單幀 id{point['track_id']} 空間最近宿主為 id{best_host}，"
                f"非 id{host['track_id']}"
            )

    appearance_ok = emb_sim >= emb_thresh
    if not appearance_ok:
        fail.append(f"③外觀 emb_sim={emb_sim:.4f}<{emb_thresh}")

    merged = time_ok and space_ok and appearance_ok
    return PairDecision(
        track_a=int(a["track_id"]),
        track_b=int(b["track_id"]),
        overlap_ratio=ratio,
        gap_sec=gap_sec,
        center_dist=center_dist,
        dist_ratio=dist_ratio,
        emb_sim=emb_sim,
        merged=merged,
        fail_reasons=fail,
    )


def merge_tracks_union_find(
    tracks: List[Dict[str, Any]],
    *,
    overlap_max: float,
    max_gap: float,
    max_dist_ratio: float,
    emb_thresh: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not tracks:
        return [], []

    n = len(tracks)
    uf = UnionFind.build(n)
    pair_logs: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = order_earlier_later(tracks[i], tracks[j])
            dec = evaluate_pair(
                a,
                b,
                overlap_max=overlap_max,
                max_gap=max_gap,
                max_dist_ratio=max_dist_ratio,
                emb_thresh=emb_thresh,
                all_tracks=tracks,
            )
            pair_logs.append(dec.to_log_dict())
            print(
                f"    PAIR {dec.track_a} → {dec.track_b}: "
                f"①overlap={dec.overlap_ratio:.3f} gap={dec.gap_sec:.1f}s "
                f"②dist_ratio={dec.dist_ratio:.3f} "
                f"③emb={dec.emb_sim:.4f} → {dec.decision.upper()} "
                f"({dec.reason})"
            )
            if dec.merged:
                uf.union(i, j)

    groups_map: Dict[int, List[Dict[str, Any]]] = {}
    for idx, tr in enumerate(tracks):
        root = uf.find(idx)
        groups_map.setdefault(root, []).append(tr)

    groups = [
        sorted(g, key=lambda t: (t["start_time"], int(t["track_id"])))
        for g in groups_map.values()
    ]
    groups.sort(key=lambda g: (g[0]["start_time"], int(g[0]["track_id"])))
    return groups, pair_logs


def combine_group(
    members: List[Dict[str, Any]],
    *,
    query_vec: Optional[np.ndarray],
) -> Dict[str, Any]:
    chain: Dict[str, Any] = {
        **members[0],
        "source_track_ids": [int(members[0]["track_id"])],
    }
    for nxt in members[1:]:
        chain = append_track(chain, nxt)
    out = finalize_merged_track(chain)
    if query_vec is not None:
        out["similarity"] = round(float(np.dot(query_vec, chain["vec"])), 6)
    return out


def save_merged_json(
    *,
    payload: dict,
    query_result_json: Path,
    merged: List[Dict[str, Any]],
    pair_logs: List[Dict[str, Any]],
    dropped_singletons: List[Dict[str, Any]],
    out_path: Path,
    emb_thresh: float,
    overlap_max: float,
    max_gap: float,
    max_dist_ratio: float,
) -> None:
    json_data = {
        "query_image": payload.get("query_image", ""),
        "video_id": payload.get("video_id", ""),
        "similarity_thresh": payload.get("similarity_thresh"),
        "merged_from": str(query_result_json.resolve()),
        "merge_params": {
            "rule": TRIPLE_RULE,
            "emb_thresh": emb_thresh,
            "overlap_max": overlap_max,
            "max_gap": max_gap,
            "max_dist_ratio": max_dist_ratio,
        },
        "merge_pairs": pair_logs,
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


def save_collage(*, merged: List[Dict[str, Any]], out_path: Path) -> None:
    collage_entries = [{**g, "track_id": g["track_label"]} for g in merged]
    save_query_collage(matched=collage_entries, out_path=out_path)


def default_emb_thresh(label: Label) -> float:
    return 0.85 if label == "person" else 0.90


def process(
    *,
    query_result_json: Path,
    output_png: Path,
    output_json: Path,
    label: Label,
    mapping_json: Optional[Path],
    fallback_mapping_json: Optional[Path],
    emb_thresh: float,
    overlap_max: float = 0.5,
    max_gap: float = 15.0,
    max_dist_ratio: float = 0.4,
    query_vec: Optional[np.ndarray] = None,
) -> None:
    print(f"[1] 讀取 query_result: {query_result_json}")
    with query_result_json.open(encoding="utf-8") as f:
        payload = json.load(f)

    video_id = str(payload["video_id"])
    cfg = LABEL_CONFIGS[label]
    print(f"[2] video_id={video_id}  label={cfg.label}  rule=triple_consistency")
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
        f"[4] 三重一致性合併（全對全 Union-Find）  "
        f"overlap<{overlap_max}  0≤gap≤{max_gap}s  "
        f"dist_ratio≤{max_dist_ratio}  emb≥{emb_thresh} …"
    )
    groups, pair_logs = merge_tracks_union_find(
        tracks,
        overlap_max=overlap_max,
        max_gap=max_gap,
        max_dist_ratio=max_dist_ratio,
        emb_thresh=emb_thresh,
    )
    merged = [combine_group(g, query_vec=query_vec) for g in groups]
    print(f"    合併後 group 數: {len(merged)}")
    for g in merged:
        print(
            f"      group {g['track_label']} (track_id={g['track_id']})  n={g['n_crops']}  "
            f"sim={g['similarity']:.3f}  {g['start_time']} ~ {g['end_time']}"
        )

    print("[5] 移除嵌套單張 track（同 frame 已存在於其他 track）…")
    merged, dropped_singletons = drop_nested_singleton_tracks(merged)
    for d in dropped_singletons:
        print(
            f"    DROP track_id={d['track_id']}  host={d['host_track_id']}  "
            f"crop={Path(d['crop_path']).name}"
        )
    print(f"    最終 group 數: {len(merged)}")

    print(f"[6] 輸出 JSON -> {output_json}")
    save_merged_json(
        payload=payload,
        query_result_json=query_result_json,
        merged=merged,
        pair_logs=pair_logs,
        dropped_singletons=dropped_singletons,
        out_path=output_json,
        emb_thresh=emb_thresh,
        overlap_max=overlap_max,
        max_gap=max_gap,
        max_dist_ratio=max_dist_ratio,
    )

    print(f"[7] 輸出拼圖 -> {output_png}")
    save_collage(merged=merged, out_path=output_png)
