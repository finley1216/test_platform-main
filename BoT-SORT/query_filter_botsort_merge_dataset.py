#!/usr/bin/env python3
"""
完整資料集：crop 0.8 篩選 → BoT-SORT → tracklet 0.9 篩選 → 相鄰 crop gap 過濾 → merge。

輸出目錄預設為 output/query_filter_merge/{dataset}/，不覆寫 output/query_results/。

範例：
  # 整包人員 0528（已有 merge 成果則跳過）
  python3 query_filter_botsort_merge_dataset.py \\
    --dataset 人員追蹤_20260528 \\
    --query-image CLIP-ReID-embed-test/data/p9.jpg

  # 整包車輛 0507，強制重跑
  python3 query_filter_botsort_merge_dataset.py \\
    --dataset 車輛追蹤_20260507 \\
    --query-image CLIP-ReID-embed-test/data/wc.png \\
    --force

  # 只跑指定鏡頭
  python3 query_filter_botsort_merge_dataset.py \\
    --dataset 人員0528 \\
    --video-ids 人員追蹤_20260528_K8-08 人員追蹤_20260528_K8-09
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CLIP-ReID"))
sys.path.insert(0, str(HERE))

from repo_paths import (  # noqa: E402
    CLIP_REID_ROOT,
    DEFAULT_PERSON_QUERY,
    DEFAULT_VEHICLE_QUERY_0507,
    DEFAULT_VEHICLE_QUERY_0528,
    OUTPUT_ROOT,
    QUERY_FILTER_OUTPUT_ROOT,
    person_embed_cache_path,
    vehicle_embed_cache_path,
)

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

from clipreid_crossmodal import ClipReIDEmbedder  # noqa: E402
from merge_query_tracks import Label, process as merge_process  # noqa: E402
from triple_consistency_merge import (  # noqa: E402
    TRIPLE_RULE,
    default_emb_thresh,
    process as triple_merge_process,
)
from query_tracklet import (  # noqa: E402
    build_tracklet_vec,
    load_person_records,
    normalize_vec,
)
from query_vehicle_tracklet import (  # noqa: E402
    load_vehicle_records,
    vehicle_crop_dir,
)
from run_k809_botsort_clipreid import (  # noqa: E402
    CLIP_REID_ROOT,
    collect_track_rows,
    default_crop_dir,
    extract_clipreid_embeddings,
    group_frames,
    init_botsort,
    run_botsort_tracking,
)

LabelKind = Literal["person", "car"]
MergeRule = Literal["chain", "triple"]

DEFAULT_DATA_DIR = OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = QUERY_FILTER_OUTPUT_ROOT

DATASET_ALIASES: Dict[str, str] = {
    "人員0528": "人員追蹤_20260528",
    "人員0507": "人員追蹤_20260507",
    "車輛0528": "車輛追蹤_20260528",
    "車輛0507": "車輛追蹤_20260507",
}

PERSON_MODEL_DEFAULTS = {
    "config_file": str(CLIP_REID_ROOT / "configs" / "person" / "vit_clipreid.yml"),
    "weight": str(CLIP_REID_ROOT / "pretrained" / "Market1501_clipreid_ViT-B-16_60.pth"),
    "num_classes": 751,
    "camera_num": 6,
    "view_num": 1,
}

VEHICLE_MODEL_DEFAULTS = {
    "config_file": str(CLIP_REID_ROOT / "configs" / "veri" / "vit_prom.yml"),
    "weight": str(CLIP_REID_ROOT / "pretrained" / "ViT_CLIP_ReID_SIE_OLP_VeRi.pth"),
    "num_classes": 576,
    "camera_num": 20,
    "view_num": 8,
}

DEFAULT_BOTSORT_KWARGS = {
    "track_high_thresh": 0.35,
    "track_low_thresh": 0.10,
    "new_track_thresh": 0.65,
    "track_buffer": 20,
    "match_thresh": 0.75,
    "proximity_thresh": 0.80,
    "appearance_thresh": 0.50,
}

DEFAULT_MERGE_KWARGS = {
    "emb_thresh": 0.88,
    "time_thresh": 10.0,
    "iou_thresh": 0.1,
    "crop_time_pair_thresh": 2.0,
}
DEFAULT_TRIPLE_MERGE_KWARGS = {
    "overlap_max": 0.5,
    "max_gap": 15.0,
    "max_dist_ratio": 0.4,
}
DEFAULT_MAX_ADJACENT_GAP_SEC = 10.0


def resolve_merge_emb_thresh(
    *,
    merge_rule: MergeRule,
    label: LabelKind,
    explicit: Optional[float],
) -> float:
    if explicit is not None:
        return explicit
    if merge_rule == "triple":
        return default_emb_thresh(label)
    return float(DEFAULT_MERGE_KWARGS["emb_thresh"])


def empty_merge_params(
    *,
    merge_rule: MergeRule,
    merge_kwargs: Dict[str, Any],
    triple_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if merge_rule == "triple":
        return {
            "rule": TRIPLE_RULE,
            "emb_thresh": triple_kwargs["emb_thresh"],
            "overlap_max": triple_kwargs["overlap_max"],
            "max_gap": triple_kwargs["max_gap"],
            "max_dist_ratio": triple_kwargs["max_dist_ratio"],
        }
    return {
        "rule": "chain_emb+tail_head_time+boundary_iou",
        "emb_thresh": merge_kwargs["emb_thresh"],
        "time_thresh": merge_kwargs["time_thresh"],
        "iou_thresh": merge_kwargs["iou_thresh"],
    }


def run_merge_step(
    *,
    merge_rule: MergeRule,
    query_result_json: Path,
    output_png: Path,
    output_json: Path,
    label: Label,
    mapping_json: Path,
    query_vec: np.ndarray,
    merge_kwargs: Dict[str, Any],
    triple_kwargs: Dict[str, Any],
) -> None:
    if merge_rule == "triple":
        triple_merge_process(
            query_result_json=query_result_json,
            output_png=output_png,
            output_json=output_json,
            label=label,
            mapping_json=mapping_json,
            fallback_mapping_json=None,
            query_vec=query_vec,
            **triple_kwargs,
        )
        return
    merge_process(
        query_result_json=query_result_json,
        output_png=output_png,
        output_json=output_json,
        label=label,
        mapping_json=mapping_json,
        fallback_mapping_json=None,
        **merge_kwargs,
    )


def max_adjacent_crop_gap_sec(crops: Sequence[Dict[str, Any]]) -> float:
    """track 內相鄰 crop 的最大時間間隔（秒）。"""
    if len(crops) < 2:
        return 0.0
    ts = sorted(datetime.fromisoformat(str(c["absolute_timestamp"])) for c in crops)
    return max((ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1))


def filter_tracks_by_adjacent_gap(
    tracks: List[Dict[str, Any]],
    *,
    max_adjacent_gap_sec: float,
    crop_ts_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if max_adjacent_gap_sec <= 0:
        return tracks, []

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for entry in tracks:
        gap = entry.get("max_adjacent_gap_sec")
        if gap is None and crop_ts_lookup is not None:
            names = [Path(p).name for p in entry.get("crop_paths", [])]
            ts = [crop_ts_lookup[n] for n in names if n in crop_ts_lookup]
            if len(ts) >= 2:
                dt = sorted(datetime.fromisoformat(t) for t in ts)
                gap = max((dt[i + 1] - dt[i]).total_seconds() for i in range(len(dt) - 1))
            else:
                gap = 0.0
            entry = {**entry, "max_adjacent_gap_sec": round(float(gap), 3)}
        gap = float(entry.get("max_adjacent_gap_sec", 0.0))
        if gap <= max_adjacent_gap_sec:
            kept.append(entry)
            continue
        dropped.append(
            {
                "track_id": entry.get("track_id"),
                "similarity": entry.get("similarity"),
                "n_crops": entry.get("n_crops"),
                "max_adjacent_gap_sec": round(gap, 3),
                "reason": "adjacent_crop_gap",
            }
        )
    return kept, dropped


def build_crop_ts_lookup(mapping_json: Path, video_id: str) -> Dict[str, str]:
    with mapping_json.open(encoding="utf-8") as f:
        data = json.load(f)
    lookup: Dict[str, str] = {}
    for seg in data.get("segments", []):
        if seg.get("video_id") != video_id:
            continue
        for crop in seg.get("crops", []):
            lookup[str(crop["crop_path"])] = str(crop.get("absolute_timestamp", ""))
    return lookup


@dataclass
class DatasetSpec:
    key: str
    prefix: str
    mapping_json: Path
    crop_root: Path
    label: LabelKind
    default_query: Path


@dataclass
class CameraResult:
    video_id: str
    status: str  # processed | skipped | merge_only | empty | error
    n_crop_candidates: int = 0
    n_botsort_tracks: int = 0
    n_tracklet_pass: int = 0
    n_merged_tracks: int = 0
    merged_png: str = ""
    merged_json: str = ""
    message: str = ""


@dataclass
class RunSummary:
    dataset: str
    query_image: str
    output_dir: str
    started_at: str
    finished_at: str = ""
    cameras: List[CameraResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "query_image": self.query_image,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stats": {
                "total": len(self.cameras),
                "processed": sum(1 for c in self.cameras if c.status == "processed"),
                "merge_only": sum(1 for c in self.cameras if c.status == "merge_only"),
                "skipped": sum(1 for c in self.cameras if c.status == "skipped"),
                "empty": sum(1 for c in self.cameras if c.status == "empty"),
                "error": sum(1 for c in self.cameras if c.status == "error"),
            },
            "cameras": [
                {
                    "video_id": c.video_id,
                    "status": c.status,
                    "n_crop_candidates": c.n_crop_candidates,
                    "n_botsort_tracks": c.n_botsort_tracks,
                    "n_tracklet_pass": c.n_tracklet_pass,
                    "n_merged_tracks": c.n_merged_tracks,
                    "merged_png": c.merged_png,
                    "merged_json": c.merged_json,
                    "message": c.message,
                }
                for c in self.cameras
            ],
        }


def resolve_dataset(name: str, data_dir: Optional[Path] = None) -> DatasetSpec:
    key = DATASET_ALIASES.get(name.strip(), name.strip())
    root = (data_dir or DEFAULT_DATA_DIR).resolve()
    mapping = root / f"{key}_crop_time_mapping.json"
    if not mapping.is_file():
        raise SystemExit(f"找不到資料集 mapping：{mapping}")

    if key.startswith("人員"):
        label: LabelKind = "person"
        default_query = DEFAULT_PERSON_QUERY
    elif key.startswith("車輛"):
        label = "car"
        default_query = (
            DEFAULT_VEHICLE_QUERY_0507 if "0507" in key else DEFAULT_VEHICLE_QUERY_0528
        )
    else:
        raise SystemExit(f"無法判斷資料集類型：{key}（需以 人員追蹤_ 或 車輛追蹤_ 開頭）")

    return DatasetSpec(
        key=key,
        prefix=key,
        mapping_json=mapping,
        crop_root=root,
        label=label,
        default_query=default_query,
    )


def list_video_ids(mapping_json: Path) -> List[str]:
    with mapping_json.open(encoding="utf-8") as f:
        data = json.load(f)
    return sorted({str(seg["video_id"]) for seg in data.get("segments", []) if seg.get("video_id")})


def output_paths(output_dir: Path, video_id: str) -> Dict[str, Path]:
    stem = video_id
    return {
        "intermediate_json": output_dir / f"{stem}_crop08_botsort09_query_result.json",
        "merged_json": output_dir / f"{stem}_crop08_botsort09_merged.json",
        "merged_png": output_dir / f"{stem}_crop08_botsort09_merged.png",
    }


def merged_exists(paths: Dict[str, Path]) -> bool:
    return paths["merged_json"].is_file() and paths["merged_png"].is_file()


def load_records(
    spec: DatasetSpec,
    mapping_json: Path,
    video_id: str,
) -> List[Dict[str, Any]]:
    crop_dir = spec.crop_root / video_id
    if spec.label == "person":
        return load_person_records(mapping_json, crop_dir, video_id=video_id)
    return load_vehicle_records(mapping_json, crop_dir, video_id=video_id)


def cache_path(spec: DatasetSpec, video_id: str) -> Path:
    root = spec.crop_root
    if spec.label == "person":
        return person_embed_cache_path(root, video_id)
    return vehicle_embed_cache_path(root, video_id)


def run_crop_botsort_tracklet_filter(
    *,
    spec: DatasetSpec,
    video_id: str,
    mapping_json: Path,
    query_vec: np.ndarray,
    embedder: ClipReIDEmbedder,
    query_image: Path,
    crop_sim_thresh: float,
    tracklet_sim_thresh: float,
    max_adjacent_gap_sec: float,
    botsort_kwargs: Dict[str, Any],
    out_json: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records = load_records(spec, mapping_json, video_id)
    stats = {
        "n_records": len(records),
        "n_crop_candidates": 0,
        "n_botsort_tracks": 0,
        "n_tracklet_pass": 0,
        "n_gap_pass": 0,
    }
    if not records:
        return [], stats

    cache = cache_path(spec, video_id)
    print(f"    embedding 快取：{cache}")
    emb_cache = extract_clipreid_embeddings(records, embedder, cache)

    filtered: List[Dict[str, Any]] = []
    for rec in records:
        vec = normalize_vec(emb_cache[rec["crop_path"]])
        sim = float(np.dot(query_vec, vec))
        if sim >= crop_sim_thresh:
            filtered.append({**rec, "sim_to_query": sim, "score": sim})
    stats["n_crop_candidates"] = len(filtered)
    print(f"    [1/4] crop ≥ {crop_sim_thresh:.2f}: {len(records)} → {len(filtered)}")

    if not filtered:
        return [], stats

    frame_groups = group_frames(filtered)
    tracker = init_botsort(video_id=video_id, **botsort_kwargs)
    rows = run_botsort_tracking(
        frame_groups,
        emb_cache,
        tracker,
        video_id=video_id,
        reset_per_segment=False,
    )
    track_rows = collect_track_rows(rows)
    stats["n_botsort_tracks"] = len(track_rows)
    print(f"    [2/4] BoT-SORT: {len(track_rows)} tracks")

    matched: List[Dict[str, Any]] = []
    for track_id, crops in track_rows:
        tracklet_vec = build_tracklet_vec(crops, emb_cache)
        mean_sim = float(np.dot(query_vec, tracklet_vec))
        if mean_sim < tracklet_sim_thresh:
            continue
        crops_sorted = sorted(crops, key=lambda r: (r["global_frame"], -r["score"]))
        gap = max_adjacent_crop_gap_sec(crops_sorted)
        matched.append(
            {
                "track_id": track_id,
                "similarity": round(mean_sim, 6),
                "n_crops": len(crops_sorted),
                "max_adjacent_gap_sec": round(gap, 3),
                "start_time": crops_sorted[0]["absolute_timestamp"],
                "end_time": crops_sorted[-1]["absolute_timestamp"],
                "crop_paths": [c["crop_path"] for c in crops_sorted],
            }
        )
    matched.sort(key=lambda x: -x["similarity"])
    stats["n_tracklet_pass"] = len(matched)
    print(f"    [3/5] tracklet ≥ {tracklet_sim_thresh:.2f}: {len(matched)} tracks")

    matched, dropped_gap = filter_tracks_by_adjacent_gap(
        matched, max_adjacent_gap_sec=max_adjacent_gap_sec
    )
    stats["n_gap_pass"] = len(matched)
    if max_adjacent_gap_sec > 0:
        print(
            f"    [4/5] 相鄰 crop gap ≤ {max_adjacent_gap_sec:.0f}s: "
            f"{stats['n_tracklet_pass']} → {len(matched)} tracks"
        )
        for d in dropped_gap:
            print(
                f"          DROP track_id={d['track_id']}  gap={d['max_adjacent_gap_sec']:.1f}s  "
                f"sim={d['similarity']:.3f}  n={d['n_crops']}"
            )
    for entry in matched:
        print(
            f"          track_id={entry['track_id']}  sim={entry['similarity']:.3f}  "
            f"gap={entry['max_adjacent_gap_sec']:.1f}s  n={entry['n_crops']}  "
            f"{entry['start_time']} ~ {entry['end_time']}"
        )

    payload = {
        "query_image": str(query_image.resolve()),
        "video_id": video_id,
        "dataset": spec.key,
        "pipeline": "crop08_botsort_tracklet09_gap",
        "crop_similarity_thresh": crop_sim_thresh,
        "tracklet_similarity_thresh": tracklet_sim_thresh,
        "max_adjacent_gap_sec": max_adjacent_gap_sec,
        "dropped_adjacent_gap": dropped_gap,
        "matched_tracks": matched,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"    中間 JSON -> {out_json.name}")

    return matched, stats


def process_camera(
    *,
    spec: DatasetSpec,
    video_id: str,
    mapping_json: Path,
    query_vec: np.ndarray,
    embedder: ClipReIDEmbedder,
    query_image: Path,
    output_dir: Path,
    crop_sim_thresh: float,
    tracklet_sim_thresh: float,
    max_adjacent_gap_sec: float,
    botsort_kwargs: Dict[str, Any],
    merge_rule: MergeRule,
    merge_kwargs: Dict[str, Any],
    triple_kwargs: Dict[str, Any],
    skip_existing: bool,
    force: bool,
) -> CameraResult:
    paths = output_paths(output_dir, video_id)
    result = CameraResult(
        video_id=video_id,
        status="processed",
        merged_png=str(paths["merged_png"]),
        merged_json=str(paths["merged_json"]),
    )

    print(f"\n{'=' * 20} [{video_id}] {'=' * 20}")

    if skip_existing and not force and merged_exists(paths):
        with paths["merged_json"].open(encoding="utf-8") as f:
            merged_data = json.load(f)
        n_tracks = len(merged_data.get("matched_tracks", []))
        result.status = "skipped"
        result.n_merged_tracks = n_tracks
        result.message = "已有 merge 成果，跳過"
        print(f"  [SKIP] {result.message}（{n_tracks} tracks）")
        return result

    try:
        need_full = force or not paths["intermediate_json"].is_file()
        if need_full:
            matched, stats = run_crop_botsort_tracklet_filter(
                spec=spec,
                video_id=video_id,
                mapping_json=mapping_json,
                query_vec=query_vec,
                embedder=embedder,
                query_image=query_image,
                crop_sim_thresh=crop_sim_thresh,
                tracklet_sim_thresh=tracklet_sim_thresh,
                max_adjacent_gap_sec=max_adjacent_gap_sec,
                botsort_kwargs=botsort_kwargs,
                out_json=paths["intermediate_json"],
            )
            result.status = "processed"
            result.n_crop_candidates = stats["n_crop_candidates"]
            result.n_botsort_tracks = stats["n_botsort_tracks"]
            result.n_tracklet_pass = stats.get("n_gap_pass", stats["n_tracklet_pass"])
        else:
            with paths["intermediate_json"].open(encoding="utf-8") as f:
                payload = json.load(f)
            matched = payload.get("matched_tracks", [])
            ts_lookup = build_crop_ts_lookup(mapping_json, video_id)
            matched, dropped_gap = filter_tracks_by_adjacent_gap(
                matched,
                max_adjacent_gap_sec=max_adjacent_gap_sec,
                crop_ts_lookup=ts_lookup,
            )
            if dropped_gap:
                print(
                    f"  [GAP] 沿用中間 JSON，再套用 gap≤{max_adjacent_gap_sec:.0f}s："
                    f"剔除 {len(dropped_gap)} tracks"
                )
            payload["matched_tracks"] = matched
            payload["max_adjacent_gap_sec"] = max_adjacent_gap_sec
            payload["dropped_adjacent_gap"] = dropped_gap
            paths["intermediate_json"].write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result.status = "merge_only"
            result.n_tracklet_pass = len(matched)
            result.message = "沿用中間 JSON，僅重跑 merge"
            print(f"  [FAST] {result.message}: {paths['intermediate_json'].name}")

        if not matched:
            result.status = "empty"
            result.message = "無符合條件的 track（tracklet / gap 篩選後為空）"
            print(f"  [EMPTY] {result.message}")
            # 仍寫空 merge 成果以便 skip 下次
            empty_merged = {
                "query_image": str(query_image.resolve()),
                "video_id": video_id,
                "dataset": spec.key,
                "similarity_thresh": tracklet_sim_thresh,
                "merged_from": str(paths["intermediate_json"].resolve()),
                "merge_params": empty_merge_params(
                    merge_rule=merge_rule,
                    merge_kwargs=merge_kwargs,
                    triple_kwargs=triple_kwargs,
                ),
                "merge_pairs": [],
                "dropped_nested_singletons": [],
                "matched_tracks": [],
            }
            paths["merged_json"].write_text(
                json.dumps(empty_merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            from query_tracklet import save_query_collage  # noqa: E402

            save_query_collage(matched=[], out_path=paths["merged_png"])
            return result

        print(f"    [5/5] merge ({merge_rule}) …")
        run_merge_step(
            merge_rule=merge_rule,
            query_result_json=paths["intermediate_json"],
            output_png=paths["merged_png"],
            output_json=paths["merged_json"],
            label=spec.label,
            mapping_json=mapping_json,
            query_vec=query_vec,
            merge_kwargs=merge_kwargs,
            triple_kwargs=triple_kwargs,
        )
        with paths["merged_json"].open(encoding="utf-8") as f:
            merged_data = json.load(f)
        result.n_merged_tracks = len(merged_data.get("matched_tracks", []))
        result.message = f"完成，merge 後 {result.n_merged_tracks} tracks"
        print(f"  [OK] {result.message}")
        return result

    except Exception as exc:
        result.status = "error"
        result.message = str(exc)
        print(f"  [ERROR] {video_id}: {exc}")
        traceback.print_exc()
        return result


def print_final_summary(summary: RunSummary) -> None:
    stats = summary.to_dict()["stats"]
    print("\n" + "=" * 60)
    print(f"資料集：{summary.dataset}")
    print(f"輸出目錄：{summary.output_dir}")
    print(f"鏡頭總數：{stats['total']}")
    print(f"  完整處理：{stats['processed']}")
    print(f"  僅 merge：{stats['merge_only']}")
    print(f"  跳過（已有成果）：{stats['skipped']}")
    print(f"  無結果：{stats['empty']}")
    print(f"  錯誤：{stats['error']}")
    print("-" * 60)

    if stats["skipped"]:
        print("跳過的鏡頭：")
        for c in summary.cameras:
            if c.status == "skipped":
                print(f"  - {c.video_id}  ({c.n_merged_tracks} tracks)")

    if stats["processed"] or stats["merge_only"]:
        print("本次產出/更新的鏡頭：")
        for c in summary.cameras:
            if c.status in ("processed", "merge_only"):
                print(
                    f"  - {c.video_id}  merge={c.n_merged_tracks} tracks  "
                    f"({c.status})  -> {Path(c.merged_png).name}"
                )

    if stats["empty"]:
        print("無符合結果的鏡頭：")
        for c in summary.cameras:
            if c.status == "empty":
                print(f"  - {c.video_id}")

    if stats["error"]:
        print("失敗的鏡頭：")
        for c in summary.cameras:
            if c.status == "error":
                print(f"  - {c.video_id}: {c.message}")

    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="完整資料集：crop0.8 → BoT-SORT → tracklet0.9 → merge"
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="例如 人員追蹤_20260528、人員0528、車輛追蹤_20260507",
    )
    p.add_argument("--query-image", type=Path, default=None)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="mapping JSON 與 crop 圖根目錄（預設 ../output）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="預設 output/query_filter_merge/{dataset}/",
    )
    p.add_argument(
        "--video-ids",
        nargs="*",
        default=None,
        help="只跑指定鏡頭；預設跑 mapping 內全部",
    )
    p.add_argument("--crop-sim-thresh", type=float, default=0.80)
    p.add_argument("--tracklet-sim-thresh", type=float, default=0.90)
    p.add_argument(
        "--max-adjacent-gap-sec",
        type=float,
        default=1.33,
        help="track 內相鄰 crop 最大時間間隔（秒）；設 0 可關閉",
    )
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--force", action="store_true", help="忽略已有成果，完整重跑")
    p.add_argument("--appearance-thresh", type=float, default=0.50)
    p.add_argument("--proximity-thresh", type=float, default=0.80)
    p.add_argument("--match-thresh", type=float, default=0.75)
    p.add_argument("--track-buffer", type=int, default=5)
    p.add_argument("--new-track-thresh", type=float, default=0.65)
    p.add_argument("--track-high-thresh", type=float, default=0.35)
    p.add_argument("--track-low-thresh", type=float, default=0.10)
    p.add_argument(
        "--merge-rule",
        choices=["chain", "triple"],
        default="chain",
        help="第 5 步 merge 規則；chain=舊鏈式 IOU，triple=三重一致性 Union-Find",
    )
    p.add_argument(
        "--merge-emb-thresh",
        type=float,
        default=None,
        help="merge embedding 門檻；預設 chain=0.88，triple 人員=0.85 車輛=0.90",
    )
    p.add_argument(
        "--merge-time-thresh",
        type=float,
        default=10.0,
        help="chain 模式：首尾最大間隔（秒）；triple 模式忽略",
    )
    p.add_argument(
        "--merge-iou-thresh",
        type=float,
        default=0.1,
        help="chain 模式：boundary IOU 門檻；triple 模式忽略",
    )
    p.add_argument(
        "--overlap-max",
        type=float,
        default=DEFAULT_TRIPLE_MERGE_KWARGS["overlap_max"],
        help="triple 模式：overlap_ratio 上限（≥ 視為並存）",
    )
    p.add_argument(
        "--max-gap",
        type=float,
        default=DEFAULT_TRIPLE_MERGE_KWARGS["max_gap"],
        help="triple 模式：A.end→B.start 最大間隔（秒）",
    )
    p.add_argument(
        "--max-dist-ratio",
        type=float,
        default=DEFAULT_TRIPLE_MERGE_KWARGS["max_dist_ratio"],
        help="triple 模式：中心距 / 畫面對角線上限",
    )
    p.add_argument(
        "--repo-root",
        default=str(CLIP_REID_ROOT),
        help="CLIP-ReID 專案根目錄",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(HERE)

    data_dir = (args.data_dir or DEFAULT_DATA_DIR).resolve()
    spec = resolve_dataset(args.dataset, data_dir)

    from download_weights import ensure_for_dataset  # noqa: E402

    ensure_for_dataset(spec.key)

    query_image = (args.query_image or spec.default_query).resolve()
    if not query_image.is_file():
        raise SystemExit(f"query 圖片不存在：{query_image}")

    output_dir = (args.output_dir or (DEFAULT_OUTPUT_ROOT / spec.prefix)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ids = args.video_ids or list_video_ids(spec.mapping_json)
    if not video_ids:
        raise SystemExit(f"資料集 {spec.key} 內找不到任何 video_id")

    model_defaults = PERSON_MODEL_DEFAULTS if spec.label == "person" else VEHICLE_MODEL_DEFAULTS
    embedder = ClipReIDEmbedder(
        repo_root=args.repo_root,
        config_file=model_defaults["config_file"],
        weight=model_defaults["weight"],
        num_classes=model_defaults["num_classes"],
        camera_num=model_defaults["camera_num"],
        view_num=model_defaults["view_num"],
    )

    print("[0] 初始化 CLIP-ReID …")
    print(f"    資料集：{spec.key}  ({spec.label})")
    print(f"    data_dir：{data_dir}")
    print(f"    mapping：{spec.mapping_json.name}")
    print(f"    query：{query_image}")
    print(f"    輸出：{output_dir}")
    print(f"    鏡頭數：{len(video_ids)}")
    merge_emb_thresh = resolve_merge_emb_thresh(
        merge_rule=args.merge_rule,
        label=spec.label,
        explicit=args.merge_emb_thresh,
    )
    print(
        f"    流程：crop≥{args.crop_sim_thresh} → BoT-SORT → tracklet≥{args.tracklet_sim_thresh} "
        f"→ gap≤{args.max_adjacent_gap_sec:.0f}s → merge({args.merge_rule})"
    )
    if args.merge_rule == "triple":
        print(
            f"    triple: emb≥{merge_emb_thresh}  overlap<{args.overlap_max}  "
            f"gap≤{args.max_gap}s  dist_ratio≤{args.max_dist_ratio}"
        )
    else:
        print(
            f"    chain: emb≥{merge_emb_thresh}  time≤{args.merge_time_thresh}s  "
            f"iou≥{args.merge_iou_thresh}"
        )
    print(f"    skip_existing={args.skip_existing}  force={args.force}")

    query_vec = normalize_vec(embedder.encode_one(str(query_image)))

    botsort_kwargs = {
        **DEFAULT_BOTSORT_KWARGS,
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh": args.track_low_thresh,
        "new_track_thresh": args.new_track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "proximity_thresh": args.proximity_thresh,
        "appearance_thresh": args.appearance_thresh,
    }
    merge_kwargs = {
        **DEFAULT_MERGE_KWARGS,
        "emb_thresh": merge_emb_thresh,
        "time_thresh": args.merge_time_thresh,
        "iou_thresh": args.merge_iou_thresh,
    }
    triple_kwargs = {
        "emb_thresh": merge_emb_thresh,
        "overlap_max": args.overlap_max,
        "max_gap": args.max_gap,
        "max_dist_ratio": args.max_dist_ratio,
    }

    summary = RunSummary(
        dataset=spec.key,
        query_image=str(query_image),
        output_dir=str(output_dir),
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    for i, video_id in enumerate(video_ids, 1):
        print(f"\n>>> 鏡頭 {i}/{len(video_ids)}")
        result = process_camera(
            spec=spec,
            video_id=video_id,
            mapping_json=spec.mapping_json,
            query_vec=query_vec,
            embedder=embedder,
            query_image=query_image,
            output_dir=output_dir,
            crop_sim_thresh=args.crop_sim_thresh,
            tracklet_sim_thresh=args.tracklet_sim_thresh,
            max_adjacent_gap_sec=args.max_adjacent_gap_sec,
            botsort_kwargs=botsort_kwargs,
            merge_rule=args.merge_rule,
            merge_kwargs=merge_kwargs,
            triple_kwargs=triple_kwargs,
            skip_existing=args.skip_existing,
            force=args.force,
        )
        summary.cameras.append(result)

    summary.finished_at = datetime.now(timezone.utc).isoformat()
    summary_path = output_dir / f"{spec.prefix}_run_summary.json"
    summary_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_final_summary(summary)
    print(f"摘要 JSON -> {summary_path}")


if __name__ == "__main__":
    main()
