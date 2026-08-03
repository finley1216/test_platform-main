#!/usr/bin/env python3
"""對 query_filter_merge 的 merged JSON 內每條 track 跑 combined intra-filter 並輸出拼圖。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CLIP_REID_ROOT = REPO_ROOT / "CLIP-ReID"
DEFAULT_PERSON_QUERY = REPO_ROOT / "p9.jpg"
OUTPUT_ROOT = REPO_ROOT.parent / "output"
QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CLIP-ReID"))
sys.path.insert(0, str(HERE))


CLIP_EMBED_ROOT = OUTPUT_ROOT
ASE_ROOT = REPO_ROOT.parent

from clipreid_crossmodal import ClipReIDEmbedder, add_common_model_args
from filter_tracklet_crops import (
    ASE_ROOT,
    CLIP_EMBED_ROOT,
    CLIP_REID_ROOT,
    FALLBACK_MAPPING_20260528,
    apply_filter_combined,
    build_records,
    compute_combined_scores,
    compute_mean_sim_to_ref,
    ensure_embeddings,
    filter_result_paths,
    load_bbox_lookup,
    load_embedding_cache,
    resolve_path,
    save_filter_collage_combined,
    select_reference_set,
    write_combined_result_json,
)
from query_tracklet import save_query_collage
from run_k809_botsort_clipreid import compute_global_frame

CROP_RE = re.compile(r"crop_s(\d+)_(\d+)_")
DEFAULT_MAPPING = OUTPUT_ROOT / "人員追蹤_crop_time_mapping.json"


def crop_record_to_collage(rec) -> Dict[str, Any]:
    m = CROP_RE.match(rec.crop_name)
    seg_i = int(m.group(1)) if m else 0
    frame = int(m.group(2)) if m else 0
    return {
        "crop_path": rec.image_path,
        "segment_idx": seg_i,
        "frame": frame,
        "global_frame": compute_global_frame(seg_i, frame),
        "box": rec.box,
        "score": float(rec.combined_score or rec.sim_to_query),
        "absolute_timestamp": rec.timestamp_str,
    }


def process_merged_json(
    merged_json: Path,
    *,
    query_image: Path,
    mapping_json: Path,
    alpha: float,
    combined_thresh: float,
    top_k: int,
    embedder: ClipReIDEmbedder,
) -> None:
    payload = json.loads(merged_json.read_text(encoding="utf-8"))
    tracks = payload.get("matched_tracks", [])
    video_id = str(payload.get("video_id", merged_json.stem))
    stem = merged_json.name.replace("_merged.json", "")
    out_dir = merged_json.parent / "filter_results" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    map_vids = ["K8-09", video_id]
    if "K8-09" not in video_id:
        map_vids.append(video_id.split("_")[-1] if "_" in video_id else video_id)
    fallback_map = FALLBACK_MAPPING_20260528 if "20260528" in video_id else None
    if fallback_map and video_id not in map_vids:
        map_vids = list(map_vids) + [video_id]

    bbox_lookup = load_bbox_lookup(
        mapping_json.resolve(),
        map_vids,
        fallback_json=fallback_map,
    )
    cache_path = out_dir / "clipreid_embeddings_cache.pkl"
    cache = load_embedding_cache(cache_path)

    print(f"\n=== {merged_json.name}  tracks={len(tracks)}  out={out_dir} ===")
    collage_entries: List[Dict[str, Any]] = []

    for track in tracks:
        track_id = int(track["track_id"])
        print(f"  track_id={track_id}  label={track.get('track_label', track_id)}  n={track.get('n_crops')}")
        records = build_records(track, bbox_lookup)
        query_emb = ensure_embeddings(
            records, cache, cache_path, embedder, query_image
        )
        ref_indices = select_reference_set(records, query_emb, top_k)

        compute_mean_sim_to_ref(records, ref_indices)
        compute_combined_scores(records, alpha)
        kept, rejected = apply_filter_combined(records, combined_thresh=combined_thresh)

        out_png, out_json = filter_result_paths(out_dir, track_id)
        save_filter_collage_combined(
            kept=kept,
            rejected=rejected,
            track_id=track_id,
            alpha=alpha,
            combined_thresh=combined_thresh,
            out_path=out_png,
        )
        write_combined_result_json(
            out_json,
            track_id=track_id,
            top_k=top_k,
            alpha=alpha,
            combined_thresh=combined_thresh,
            ref_indices=ref_indices,
            records=records,
            kept=kept,
            rejected=rejected,
        )

        if kept:
            kept_recs = sorted(kept, key=lambda r: (r.timestamp, r.crop_name))
            collage_entries.append(
                {
                    "track_id": track.get("track_label", str(track_id)),
                    "similarity": float(track.get("similarity", 0.0)),
                    "n_crops": len(kept),
                    "start_time": kept_recs[0].timestamp_str,
                    "end_time": kept_recs[-1].timestamp_str,
                    "crops": [crop_record_to_collage(r) for r in kept_recs],
                }
            )

    summary_png = merged_json.parent / f"{stem}_filtered_merged.png"
    save_query_collage(matched=collage_entries, out_path=summary_png)
    print(f"  summary -> {summary_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批次 filter merged tracklets")
    p.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
    )
    p.add_argument(
        "--query-image",
        type=Path,
        default=DEFAULT_PERSON_QUERY,
    )
    p.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--combined-thresh", type=float, default=0.90)
    p.add_argument("--top-k", type=int, default=3)
    add_common_model_args(p)
    p.set_defaults(
        repo_root=str(CLIP_REID_ROOT),
        config_file=str(CLIP_REID_ROOT / "configs" / "person" / "vit_clipreid.yml"),
        weight=str(CLIP_REID_ROOT / "pretrained" / "Market1501_clipreid_ViT-B-16_60.pth"),
        num_classes=751,
        camera_num=6,
        view_num=1,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge_dir = args.merge_dir.resolve()
    query_image = resolve_path(str(args.query_image))
    merged_jsons = sorted(merge_dir.glob("*_merged.json"))

    embedder = ClipReIDEmbedder(
        repo_root=args.repo_root,
        config_file=args.config_file,
        weight=args.weight,
        num_classes=args.num_classes,
        camera_num=args.camera_num,
        view_num=args.view_num,
    )

    for merged_json in merged_jsons:
        process_merged_json(
            merged_json,
            query_image=query_image,
            mapping_json=args.mapping_json,
            alpha=args.alpha,
            combined_thresh=args.combined_thresh,
            top_k=args.top_k,
            embedder=embedder,
        )


if __name__ == "__main__":
    main()
