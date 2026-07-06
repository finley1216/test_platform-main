#!/usr/bin/env python3
"""Export dated category crop_time_mapping (e.g. 人員追蹤_20260528_K8-*)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

ASE_ROOT = Path(__file__).resolve().parents[3]
CLIP_DIR = ASE_ROOT / "CLIP-ReID-embed-test"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

from export_crop_time_mapping import (  # noqa: E402
    _box_json,
    _crop_file_exists,
    _load_pipeline_json,
    _parse_time_range_hms,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export dated category crop_time_mapping.json")
    p.add_argument("--category", required=True, help="e.g. 人員追蹤_20260528")
    p.add_argument(
        "--segment_root",
        default="/home/M133040024/ASE/test_platform-main/backend/segment",
    )
    p.add_argument(
        "--output_dir",
        default="/home/M133040024/ASE/CLIP-ReID-embed-test/data",
    )
    p.add_argument("--base_date", required=True, help="e.g. 2026-05-28")
    p.add_argument("--labels", default="person", help="Comma-separated labels to keep")
    p.add_argument("--mode", default="", help="mode field in output json")
    return p.parse_args()


def _iter_category_stems(segment_root: Path, category: str) -> Iterator[Path]:
    pat = re.compile(rf"^{re.escape(category)}_K8-\d+$", re.IGNORECASE)
    for p in sorted(segment_root.iterdir()):
        if p.is_dir() and pat.match(p.name):
            yield p


def build_category_mapping(
    segment_root: Path,
    category: str,
    base_date_str: str,
    label_filter: Set[str],
    mode: str,
) -> Dict[str, Any]:
    base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()
    segments_out: List[Dict[str, Any]] = []
    crop_id = 0
    missing_files = 0
    skipped_labels = 0

    for stem_dir in _iter_category_stems(segment_root, category):
        video_id = stem_dir.name
        payload = _load_pipeline_json(stem_dir)
        if not payload:
            print(f"[Export Skip] {video_id}: no json", flush=True)
            continue

        results = payload.get("results") or []
        if not isinstance(results, list):
            continue

        for res in results:
            if not res.get("success", True):
                continue
            segment_name = str(res.get("segment") or "")
            time_range = str(res.get("time_range") or "")
            seg_start, seg_end = _parse_time_range_hms(time_range, base_date)
            seg_start_iso = seg_start.isoformat() if seg_start else None
            seg_end_iso = seg_end.isoformat() if seg_end else None

            yolo = (res.get("raw_detection") or {}).get("yolo") or {}
            dets = yolo.get("detections") or yolo.get("crop_paths") or []
            if not isinstance(dets, list):
                continue

            crops_out: List[Dict[str, Any]] = []
            for d in dets:
                if not isinstance(d, dict):
                    continue
                label = str(d.get("label") or "").strip().lower()
                if label_filter and label not in label_filter:
                    skipped_labels += 1
                    continue
                crop_path = str(d.get("path") or "").strip()
                if not crop_path:
                    continue
                if not _crop_file_exists(stem_dir, crop_path):
                    missing_files += 1
                    continue

                rel_ts = d.get("timestamp")
                abs_ts = d.get("absolute_timestamp")
                if not abs_ts and seg_start is not None and rel_ts is not None:
                    try:
                        abs_ts = (seg_start + timedelta(seconds=float(rel_ts))).isoformat()
                    except Exception:
                        abs_ts = None

                crop_id += 1
                crops_out.append(
                    {
                        "crop_id": crop_id,
                        "crop_path": Path(crop_path).name,
                        "absolute_timestamp": abs_ts,
                        "label": d.get("label"),
                        "score": d.get("score"),
                        "relative_seconds": rel_ts,
                        "frame": d.get("frame"),
                        "box": _box_json(d.get("box")),
                        "segment_start_timestamp": seg_start_iso,
                        "segment_end_timestamp": seg_end_iso,
                    }
                )

            if crops_out:
                segments_out.append(
                    {
                        "video_id": video_id,
                        "segment": segment_name,
                        "time_range": time_range,
                        "segment_start_timestamp": seg_start_iso,
                        "segment_end_timestamp": seg_end_iso,
                        "crops": crops_out,
                    }
                )

    return {
        "generated_at": datetime.now().isoformat(),
        "source_segment_root": str(segment_root.resolve()),
        "base_date": base_date_str,
        "mode": mode or f"person_{base_date_str.replace('-', '')}",
        "segments": segments_out,
        "_stats": {
            "segments": len(segments_out),
            "crops": crop_id,
            "missing_files": missing_files,
            "skipped_labels": skipped_labels,
        },
    }


def main() -> None:
    args = parse_args()
    segment_root = Path(args.segment_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_filter = {x.strip().lower() for x in args.labels.split(",") if x.strip()}
    payload = build_category_mapping(
        segment_root,
        args.category,
        args.base_date,
        label_filter,
        args.mode or f"person_{args.base_date.replace('-', '')}",
    )
    stats = payload.pop("_stats")
    out_name = f"{args.category}_crop_time_mapping.json"
    out_path = output_dir / out_name
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Export Done] {out_path} segments={stats['segments']} crops={stats['crops']} "
        f"missing_files={stats['missing_files']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
