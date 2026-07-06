#!/usr/bin/env python3
"""Export crop time mapping JSON from test_platform segment pipeline results."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

PERSON_STEM_RE = re.compile(r"^K8-\d+$", re.IGNORECASE)
VEHICLE_STEM_RE = re.compile(r"^車輛追蹤_K8-\d+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 人員/車輛 crop_time_mapping.json from segment dirs.")
    p.add_argument(
        "--segment_root",
        default=str(Path(__file__).resolve().parents[1] / "backend" / "segment"),
        help="test_platform segment root (contains K8-xx / 車輛追蹤_K8-xx).",
    )
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parents[1].parent / "output"),
        help="Where to write mapping json files.",
    )
    p.add_argument(
        "--person_base_date",
        default="2026-05-07",
        help="Base calendar date for 人員追蹤 (time_range is offset from video start).",
    )
    p.add_argument(
        "--vehicle_base_date",
        default="2026-05-01",
        help="Base calendar date for 車輛追蹤 (matches legacy crop_embeddings_db).",
    )
    p.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels to keep; empty = keep all labels.",
    )
    p.add_argument(
        "--export_person",
        action="store_true",
        help="Write 人員追蹤_crop_time_mapping.json (K8-* stems).",
    )
    p.add_argument(
        "--export_vehicle",
        action="store_true",
        help="Write 車輛追蹤_crop_time_mapping.json (車輛追蹤_K8-* stems).",
    )
    return p.parse_args()


def _parse_time_range_hms(time_range_str: str, base_date: datetime.date) -> Tuple[Optional[datetime], Optional[datetime]]:
    if not time_range_str or " - " not in time_range_str:
        return None, None
    try:
        start_str, end_str = time_range_str.split(" - ", 1)
        t0 = datetime.strptime(start_str.strip(), "%H:%M:%S").time()
        t1 = datetime.strptime(end_str.strip(), "%H:%M:%S").time()
        return (
            datetime.combine(base_date, t0),
            datetime.combine(base_date, t1),
        )
    except ValueError:
        return None, None


def _box_json(box: Any) -> Optional[str]:
    if box is None:
        return None
    if isinstance(box, str):
        return box
    try:
        return json.dumps(box, ensure_ascii=False)
    except Exception:
        return None


def _iter_stems(segment_root: Path, mode: str) -> Iterator[Path]:
    pat = PERSON_STEM_RE if mode == "person" else VEHICLE_STEM_RE
    for p in sorted(segment_root.iterdir()):
        if p.is_dir() and pat.match(p.name):
            yield p


def _load_pipeline_json(stem_dir: Path) -> Optional[Dict[str, Any]]:
    cand = stem_dir / f"{stem_dir.name}.json"
    if not cand.is_file():
        return None
    return json.loads(cand.read_text(encoding="utf-8"))


def _crop_file_exists(stem_dir: Path, crop_path: str) -> bool:
    rel = Path(crop_path)
    checks = [
        stem_dir / rel,
        stem_dir / "yolo_output" / "object_crops" / rel.name,
        stem_dir / rel.name,
    ]
    return any(c.is_file() for c in checks)


def build_mapping_for_mode(
    segment_root: Path,
    mode: str,
    base_date_str: str,
    label_filter: Set[str],
) -> Dict[str, Any]:
    base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()
    segments_out: List[Dict[str, Any]] = []
    crop_id = 0
    missing_files = 0
    skipped_labels = 0

    for stem_dir in _iter_stems(segment_root, mode):
        video_id = stem_dir.name
        payload = _load_pipeline_json(stem_dir)
        if not payload:
            print(f"[Skip] {video_id}: no {video_id}.json")
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

    print(
        f"[{mode}] stems processed, segments={len(segments_out)}, crops={crop_id}, "
        f"missing_files={missing_files}, skipped_labels={skipped_labels}"
    )
    return {
        "generated_at": datetime.now().isoformat(),
        "source_segment_root": str(segment_root.resolve()),
        "base_date": base_date_str,
        "mode": mode,
        "segments": segments_out,
    }


def main() -> None:
    args = parse_args()
    segment_root = Path(args.segment_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_filter: Set[str] = set()
    if args.labels.strip():
        label_filter = {x.strip().lower() for x in args.labels.split(",") if x.strip()}

    do_person = args.export_person or (not args.export_person and not args.export_vehicle)
    do_vehicle = args.export_vehicle or (not args.export_person and not args.export_vehicle)

    if do_person:
        person_labels = label_filter or {"person"}
        person_payload = build_mapping_for_mode(
            segment_root, "person", args.person_base_date, person_labels
        )
        person_path = output_dir / "人員追蹤_crop_time_mapping.json"
        person_path.write_text(
            json.dumps(person_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_person = sum(len(s["crops"]) for s in person_payload["segments"])
        print(f"[Done] {person_path} ({n_person} crops)")

    if do_vehicle:
        vehicle_labels = label_filter or {"car", "truck", "bus", "motorcycle", "bicycle"}
        vehicle_payload = build_mapping_for_mode(
            segment_root, "vehicle", args.vehicle_base_date, vehicle_labels
        )
        vehicle_path = output_dir / "車輛追蹤_crop_time_mapping.json"
        vehicle_path.write_text(
            json.dumps(vehicle_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_crops = sum(len(s["crops"]) for s in vehicle_payload["segments"])
        print(f"[Done] {vehicle_path} ({n_crops} crops)")


if __name__ == "__main__":
    main()
