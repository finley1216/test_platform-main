#!/usr/bin/env python3
"""test_platform-main 路徑常數（repo 與上一層 output/）。"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT.parent / "output"
BOTSORT_ROOT = REPO_ROOT / "BoT-SORT"
CLIP_REID_ROOT = REPO_ROOT / "CLIP-ReID"
SEGMENT_ROOT = REPO_ROOT / "backend" / "segment"

DEFAULT_PERSON_QUERY = REPO_ROOT / "p9.jpg"
DEFAULT_VEHICLE_QUERY_0528 = REPO_ROOT / "wc.png"
DEFAULT_VEHICLE_QUERY_0507 = REPO_ROOT / "BSH-5613.jpg"

QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"
