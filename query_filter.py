#!/usr/bin/env python3
"""
一鍵執行 query 篩選完整流程（兩段合併）：

  Step 1  BoT-SORT/query_filter_botsort_merge_dataset.py
          crop≥0.8 → BoT-SORT → tracklet≥0.9 → gap → merge

  Step 2  BoT-SORT/batch_filter_merged_tracks.py
          每條 merged track 做 combined intra-filter（α + combined thresh）

範例：
  cd test_platform-main
  python3 query_filter_botsort_merge_filter_dataset.py \\
    --dataset 人員追蹤_20260528 \\
    --merge-rule triple \\
    --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from repo_paths import (  # noqa: E402
    BOTSORT_ROOT,
    DEFAULT_PERSON_QUERY,
    DEFAULT_VEHICLE_QUERY_0507,
    DEFAULT_VEHICLE_QUERY_0528,
    OUTPUT_ROOT,
    QUERY_FILTER_OUTPUT_ROOT,
)

STEP1 = BOTSORT_ROOT / "query_filter_botsort_merge_dataset.py"
STEP2 = BOTSORT_ROOT / "batch_filter_merged_tracks.py"

DATASET_ALIASES = {
    "人員0528": "人員追蹤_20260528",
    "人員0507": "人員追蹤_20260507",
    "車輛0528": "車輛追蹤_20260528",
    "車輛0507": "車輛追蹤_20260507",
}


def resolve_dataset_key(name: str) -> str:
    return DATASET_ALIASES.get(name.strip(), name.strip())


def default_query_image(dataset_key: str) -> Path:
    if dataset_key.startswith("人員"):
        return DEFAULT_PERSON_QUERY
    if dataset_key.startswith("車輛"):
        if "0507" in dataset_key:
            return DEFAULT_VEHICLE_QUERY_0507
        return DEFAULT_VEHICLE_QUERY_0528
    raise SystemExit(f"無法判斷資料集類型：{dataset_key}")


def default_mapping_json(dataset_key: str, data_dir: Path) -> Path:
    path = data_dir / f"{dataset_key}_crop_time_mapping.json"
    if not path.is_file():
        raise SystemExit(f"找不到 mapping：{path}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="一鍵：crop 篩選 → BoT-SORT → merge → combined intra-filter"
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
        help=f"mapping + crop 根目錄（預設 {OUTPUT_ROOT}）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"merge/filter 結果目錄（預設 {QUERY_FILTER_OUTPUT_ROOT}/{{dataset}}/）",
    )

    # Step 1
    p.add_argument("--video-ids", nargs="*", default=None)
    p.add_argument("--crop-sim-thresh", type=float, default=0.80)
    p.add_argument("--tracklet-sim-thresh", type=float, default=0.85)
    p.add_argument("--max-adjacent-gap-sec", type=float, default=1.33)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--force", action="store_true")
    p.add_argument("--merge-rule", choices=["chain", "triple"], default="triple")
    p.add_argument("--merge-emb-thresh", type=float, default=None)
    p.add_argument("--merge-time-thresh", type=float, default=10.0)
    p.add_argument("--merge-iou-thresh", type=float, default=0.1)
    p.add_argument("--overlap-max", type=float, default=0.5)
    p.add_argument("--max-gap", type=float, default=15.0)
    p.add_argument("--max-dist-ratio", type=float, default=0.4)
    p.add_argument("--appearance-thresh", type=float, default=0.50)
    p.add_argument("--proximity-thresh", type=float, default=0.80)
    p.add_argument("--match-thresh", type=float, default=0.75)
    p.add_argument("--track-buffer", type=int, default=5)
    p.add_argument("--new-track-thresh", type=float, default=0.65)
    p.add_argument("--track-high-thresh", type=float, default=0.35)
    p.add_argument("--track-low-thresh", type=float, default=0.10)

    # Step 2
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--combined-thresh", type=float, default=0.90)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help="覆寫 mapping 路徑（預設 {data-dir}/{dataset}_crop_time_mapping.json）",
    )

    p.add_argument("--skip-step1", action="store_true", help="略過 merge，只跑 intra-filter")
    p.add_argument("--skip-step2", action="store_true", help="略過 intra-filter，只跑 merge")

    return p.parse_args()


def _append_opt(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_step1_cmd(
    args: argparse.Namespace,
    dataset_key: str,
    data_dir: Path,
    output_dir: Path,
) -> list[str]:
    query_image = (args.query_image or default_query_image(dataset_key)).resolve()
    cmd = [
        sys.executable,
        str(STEP1),
        "--dataset",
        dataset_key,
        "--data-dir",
        str(data_dir),
        "--query-image",
        str(query_image),
        "--output-dir",
        str(output_dir),
        "--crop-sim-thresh",
        str(args.crop_sim_thresh),
        "--tracklet-sim-thresh",
        str(args.tracklet_sim_thresh),
        "--max-adjacent-gap-sec",
        str(args.max_adjacent_gap_sec),
        "--merge-rule",
        args.merge_rule,
        "--merge-time-thresh",
        str(args.merge_time_thresh),
        "--merge-iou-thresh",
        str(args.merge_iou_thresh),
        "--overlap-max",
        str(args.overlap_max),
        "--max-gap",
        str(args.max_gap),
        "--max-dist-ratio",
        str(args.max_dist_ratio),
        "--appearance-thresh",
        str(args.appearance_thresh),
        "--proximity-thresh",
        str(args.proximity_thresh),
        "--match-thresh",
        str(args.match_thresh),
        "--track-buffer",
        str(args.track_buffer),
        "--new-track-thresh",
        str(args.new_track_thresh),
        "--track-high-thresh",
        str(args.track_high_thresh),
        "--track-low-thresh",
        str(args.track_low_thresh),
    ]
    if args.force:
        cmd.append("--force")
    if not args.skip_existing:
        cmd.append("--no-skip-existing")
    if args.video_ids:
        cmd.extend(["--video-ids", *args.video_ids])
    _append_opt(cmd, "--merge-emb-thresh", args.merge_emb_thresh)
    return cmd


def build_step2_cmd(
    args: argparse.Namespace,
    dataset_key: str,
    data_dir: Path,
    output_dir: Path,
) -> list[str]:
    query_image = (args.query_image or default_query_image(dataset_key)).resolve()
    mapping_json = (args.mapping_json or default_mapping_json(dataset_key, data_dir)).resolve()
    return [
        sys.executable,
        str(STEP2),
        "--merge-dir",
        str(output_dir),
        "--query-image",
        str(query_image),
        "--mapping-json",
        str(mapping_json),
        "--alpha",
        str(args.alpha),
        "--combined-thresh",
        str(args.combined_thresh),
        "--top-k",
        str(args.top_k),
    ]


def run_cmd(label: str, cmd: list[str], *, cwd: Path) -> None:
    print("\n" + "=" * 60)
    print(f"{label}")
    print(" ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> None:
    args = parse_args()
    dataset_key = resolve_dataset_key(args.dataset)
    data_dir = (args.data_dir or OUTPUT_ROOT).resolve()
    output_dir = (args.output_dir or (QUERY_FILTER_OUTPUT_ROOT / dataset_key)).resolve()

    if args.skip_step1 and args.skip_step2:
        raise SystemExit("不能同時 --skip-step1 與 --skip-step2")

    if not STEP1.is_file():
        raise SystemExit(f"找不到 Step1 腳本：{STEP1}")
    if not STEP2.is_file():
        raise SystemExit(f"找不到 Step2 腳本：{STEP2}")

    mapping_json = (args.mapping_json or (data_dir / f"{dataset_key}_crop_time_mapping.json")).resolve()
    if not mapping_json.is_file():
        raise SystemExit(
            f"找不到 mapping：{mapping_json}\n"
            "請先執行 backend/scripts/export_category_crop_time_mapping.py\n"
            "詳見 docs/追蹤流程快速開始.md"
        )

    print(f"資料集：{dataset_key}")
    print(f"data_dir（mapping + crop）：{data_dir}")
    print(f"輸出目錄（merge/filter）：{output_dir}")

    if not args.skip_step1:
        run_cmd(
            "Step 1/2：query 篩選 + BoT-SORT + merge",
            build_step1_cmd(args, dataset_key, data_dir, output_dir),
            cwd=BOTSORT_ROOT,
        )
    else:
        print("\n[SKIP] Step 1（merge）")

    if not args.skip_step2:
        run_cmd(
            "Step 2/2：combined intra-filter",
            build_step2_cmd(args, dataset_key, data_dir, output_dir),
            cwd=BOTSORT_ROOT,
        )
    else:
        print("\n[SKIP] Step 2（intra-filter）")

    print("\n" + "=" * 60)
    print("全部完成")
    print(f"  merged 結果：{output_dir}/*_merged.json")
    if not args.skip_step2:
        print(f"  filter 結果：{output_dir}/filter_results/")
        print(f"  總覽拼圖：{output_dir}/*_filtered_merged.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
