#!/usr/bin/env python3
"""從 Google Drive 下載追蹤用影片，整理成 backend API 可讀的路徑。

預設輸出：../video/{category}/K8-XX.avi
（與 VideoService.prepare_segments 的 video_id={category}/K8-XX 一致）

範例：
  python3 scripts/download_tracking_videos.py --dataset 人員追蹤_20260507
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VIDEO_ROOT = REPO_ROOT.parent / "video"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".flv"}
CAMERA_RE = re.compile(r"^(K8-\d+)\.", re.IGNORECASE)

# 公開測試資料集（Google Drive 資料夾）
DATASETS: Dict[str, Dict[str, str]] = {
    "人員追蹤_20260507": {
        "folder_id": "1HKUcJzfewkqHCJba4WH5ShR50iHEHYHu",
        "url": "https://drive.google.com/drive/folders/1HKUcJzfewkqHCJba4WH5ShR50iHEHYHu",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gdown 下載追蹤影片至 ../video/{category}/")
    p.add_argument(
        "--dataset",
        default="人員追蹤_20260507",
        help=f"內建資料集：{', '.join(DATASETS)}",
    )
    p.add_argument("--folder-url", default=None, help="覆寫 Google Drive 資料夾 URL")
    p.add_argument("--folder-id", default=None, help="覆寫 Google Drive folder id")
    p.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="輸出根目錄（預設 ../video/{category}/）",
    )
    p.add_argument(
        "--videos-only",
        action="store_true",
        default=True,
        help="只保留 K8-XX 影片（略過 jpg/txt 等，預設開啟）",
    )
    p.add_argument(
        "--keep-all",
        action="store_true",
        help="保留資料夾內所有下載檔案（含時間軸 txt、截圖 jpg）",
    )
    p.add_argument("--dry-run", action="store_true", help="只顯示將執行的動作")
    return p.parse_args()


def resolve_folder_url(args: argparse.Namespace) -> tuple[str, str]:
    category = args.dataset.strip()
    if args.folder_url:
        return category, args.folder_url.strip()
    if args.folder_id:
        fid = args.folder_id.strip()
        return category, f"https://drive.google.com/drive/folders/{fid}"
    spec = DATASETS.get(category)
    if not spec:
        known = ", ".join(DATASETS)
        raise SystemExit(f"未知資料集 {category!r}，請用 --folder-url 或已知：{known}")
    return category, spec["url"]


def download_folder(url: str, tmp_dir: Path) -> None:
    try:
        import gdown
    except ImportError as e:
        raise SystemExit("請安裝 gdown: pip install gdown") from e

    print(f"[download] {url}", flush=True)
    print(f"[download] 暫存目錄 {tmp_dir}", flush=True)
    gdown.download_folder(url, output=str(tmp_dir), quiet=False, use_cookies=False)


def collect_video_files(src_dir: Path, *, videos_only: bool) -> List[Path]:
    files = sorted(src_dir.iterdir()) if src_dir.is_dir() else []
    out: List[Path] = []
    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() not in VIDEO_EXTS:
            if videos_only:
                continue
        if videos_only and not CAMERA_RE.match(f.name):
            continue
        out.append(f)
    return out


def install_videos(
    src_files: List[Path],
    dest_dir: Path,
    *,
    dry_run: bool,
) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    installed: List[Path] = []
    for src in src_files:
        dst = dest_dir / src.name
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            print(f"[skip] {dst.name} 已存在且大小相同", flush=True)
            installed.append(dst)
            continue
        print(f"[install] {src.name} -> {dst}", flush=True)
        if not dry_run:
            shutil.copy2(src, dst)
        installed.append(dst)
    return installed


def main() -> None:
    args = parse_args()
    category, folder_url = resolve_folder_url(args)
    dest_dir = (args.dest or (VIDEO_ROOT / category)).resolve()
    videos_only = not args.keep_all

    if args.dry_run:
        print(f"[dry-run] category={category} dest={dest_dir} videos_only={videos_only}")

    with tempfile.TemporaryDirectory(prefix="tracking_videos_") as tmp:
        tmp_dir = Path(tmp)
        if not args.dry_run:
            download_folder(folder_url, tmp_dir)
        else:
            print(f"[dry-run] 將下載至 {tmp_dir}")

        if args.dry_run:
            print(
                "\n[Next] 啟動 backend 後執行 segment API：\n"
                f"  python3 backend/scripts/rerun_segment_cameras.py \\\n"
                f"    --category {category} \\\n"
                f"    --cameras K8-01 K8-05 K8-07 K8-08 K8-09 K8-20 K8-21 K8-22 K8-23"
            )
            return

        src_files = collect_video_files(tmp_dir, videos_only=videos_only)
        if not src_files:
            raise SystemExit(
                f"下載完成但未找到影片（videos_only={videos_only}）。"
                f"請檢查 Drive 資料夾內容：{folder_url}"
            )

        installed = install_videos(src_files, dest_dir, dry_run=False)
        cameras = sorted({CAMERA_RE.match(p.name).group(1) for p in installed if CAMERA_RE.match(p.name)})

        print(
            f"\n[Done] {len(installed)} 支影片 -> {dest_dir}\n"
            f"  鏡頭：{' '.join(cameras)}",
            flush=True,
        )
        print(
            "\n[Next] 啟動 backend，再跑 segment pipeline：\n"
            f"  python3 backend/scripts/rerun_segment_cameras.py \\\n"
            f"    --category {category} \\\n"
            f"    --cameras {' '.join(cameras)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
