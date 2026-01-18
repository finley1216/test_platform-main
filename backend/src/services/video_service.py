# -*- coding: utf-8 -*-
import os
import requests
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import HTTPException
from src.config import config
from src.utils.video_utils import _probe_duration_seconds, _split_one_video

class VideoService:
    @staticmethod
    def download_to_temp(url: str) -> str:
        """下載 URL 影片到暫存檔"""
        suffix = Path(url).suffix or ".mp4"
        fd, tmp = tempfile.mkstemp(prefix="download_", suffix=suffix)
        os.close(fd)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return tmp

    @staticmethod
    def prepare_segments(
        video_path: Optional[str],
        video_id: Optional[str],
        target_filename: str,
        segment_duration: float,
        overlap: float,
        target_short: int,
        strict_segmentation: bool
    ):
        """準備影片片段，回傳 (seg_dir, seg_files, stem, total_duration)"""
        if video_id and video_id.strip():
            video_id_clean = video_id.strip()
            if "/" in video_id_clean:
                category, video_name = video_id_clean.split("/", 1)
                stem = f"{category}_{video_name}"
                seg_dir = Path("segment") / stem
                
                if seg_dir.exists() and list(seg_dir.glob("segment_*.mp4")):
                    seg_files = sorted(seg_dir.glob("segment_*.mp4"))
                    total_duration = len(seg_files) * segment_duration # Approximation
                else:
                    video_lib_path = config.VIDEO_LIB_DIR / category / f"{video_name}.mp4"
                    if not video_lib_path.exists():
                        for ext in ['.avi', '.mov', '.mkv', '.flv']:
                            p = config.VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                            if p.exists(): video_lib_path = p; break
                    
                    if not video_lib_path.exists():
                        raise HTTPException(status_code=404, detail=f"Video {video_id_clean} not found")
                    
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    temp_video = seg_dir / video_lib_path.name
                    shutil.copy2(video_lib_path, temp_video)
                    seg_files = _split_one_video(
                        str(temp_video), seg_dir, segment_duration, overlap,
                        resolution=target_short if strict_segmentation else None
                    )
                    total_duration = _probe_duration_seconds(str(temp_video))
            else:
                stem = video_id_clean
                seg_dir = Path("segment") / stem
                seg_files = sorted(seg_dir.glob("segment_*.mp4"))
                if not seg_files:
                    raise HTTPException(status_code=404, detail=f"No segments for {video_id_clean}")
                total_duration = len(seg_files) * segment_duration
        else:
            stem = Path(target_filename).stem
            seg_dir = Path("segment") / stem
            seg_dir.mkdir(parents=True, exist_ok=True)
            total_duration = _probe_duration_seconds(video_path)
            seg_files = _split_one_video(
                video_path, seg_dir, segment_duration, overlap,
                resolution=target_short if strict_segmentation else None
            )
            
        return seg_dir, seg_files, stem, total_duration
