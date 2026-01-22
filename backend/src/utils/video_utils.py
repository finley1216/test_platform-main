# -*- coding: utf-8 -*-
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image

def _fmt_hms(seconds: float) -> str:
    """將秒數轉為 HH:MM:SS 格式"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _probe_duration_seconds(video_path: str) -> float:
    """使用 ffprobe 獲取影片長度"""
    if not video_path or not Path(video_path).exists():
        print(f"--- [VideoUtils] 錯誤：影片檔案不存在: {video_path} ---")
        return 0.0
        
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        if not out:
            print(f"--- [VideoUtils] 警告：ffprobe 回傳空值，嘗試使用 OpenCV 獲取長度 ---")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return count / fps if fps > 0 else 0.0
        return float(out)
    except Exception as e:
        print(f"--- [VideoUtils] 獲取影片長度失敗: {e} ---")
        return 0.0

def _split_one_video(video_path: str, out_dir: Path, segment_duration: float, overlap: float, prefix: str = "segment", resolution: Optional[int] = None, strict_mode: bool = False) -> List[Path]:
    """將影片切割成多個片段"""
    duration = _probe_duration_seconds(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    outs = []
    step = segment_duration - overlap
    if step <= 0: step = segment_duration
    
    # 獲取原始影片資訊
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    i = 0
    start = 0.0
    while start < duration:
        out_file = out_dir / f"{prefix}_{i:03d}.mp4"
        
        # 建立 ffmpeg 指令
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(segment_duration),
            "-i", video_path
        ]
        
        if resolution:
            ffmpeg_cmd += ["-vf", f"scale='if(gt(iw,ih),{resolution},-1)':'if(gt(iw,ih),-1,{resolution})'"]
        
        ffmpeg_cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy", str(out_file)
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        outs.append(out_file)
        
        i += 1
        start += step
        if start >= duration - 0.1: break
        
    return outs

def _sample_frames_evenly_to_pil(video_path: str, max_frames: int=8, sampling_fps: Optional[float] = None) -> List[Image.Image]:
    """從影片中抓取截圖"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0: raise RuntimeError("invalid or zero-frame video")
        
        duration_sec = total_frames / fps
        frames = []
        
        if sampling_fps and sampling_fps > 0:
            interval_sec = 1.0 / sampling_fps
            t = 0.0
            while t < duration_sec:
                frame_number = int(round(t * fps))
                if frame_number >= total_frames: frame_number = total_frames - 1
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, bgr = cap.read()
                if ok:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                t += interval_sec
        else:
            n = min(max_frames, total_frames)
            idxs = np.linspace(0, total_frames-1, num=n, dtype=np.int64)
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, bgr = cap.read()
                if ok:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
        
        if not frames: raise RuntimeError("no frames sampled")
        return frames
    finally:
        cap.release()
