import os
import time
import subprocess
import threading
import signal
import shutil
from pathlib import Path
from datetime import datetime
from src.services.analysis_service import AnalysisService
from src.main import _save_results_to_postgres, SegmentAnalysisRequest
from src.database import SessionLocal

class RTSPStreamManager:
    _streams = {}  # video_id -> {process, thread, stop_event}

    @staticmethod
    def start_stream(rtsp_url: str, video_id: str, segment_duration: int = 10):
        if video_id in RTSPStreamManager._streams:
            return {"status": "already_running", "pid": RTSPStreamManager._streams[video_id]["process"].pid}

        # 1. 準備輸出目錄
        output_dir = Path("segment") / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. 啟動 FFmpeg 進行切片 (使用 segment muxer)
        # -strftime 1 允許使用 %Y%m%d 等時間格式命名檔案
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-reset_timestamps", "1",
            "-c", "copy",
            "-strftime", "1",
            f"{output_dir}/%Y%m%d_%H%M%S.mp4"
        ]

        process = subprocess.Popen(cmd)
        
        # 3. 啟動監控執行緒 (Watcher)
        stop_event = threading.Event()
        watcher_thread = threading.Thread(
            target=RTSPStreamManager._watcher_loop,
            args=(video_id, output_dir, stop_event, segment_duration)
        )
        watcher_thread.start()

        RTSPStreamManager._streams[video_id] = {
            "process": process,
            "thread": watcher_thread,
            "stop_event": stop_event,
            "start_time": time.time()
        }
        
        print(f"--- [RTSP] Stream started: {video_id} (PID: {process.pid}) ---")
        return {"status": "started", "pid": process.pid}

    @staticmethod
    def stop_stream(video_id: str):
        if video_id not in RTSPStreamManager._streams:
            return {"status": "not_found"}

        stream = RTSPStreamManager._streams[video_id]
        
        # 停止 FFmpeg
        stream["process"].terminate()
        try:
            stream["process"].wait(timeout=2)
        except subprocess.TimeoutExpired:
            stream["process"].kill()

        # 停止 Watcher
        stream["stop_event"].set()
        stream["thread"].join()

        del RTSPStreamManager._streams[video_id]
        print(f"--- [RTSP] Stream stopped: {video_id} ---")
        return {"status": "stopped"}

    @staticmethod
    def get_status():
        return {
            vid: {
                "pid": info["process"].pid,
                "uptime": round(time.time() - info["start_time"], 2)
            } for vid, info in RTSPStreamManager._streams.items()
        }

    @staticmethod
    def _watcher_loop(video_id, output_dir, stop_event, segment_duration):
        """監控資料夾，當發現新檔案完成寫入時觸發分析"""
        processed_files = set()
        
        # 載入已存在的檔案以免重複分析
        for f in output_dir.glob("*.mp4"):
            processed_files.add(f.name)

        print(f"--- [RTSP Watcher] Started for {video_id} ---")

        while not stop_event.is_set():
            # 獲取所有 mp4，按修改時間排序
            files = sorted(list(output_dir.glob("*.mp4")), key=lambda f: f.stat().st_mtime)
            
            # 策略：只處理「不是最新」的檔案 (假設最新的正在寫入)
            # 或者檢查檔案大小在幾秒內沒有變化
            
            if len(files) > 1:
                # 取出所有非最新的檔案 (candidates)
                candidates = files[:-1]
                
                for file_path in candidates:
                    if file_path.name not in processed_files:
                        print(f"--- [RTSP] New segment detected: {file_path.name} ---")
                        
                        # 等待一小段時間確保寫入完全 flush
                        time.sleep(1)
                        
                        # 觸發分析
                        try:
                            RTSPStreamManager._analyze_file(video_id, file_path, segment_duration)
                            processed_files.add(file_path.name)
                        except Exception as e:
                            print(f"--- [RTSP ERROR] Analysis failed for {file_path.name}: {e} ---")
            
            time.sleep(2)

    @staticmethod
    def _analyze_file(video_id, file_path, duration):
        # 1. 建立 Request 物件
        # 檔名範例: 20260118_120001.mp4
        stem = file_path.stem 
        
        req = SegmentAnalysisRequest(
            segment_path=str(file_path),
            segment_index=0, # RTSP 流不需要 index
            start_time=0.0,  # 這些在 RTSP 場景相對不重要，可填 0
            end_time=float(duration),
            model_type="qwen", # 預設模型
            qwen_model="qwen2.5-vl:7b",
            frames_per_segment=8,
            target_short=720,
            event_detection_prompt="請判斷畫面中是否有異常事件(火災、倒地、鬥毆等)。",
            summary_prompt="請簡短描述畫面內容。",
            yolo_labels="person,car", # 預設 YOLO
            yolo_every_sec=2.0,
            yolo_score_thr=0.25
        )

        # 2. 執行分析 (Blocking)
        print(f"--- [RTSP] Analyzing {file_path.name}... ---")
        result = AnalysisService.analyze_segment(req)
        
        # 補上 video_id 資訊，確保 DB 知道這是哪個 stream
        # 注意：我們使用 video_id 作為 summary 表的 video 欄位
        result["video"] = video_id 
        result["time_range"] = stem # 用檔名當時間標記

        # 3. 存入資料庫
        db = SessionLocal()
        try:
            # 這裡我們把單一結果包成 list 傳進去
            _save_results_to_postgres(db, [result], video_id)
            print(f"--- [RTSP] Saved {file_path.name} to DB ---")
        finally:
            db.close()