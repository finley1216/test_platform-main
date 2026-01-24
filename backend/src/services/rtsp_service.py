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
    _streams = {}  # video_id -> {process, thread, stop_event, status, rtsp_url}

    @staticmethod
    def start_stream(rtsp_url: str, video_id: str, segment_duration: int = 10):
        # 啟動前先嘗試清理可能存在的舊進程
        if video_id in RTSPStreamManager._streams:
            print(f"--- [RTSP] Stream {video_id} already exists, stopping first... ---")
            RTSPStreamManager.stop_stream(video_id)
        
        # 驗證 RTSP URL 是否可訪問（快速檢查，但允許繼續嘗試）
        print(f"--- [RTSP] Validating RTSP URL: {rtsp_url} ---")
        rtsp_valid = False
        try:
            test_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-t", "0.1",  # 只讀取 0.1 秒
                "-f", "null", "-"
            ]
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                rtsp_valid = True
                print(f"--- [RTSP] ✓ RTSP URL validation passed ---")
            else:
                error_msg = result.stderr or result.stdout
                print(f"--- [RTSP] ⚠️ RTSP URL validation failed (will still attempt): {error_msg[:150]} ---")
                # 不拋出異常，允許繼續嘗試（可能是暫時的連接問題）
        except subprocess.TimeoutExpired:
            print(f"--- [RTSP] ⚠️ RTSP URL validation timeout (will still attempt) ---")
        except Exception as e:
            print(f"--- [RTSP] ⚠️ RTSP URL validation error (will still attempt): {str(e)[:150]} ---")

        # 1. 準備輸出目錄
        output_dir = Path("segment") / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. 啟動兩個 FFmpeg 進程：
        #    - 進程1：推流到 MediaMTX (供前端 HLS 觀看)
        #    - 進程2：切片到本地文件 (供分析)
        # 使用 NVENC (h264_nvenc) 進行 GPU 加速編碼，並加入低延遲參數
        
        # MediaMTX 推流地址（Docker 內部地址）
        # 使用不同的路徑避免與輸入源衝突
        # 前端可以通過 http://<hostname>:8888/stream_<video_id>/ 訪問 HLS
        mediamtx_rtsp_url = f"rtsp://rtsp-server:8554/stream_{video_id}"
        
        # 檢測輸入流的編碼格式
        input_codec = None
        input_width = None
        input_height = None
        try:
            probe_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height",
                "-of", "default=noprint_wrappers=1",
                rtsp_url
            ]
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('codec_name='):
                        input_codec = line.split('=')[1]
                    elif line.startswith('width='):
                        input_width = int(line.split('=')[1])
                    elif line.startswith('height='):
                        input_height = int(line.split('=')[1])
                print(f"--- [RTSP] Input stream: codec={input_codec}, resolution={input_width}x{input_height} ---")
        except Exception as e:
            print(f"--- [RTSP] Could not probe input stream: {e}, will attempt to use copy mode ---")
        
        # 如果輸入已經是 H.264，直接使用 copy（最快且最穩定）
        # 否則需要重新編碼
        use_copy = (input_codec == "h264")
        use_gpu = False  # 初始化變數，避免作用域錯誤
        
        # 根據編碼模式決定編碼器名稱（提前定義，避免作用域錯誤）
        if use_copy:
            encoder_name = "copy (no re-encoding)"
        else:
            # 暫時設置為 CPU，稍後會根據實際檢測結果更新
            encoder_name = "libx264 (CPU)"
        
        if use_copy:
            # 直接複製視頻流，不需要重新編碼
            encoding_params = ["-c:v", "copy"]
            print("--- [RTSP] Input is H.264, using copy mode (no re-encoding) ---")
        else:
            # 需要重新編碼，檢測 GPU 編碼器是否可用
            try:
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if "h264_nvenc" in result.stdout:
                    use_gpu = True
                    print("--- [RTSP] GPU encoder (h264_nvenc) detected, using GPU acceleration ---")
                else:
                    print("--- [RTSP] GPU encoder not available, falling back to CPU (libx264) ---")
            except Exception as e:
                print(f"--- [RTSP] Could not detect encoder, using CPU fallback: {e} ---")
            
            # 根據可用性和輸入解析度選擇編碼參數
            if use_gpu and input_width and input_height:
                # GPU 編碼需要指定解析度
                encoder_name = "h264_nvenc (GPU)"
                encoding_params = [
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",  # p4 是 NVENC 的預設，比 ultrafast 更穩定
                    "-g", "30",
                    "-profile:v", "baseline",
                    "-level", "3.1",
                    "-s", f"{input_width}x{input_height}"  # 指定解析度
                ]
            elif use_gpu:
                # GPU 可用但無法獲取解析度，使用 CPU 編碼
                encoder_name = "libx264 (CPU)"  # GPU 不可用，回退到 CPU
                encoding_params = [
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-g", "30",
                    "-profile:v", "baseline",
                    "-level", "3.1"
                ]
            else:
                # CPU 編碼
                encoder_name = "libx264 (CPU)"
                encoding_params = [
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-g", "30",
                    "-profile:v", "baseline",
                    "-level", "3.1"
                ]
        
        # 進程1：推流到 MediaMTX (供前端 HLS 觀看)
        # 注意：如果輸入已經是 H.264，可以直接 copy，否則需要重新編碼
        # 添加重試和超時參數以處理連接問題
        push_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-rtsp_transport", "tcp",  # 輸入端使用 TCP
            "-i", rtsp_url,
            "-fflags", "+genpts",
            *encoding_params,
            "-bsf:v", "h264_mp4toannexb",  # RTSP 需要 Annex-B 格式
            "-f", "rtsp",
            "-rtsp_transport", "tcp",  # 輸出端也使用 TCP
            mediamtx_rtsp_url
        ]
        
        print(f"--- [RTSP] Push command: {' '.join(push_cmd)} ---")
        print(f"--- [RTSP] Input RTSP URL: {rtsp_url} ---")
        print(f"--- [RTSP] Output RTSP URL: {mediamtx_rtsp_url} ---")
        
        # 進程2：切片到本地文件 (供分析)
        segment_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-rtsp_transport", "tcp",  # 使用 TCP 傳輸
            "-i", rtsp_url,
            "-fflags", "+genpts",
            *encoding_params,
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-reset_timestamps", "1",
            "-strftime", "1",
            f"{output_dir}/%Y%m%d_%H%M%S.mp4"
        ]
        
        print(f"--- [RTSP] Segment command: {' '.join(segment_cmd)} ---")
        
        # 啟動兩個 FFmpeg 進程
        push_process = subprocess.Popen(push_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        segment_process = subprocess.Popen(segment_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # 使用 segment_process 作為主要進程（用於狀態檢查）
        process = segment_process
        
        # 3. 先創建 stream 資訊結構（在啟動日誌線程前）
        stop_event = threading.Event()
        RTSPStreamManager._streams[video_id] = {
            "process": process,  # segment 進程（主要）
            "push_process": push_process,  # 推流進程
            "thread": None,  # 稍後設置
            "stop_event": stop_event,
            "start_time": time.time(),
            "status": "running",  # running, stopped, error, ended
            "rtsp_url": rtsp_url,
            "error_message": None,
            "encoding": encoder_name  # 存儲編碼器信息
        }
        
        # 啟動執行緒來讀取並打印 FFmpeg 的輸出，並檢測錯誤
        def log_ffmpeg_output(proc, vid, proc_type="Segment"):
            error_lines = []
            try:
                for line in proc.stdout:
                    line_lower = line.lower()
                    # 收集所有錯誤和警告訊息
                    if "error" in line_lower or "warning" in line_lower:
                        print(f"--- [FFmpeg {vid} - {proc_type}] {line.strip()} ---")
                        error_lines.append(line.strip())
                    # 檢測串流結束或找不到串流的錯誤
                    if any(keyword in line_lower for keyword in [
                        "stream not found", "connection refused", "connection timed out",
                        "end of file", "server returned 404", "server returned 400", 
                        "unable to open", "error opening input", "option not found",
                        "error splitting", "unrecognized option"
                    ]):
                        print(f"--- [FFmpeg {vid} - {proc_type}] ⚠️ 串流錯誤檢測: {line.strip()} ---")
                        stream_info = RTSPStreamManager._streams.get(vid)
                        if stream_info:
                            stream_info["status"] = "error"
                            # 提供更友好的錯誤訊息
                            if "404" in line:
                                stream_info["error_message"] = f"RTSP 源不可用 (404): {rtsp_url}。請確認 RTSP 源正在運行。"
                            elif "400" in line:
                                stream_info["error_message"] = f"RTSP 源連接失敗 (400): {rtsp_url}。請檢查 RTSP URL 是否正確。"
                            elif "option not found" in line_lower or "unrecognized option" in line_lower:
                                stream_info["error_message"] = f"FFmpeg 參數錯誤: {line.strip()}"
                            else:
                                stream_info["error_message"] = line.strip()
            except Exception as e:
                print(f"--- [FFmpeg {vid} - {proc_type}] Log thread error: {e} ---")
            finally:
                # 如果進程結束時有錯誤但沒有設置錯誤訊息，使用收集到的錯誤
                if proc.poll() is not None and proc.returncode != 0:
                    stream_info = RTSPStreamManager._streams.get(vid)
                    if stream_info:
                        if not stream_info.get("error_message") or stream_info.get("error_message") == "None":
                            if error_lines:
                                stream_info["error_message"] = "; ".join(error_lines[-3:])  # 取最後3條錯誤訊息
                            else:
                                stream_info["error_message"] = f"FFmpeg 進程異常退出 (返回碼: {proc.returncode})"
        
        # 為兩個進程分別啟動日誌線程
        threading.Thread(target=log_ffmpeg_output, args=(push_process, video_id, "Push"), daemon=True).start()
        threading.Thread(target=log_ffmpeg_output, args=(segment_process, video_id, "Segment"), daemon=True).start()

        # 4. 啟動監控執行緒 (Watcher)
        watcher_thread = threading.Thread(
            target=RTSPStreamManager._watcher_loop,
            args=(video_id, output_dir, stop_event, segment_duration, process)
        )
        watcher_thread.start()
        
        # 更新 stream 資訊中的 thread 引用
        RTSPStreamManager._streams[video_id]["thread"] = watcher_thread
        
        # encoder_name 已經在上面定義了，這裡直接使用
        print(f"--- [RTSP] Stream started: {video_id} (Segment PID: {segment_process.pid}, Push PID: {push_process.pid}, Encoder: {encoder_name}) ---")
        print(f"--- [RTSP] Pushing to MediaMTX: {mediamtx_rtsp_url} ---")
        print(f"--- [RTSP] HLS will be available at: http://<hostname>:8888/stream_{video_id}/ ---")
        
        # 等待一小段時間，檢查推流進程是否成功啟動
        time.sleep(2)
        if push_process.poll() is not None:
            print(f"--- [RTSP] ⚠️ Push process exited early with code {push_process.returncode} ---")
            return {"status": "error", "message": f"Push process failed: return code {push_process.returncode}"}
        
        return {"status": "started", "pid": segment_process.pid, "push_pid": push_process.pid}

    @staticmethod
    def stop_stream(video_id: str):
        if video_id not in RTSPStreamManager._streams:
            return {"status": "not_found"}

        stream = RTSPStreamManager._streams[video_id]
        
        # 1. 立即標記停止，讓 Watcher 退出
        stream["stop_event"].set()
        
        # 2. 停止兩個 FFmpeg 進程
        processes_to_stop = [
            ("Segment", stream["process"]),
            ("Push", stream.get("push_process"))
        ]
        
        for proc_name, proc in processes_to_stop:
            if proc is None:
                continue
            print(f"--- [RTSP] Stopping {proc_name} FFmpeg process (PID: {proc.pid})... ---")
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                print(f"--- [RTSP] {proc_name} FFmpeg terminate timeout, killing... ---")
                proc.kill()
            except Exception as e:
                print(f"--- [RTSP] Error stopping {proc_name} FFmpeg: {e} ---")

        # 3. 快速清理
        del RTSPStreamManager._streams[video_id]
        print(f"--- [RTSP] Stream stopped: {video_id} ---")
        
        # 4. 非同步清理殭屍進程 (不阻塞 API 回傳)
        def cleanup():
            try:
                while True:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0: break
            except: pass
        
        threading.Thread(target=cleanup).start()
            
        return {"status": "stopped"}

    @staticmethod
    def get_status():
        """獲取所有串流的狀態，包含運行狀態和錯誤信息"""
        result = {}
        for vid, info in RTSPStreamManager._streams.items():
            # 檢查進程是否還在運行
            process_status = "running"
            if info["process"].poll() is not None:
                # 進程已結束
                if info["process"].returncode == 0:
                    process_status = "ended"
                else:
                    process_status = "error"
                # 更新內部狀態
                if info["status"] == "running":
                    info["status"] = process_status
            
            stream_data = {
                "pid": info["process"].pid,
                "uptime": round(time.time() - info["start_time"], 2),
                "status": info.get("status", process_status),
                "rtsp_url": info.get("rtsp_url", "unknown"),
                "encoding": info.get("encoding", "unknown")  # 添加編碼器信息
            }
            
            # 如果有錯誤訊息，加入結果
            if info.get("error_message"):
                stream_data["error_message"] = info["error_message"]
            
            # 如果進程已結束，加入明確的結束信號
            if process_status in ["ended", "error"]:
                stream_data["ended"] = True
                if process_status == "ended":
                    stream_data["message"] = "Stream Ended"
                else:
                    stream_data["message"] = f"Stream Error (code: {info['process'].returncode})"
            
            result[vid] = stream_data
        
        return result

    @staticmethod
    def _watcher_loop(video_id, output_dir, stop_event, segment_duration, process):
        """監控資料夾，當發現新檔案完成寫入時觸發分析"""
        processed_files = set()
        
        # 載入已存在的檔案以免重複分析
        for f in output_dir.glob("*.mp4"):
            processed_files.add(f.name)

        print(f"--- [RTSP Watcher] Started for {video_id} ---")

        while not stop_event.is_set():
            # 檢查兩個 FFmpeg 進程是否還在
            stream_info = RTSPStreamManager._streams.get(video_id)
            if not stream_info:
                break
            
            segment_dead = process.poll() is not None
            push_dead = stream_info.get("push_process") and stream_info["push_process"].poll() is not None
            
            if segment_dead or push_dead:
                # 至少有一個進程結束了
                if segment_dead:
                    return_code = process.returncode
                    if return_code == 0:
                        # 正常結束（可能是串流源結束）
                        print(f"--- [RTSP Watcher] Segment process ended normally for {video_id} (Return code: 0) ---")
                        if stream_info:
                            stream_info["status"] = "ended"
                            stream_info["error_message"] = "Stream Ended - Source stream has finished"
                    else:
                        # 異常結束（return code != 0 表示錯誤）
                        print(f"--- [RTSP Watcher] Segment FFmpeg process died for {video_id} (Return code: {return_code}) ---")
                        if stream_info:
                            stream_info["status"] = "error"
                            # 獲取更詳細的錯誤訊息
                            error_msg = stream_info.get("error_message")
                            if not error_msg or error_msg == "None":
                                error_msg = f"FFmpeg 進程異常退出 (返回碼: {return_code})"
                            else:
                                error_msg = f"Segment FFmpeg 錯誤 (code: {return_code}): {error_msg}"
                            stream_info["error_message"] = error_msg
                
                if push_dead:
                    push_return_code = stream_info["push_process"].returncode
                    print(f"--- [RTSP Watcher] Push FFmpeg process died for {video_id} (Return code: {push_return_code}) ---")
                    # Push 進程結束也應該標記為錯誤（如果 segment 還沒結束）
                    if stream_info and stream_info.get("status") == "running":
                        stream_info["status"] = "error"
                        push_error = stream_info.get("error_message", "")
                        stream_info["error_message"] = f"Push FFmpeg 錯誤 (code: {push_return_code})" + (f": {push_error}" if push_error else "")
                
                break

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
        # 1. 獲取系統預設的 Event Prompt
        from src.main import EVENT_DETECTION_PROMPT
        
        # 2. 建立 Request 物件 (只進行事件偵測，跳過摘要以加速)
        stem = file_path.stem 
        
        req = SegmentAnalysisRequest(
            segment_path=str(file_path),
            segment_index=0, # RTSP 流不需要 index
            start_time=0.0,  # 這些在 RTSP 場景相對不重要，可填 0
            end_time=float(duration),
            model_type="qwen", # 預設模型
            qwen_model="qwen2.5vl:latest",
            frames_per_segment=8,
            target_short=720,
            event_detection_prompt=EVENT_DETECTION_PROMPT,
            summary_prompt="",  # [加速] 空字串代表跳過摘要生成
            yolo_labels="person,car", # 預設 YOLO
            yolo_every_sec=2.0,
            yolo_score_thr=0.25
        )

        # 2. 執行分析 (Blocking)
        print(f"--- [RTSP] ⚡ 極速模式分析中: {file_path.name}... ---")
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
            print(f"--- [RTSP] ✓ 分析完成並存檔: {file_path.name} ---")
        finally:
            db.close()
