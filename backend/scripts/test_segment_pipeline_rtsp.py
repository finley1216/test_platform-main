#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
process_duration_sec：傳輸時間 (Upload) + 後端分析時間 (time_sec)。
wait_in_queue_time: 0.0 : 數值為 0 代表影片一被切好，Worker 就立刻把它拿走去處理了。
time_sec：「後端實際 AI 運算耗時」（由 API 回傳）。
queue_to_db_sec：每筆從「放入 queue」到「處理完並寫入 DB」的總時間 = wait_in_queue_time + process_duration_sec。
avg_process_duration_sec : 平均每個片段從上傳到分析完畢的時間。
avg_wait_in_queue_time : 平均排隊時間。
avg_queue_to_db_sec : 平均每段從入隊到存 DB 的延遲（realtime 指標：若 < segment_sec 且穩定，代表能即時消化）。
KPI Realtime：在每個 10 秒影片放入佇列前，檢查上一段是否已處理並寫入 DB；有則在 terminal 輸出「Realtime: 是」。

Realtime 指標建議：
- 單段：avg_queue_to_db_sec < segment_sec（例如 10s）→ 單路可即時。
- 多路（如三路）：可將「同一時間窗內完成的各段 queue_to_db_sec」加總，若加總 < segment_sec，
  表示在一個 segment 週期內能處理完該時間窗內所有路的一段，即多路整體可即時；亦可看 realtime_ratio（Realtime: 是 比例）。
'''
# --- 標準庫：環境、CLI、時間、佇列、多執行緒、JSON、路徑、時間戳 ---
import os
import sys
import argparse
import time
import queue
import threading
import json
from pathlib import Path
from datetime import datetime

# 依賴 requests 發送 HTTP；若未安裝則提示並結束
try:
    import requests
except ImportError:
    print("請安裝 requests: pip install requests")
    sys.exit(1)


def _load_dotenv():
    """從 backend 目錄的 .env 載入環境變數（與後端共用 MY_API_KEY 等），僅在變數尚未設定時寫入 os.environ。"""
    # 腳本在 backend/scripts/，故 parent.parent = backend/
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        # 略過空行與註解
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            # 不覆蓋既有環境變數，讓命令列或系統設定優先
            if k and k not in os.environ:
                os.environ[k] = v


# 程式啟動時先載入 .env，後續 DEFAULT_BASE / API_KEY 等才會用到
_load_dotenv()

# 後端 base URL：可於 .env 設 BACKEND_URL 覆蓋（例如 nginx 代理位址）
DEFAULT_BASE = os.environ.get("BACKEND_URL", "http://140.117.176.42:3000/api")
# 預設 RTSP 來源；可於 .env 設 RTSP_URL 覆蓋
DEFAULT_RTSP = os.environ.get("RTSP_URL", "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")
# API Key：與後端驗證用，優先 API_KEY，其次 MY_API_KEY
API_KEY = os.environ.get("API_KEY") or os.environ.get("MY_API_KEY", "")


# 函式定義：輸入 RTSP URL、要錄幾秒、video_id（目前只用在錯誤/日誌），回傳型別是 Path（暫存 .mp4 的路徑）。
def capture_rtsp_local(rtsp_url: str, duration_sec: float, video_id: str) -> Path:
    '''從 RTSP 網址拉一段影片（例如 10 秒），存成一個本機暫存 .mp4，把路徑回傳給呼叫端， 呼叫端之後會把這個檔案上傳到後端 API'''
    # 使用 tempfile 建立暫存檔、subprocess 呼叫 FFmpeg。
    import tempfile
    import subprocess

    # 建立暫存檔，prefix/suffix 方便辨識；fd 用完即關閉，只保留路徑 tmp
    fd, tmp = tempfile.mkstemp(prefix="rtsp_capture_", suffix=".mp4")
    os.close(fd)

    # RTSP 連線逾時（微秒），60 秒
    stimeout_us = 60_000_000
    cmd = [
        "ffmpeg", "-y",                           # -y 覆蓋已存在檔案
        "-rtsp_transport", "tcp",                  # RTSP 用 TCP 較穩定
        "-stimeout", str(stimeout_us),             # 來源逾時
        "-i", rtsp_url,                            # 輸入 RTSP URL
        "-t", str(duration_sec),                   # 擷取時長（秒）
        "-c", "copy",                              # 不重新編碼，直接 copy
        "-movflags", "+faststart",                 # 將 moov 放檔頭，利於串流/上傳
        tmp,                                       # 輸出檔路徑
    ]

    # 用 subprocess.run 執行 FFmpeg，timeout=int(duration_sec)+60，避免擷取或卡住時腳本永遠不結束。
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(duration_sec) + 60)
    if proc.returncode != 0:
        try:
            os.remove(tmp)
        except OSError:
            pass
        # 取 stderr 最後 500 字元當錯誤訊息
        err = (proc.stderr or proc.stdout or "")[-500:]
        raise ValueError(f"FFmpeg RTSP capture failed: {err}")
    return Path(tmp)

# 單一影片擷取後上傳到後端 API
def run_one_local(base_url: str, rtsp_url: str, video_id: str, capture_duration: float, api_key: str, qwen_model: str = "qwen2.5vl:latest") -> dict:
    """本機 FFmpeg 擷取 RTSP → 上傳 POST /v1/segment_pipeline_multipart。後端只做切割+分析，不需連外。"""
    tmp_path = None
    try:
        print(f"  [本機] FFmpeg 擷取 RTSP {capture_duration}s...")
        tmp_path = capture_rtsp_local(rtsp_url, capture_duration, video_id)
        return upload_segment_to_api(base_url, video_id, tmp_path, api_key, qwen_model=qwen_model)
    finally:
        # 不論成功或失敗，刪除本機暫存檔，避免佔碟
        if tmp_path is not None:
            p = Path(tmp_path)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

def _run_multi_stream_duration(base_url: str, rtsp_url: str, num_streams: int, duration_minutes: float, segment_sec: float, num_workers: int, api_key: str, qwen_model: str = "qwen2.5vl:latest"):
    """
    N 路 RTSP 跑 M 分鐘：多個擷取 thread 寫入共用佇列，多個 worker thread 從佇列取件並 call API。
    流程：擷取 thread 每 segment_sec 秒產出一段 → 放入 seg_q → worker 取件 → POST segment_pipeline_multipart。
    結束後將每段處理時間寫入 segment_timing_YYYYMMDD_HHMMSS.json。
    """
    # 共用佇列：每筆為 (video_id, 本機暫存路徑, 檔名, put_timestamp) 供計算 wait_in_queue_time
    seg_q = queue.Queue()
    # 每路應產出的段數（1 分鐘 / 10 秒 = 6 段），避免因擷取逾時只產 5 段
    segments_per_stream = max(1, int(round((duration_minutes * 60) / segment_sec)))
    # 每路一個 ID：RTSP_01, RTSP_02, ...
    video_ids = [f"RTSP_{i+1:02d}" for i in range(num_streams)]
    # 佇列結束標記：worker 收到此值就結束迴圈
    SENTINEL = (None, None, None)
    # 每段 API 的開始/結束時間與成功與否，供最後寫 JSON
    timing_records = []
    timing_lock = threading.Lock()
    # KPI：每個 video_id 已「處理完並寫入 DB」的最大 segment 編號（API 回傳 success 即視為已寫入 DB）
    last_completed_segment_index = {}  # video_id -> int
    kpi_lock = threading.Lock()

    def capture_loop(stream_index: int):
        """單一 RTSP 路的擷取迴圈：依段數產出 exactly segments_per_stream 段（每段 segment_sec 秒），放入 seg_q。"""
        video_id = video_ids[stream_index]
        for count in range(segments_per_stream):
            try:
                path = capture_rtsp_local(rtsp_url, segment_sec, video_id)
                # KPI：放入佇列前，確認上一段是否已處理並寫入 DB → 有則代表 realtime
                with kpi_lock:
                    prev_done = (count - 1) <= last_completed_segment_index.get(video_id, -1) if count > 0 else True
                if count > 0:
                    if prev_done:
                        print(f"  [KPI] Realtime: 是（上一段 {video_id}_{count-1} 已處理並寫入 DB，再放入本段 {video_id}_{count}）")
                    else:
                        print(f"  [KPI] Realtime: 否（上一段 {video_id}_{count-1} 尚未完成，仍放入本段 {video_id}_{count}）")
                seg_q.put((video_id, path, f"{video_id}_{count}.mp4", time.time()))
            except Exception as e:
                print(f"  [擷取 {video_id}] 錯誤: {e}")

    def worker():
        """從佇列取 (video_id, path, filename, put_time)，上傳並 call API，紀錄 wait_in_queue_time、upload_time、time_sec。"""
        while True:
            item = seg_q.get()
            if item[0] is None:  # SENTINEL
                seg_q.task_done()
                return
            video_id, path, filename, put_time = item
            t_get = time.time()
            wait_in_queue_time = round(t_get - put_time, 3)
            success = False
            api_resp = None
            time_sec = 0.0
            t_upload_start = time.time()
            try:
                api_resp = upload_segment_to_api(base_url, video_id, path, api_key, filename, qwen_model=qwen_model)
                success = True
                # 從 API 回傳的 results 取得 total_api_time（或 time_sec）
                results = api_resp.get("results") or []
                if results:
                    time_sec = results[0].get("total_api_time") or results[0].get("time_sec") or 0.0
                print(f"  [API] {video_id} 已分析並寫入 DB")
                # KPI：成功即視為已寫入 DB，更新該 video_id 的「已完成」最大 segment 編號
                try:
                    seg_idx = int(Path(filename).stem.split("_")[-1])
                    with kpi_lock:
                        last_completed_segment_index[video_id] = max(
                            last_completed_segment_index.get(video_id, -1), seg_idx
                        )
                except (ValueError, IndexError):
                    pass
            except Exception as e:
                print(f"  [API] {video_id} 失敗: {e}")
            finally:
                upload_time = round(time.time() - t_upload_start, 3)
                t_end = time.time()
                queue_to_db_sec = round(t_end - put_time, 3)  # 從入隊到處理完存 DB 的總時間
                rec = {
                    "video_id": video_id,
                    "filename": filename,
                    "segment_sec": segment_sec,
                    "process_start_sec": round(t_get, 3),
                    "process_end_sec": round(t_end, 3),
                    "process_duration_sec": round(t_end - t_get, 3),
                    "queue_to_db_sec": queue_to_db_sec,
                    "success": success,
                    "wait_in_queue_time": wait_in_queue_time,
                    "upload_time": upload_time,
                    "time_sec": time_sec,
                }
                with timing_lock:
                    timing_records.append(rec)
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except OSError:
                        pass
            seg_q.task_done()

    print(f"--- {num_streams} 路 RTSP，每路 {segments_per_stream} 段（每段 {segment_sec}s），{num_workers} 個 API 並行 ---")
    t0 = time.time()

    # 啟動 num_streams 個擷取 thread，每路獨立跑 capture_loop
    capture_threads = []
    for i in range(num_streams):
        t = threading.Thread(target=capture_loop, args=(i,), daemon=True)
        t.start()
        capture_threads.append(t)

    # 啟動 num_workers 個 worker thread，共同消費 seg_q
    workers = [threading.Thread(target=worker, daemon=True) for _ in range(num_workers)]
    for w in workers:
        w.start()

    # 等所有擷取 thread 結束（時間到就不再 put）
    for t in capture_threads:
        t.join()

    # 對每個 worker 各放一個 SENTINEL，讓它們依序結束
    for _ in range(num_workers):
        seg_q.put(SENTINEL)
    for w in workers:
        w.join()
    elapsed = time.time() - t0

    # 計算平均佇列等待時間（證明佇列有無積壓）
    avg_wait_in_queue = round(sum(r.get("wait_in_queue_time", 0) for r in timing_records) / len(timing_records), 3) if timing_records else 0
    avg_queue_to_db = round(sum(r.get("queue_to_db_sec", 0) for r in timing_records) / len(timing_records), 3) if timing_records else 0

    out = {
        "run_config": {
            "num_streams": num_streams,
            "duration_minutes": duration_minutes,
            "segment_sec": segment_sec,
            "segments_per_stream": segments_per_stream,
            "num_workers": num_workers,
            "total_elapsed_sec": round(elapsed, 3),
        },
        "segments": timing_records,
        "summary": {
            "total_segments": len(timing_records),
            "success_count": sum(1 for r in timing_records if r["success"]),
            "fail_count": sum(1 for r in timing_records if not r["success"]),
            "avg_process_duration_sec": round(sum(r["process_duration_sec"] for r in timing_records) / len(timing_records), 3) if timing_records else 0,
            "avg_wait_in_queue_time": avg_wait_in_queue,
            "avg_queue_to_db_sec": avg_queue_to_db,
        },
    }
    out_path = Path(__file__).resolve().parent.parent / "segment_timing_{}.json".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，總耗時 {elapsed:.1f}s")
    print(f"  平均佇列等待時間: {avg_wait_in_queue:.2f}s（愈高表示佇列積壓愈嚴重）")
    print(f"  平均入隊→存 DB: {avg_queue_to_db:.2f}s（realtime 指標：若 < {segment_sec}s 表示可即時消化）")
    print(f"每段處理時間已寫入: {out_path}")

def upload_segment_to_api(
    base_url: str, video_id: str, file_path: Path, api_key: str, filename: str = None, qwen_model: str = "qwen2.5vl:latest", max_retries: int = 3
) -> dict:
    """上傳單一影片檔到 POST /v1/segment_pipeline_multipart。503/429 時自動重試。
    說明：503 通常來自「請求進到後端應用前」被擋（proxy/連線數上限），不一定是記憶體。
    """
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_multipart"
    headers = {"X-API-Key": api_key} if api_key else {}
    name = filename or f"{video_id}.mp4"
    data = {
        "model_type": "qwen",
        "segment_duration": "10.0",
        "overlap": "0.0",
        "qwen_model": qwen_model,
        "frames_per_segment": "4",
        "target_short": "480",
        "strict_segmentation": "False",
        "yolo_labels": "person,car",
        "yolo_every_sec": "2.0",
        "yolo_score_thr": "0.25",
        "event_detection_prompt": "請根據提供的影格輸出事件 JSON。",
        "summary_prompt": "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。",
        "save_json": "True",
    }
    last_exc = None
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                files = {"file": (name, f, "video/mp4")}
                r = requests.post(url, data=data, files=files, headers=headers, timeout=600)
            if r.status_code in (503, 429) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            last_exc = e
            if e.response is not None and e.response.status_code in (503, 429) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("upload_segment_to_api: max retries exceeded")

def main():
    wall_start = time.time()  # 從執行開始計時，供最後輸出「執行到 JSON 輸出」總耗時
    # ---------- 命令列參數 ----------
    parser = argparse.ArgumentParser(description="Test /v1/segment_pipeline_rtsp")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Backend base URL")
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="RTSP URL")
    parser.add_argument("--capture", type=float, default=float(os.environ.get("CAPTURE_SEC", "10")), help="每段擷取秒數（單次或每段長）")
    parser.add_argument("--api-key", default=API_KEY, help="X-API-Key（或設 API_KEY / MY_API_KEY）")
    parser.add_argument("--streams", type=int, default=1, help="多路長時間模式：幾路 RTSP，需搭配 --duration")
    parser.add_argument("--duration", type=float, default=1, help="多路長時間模式：跑幾分鐘，需搭配 --streams")
    parser.add_argument("--workers", type=int, default=1, help="多路長時間模式：同時幾個 API；0 則用 max(32, streams*8)")
    parser.add_argument("--model", default="qwen2.5vl:latest", help="Ollama VLM 模型名稱")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    api_key = args.api_key or API_KEY
    print(f"Backend: {base}")
    print(f"RTSP:   {args.rtsp[:60]}...")
    print(f"Capture: {args.capture}s, 本機擷取後上傳, model: {args.model}")
    if not api_key:
        print("Warning: 未設定 API Key（請在 backend/.env 設 MY_API_KEY 或傳 --api-key）")
    print()

    # ---------- 分支 1：多路長時間模式（--streams N --duration M）----------
    # 擷取端 N 個 thread 持續產出片段進佇列，消費端 num_workers 個 thread 平行 call API
    if args.streams and args.duration and args.duration > 0:
        w = args.workers if args.workers > 0 else max(32, args.streams * 8)  # 0 = 盡可能吃滿資源
        _run_multi_stream_duration(base, args.rtsp, args.streams, args.duration, args.capture, w, api_key, args.model)
        print(f"總耗時（執行到 JSON 輸出）: {time.time() - wall_start:.1f}s")
        return

    # ---------- 分支 2：單一請求（預設）----------
    # 本機擷取一段 RTSP → 上傳到後端 API → 印出 stem、total_segments 等
    print("--- 單一 RTSP 測試 ---")
    t0 = time.time()
    try:
        out = run_one_local(base, args.rtsp, "RTSP_TEST_01", args.capture, api_key, args.model)
        elapsed = time.time() - t0
        print(f"OK 耗時: {elapsed:.1f}s")
        print(f"  stem: {out.get('stem')}")
        print(f"  total_segments: {out.get('total_segments')}")
        print(f"  success_segments: {out.get('success_segments')}")
        print(f"  total_time_sec: {out.get('total_time_sec')}")
        if out.get("diagnostics"):
            print(f"  diagnostics: {out['diagnostics']}")
    except requests.exceptions.RequestException as e:
        print(f"FAIL: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  body: {e.response.text[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
