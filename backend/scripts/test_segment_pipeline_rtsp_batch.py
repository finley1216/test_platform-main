#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多路 RTSP + 批次 API 測試：每批 N 段影片一次送給後端 /v1/segment_pipeline_batch，
後端使用 qwen_hf（本機 Hugging Face Qwen2.5-VL）batch 推論，VLM 與 YOLO 同批處理同段數。

用法：
- 單批測試（擷取 N 段後送一次 batch）：
  python test_segment_pipeline_rtsp_batch.py --base http://localhost:3000/api --rtsp rtsp://... --batch-once

- 多路長時間（每 4 段湊成一批送 API，較省 GPU）：
  python test_segment_pipeline_rtsp_batch.py --base http://localhost:3000/api --streams 10 --duration 1 --batch-size 4
"""
import os
import sys
import argparse
import time
import queue
import threading
import json
from pathlib import Path
from datetime import datetime

from typing import Optional

try:
    import requests
except ImportError:
    print("請安裝 requests: pip install requests")
    sys.exit(1)


def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()

DEFAULT_BASE = os.environ.get("BACKEND_URL", "http://140.117.176.42:3000/api")
DEFAULT_RTSP = os.environ.get("RTSP_URL", "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")
API_KEY = os.environ.get("API_KEY") or os.environ.get("MY_API_KEY", "")
BATCH_SIZE = 4  # 每批幾段（4 段較省 GPU；10 段易 OOM）

# 接收三個參數：RTSP 網址、擷取時長（秒）、以及影片 ID。回傳值是一個 Path 物件（指向存好的檔案路徑）。
def capture_rtsp_local(rtsp_url: str, duration_sec: float, video_id: str) -> Path:

    # 用來產生唯一的暫存檔名，避免多執行緒時檔案名稱衝突。
    import tempfile

    # 用來執行 ffmpeg 指令
    import subprocess

    # 在系統暫存目錄中建立一個檔案。fd 是檔案描述符（低階控制用），tmp 是該檔案的完整路徑字串。檔案前綴為 rtsp_batch_，副檔名為 .mp4。
    fd, tmp = tempfile.mkstemp(prefix="rtsp_batch_", suffix=".mp4")

    # mkstemp 開啟檔案後會回傳一個開啟的控制權，但我們之後要交給 FFmpeg 寫入，所以這裡先關閉 fd，避免檔案被鎖定導致 FFmpeg 無法寫入。
    os.close(fd)
    stimeout_us = 60_000_000

    '''
    "ffmpeg", "-y": 執行 FFmpeg 並強制覆寫已存在的輸出檔。
    "-rtsp_transport", "tcp": 強制使用 TCP 協定傳輸
    "-stimeout", str(stimeout_us): 設定連線逾時。
    "-i", rtsp_url: 輸入端，即 RTSP 串流網址。
    "-t", str(duration_sec): 設定擷取的持續時間。
    "-c", "copy": 串流拷貝模式。直接複製編碼而不重轉碼，速度極快且不耗 CPU。
    "-movflags", "+faststart": 將 MP4 的 Meta data 移至檔案開頭，讓影片能更快被讀取播放。
    tmp: 指定輸出路徑為剛才建立的暫存檔路徑。
    '''
    cmd = [
        "ffmpeg", "-y", "-rtsp_transport", "tcp", "-stimeout", str(stimeout_us),
        "-i", rtsp_url, "-t", str(duration_sec), "-c", "copy", "-movflags", "+faststart", tmp,
    ]

    # 正式執行指令。capture_output=True 會擷取 FFmpeg 的日誌內容。如果 FFmpeg 卡死超過「影片長度 + 60秒」，則強制終止。
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(duration_sec) + 60)

    # 判斷 FFmpeg 的回傳碼。如果不等於 0，表示執行過程中發生錯誤（如網址無效、連線中斷等）。
    if proc.returncode != 0:
        try:
            # 如果失敗了，嘗試把剛才建立的空白暫存檔刪掉，保持系統乾淨。
            os.remove(tmp)
        except OSError:
            pass
        # 取 stderr 最後 500 字元當錯誤訊息
        err = (proc.stderr or proc.stdout or "")[-500:]
        raise ValueError(f"FFmpeg RTSP capture failed: {err}")

    # 若成功執行，將暫存檔路徑轉換為 pathlib.Path 物件並回傳。
    return Path(tmp)

# 將一批影片檔案（batch）連同相關的推論參數，透過 POST 請求發送到一個後端 API 伺服器進行處理
def upload_batch_to_api(
    base_url: str,
    batch: list,
    api_key: str,
    segment_sec: float = 10.0,
    qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_retries: int = 2,
    qwen_inference_batch_size: Optional[int] = None,
    yolo_batch_size: Optional[int] = None,
    yolo_every_sec: float = 2.0,
) -> dict:
   
    # 組合完整的 API 端點網址
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_batch"

    # 設定 API 金鑰
    headers = {"X-API-Key": api_key} if api_key else {}

    # 建立一個字典，存放要傳送給 API 的非檔案參數（Form Data），包括影片片段時長、模型名稱、YOLO 標籤（人、車）以及信心門檻等。
    data = {
        "segment_duration": str(segment_sec),
        "qwen_model": qwen_model,
        "frames_per_segment": "5",
        "target_short": "432",
        "yolo_labels": "person,car",
        "yolo_every_sec": str(yolo_every_sec),
        "yolo_score_thr": "0.25",
        "event_detection_prompt": "",
        "summary_prompt": "",
        "save_json": "False",}

    # 如果有特別指定 VLM 或 YOLO 的推論批次大小，則加入 data 字典。
    if qwen_inference_batch_size is not None:
        data["qwen_inference_batch_size"] = str(qwen_inference_batch_size)
    if yolo_batch_size is not None:
        data["yolo_batch_size"] = str(yolo_batch_size)

    # 初始化一個清單，準備存放待上傳的檔案物件。
    files_list = []
    try:

        # 遍歷傳入的 batch 清單，解構出影片 ID 和本地檔案路徑。
        for _i, (video_id, path, filename, _put_time) in enumerate(batch):
            name = filename or f"{video_id}.mp4"

            # 以二進位讀取模式 (rb) 開啟檔案，並按照 FastAPI 接受的多檔案格式 ("欄位名", (檔名, 檔案物件, MIME類型)) 加入清單。
            files_list.append(("files", (name, open(path, "rb"), "video/mp4")))

        # 進入重試迴圈，根據 max_retries 設定執行次數。
        for attempt in range(max_retries):
            try:

                # 第一次嘗試時，在終端機印出進度訊息。
                if attempt == 0:
                    print(f"  [API] POST {url}（{len(batch)} 個檔案）…", flush=True)

                # 用 requests 發送 POST 請求，包含表單資料與多個檔案。設定 timeout 為 600 秒（10 分鐘），給予後端足夠的時間處理大檔案。
                r = requests.post(url, data=data, files=files_list, headers=headers, timeout=600)

                # 第一次嘗試時，在終端機印出 HTTP 狀態碼。
                if attempt == 0:
                    print(f"  [API] HTTP {r.status_code}", flush=True)

                # 如果遇到 503 (服務暫不可用) 或 429 (請求過多)，再重試：
                if r.status_code in (503, 429) and attempt < max_retries - 1:

                    # 等待 2 的冪次倍數秒（第一次 2 秒，第二次 4 秒，第三次 8 秒，依此類推），避免過度頻繁重試。
                    time.sleep(2.0 * (attempt + 1))
                    continue

                # 如果請求成功，將回應內容轉換為 JSON 格式並回傳。
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ConnectionError as e:
                print(f"  [API] 連線失敗（無法連到後端）: {e}", flush=True)
                raise
            except requests.exceptions.Timeout as e:
                print(f"  [API] 逾時（後端未在 600s 內回應）: {e}", flush=True)
                raise

            # 捕捉其他 HTTP 錯誤，並嘗試印出前 500 個字元的錯誤訊息正文（Body），方便除錯。
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (503, 429) and attempt < max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                body = (e.response.text[:500] if e.response is not None else "") or ""
                print(f"  [API] HTTP {e.response.status_code if e.response else '?'}: {body}", flush=True)
                raise

    # 遍歷剛才開啟的所有檔案物件並將其關閉。若不執行這步，可能會導致系統「開啟檔案過多」的錯誤
    finally:
        for _name, tup in files_list:
            if len(tup) >= 2 and hasattr(tup[1], "close"):
                tup[1].close()

    # 如果所有嘗試都失敗，回傳一個空字典。
    return {}

# 多路 RTSP：擷取寫入佇列，worker 每湊滿 batch_size 段就送一次 /v1/segment_pipeline_batch。
def _run_batch_mode(
    base_url: str,
    rtsp_url: str,
    num_streams: int,
    duration_minutes: float,
    segment_sec: float,
    batch_size: int,
    api_key: str,
    qwen_model: str,
    qwen_inference_batch_size: Optional[int] = None,
    yolo_batch_size: Optional[int] = None,
    yolo_every_sec: float = 2.0,
):
    
    # 建立一個執行緒安全的佇列，用來傳放「已擷取的影片資訊」。
    seg_q = queue.Queue()

    # 計算每路 RTSP 需要擷取的片段數。
    segments_per_stream = max(1, int(round((duration_minutes * 60) / segment_sec)))

    # 計算總片段數。
    total_segments = num_streams * segments_per_stream

    # 建立一個清單，存放每路 RTSP 的 ID。
    video_ids = [f"RTSP_{i+1:02d}" for i in range(num_streams)]

    # 定義一個「哨兵值」，用來通知 Worker 執行緒可以停止了。
    SENTINEL = (None, None, None, None)

    # 建立一個清單，存放每段 API 的開始/結束時間與成功與否，供最後寫 JSON。
    timing_records = []

    # 建立一個執行緒安全的鎖，用來保護 timing_records 的存取。
    timing_lock = threading.Lock()

    # 建立一個計數器，用來記錄已擷取的片段數。
    capture_count = [0]

    # 定義一個擷取迴圈，用來擷取每路 RTSP 的影片片段。
    def capture_loop(stream_index: int):

        # 取得該路 RTSP 的 ID。
        video_id = video_ids[stream_index]

        # 從 RTSP 抓取影像，擷取成功計算成功次數，並放入佇列。
        for count in range(segments_per_stream):
            try:
                path = capture_rtsp_local(rtsp_url, segment_sec, video_id)
                seg_q.put((video_id, path, f"{video_id}_{count}.mp4", time.time()))
                with timing_lock:
                    capture_count[0] += 1
                    n = capture_count[0]
                print(f"  [擷取] 第 {n}/{total_segments} 段完成（{video_id}）", flush=True)
            except Exception as e:
                print(f"  [擷取 {video_id}] 錯誤: {e}", flush=True)

    def worker():
        batch = []
        while True:
            item = seg_q.get()

            # 如果收到哨兵值，表示佇列結束，可以結束這個 Worker。
            if item[0] is None:
                seg_q.task_done()

                # if batch: ...: 處理最後一批不足 batch_size 的殘餘檔案，上傳後結束迴圈。
                if batch:
                    t_upload = time.time()
                    print(f"  [上傳] 最後一批 {len(batch)} 段，送 API…", flush=True)
                    try:
                        api_resp = upload_batch_to_api(
                            base_url, batch, api_key, segment_sec, qwen_model,
                            qwen_inference_batch_size=qwen_inference_batch_size,
                            yolo_batch_size=yolo_batch_size,
                            yolo_every_sec=yolo_every_sec,
                        )
                        results = api_resp.get("results") or []
                        t_end = time.time()
                        for j, (vid, _path, fname, put_time) in enumerate(batch):
                            rec = {
                                "video_id": vid,
                                "filename": fname,
                                "segment_sec": segment_sec,
                                "queue_to_db_sec": round(t_end - put_time, 3),
                                "success": results[j].get("success", False) if j < len(results) else False,
                                "time_sec": results[j].get("total_api_time", 0) if j < len(results) else 0,
                            }
                            with timing_lock:
                                timing_records.append(rec)
                    except Exception as e:
                        print(f"  [API] Batch 失敗: {e}")
                        for (_vid, _p, fname, put_time) in batch:
                            with timing_lock:
                                timing_records.append({
                                    "video_id": _vid, "filename": fname, "segment_sec": segment_sec,
                                    "queue_to_db_sec": round(time.time() - put_time, 3), "success": False, "time_sec": 0,
                                })
                    for _, p, _, _ in batch:
                        if Path(p).exists():
                            try:
                                Path(p).unlink()
                            except OSError:
                                pass
                return

            # 將剛收到的片段加入 batch，並通知佇列已處理完畢。
            batch.append(item)
            seg_q.task_done()

            # 如果 batch 已經湊滿 batch_size，則上傳 API。
            if len(batch) >= batch_size:
                t_upload = time.time()
                print(f"  [上傳] 湊滿 {len(batch)} 段，送 API…（後端收到後才會載入 GPU；若後端在別台機器請在該台看 nvidia-smi）", flush=True)
                try:

                    # 呼叫 upload_batch_to_api 發送請求。
                    api_resp = upload_batch_to_api(
                        base_url, batch, api_key, segment_sec, qwen_model,
                        qwen_inference_batch_size=qwen_inference_batch_size,
                        yolo_batch_size=yolo_batch_size,
                        yolo_every_sec=yolo_every_sec,
                    )
                    results = api_resp.get("results") or []
                    t_end = time.time()
                    for j, (vid, _path, fname, put_time) in enumerate(batch):
                        rec = {
                            "video_id": vid,
                            "filename": fname,
                            "segment_sec": segment_sec,
                            "queue_to_db_sec": round(t_end - put_time, 3),
                            "success": results[j].get("success", False) if j < len(results) else False,
                            "time_sec": results[j].get("total_api_time", 0) if j < len(results) else 0,
                        }
                        with timing_lock:
                            timing_records.append(rec)
                    print(f"  [API] Batch 已分析 {len(batch)} 段並寫入 DB")
                except Exception as e:
                    print(f"  [API] Batch 失敗: {e}")
                    for (_vid, _p, fname, put_time) in batch:
                        with timing_lock:
                            timing_records.append({
                                "video_id": _vid, "filename": fname, "segment_sec": segment_sec,
                                "queue_to_db_sec": round(time.time() - put_time, 3), "success": False, "time_sec": 0,
                            })
                for _, p, _, _ in batch:

                    # 清理磁碟，上傳完成後立即刪除本地暫存檔，避免磁碟爆滿。
                    if Path(p).exists():
                        try:
                            Path(p).unlink()
                        except OSError:
                            pass

                # 清空批次，準備下一輪。
                batch = []

    print(f"--- {num_streams} 路 RTSP，每路 {segments_per_stream} 段（每段 {segment_sec}s），每 {batch_size} 段送一次 batch API ---")
    print("擷取中（10 條並行）… 後端在別台機器時，GPU 會在該台載入，請在該台看 nvidia-smi。", flush=True)

    # 開始擷取。
    t0 = time.time()

    # 開始擷取每路 RTSP 的影片片段。
    capture_threads = [threading.Thread(target=capture_loop, args=(i,), daemon=True) for i in range(num_streams)]
    for t in capture_threads:
        t.start()

    # 開始 worker，每湊滿 batch_size 段就送一次 /v1/segment_pipeline_batch。
    num_workers = max(1, (num_streams * segments_per_stream + batch_size - 1) // batch_size)
    workers = [threading.Thread(target=worker, daemon=True) for _ in range(num_workers)]

    # 啟動所有 worker。
    for w in workers:
        w.start()

    # 等待所有擷取 thread 結束（時間到就不再 put）。
    for t in capture_threads:
        t.join()

    # 對每個 worker 各放一個 SENTINEL，讓它們依序結束。
    for _ in range(num_workers):
        seg_q.put(SENTINEL)

    # 等待所有 worker 結束。
    for w in workers:
        w.join()

    # 計算總耗時。
    elapsed = time.time() - t0

    # 計算平均入隊→存 DB 時間。
    avg_queue_to_db = round(sum(r.get("queue_to_db_sec", 0) for r in timing_records) / len(timing_records), 3) if timing_records else 0

    # 建立輸出字典，包含執行設定、各段分析結果、以及摘要統計。
    out = {
        "run_config": {
            "num_streams": num_streams,
            "duration_minutes": duration_minutes,
            "segment_sec": segment_sec,
            "batch_size": batch_size,
            "total_elapsed_sec": round(elapsed, 3),
        },
        "segments": timing_records,
        "summary": {
            "total_segments": len(timing_records),
            "success_count": sum(1 for r in timing_records if r.get("success")),
            "avg_queue_to_db_sec": avg_queue_to_db,
        },
    }
    out_path = Path(__file__).resolve().parent.parent / f"segment_timing_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，總耗時 {elapsed:.1f}s")
    print(f"  平均入隊→存 DB: {avg_queue_to_db:.2f}s（realtime: 若 < {segment_sec}s 可即時消化）")
    print(f"結果: {out_path}")

# 單次測試：擷取 N 段（N=BATCH_SIZE）後一次送 batch API。
def run_batch_once(
    base_url: str,
    rtsp_url: str,
    segment_sec: float,
    api_key: str,
    qwen_model: str,
    qwen_inference_batch_size: Optional[int] = None,
    yolo_batch_size: Optional[int] = None,
    yolo_every_sec: float = 2.0,
):

    batch = []
    print(f"擷取 {BATCH_SIZE} 段（每段 {segment_sec}s）… 此時為本機 FFmpeg 從 RTSP 抓檔，GPU 要等「上傳後」後端收到請求才會載入。")
    for i in range(BATCH_SIZE):
        try:
            print(f"  擷取第 {i+1}/{BATCH_SIZE} 段…", end=" ", flush=True)
            t_seg = time.time()
            path = capture_rtsp_local(rtsp_url, segment_sec, f"BATCH_01")
            batch.append((f"BATCH_01", path, f"segment_{i:03d}.mp4", time.time()))
            print(f"OK ({time.time() - t_seg:.1f}s)", flush=True)
        except Exception as e:
            print(f"失敗: {e}", flush=True)
            break
    if len(batch) < 1:
        print("無有效片段")
        return
    print(f"上傳 {len(batch)} 段到後端（此時後端才會載入 GPU 模型）…")
    t0 = time.time()
    try:
        api_resp = upload_batch_to_api(
            base_url, batch, api_key, segment_sec, qwen_model,
            qwen_inference_batch_size=qwen_inference_batch_size,
            yolo_batch_size=yolo_batch_size,
            yolo_every_sec=yolo_every_sec,
        )
        elapsed = time.time() - t0
        print(f"OK 耗時: {elapsed:.1f}s")
        print(f"  total_segments: {api_resp.get('total_segments')}")
        print(f"  success_segments: {api_resp.get('success_segments')}")
        print(f"  total_time_sec: {api_resp.get('total_time_sec')}")
        if api_resp.get("diagnostics"):
            print(f"  diagnostics: {api_resp['diagnostics']}")
    finally:
        for _, p, _, _ in batch:
            if Path(p).exists():
                try:
                    Path(p).unlink()
                except OSError:
                    pass

def main():

    # python test_segment_pipeline_rtsp_batch.py --qwen-inference-batch-size 4 --yolo-batch-size 20
    parser = argparse.ArgumentParser(description="Test /v1/segment_pipeline_batch (qwen_hf batch)")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Backend base URL")
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="RTSP URL")
    parser.add_argument("--capture", type=float, default=10.0, help="每段擷取秒數")
    parser.add_argument("--api-key", default=API_KEY, help="X-API-Key")
    parser.add_argument("--streams", type=int, default=4, help="多路模式：幾路「並行」擷取（僅在不加 --batch-once 時使用）")
    parser.add_argument("--duration", type=float, default=1, help="多路模式：跑幾分鐘（僅在不加 --batch-once 時使用）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="每批送幾段（預設 4，較省 GPU）")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF 模型名稱")
    parser.add_argument("--batch-once", action="store_true", help="單次測試：本機依序擷取 N 段再送 API（N=--batch-size）")
    parser.add_argument("--qwen-inference-batch-size", type=int, default=None, help="VLM 一次處理幾段（與 batch-size 同即 10 段一 batch，預設=本批段數）")
    parser.add_argument("--yolo-batch-size", type=int, default=None, help="YOLO 一次處理幾幀（10 段×0.5 FPS=50 幀則填 50，預設由後端依段數與 yolo-every-sec 計算）")
    parser.add_argument("--yolo-every-sec", type=float, default=2.0, help="YOLO 每幾秒取一幀（2.0 = 0.5 FPS，10 秒段=5 幀）")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    api_key = args.api_key or API_KEY
    print(f"Backend: {base}")
    if "127.0.0.1" not in base and "localhost" not in base:
        print("  若後端與本機同一台，請改用 --base http://127.0.0.1:3000/api 避免防火牆擋外連")
    print(f"RTSP: {args.rtsp[:60]}...")
    print(f"Model: {args.model}, batch_size: {args.batch_size}")
    if args.qwen_inference_batch_size is not None or args.yolo_batch_size is not None:
        print(f"  VLM batch: {args.qwen_inference_batch_size}, YOLO 幀 batch: {args.yolo_batch_size}, yolo_every_sec: {args.yolo_every_sec}")
    if not api_key:
        print("Warning: 未設定 API Key")
    print()

    if args.batch_once:
        run_batch_once(
            base, args.rtsp, args.capture, api_key, args.model,
            qwen_inference_batch_size=args.qwen_inference_batch_size,
            yolo_batch_size=args.yolo_batch_size,
            yolo_every_sec=args.yolo_every_sec,
        )
        return

    # 多路模式：10 條 thread 並行擷取，約 10 秒就有第一批 10 段 → 送 API → GPU 開始載入
    if args.streams and args.duration > 0:
        _run_batch_mode(
            base, args.rtsp, args.streams, args.duration, args.capture, args.batch_size, api_key, args.model,
            qwen_inference_batch_size=args.qwen_inference_batch_size,
            yolo_batch_size=args.yolo_batch_size,
            yolo_every_sec=args.yolo_every_sec,
        )
        return

    print("請使用 --batch-once（單次依序 10 段，約 100s）或 --streams N --duration M（多路並行，約 10s 就有第一批 10 段送 API）")
    sys.exit(1)

if __name__ == "__main__":
    main()