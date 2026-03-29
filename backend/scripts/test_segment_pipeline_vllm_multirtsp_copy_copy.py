#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
無限制 Worker 派發版：
當影片由生產者產出後，立刻開啟獨立執行緒發送 API 請求，不設上限。
旨在觀察 30 路 RTSP 同時衝擊下，vLLM Scheduler 如何執行 Continuous Batching。

總吞吐(段/秒)：成功送完 segment_pipeline_multipart 的「段」數 ÷ 從啟動第一路 producer
到最後一筆 API 完成的牆鐘秒數。調整負載請改 --streams（路數）為單一控制變因。
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("請安裝 requests: pip install requests")
    sys.exit(1)

# --- 環境變數載入 ---
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

_load_dotenv()

# 預設參數
DEFAULT_BASE = os.environ.get("BACKEND_URL", "http://140.117.176.42:3000/api")
DEFAULT_RTSP = os.environ.get("RTSP_URL", "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")
API_KEY = os.environ.get("API_KEY") or ""
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

@dataclass
class SegmentTask:
    stream_id: str
    chunk_id: int
    file_path: Path
    put_time: float
    capture_elapsed_sec: float

@dataclass
class SegmentRecord:
    stream_id: str
    chunk_id: int
    ok: bool
    capture_elapsed_sec: float
    queue_wait_sec: float
    upload_plus_backend_sec: float
    queue_to_done_sec: float
    total_segments: int
    success_segments: int
    api_process_time_sec: Optional[float]
    api_total_time_sec: Optional[float]
    error: str = ""

# --- FFmpeg 擷取：增加穩定性 ---
def capture_rtsp_local(rtsp_url: str, duration_sec: float) -> Path:
    import subprocess
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix="unlimited_vllm_", suffix=".mp4")
    os.close(fd)
    # 增加 stimeout (微秒) 以應對網路波動
    cmd = [
        "ffmpeg", "-y", "-rtsp_transport", "tcp", "-stimeout", "60000000",
        "-i", rtsp_url, "-t", str(duration_sec), "-c", "copy", "-movflags", "+faststart", tmp
    ]
    # 設定超時時間，給予足夠緩衝
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(duration_sec) + 60)
    if proc.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        err = (proc.stderr or "")[-500:]
        raise ValueError(f"FFmpeg capture failed: {err}")
    return Path(tmp)

# --- API 請求：支援高併發 ---
def upload_segment_to_api_vllm(base_url: str, file_path: Path, args, event_p, summary_p) -> Dict:
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_multipart"
    headers = {"X-API-Key": args.api_key or API_KEY}
    data = {
        "model_type": "vllm_qwen",
        "segment_duration": str(args.segment_duration),
        "qwen_model": args.model,
        "frames_per_segment": str(args.frames_per_segment),
        "target_short": str(args.target_short),
        "event_detection_prompt": event_p,
        "summary_prompt": summary_p,
        "save_json": "True"
    }
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "video/mp4")}
        # 設定較長的 timeout 以應對 vLLM 佇列排隊
        r = requests.post(url, data=data, files=files, headers=headers, timeout=900)
    r.raise_for_status()
    return r.json()

def main():
    parser = argparse.ArgumentParser(description="vLLM Unlimited Concurrency Test")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--rtsp", default=DEFAULT_RTSP)
    parser.add_argument("--streams", type=int, default=17, help="RTSP 路數")
    parser.add_argument("--chunks", type=int, default=6, help="每路總段數")
    parser.add_argument("--duration", type=float, default=10.0, help="擷取秒數")
    parser.add_argument("--segment-duration", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--frames-per-segment", type=int, default=5)
    parser.add_argument("--target-short", type=int, default=432)
    args = parser.parse_args()

    seg_q = queue.Queue()
    records = []
    records_lock = threading.Lock()
    sentinel = object()
    
    default_event = "請根據提供的影格輸出事件 JSON。"
    default_summary = "請產出 50-100 字繁體中文畫面摘要。"

    # 執行單次 API 請求的 Worker
    def fire_request_job(item: SegmentTask):
        t_req_start = time.time()
        try:
            out = upload_segment_to_api_vllm(args.base, item.file_path, args, default_event, default_summary)
            t_done = time.time()
            rec = SegmentRecord(
                item.stream_id, item.chunk_id, True, item.capture_elapsed_sec,
                round(t_req_start - item.put_time, 3), round(t_done - t_req_start, 3),
                round(t_done - item.put_time, 3), 
                int(out.get("total_segments", 0)), int(out.get("success_segments", 0)),
                out.get("process_time_sec"), out.get("total_time_sec")
            )
            print(f"[FIRE-OK] {item.stream_id}#{item.chunk_id} wait={rec.queue_wait_sec}s backend={rec.upload_plus_backend_sec}s", flush=True)
        except Exception as e:
            rec = SegmentRecord(item.stream_id, item.chunk_id, False, item.capture_elapsed_sec, 
                                round(t_req_start - item.put_time, 3), 0, 0, 0, 0, None, None, str(e))
            print(f"[FIRE-FAIL] {item.stream_id}#{item.chunk_id} err={e}", flush=True)
        finally:
            with records_lock:
                records.append(rec)
            if item.file_path.exists():
                item.file_path.unlink()
            seg_q.task_done()

    # 派發器：監控 Queue，一有影片就開新執行緒「秒發」
    def dispatcher():
        while True:
            item = seg_q.get()
            if item is sentinel:
                seg_q.task_done()
                break
            # 關鍵：不限制數量，為每個任務開一個 Thread
            threading.Thread(target=fire_request_job, args=(item,), daemon=True).start()

    threading.Thread(target=dispatcher, daemon=True).start()

    # 生產者：負責擷取影片片段
    def producer(sid):
        for cid in range(1, args.chunks + 1):
            t0 = time.time()
            try:
                path = capture_rtsp_local(args.rtsp, args.duration)
                seg_q.put(SegmentTask(sid, cid, path, time.time(), round(time.time() - t0, 3)))
                print(f"[PRODUCE] {sid} chunk={cid} captured", flush=True)
            except Exception as e:
                print(f"[PRODUCE-ERR] {sid} chunk={cid} {e}", flush=True)

    # 啟動各路生產者，並透過微調錯開啟動時間（總吞吐牆鐘：自此刻至 queue 排空）
    wall_start = time.time()
    p_threads = []
    for i in range(args.streams):
        pt = threading.Thread(target=producer, args=(f"RTSP_{i+1:02d}",))
        p_threads.append(pt)
        pt.start()
        time.sleep(0.2) # 每 0.2 秒開一路，防止 FFmpeg 瞬間擠爆系統

    for pt in p_threads:
        pt.join()
    
    seg_q.put(sentinel)
    seg_q.join()
    wall_end = time.time()
    wall_sec = max(wall_end - wall_start, 1e-9)

    print(f"\n--- 測試報告 ---")
    ok_list = [r for r in records if r.ok]
    print(f"總請求數: {len(records)}, 成功數: {len(ok_list)}")
    # 總吞吐(段/秒) = 成功完成的「段」數 / 測試總牆鐘時間（見檔案開頭 docstring）
    throughput_sps = len(ok_list) / wall_sec
    print(f"測試總牆鐘時間: {wall_sec:.2f}s（第一路 producer 啟動 → 最後一筆 API 完成）")
    print(f"總吞吐: {throughput_sps:.3f} 段/秒 (req/s) = 成功段數 {len(ok_list)} / {wall_sec:.2f}s")
    if ok_list:
        avg_wait = sum(r.queue_wait_sec for r in ok_list) / len(ok_list)
        avg_backend = sum(r.upload_plus_backend_sec for r in ok_list) / len(ok_list)
        print(f"平均 Queue 等待: {avg_wait:.3f}s (應趨近於 0)")
        print(f"平均 API 響應: {avg_backend:.2f}s")


if __name__ == "__main__":
    main()