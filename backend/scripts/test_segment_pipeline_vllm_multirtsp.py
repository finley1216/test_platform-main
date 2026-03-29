#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真即時 Queue 版多路 RTSP 測試（固定走 /v1/segment_pipeline_multipart）。"""
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


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()

DEFAULT_BASE = os.environ.get("BACKEND_URL", "http://140.117.176.42:3000/api")
DEFAULT_RTSP = os.environ.get("RTSP_URL", "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")
API_KEY = os.environ.get("API_KEY") or os.environ.get("MY_API_KEY", "")
# DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"


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


def capture_rtsp_local(rtsp_url: str, duration_sec: float) -> Path:
    import subprocess
    import tempfile

    fd, tmp = tempfile.mkstemp(prefix="rtsp_vllm_multi_", suffix=".mp4")
    os.close(fd)
    stimeout_us = 60_000_000
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-stimeout", str(stimeout_us),
        "-i", rtsp_url,
        "-t", str(duration_sec),
        "-c", "copy",
        "-movflags", "+faststart",
        tmp,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(duration_sec) + 60)
    if proc.returncode != 0:
        try:
            os.remove(tmp)
        except OSError:
            pass
        err = (proc.stderr or proc.stdout or "")[-500:]
        raise ValueError(f"FFmpeg RTSP capture failed: {err}")
    return Path(tmp)


def upload_segment_to_api_vllm(
    base_url: str,
    file_path: Path,
    api_key: str,
    *,
    qwen_model: str,
    segment_duration: float,
    frames_per_segment: int,
    target_short: int,
    sampling_fps: Optional[float],
    event_detection_prompt: str,
    summary_prompt: str,
    save_json: bool,
    max_retries: int,
) -> Dict:
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_multipart"
    headers = {"X-API-Key": api_key} if api_key else {}
    data = {
        "model_type": "vllm_qwen",
        "segment_duration": str(segment_duration),
        "overlap": "0.0",
        "qwen_model": qwen_model,
        "frames_per_segment": str(frames_per_segment),
        "target_short": str(target_short),
        "strict_segmentation": "False",
        "yolo_labels": "person,car",
        "yolo_every_sec": "2.0",
        "yolo_score_thr": "0.25",
        "event_detection_prompt": event_detection_prompt,
        "summary_prompt": summary_prompt,
        "save_json": "True" if save_json else "False",
    }
    if sampling_fps is not None:
        data["sampling_fps"] = str(sampling_fps)

    last_exc = None
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "video/mp4")}
                r = requests.post(url, data=data, files=files, headers=headers, timeout=900)
            if r.status_code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            last_exc = e
            if e.response is not None and e.response.status_code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("upload_segment_to_api_vllm: max retries exceeded")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time queue test for /v1/segment_pipeline_multipart (vllm_qwen)"
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Backend base URL")
    parser.add_argument("--api-key", default=API_KEY, help="X-API-Key")
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="RTSP URL")
    parser.add_argument("--streams", type=int, default=30, help="RTSP 路數")
    parser.add_argument("--workers", type=int, default=30, help="上傳 worker 數")
    parser.add_argument("--chunks-per-stream", type=int, default=6, help="每路要產生幾個 chunk")
    parser.add_argument("--capture-per-chunk", type=float, default=10.0, help="每個 chunk 擷取秒數")
    parser.add_argument("--segment-duration", type=float, default=10.0, help="後端切段秒數（multipart 參數）")
    parser.add_argument("--frames-per-segment", type=int, default=5, help="每段取樣影格數")
    parser.add_argument("--target-short", type=int, default=432, help="影格短邊")
    parser.add_argument("--sampling-fps", type=float, default=None, help="固定抽幀 fps（可不給）")
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL, help="vLLM 模型名")
    parser.add_argument("--max-retries", type=int, default=3, help="API 重試次數")
    parser.add_argument("--queue-maxsize", type=int, default=0, help="Queue 長度上限（0=無上限）")
    parser.add_argument("--out", type=Path, default=None, help="輸出 JSON 報告檔")
    args = parser.parse_args()

    stream_ids = [f"RTSP_{i+1:02d}" for i in range(args.streams)]
    seg_q: queue.Queue = queue.Queue(maxsize=max(0, args.queue_maxsize))
    records: List[SegmentRecord] = []
    records_lock = threading.Lock()

    print("--- vLLM 即時 Queue 測試開始 ---")
    print(f"Backend: {args.base}")
    print(f"Model:   {args.model}")
    print(f"RTSP:    {args.rtsp[:80]}...")
    print(f"Streams: {args.streams}, Workers: {args.workers}")
    print(f"Chunks/Stream: {args.chunks_per_stream}, Capture/Chunk: {args.capture_per_chunk}s")
    print(f"Segment Duration (backend): {args.segment_duration}s")
    print()

    default_event = "請根據提供的影格輸出事件 JSON。"
    default_summary = "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。"
    sentinel = object()

    def producer(stream_id: str) -> None:
        for chunk_id in range(1, args.chunks_per_stream + 1):
            t_cap0 = time.time()
            try:
                p = capture_rtsp_local(args.rtsp, args.capture_per_chunk)
                cap_elapsed = round(time.time() - t_cap0, 3)
                seg_q.put(
                    SegmentTask(
                        stream_id=stream_id,
                        chunk_id=chunk_id,
                        file_path=p,
                        put_time=time.time(),
                        capture_elapsed_sec=cap_elapsed,
                    )
                )
                print(f"[PRODUCE] {stream_id} chunk={chunk_id} captured={cap_elapsed}s queue={seg_q.qsize()}", flush=True)
            except Exception as e:
                print(f"[PRODUCE-FAIL] {stream_id} chunk={chunk_id} err={e}", flush=True)

    def worker(worker_id: int) -> None:
        while True:
            item = seg_q.get()
            if item is sentinel:
                seg_q.task_done()
                break

            t_upload0 = time.time()
            try:
                out = upload_segment_to_api_vllm(
                    args.base.rstrip("/"),
                    item.file_path,
                    args.api_key or API_KEY,
                    qwen_model=args.model,
                    segment_duration=args.segment_duration,
                    frames_per_segment=args.frames_per_segment,
                    target_short=args.target_short,
                    sampling_fps=args.sampling_fps,
                    event_detection_prompt=default_event,
                    summary_prompt=default_summary,
                    save_json=False,
                    max_retries=args.max_retries,
                )
                t_done = time.time()
                rec = SegmentRecord(
                    stream_id=item.stream_id,
                    chunk_id=item.chunk_id,
                    ok=True,
                    capture_elapsed_sec=item.capture_elapsed_sec,
                    queue_wait_sec=round(t_upload0 - item.put_time, 3),
                    upload_plus_backend_sec=round(t_done - t_upload0, 3),
                    queue_to_done_sec=round(t_done - item.put_time, 3),
                    total_segments=int(out.get("total_segments") or 0),
                    success_segments=int(out.get("success_segments") or 0),
                    api_process_time_sec=out.get("process_time_sec"),
                    api_total_time_sec=out.get("total_time_sec"),
                )
                print(
                    f"[WORKER-{worker_id} OK] {item.stream_id}#{item.chunk_id} "
                    f"wait={rec.queue_wait_sec}s up+backend={rec.upload_plus_backend_sec}s "
                    f"segments={rec.success_segments}/{rec.total_segments}",
                    flush=True,
                )
            except Exception as e:
                t_done = time.time()
                rec = SegmentRecord(
                    stream_id=item.stream_id,
                    chunk_id=item.chunk_id,
                    ok=False,
                    capture_elapsed_sec=item.capture_elapsed_sec,
                    queue_wait_sec=round(t_upload0 - item.put_time, 3),
                    upload_plus_backend_sec=round(t_done - t_upload0, 3),
                    queue_to_done_sec=round(t_done - item.put_time, 3),
                    total_segments=0,
                    success_segments=0,
                    api_process_time_sec=None,
                    api_total_time_sec=None,
                    error=str(e),
                )
                print(f"[WORKER-{worker_id} FAIL] {item.stream_id}#{item.chunk_id} err={e}", flush=True)
            finally:
                with records_lock:
                    records.append(rec)
                try:
                    if item.file_path.exists():
                        item.file_path.unlink()
                except OSError:
                    pass
                seg_q.task_done()

    t_global_0 = time.time()
    worker_threads = [threading.Thread(target=worker, args=(i + 1,), daemon=True) for i in range(args.workers)]
    for wt in worker_threads:
        wt.start()

    producer_threads = [threading.Thread(target=producer, args=(sid,), daemon=True) for sid in stream_ids]
    for pt in producer_threads:
        pt.start()
    for pt in producer_threads:
        pt.join()

    for _ in range(args.workers):
        seg_q.put(sentinel)

    seg_q.join()
    for wt in worker_threads:
        wt.join()

    total_elapsed = round(time.time() - t_global_0, 3)
    ok_list = [x for x in records if x.ok]
    fail_list = [x for x in records if not x.ok]

    avg_capture = round(sum(x.capture_elapsed_sec for x in ok_list) / len(ok_list), 3) if ok_list else 0.0
    avg_queue_wait = round(sum(x.queue_wait_sec for x in ok_list) / len(ok_list), 3) if ok_list else 0.0
    avg_upload_backend = round(sum(x.upload_plus_backend_sec for x in ok_list) / len(ok_list), 3) if ok_list else 0.0
    avg_queue_to_done = round(sum(x.queue_to_done_sec for x in ok_list) / len(ok_list), 3) if ok_list else 0.0
    avg_api_process = round(
        sum(float(x.api_process_time_sec or 0) for x in ok_list) / len(ok_list), 3
    ) if ok_list else 0.0
    total_success_segments = sum(x.success_segments for x in ok_list)
    total_segments = sum(x.total_segments for x in ok_list)

    summary = {
        "config": {
            "base": args.base,
            "model": args.model,
            "rtsp": args.rtsp,
            "streams": args.streams,
            "workers": args.workers,
            "chunks_per_stream": args.chunks_per_stream,
            "capture_per_chunk_sec": args.capture_per_chunk,
            "segment_duration_sec": args.segment_duration,
            "frames_per_segment": args.frames_per_segment,
            "target_short": args.target_short,
            "sampling_fps": args.sampling_fps,
            "queue_maxsize": args.queue_maxsize,
        },
        "result": {
            "total_runs": len(records),
            "ok_runs": len(ok_list),
            "fail_runs": len(fail_list),
            "success_rate": round((len(ok_list) / len(records)) * 100, 2) if records else 0.0,
            "total_elapsed_sec": total_elapsed,
            "avg_capture_sec": avg_capture,
            "avg_queue_wait_sec": avg_queue_wait,
            "avg_upload_plus_backend_sec": avg_upload_backend,
            "avg_queue_to_done_sec": avg_queue_to_done,
            "avg_api_process_time_sec": avg_api_process,
            "total_success_segments": total_success_segments,
            "total_segments": total_segments,
        },
        "records": [asdict(r) for r in records],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=== Summary ===")
    print(f"total_runs: {summary['result']['total_runs']}")
    print(f"ok_runs: {summary['result']['ok_runs']}")
    print(f"fail_runs: {summary['result']['fail_runs']}")
    print(f"success_rate: {summary['result']['success_rate']}%")
    print(f"total_elapsed_sec: {summary['result']['total_elapsed_sec']}")
    print(f"avg_capture_sec: {summary['result']['avg_capture_sec']}")
    print(f"avg_queue_wait_sec: {summary['result']['avg_queue_wait_sec']}")
    print(f"avg_upload_plus_backend_sec: {summary['result']['avg_upload_plus_backend_sec']}")
    print(f"avg_queue_to_done_sec: {summary['result']['avg_queue_to_done_sec']}")
    print(f"avg_api_process_time_sec: {summary['result']['avg_api_process_time_sec']}")
    print(f"segments(success/total): {total_success_segments}/{total_segments}")

    out_path = args.out
    if out_path is None:
        out_path = Path(__file__).resolve().parent.parent / f"vllm_multirtsp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n報告已輸出: {out_path}")


if __name__ == "__main__":
    main()
