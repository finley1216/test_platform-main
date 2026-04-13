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
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct-AWQ"
_ENCODER_PROBE_LOCK = threading.Lock()
_AVAILABLE_VIDEO_ENCODERS = None

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
def _get_ffmpeg_encoders() -> set:
    import subprocess

    global _AVAILABLE_VIDEO_ENCODERS
    if _AVAILABLE_VIDEO_ENCODERS is not None:
        return _AVAILABLE_VIDEO_ENCODERS

    with _ENCODER_PROBE_LOCK:
        if _AVAILABLE_VIDEO_ENCODERS is not None:
            return _AVAILABLE_VIDEO_ENCODERS
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            encoders = set()
            if proc.returncode == 0:
                for line in (proc.stdout or "").splitlines():
                    line = line.strip()
                    if not line or line.startswith("-"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        encoders.add(parts[1])
            _AVAILABLE_VIDEO_ENCODERS = encoders
        except Exception:
            _AVAILABLE_VIDEO_ENCODERS = set()
        return _AVAILABLE_VIDEO_ENCODERS


def capture_rtsp_local(rtsp_url: str, duration_sec: float) -> Path:
    import subprocess
    import tempfile

    fd, tmp = tempfile.mkstemp(prefix="unlimited_vllm_", suffix=".mp4")
    os.close(fd)
    base_cmd = [
        "ffmpeg",
        "-y",
        "-rtsp_transport",
        "tcp",
        "-stimeout",
        "60000000",
        "-i",
        rtsp_url,
        "-t",
        str(duration_sec),
    ]
    encoders = _get_ffmpeg_encoders()
    cmd_variants = []

    if "libx264" in encoders:
        cmd_variants.append(
            ("libx264", ["-vf", "fps=0.5", "-r", "0.5", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"])
        )
    if "libopenh264" in encoders:
        cmd_variants.append(("libopenh264", ["-vf", "fps=0.5", "-r", "0.5", "-c:v", "libopenh264"]))
    if "mpeg4" in encoders:
        cmd_variants.append(("mpeg4", ["-vf", "fps=0.5", "-r", "0.5", "-c:v", "mpeg4", "-q:v", "5"]))

    # 最後保底：不重編碼，避免因編碼器缺失造成整批失敗。
    cmd_variants.append(("copy", ["-c", "copy"]))

    errors = []
    for name, options in cmd_variants:
        cmd = base_cmd + options + ["-movflags", "+faststart", tmp]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(duration_sec) + 60)
        if proc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            return Path(tmp)
        err_tail = (proc.stderr or "")[-500:]
        errors.append(f"{name}: {err_tail}")
        if os.path.exists(tmp):
            os.remove(tmp)

    raise ValueError("FFmpeg capture failed (all variants): " + " | ".join(errors[-3:]))


# --- API 請求：支援高併發 ---
def upload_segment_to_api_vllm(base_url: str, file_path: Path, args, event_p, summary_p) -> Dict:
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_multipart_batch"
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
    parser.add_argument("--streams", type=int, default=15, help="RTSP 路數")
    parser.add_argument("--chunks", type=int, default=6, help="每路總段數")
    parser.add_argument("--duration", type=float, default=10.0, help="擷取秒數")
    parser.add_argument("--segment-duration", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--frames-per-segment", type=int, default=5)
    parser.add_argument("--target-short", type=int, default=432)
    args = parser.parse_args()

    # 每輪收集各路錄好的檔案
    round_files: Dict[int, List[Path]] = {}
    round_counts: Dict[int, int] = {}
    round_lock = threading.Lock()

    records = []
    records_lock = threading.Lock()
    batch_threads = []
    batch_threads_lock = threading.Lock()
    batch_timeline: Dict[int, Dict] = {}
    batch_timeline_lock = threading.Lock()

    default_event = "請根據提供的影格輸出事件 JSON。"
    default_summary = "請產出 50-100 字繁體中文畫面摘要。"

    def send_batch(cid: int, files: List[Path]):
        t_req_start = time.time()
        with batch_timeline_lock:
            slot = batch_timeline.setdefault(cid, {})
            slot["start"] = t_req_start
            slot["files"] = len(files)
        url = f"{args.base.rstrip('/')}/v1/segment_pipeline_multipart_batch"
        headers = {"X-API-Key": args.api_key or API_KEY}
        data = {
            "model_type": "vllm_qwen",
            "segment_duration": str(args.segment_duration),
            "qwen_model": args.model,
            "frames_per_segment": str(args.frames_per_segment),
            "target_short": str(args.target_short),
            "event_detection_prompt": default_event,
            "summary_prompt": default_summary,
            "save_json": "True",
        }
        file_handles = []
        try:
            multipart_files = []
            for fp in files:
                fh = open(fp, "rb")
                file_handles.append(fh)
                multipart_files.append(("files", (fp.name, fh, "video/mp4")))

            r = requests.post(url, data=data, files=multipart_files, headers=headers, timeout=900)
            r.raise_for_status()
            out = r.json()
            t_done = time.time()
            elapsed = round(t_done - t_req_start, 2)
            with batch_timeline_lock:
                slot = batch_timeline.setdefault(cid, {})
                slot["done"] = t_done
                slot["elapsed"] = elapsed
                slot["ok"] = True
            print(
                f"[BATCH-OK] chunk={cid} files={len(files)} "
                f"backend={elapsed}s "
                f"success={out.get('success_segments')}/{out.get('total_segments')}",
                flush=True,
            )
            with records_lock:
                records.append({
                    "chunk_id": cid,
                    "ok": True,
                    "elapsed": elapsed,
                    "total_segments": out.get("total_segments", 0),
                    "success_segments": out.get("success_segments", 0),
                    "process_time_sec": out.get("process_time_sec"),
                    "total_time_sec": out.get("total_time_sec"),
                })
        except Exception as e:
            t_done = time.time()
            elapsed = round(t_done - t_req_start, 2)
            with batch_timeline_lock:
                slot = batch_timeline.setdefault(cid, {})
                slot["done"] = t_done
                slot["elapsed"] = elapsed
                slot["ok"] = False
            print(f"[BATCH-FAIL] chunk={cid} err={e}", flush=True)
            with records_lock:
                records.append({
                    "chunk_id": cid,
                    "ok": False,
                    "elapsed": elapsed,
                    "error": str(e),
                })
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass
            for fp in files:
                try:
                    if fp.exists():
                        fp.unlink()
                except Exception:
                    pass

    def producer(sid: str):
        for cid in range(1, args.chunks + 1):
            t0 = time.time()
            try:
                path = capture_rtsp_local(args.rtsp, args.duration)
                print(f"[PRODUCE] {sid} chunk={cid} captured elapsed={round(time.time()-t0,2)}s", flush=True)

                trigger = False
                files_to_send = []
                with round_lock:
                    if cid not in round_files:
                        round_files[cid] = []
                        round_counts[cid] = 0
                    round_files[cid].append(path)
                    round_counts[cid] += 1
                    if round_counts[cid] >= args.streams:
                        trigger = True
                        files_to_send = list(round_files[cid])

                if trigger:
                    if cid > 1:
                        with batch_timeline_lock:
                            prev = batch_timeline.get(cid - 1, {})
                            prev_done = bool(prev.get("done"))
                            prev_elapsed = prev.get("elapsed")
                        status = "PASS" if prev_done else "WAITING"
                        elapsed_txt = f"{prev_elapsed:.2f}s" if isinstance(prev_elapsed, (int, float)) else "N/A"
                        print(
                            f"[BATCH-CHECK] chunk={cid} 觸發時，上一批 chunk={cid-1} 狀態={status} elapsed={elapsed_txt}",
                            flush=True,
                        )
                    print(f"[BATCH-TRIGGER] chunk={cid} 已收齊 {len(files_to_send)} 路，送出 batch", flush=True)
                    t = threading.Thread(target=send_batch, args=(cid, files_to_send), daemon=True)
                    with batch_threads_lock:
                        batch_threads.append(t)
                    t.start()
            except Exception as e:
                print(f"[PRODUCE-ERR] {sid} chunk={cid} {e}", flush=True)
                trigger = False
                files_to_send = []
                with round_lock:
                    if cid not in round_counts:
                        round_counts[cid] = 0
                    round_counts[cid] += 1
                    if round_counts[cid] >= args.streams:
                        trigger = True
                        files_to_send = list(round_files.get(cid, []))
                if trigger and files_to_send:
                    t = threading.Thread(target=send_batch, args=(cid, files_to_send), daemon=True)
                    with batch_threads_lock:
                        batch_threads.append(t)
                    t.start()

    wall_start = time.time()
    p_threads = []
    for i in range(args.streams):
        pt = threading.Thread(target=producer, args=(f"RTSP_{i+1:02d}",))
        p_threads.append(pt)
        pt.start()
        time.sleep(0.2)

    for pt in p_threads:
        pt.join()

    # 等所有 batch 請求都完成
    with batch_threads_lock:
        threads_to_wait = list(batch_threads)
    for t in threads_to_wait:
        t.join()

    wall_end = time.time()
    wall_sec = max(wall_end - wall_start, 1e-9)

    ok_list = [r for r in records if r.get("ok")]
    fail_list = [r for r in records if not r.get("ok")]
    total_segs = args.streams * args.chunks

    print(f"\n--- 測試報告 ---")
    print(f"RTSP 路數：{args.streams}")
    print(f"總段數：{total_segs} ({args.streams} 路 x {args.chunks} 組 x 30 段）")
    ok_segs = sum(r.get("success_segments", 0) for r in ok_list)
    print(f"10s 內處理完：{ok_segs} / {total_segs} ({int(ok_segs/total_segs*100) if total_segs else 0}%)")
    print(f"Batch 請求：{len(ok_list)} 成功, {len(fail_list)} 失敗（共 {args.chunks} 輪）")
    print(f"測試總牆鐘: {wall_sec:.2f}s")
    if ok_list:
        avg_backend = sum(r["elapsed"] for r in ok_list) / len(ok_list)
        print(f"平均 batch API 響應: {avg_backend:.2f}s")
    print("\n批次銜接檢查（是否在下一批觸發前完成）：")
    all_ready = True
    with batch_timeline_lock:
        timeline = dict(batch_timeline)
    for cid in range(1, args.chunks + 1):
        cur = timeline.get(cid, {})
        start = cur.get("start")
        done = cur.get("done")
        if cid == args.chunks:
            status = "N/A（最後一批）"
        else:
            next_start = timeline.get(cid + 1, {}).get("start")
            if done and next_start:
                in_time = done <= next_start
                status = "PASS" if in_time else "LATE"
                if not in_time:
                    all_ready = False
            else:
                status = "UNKNOWN"
                all_ready = False
        elapsed_txt = f"{cur.get('elapsed', 0):.2f}s" if isinstance(cur.get("elapsed"), (int, float)) else "N/A"
        print(f"- chunk={cid}: backend={elapsed_txt}, status={status}")
    print(f"批次銜接總結：{'PASS' if all_ready else '有延遲或資料不足'}")


if __name__ == "__main__":
    main()