#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main
chmod +x scripts/restart_backend_worker_on_vram_high.sh   # 若尚未可執行
./scripts/restart_backend_worker_on_vram_high.sh
'''




"""
生產者擷取 RTSP → Queue → ThreadPoolExecutor（--max-concurrent-http 個 worker）送 API。
每個 worker thread 使用 thread-local requests.Session 重用連線；避免「每請求新建 Session」
導致大量短命 TCP（易 ConnectTimeout / ephemeral port 耗盡）。
連線重試帶 --http-retry-jitter 抖動；預設 --max-inflight-http 10、--connect-timeout 120、--http-gate-jitter-sec。
可用 --fixture-mp4 略過 FFmpeg，專測後端連線與吞吐。

吞吐指標（見下方 run_one_round 回傳與報表）：
- 平均段/秒：成功段數 ÷ 總牆鐘（含起頭錯開與尾段排空，易低估「全速」）。
- 穩態段/秒：僅統計 [wall_start+warmup, wall_end-cooldown] 內完成的段 ÷ 該區間秒數（較接近持續負載下的吞吐）。
- 峰值：任意滑動時間窗內「完成」段數最多者（10 秒窗 / 1 秒窗），較能反映 vLLM 瞬間 batch 能力。
- **即時（截止）**：每一路徑獨立計算——該路徑上每次「下一段入隊前，上一段 API 須完成」；30 路即時 = **每一路**的 chunk→chunk+1 皆達成（取最差一路比例 + 缺段檢查）。見 rtsp_* 欄位。
- **緩判（可選）**：`--relaxed-api-sum-budget-sec` 若 >0，則每路徑「各段 API 秒數加總」≤ 門檻時視為緩判通過（與嚴格截止分開，可接受單對延遲）。

調整負載請改 --streams 或 --ladder（路數為主要控制變因）。
自動搜尋：--auto-find-streams 以二分搜尋找最大可行路數（可搭配 --fixture-mp4 縮短時間）。
"""
import argparse
import json
import math
import os
import queue
import random
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import ConnectTimeout
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
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


_load_dotenv()

DEFAULT_BASE = os.environ.get("BACKEND_URL", "http://140.117.176.42:3000/api")
DEFAULT_RTSP = os.environ.get("RTSP_URL", "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")
API_KEY = os.environ.get("API_KEY") or ""
# DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct-AWQ"

# 當錄製好一段影片後，會將所有相關資訊打包成這個物件，丟入佇列（Queue）
@dataclass
class SegmentTask:
    stream_id: str               # 串流的唯一識別碼（例如：RTSP_01、RTSP_02）
    chunk_id: int                # 該路串流的影片片段序號（例如：第 1 段、第 2 段）。
    file_path: Path              # 影片檔案的路徑。
    put_time: float              # 該片段被放入佇列的時間戳。
    capture_elapsed_sec: float   # 這段影片實際花在 FFmpeg 擷取（錄製）上的秒數。

# Task 是做之前的紀錄，而 Record 是做完之後的結果。
@dataclass
class SegmentRecord:
    stream_id: str
    chunk_id: int
    ok: bool                                    # 這段影片是否成功被 API 處理完成。
    capture_elapsed_sec: float                  
    queue_wait_sec: float                       # 影片錄製完成後，在 Queue 裡面「排隊等待 Worker 抓取」的時間。
    upload_plus_backend_sec: float              # 從發送 POST 請求開始，到收到後端 Response 的總時長（含網路傳輸）。
    queue_to_done_sec: float                    # 從錄製完（進入 Queue）到存入 DB 的總時長
    total_segments: int                         # 這段影片總共有多少段。
    success_segments: int                       # 這段影片成功被 API 處理完成的段數。
    api_process_time_sec: Optional[float]       # 僅包含模型 Forward 推論秒數。不含圖片解碼、不含 Prompt 組合、不含存入資料庫。
    api_total_time_sec: Optional[float]         # 這代表了 後端程式碼的邏輯效率。如果 api_total_time 遠大於 api_process_time，代表你的程式碼在處理非 AI 的雜事。
    error: str = ""                             # 這段影片被 API 處理失敗的原因。
    put_unix: float = 0.0                       # 進入隊列的絕對時間戳
    done_unix: float = 0.0                      # 處理完成的絕對時間戳，用來比較「Segment 1 的 done_unix」是否小於「Segment 2 的 put_unix」。


def _fmt_opt_sec(v: Optional[float]) -> str:
    """Optional 秒數欄位轉成可讀字串（僅供列印，資料仍取自 SegmentRecord）。"""
    return f"{v}s" if v is not None else "N/A"


def should_emit_segment_detail(rec: SegmentRecord, print_slow_only: bool) -> bool:
    """失敗一律列印；成功時 print_slow_only 為 True 則僅 queue_to_done_sec > 10 時列印。"""
    if not rec.ok:
        return True
    if not print_slow_only:
        return True
    return rec.queue_to_done_sec > 10.0


def emit_segment_record_detail(rec: SegmentRecord) -> None:
    """成功/失敗共用：欄位皆來自 SegmentRecord。"""
    print(
        "\n".join(
            [
                f"[SLOW-DETECTED] Stream: {rec.stream_id}#{rec.chunk_id} | OK: {rec.ok}",
                f"- 錄製耗時 (capture_elapsed): {rec.capture_elapsed_sec}s",
                f"- 隊列排隊 (queue_wait): {rec.queue_wait_sec}s",
                f"- API響應 (upload_plus_backend): {rec.upload_plus_backend_sec}s",
                f"- 總生命週期 (queue_to_done): {rec.queue_to_done_sec}s",
                f"- 片段統計: {rec.success_segments}/{rec.total_segments} 成功",
                f"- 模型推論耗時 (api_process_time): {_fmt_opt_sec(rec.api_process_time_sec)}",
                f"- 後端邏輯總計 (api_total_time): {_fmt_opt_sec(rec.api_total_time_sec)}",
                f"- 錯誤訊息: {rec.error}",
            ]
        ),
        flush=True,
    )


# 輸入：rtsp_url（攝影機網址）、duration_sec（要錄幾秒，通常是 10 秒）。
# 輸出：Path（錄好的影片檔案路徑）。
def capture_rtsp_local(rtsp_url: str, duration_sec: float) -> Path:
    import subprocess
    import tempfile

    # mkstemp: 在系統暫存目錄建立一個唯一的檔案。
    # prefix: 檔案開頭叫 unlimited_vllm_，副檔名是 .mp4。
    fd, tmp = tempfile.mkstemp(prefix="unlimited_vllm_", suffix=".mp4")
    os.close(fd)
    cmd = [
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
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        tmp,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(duration_sec) + 60)
    if proc.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        err = (proc.stderr or "")[-500:]
        raise ValueError(f"FFmpeg capture failed: {err}")
    return Path(tmp)


def _build_pooled_session(pool_max: int) -> requests.Session:
    """每個 worker thread 一個 Session；pool_block 避免瞬間建立過多連線。"""
    s = requests.Session()
    # 每 thread 通常同時只有一個 multipart；pool 小一點 + block 可降低對遠端 SYN 風暴
    pc = max(1, min(pool_max, 4))
    adapter = HTTPAdapter(
        pool_connections=pc,
        pool_maxsize=pc,
        pool_block=True,
        max_retries=0,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def upload_segment_to_api_vllm(
    base_url: str,
    file_path: Path,
    args: Any,
    event_p: str,
    summary_p: str,
    timeout: tuple,
    session: Optional[requests.Session] = None,
) -> Dict:
    """單次 POST；重試邏輯由呼叫端處理（需重開檔案指標）。"""
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
        "save_json": "True",
    }
    request_debug = {
        "method": "POST",
        "url": url,
        "headers": headers,
        "data": data,
        "files": {"file": {"filename": file_path.name, "content_type": "video/mp4"}},
        "timeout": {"connect": timeout[0], "read": timeout[1]} if isinstance(timeout, tuple) and len(timeout) == 2 else timeout,
    }
    print("[API-REQUEST]\n" + json.dumps(request_debug, ensure_ascii=False, indent=2), flush=True)
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "video/mp4")}
        post = session.post if session is not None else requests.post
        r = post(url, data=data, files=files, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _reset_thread_local_session(tls: threading.local) -> None:
    """連線層失敗時丟棄 Session，下次會建新 TCP，避免卡死在壞掉的 pool。"""
    if hasattr(tls, "session"):
        try:
            tls.session.close()
        except Exception:
            pass
        del tls.session


def upload_segment_with_connect_retries(
    base_url: str,
    file_path: Path,
    args: Any,
    event_p: str,
    summary_p: str,
    timeout: tuple,
    tls: threading.local,
    get_session: Any,
) -> Dict:
    """
    對 ConnectTimeout / ConnectionError 做指數退避重試；每次重試前重置該 thread 的 Session。
    （multipart 無法安全用 urllib3 自動重送同一個 body，故採手動重開檔案。）
    """
    retries = max(0, int(getattr(args, "http_connect_retries", 5)))
    backoff = float(getattr(args, "http_retry_backoff_sec", 0.75))
    jitter = float(getattr(args, "http_retry_jitter", 0.5))
    jitter = max(0.0, min(jitter, 1.0))
    for attempt in range(retries + 1):
        try:
            return upload_segment_to_api_vllm(
                base_url,
                file_path,
                args,
                event_p,
                summary_p,
                timeout,
                session=get_session(),
            )
        except (ConnectTimeout, RequestsConnectionError, OSError) as e:
            if attempt >= retries:
                raise
            # 第一次重試先保留 Session（僅 sleep），減少「失敗後立刻再開一批新 TCP」
            if attempt >= 1:
                _reset_thread_local_session(tls)
            delay = backoff * (2**attempt)
            # 隨機抖動：避免多 thread 同一時間醒來再次同時連線（thundering herd）
            lo = delay * (1.0 - jitter)
            hi = delay * (1.0 + jitter)
            sleep_sec = random.uniform(lo, hi)
            print(
                f"[HTTP-RETRY] {file_path.name} attempt {attempt + 1}/{retries + 1} "
                f"after {type(e).__name__}; sleep {sleep_sec:.2f}s (base {delay:.2f}s, jitter ±{jitter:.0%})",
                flush=True,
            )
            time.sleep(sleep_sec)
    raise RuntimeError("upload_segment_with_connect_retries: unreachable")


def compute_rtsp_deadline_metrics(
    records: List[SegmentRecord], chunks: int, streams: int
) -> Dict[str, Any]:
    """
    每一路徑獨立：chunk k 的 done_unix ≤ chunk k+1 的 put_unix（下一段進 queue 前上一段須做完）。
    「30 路都即時」= 每一路徑上**每一對** chunk→chunk+1 皆達成，且各路徑資料完整；
    整體用 **最差一路** 比例 rtsp_worst_stream_ratio（任一路落後則此值 < 1）。
    """
    expected_ids = [f"RTSP_{i+1:02d}" for i in range(streams)]
    by_stream: Dict[str, Dict[int, SegmentRecord]] = {}
    for r in records:
        if not r.ok or r.done_unix <= 0:
            continue
        by_stream.setdefault(r.stream_id, {})[r.chunk_id] = r

    total_pairs = 0
    on_time = 0
    slacks: List[float] = []
    worst_line = ""
    worst_slack_val = float("inf")

    per_stream_on: Dict[str, int] = {}
    per_stream_pairs: Dict[str, int] = {}
    incomplete_streams: List[str] = []

    need_chunks = set(range(1, chunks + 1)) if chunks >= 1 else set()

    for sid in expected_ids:
        cmap = by_stream.get(sid, {})
        so = 0
        sp = 0
        for k in range(1, chunks):
            a = cmap.get(k)
            b = cmap.get(k + 1)
            if a is None or b is None:
                continue
            sp += 1
            total_pairs += 1
            slack = b.put_unix - a.done_unix
            slacks.append(slack)
            if a.done_unix <= b.put_unix:
                so += 1
                on_time += 1
            if slack < worst_slack_val:
                worst_slack_val = slack
                worst_line = f"{sid} #{k}→#{k+1} slack={slack:.3f}s"
        per_stream_on[sid] = so
        per_stream_pairs[sid] = sp
        if chunks >= 2 and not need_chunks.issubset(set(cmap.keys())):
            incomplete_streams.append(sid)

    ratio_global = (on_time / total_pairs) if total_pairs else float("nan")
    per_stream_ratio: Dict[str, float] = {
        s: per_stream_on[s] / per_stream_pairs[s] for s in expected_ids if per_stream_pairs[s] > 0
    }
    worst_stream_ratio = min(per_stream_ratio.values()) if per_stream_ratio else float("nan")
    expected_pairs_per_stream = max(0, chunks - 1)
    all_streams_full_data = chunks < 2 or all(
        per_stream_pairs[sid] == expected_pairs_per_stream for sid in expected_ids
    )

    every_ok = False
    if chunks < 2:
        every_ok = False
    else:
        every_ok = (
            len(incomplete_streams) == 0
            and all_streams_full_data
            and not math.isnan(worst_stream_ratio)
            and worst_stream_ratio >= 1.0 - 1e-12
        )

    return {
        "rtsp_deadline_pairs": total_pairs,
        "rtsp_on_time_count": on_time,
        "rtsp_on_time_ratio": ratio_global,
        "rtsp_min_slack_sec": min(slacks) if slacks else float("nan"),
        "rtsp_all_on_time": (on_time == total_pairs) if total_pairs > 0 else False,
        "rtsp_worst_case": worst_line,
        "rtsp_per_stream_ratio": per_stream_ratio,
        "rtsp_worst_stream_ratio": worst_stream_ratio,
        "rtsp_incomplete_streams": incomplete_streams,
        "rtsp_every_stream_deadline_ok": every_ok,
    }


def compute_relaxed_api_sum_metrics(
    records: List[SegmentRecord], chunks: int, streams: int, budget_sec: float
) -> Dict[str, Any]:
    """
    緩判（可選）：每路徑將「各段 API 上傳+後端」秒數加總，若每路總和 ≤ budget_sec 則視為緩判通過
    （不要求每一對 chunk→chunk+1 都先於下一段入隊；與嚴格截止並列參考）。

    註：5 對 × 每段 10 秒「內容」≈ 50 秒**間隔**，與 API 總耗時是不同量；門檻請依實測調整（例如 6 段×11s≈66s）。
    """
    if budget_sec <= 0:
        return {}
    expected_ids = [f"RTSP_{i+1:02d}" for i in range(streams)]
    by_stream: Dict[str, Dict[int, SegmentRecord]] = {}
    for r in records:
        if not r.ok:
            continue
        by_stream.setdefault(r.stream_id, {})[r.chunk_id] = r

    need = set(range(1, chunks + 1))
    sums: Dict[str, float] = {}
    for sid in expected_ids:
        cmap = by_stream.get(sid, {})
        if not need.issubset(set(cmap.keys())):
            sums[sid] = float("nan")
            continue
        sums[sid] = sum(cmap[cid].upload_plus_backend_sec for cid in range(1, chunks + 1))

    valid: List[float] = []
    for s in expected_ids:
        v = sums.get(s, float("nan"))
        if not math.isnan(v):
            valid.append(v)
    if not valid:
        return {
            "relaxed_budget_sec": budget_sec,
            "rtsp_per_stream_sum_api_sec": sums,
            "rtsp_max_sum_api_per_stream": float("nan"),
            "rtsp_min_sum_api_per_stream": float("nan"),
            "rtsp_relaxed_sum_api_per_stream_ok": False,
            "rtsp_worst_stream_sum_api_id": "",
        }

    mx = max(valid)
    mn = min(valid)

    def _sum_key(sid: str) -> float:
        v = sums.get(sid, float("nan"))
        return -1e100 if math.isnan(v) else v

    worst_sid = max(expected_ids, key=_sum_key)
    all_streams_complete = len(valid) == len(expected_ids)
    all_ok = all_streams_complete and all(v <= budget_sec + 1e-9 for v in valid)

    return {
        "relaxed_budget_sec": budget_sec,
        "rtsp_per_stream_sum_api_sec": sums,
        "rtsp_max_sum_api_per_stream": mx,
        "rtsp_min_sum_api_per_stream": mn,
        "rtsp_relaxed_sum_api_per_stream_ok": all_ok,
        "rtsp_worst_stream_sum_api_id": worst_sid,
    }


def max_events_in_sliding_window(sorted_ts: List[float], window_sec: float) -> int:
    """已排序的完成時間戳中，任意長度 window_sec 的滑動窗內最多包含幾個點。"""
    if not sorted_ts or window_sec <= 0:
        return 0
    n = len(sorted_ts)
    j = 0
    best = 0
    for i in range(n):
        while j < n and sorted_ts[j] - sorted_ts[i] <= window_sec:
            j += 1
        best = max(best, j - i)
    return best


def run_one_round(args: Any, streams: int) -> Dict[str, Any]:
    """跑一輪指定路數，回傳吞吐與延遲統計。"""
    seg_q: queue.Queue = queue.Queue()
    records: List[SegmentRecord] = []
    records_lock = threading.Lock()
    completion_times: List[float] = []
    completion_lock = threading.Lock()
    sentinel = object()

    max_http = max(1, int(getattr(args, "max_concurrent_http", 64)))
    timeout_tuple = (float(args.connect_timeout), float(args.read_timeout))

    tls = threading.local()

    def get_worker_session() -> requests.Session:
        if not hasattr(tls, "session"):
            tls.session = _build_pooled_session(max_http)
        return tls.session

    default_event = "請根據提供的影格輸出事件 JSON。"
    default_summary = "請產出 50-100 字繁體中文畫面摘要。"

    executor = ThreadPoolExecutor(max_workers=max_http)

    inflight_cap = int(getattr(args, "max_inflight_http", 0) or 0)
    http_gate: Optional[threading.BoundedSemaphore] = (
        threading.BoundedSemaphore(inflight_cap) if inflight_cap > 0 else None
    )

    def fire_request_job(item: SegmentTask) -> None:
        rec: Optional[SegmentRecord] = None
        t_req_start = time.time()
        try:
            try:

                def _upload_once() -> Dict:
                    return upload_segment_with_connect_retries(
                        args.base,
                        item.file_path,
                        args,
                        default_event,
                        default_summary,
                        timeout_tuple,
                        tls,
                        get_worker_session,
                    )

                gj = float(getattr(args, "http_gate_jitter_sec", 0.0) or 0.0)
                if http_gate is not None:
                    with http_gate:
                        if gj > 0:
                            time.sleep(random.uniform(0.0, gj))
                        out = _upload_once()
                else:
                    out = _upload_once()
                t_done = time.time()
                with completion_lock:
                    completion_times.append(t_done)
                rec = SegmentRecord(
                    item.stream_id,
                    item.chunk_id,
                    True,
                    item.capture_elapsed_sec,
                    round(t_req_start - item.put_time, 3),
                    round(t_done - t_req_start, 3),
                    round(t_done - item.put_time, 3),
                    int(out.get("total_segments", 0)),
                    int(out.get("success_segments", 0)),
                    out.get("process_time_sec"),
                    out.get("total_time_sec"),
                    "",
                    item.put_time,
                    t_done,
                )
                pso = bool(getattr(args, "print_slow_only", False))
                if should_emit_segment_detail(rec, pso):
                    emit_segment_record_detail(rec)
            except Exception as e:
                rec = SegmentRecord(
                    item.stream_id,
                    item.chunk_id,
                    False,
                    item.capture_elapsed_sec,
                    round(t_req_start - item.put_time, 3),
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    str(e),
                    item.put_time,
                    0.0,
                )
                emit_segment_record_detail(rec)
        except Exception as e:
            if rec is None:
                rec = SegmentRecord(
                    item.stream_id,
                    item.chunk_id,
                    False,
                    item.capture_elapsed_sec,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    str(e),
                    item.put_time,
                    0.0,
                )
                emit_segment_record_detail(rec)
        finally:
            if rec is not None:
                with records_lock:
                    records.append(rec)
            if item.file_path.exists():
                item.file_path.unlink()
            seg_q.task_done()

    def dispatcher() -> None:
        while True:
            item = seg_q.get()
            if item is sentinel:
                seg_q.task_done()
                break
            executor.submit(fire_request_job, item)

    def _make_chunk_file() -> Path:
        """擷取 RTSP 或複製夾具影片（夾具可略過 FFmpeg，專測後端連線／吞吐）。"""
        fixture = getattr(args, "fixture_mp4", None)
        if fixture is not None:
            fp = Path(fixture)
            if not fp.is_file():
                raise FileNotFoundError(f"--fixture-mp4 不存在: {fp}")
            import tempfile

            fd, tmp = tempfile.mkstemp(prefix="fixture_vllm_", suffix=".mp4")
            os.close(fd)
            shutil.copy2(fp, tmp)
            return Path(tmp)
        return capture_rtsp_local(args.rtsp, args.duration)

    def producer(sid: str) -> None:
        for cid in range(1, args.chunks + 1):
            t0 = time.time()
            try:
                path = _make_chunk_file()
                seg_q.put(SegmentTask(sid, cid, path, time.time(), round(time.time() - t0, 3)))
                print(f"[PRODUCE] {sid} chunk={cid} captured", flush=True)
            except Exception as e:
                print(f"[PRODUCE-ERR] {sid} chunk={cid} {e}", flush=True)

    threading.Thread(target=dispatcher, daemon=True).start()

    hc_retries = int(getattr(args, "http_connect_retries", 5))
    hc_back = float(getattr(args, "http_retry_backoff_sec", 0.75))
    hj = float(getattr(args, "http_retry_jitter", 0.5))
    gate_extra = (
        f" 全域同時 multipart≤{inflight_cap}（其餘 worker 在門外等待）"
        if inflight_cap > 0
        else ""
    )
    gjit = float(getattr(args, "http_gate_jitter_sec", 0.0) or 0.0)
    gjit_s = f" gate抖動≤{gjit:.2f}s" if gjit > 0 else ""
    print(
        f"[設定] HTTP worker={max_http}（ThreadPoolExecutor + thread-local Session 連線重用）；"
        f" timeout 連線={timeout_tuple[0]}s 讀取={timeout_tuple[1]}s；"
        f" 連線失敗重試={hc_retries} 次、退避={hc_back}s×2^n、抖動±{hj:.0%}。"
        f"{gate_extra}{gjit_s}",
        flush=True,
    )

    wall_start = time.time()
    p_threads = []
    for i in range(streams):
        pt = threading.Thread(target=producer, args=(f"RTSP_{i+1:02d}",))
        p_threads.append(pt)
        pt.start()
        time.sleep(0.2)

    for pt in p_threads:
        pt.join()

    seg_q.put(sentinel)
    seg_q.join()
    executor.shutdown(wait=True)
    wall_end = time.time()
    wall_sec = max(wall_end - wall_start, 1e-9)

    ok_list = [r for r in records if r.ok]
    ts = sorted(completion_times)
    avg_sps = len(ok_list) / wall_sec

    warmup = float(getattr(args, "warmup_sec", 5.0))
    cooldown = float(getattr(args, "cooldown_sec", 5.0))
    steady_start = wall_start + warmup
    steady_end = wall_end - cooldown
    if steady_end > steady_start:
        steady_ts = [t for t in completion_times if steady_start <= t <= steady_end]
        steady_sps = len(steady_ts) / (steady_end - steady_start)
    else:
        steady_ts = []
        steady_sps = float("nan")

    peak10 = max_events_in_sliding_window(ts, 10.0)
    peak1 = max_events_in_sliding_window(ts, 1.0)
    peak10_sps = peak10 / 10.0
    peak1_sps = peak1 / 1.0

    p95_backend = float(np.percentile([r.upload_plus_backend_sec for r in ok_list], 95)) if ok_list else 0.0

    chunk_n = max(1, int(args.chunks))
    dm = compute_rtsp_deadline_metrics(records, chunk_n, streams)

    seg_dur = float(getattr(args, "segment_duration", 10.0))
    out: Dict[str, Any] = {
        "streams": streams,
        "chunks": chunk_n,
        "segment_duration_sec": seg_dur,
        "wall_sec": wall_sec,
        "total_records": len(records),
        "ok_count": len(ok_list),
        "fail_count": len(records) - len(ok_list),
        "avg_sps": avg_sps,
        "steady_sps": steady_sps,
        "steady_window_sec": max(steady_end - steady_start, 0) if steady_end > steady_start else 0.0,
        "peak10_count": peak10,
        "peak10_sps": peak10_sps,
        "peak1_count": peak1,
        "peak1_sps": peak1_sps,
        "p95_backend_sec": p95_backend,
        "avg_backend_sec": statistics.mean([r.upload_plus_backend_sec for r in ok_list]) if ok_list else 0.0,
        "avg_queue_wait_sec": statistics.mean([r.queue_wait_sec for r in ok_list]) if ok_list else 0.0,
    }
    out.update(dm)
    budget = float(getattr(args, "relaxed_api_sum_budget_sec", 0.0) or 0.0)
    if budget > 0:
        out.update(compute_relaxed_api_sum_metrics(records, chunk_n, streams, budget))
    return out


def print_round_report(r: Dict[str, Any]) -> None:
    print(f"\n--- 單輪報告 (路數={r['streams']}) ---")
    print(f"總請求數: {r['total_records']}, 成功: {r['ok_count']}, 失敗: {r['fail_count']}")
    print(f"測試總牆鐘: {r['wall_sec']:.2f}s")
    print(f"平均吞吐: {r['avg_sps']:.3f} 段/秒（成功段數 / 總牆鐘）")
    if not math.isnan(r["steady_sps"]):
        print(
            f"穩態吞吐: {r['steady_sps']:.3f} 段/秒（牆鐘中段 {r['steady_window_sec']:.1f}s 內完成數/該區間）"
        )
    else:
        print("穩態吞吐: N/A（warmup/cooldown 過大，無有效中段窗）")
    print(
        f"峰值(10秒窗): 最多 {r['peak10_count']} 段/窗 → 約 {r['peak10_sps']:.3f} 段/秒（該窗內）"
    )
    print(
        f"峰值(1秒窗):  最多 {r['peak1_count']} 段/窗 → 約 {r['peak1_sps']:.3f} 段/秒（該窗內）"
    )
    print(f"P95 API 響應: {r['p95_backend_sec']:.2f}s, 平均 API: {r['avg_backend_sec']:.2f}s")
    print(f"平均 Queue 等待: {r['avg_queue_wait_sec']:.3f}s")
    pairs = int(r.get("rtsp_deadline_pairs", 0) or 0)
    if pairs > 0:
        ratio = r.get("rtsp_on_time_ratio")
        rs = f"{ratio:.1%}" if isinstance(ratio, float) and not math.isnan(ratio) else "N/A"
        wsr = r.get("rtsp_worst_stream_ratio")
        wsr_s = f"{wsr:.1%}" if isinstance(wsr, float) and not math.isnan(wsr) else "N/A"
        ms = r.get("rtsp_min_slack_sec")
        ms_s = f"{ms:.3f}s" if isinstance(ms, float) and not math.isnan(ms) else "N/A"
        ev = r.get("rtsp_every_stream_deadline_ok", False)
        print(
            f"即時(截止)：每路徑「下一段入隊前上一段須做完」— "
            f"全體 {r.get('rtsp_on_time_count', 0)}/{pairs} ({rs})，"
            f"最差一路 {wsr_s}，最小 slack {ms_s}，"
            f"每路皆達標={'是' if ev else '否'}"
        )
        wc = r.get("rtsp_worst_case") or ""
        if wc:
            print(f"  最緊一筆：{wc}")
        inc = r.get("rtsp_incomplete_streams") or []
        if inc:
            print(f"  缺段路徑（無法驗證每對）：{', '.join(inc[:12])}{'…' if len(inc) > 12 else ''}")
        ch = int(r.get("chunks") or 0)
        seg = float(r.get("segment_duration_sec") or 10.0)
        if ch >= 2:
            pp = ch - 1
            approx = pp * seg
            print(
                f"  說明：每路 {pp} 對 = {ch} 段之間的 {pp} 次「上一段須先做完」；"
                f"「{pp}×{seg:g}s≈{approx:g}s」指內容間隔直覺，與 API 總耗時不同。"
            )
    elif int(r.get("streams", 0)) > 0 and r.get("ok_count", 0) > 0:
        print("即時(截止)：N/A（每路需至少 2 段才有 chunk→chunk+1 可比對）")

    rb = float(r.get("relaxed_budget_sec") or 0.0)
    if rb > 0:
        mx = r.get("rtsp_max_sum_api_per_stream")
        mn = r.get("rtsp_min_sum_api_per_stream")
        ok = r.get("rtsp_relaxed_sum_api_per_stream_ok", False)
        wsum = r.get("rtsp_worst_stream_sum_api_id") or ""
        mx_s = f"{mx:.2f}s" if isinstance(mx, float) and not math.isnan(mx) else "N/A"
        mn_s = f"{mn:.2f}s" if isinstance(mn, float) and not math.isnan(mn) else "N/A"
        print(
            f"緩判(API 總和)：每路徑各段 API 秒數加總 ≤ {rb:.1f}s 即通過（不論單對 slack 正負）— "
            f"各路 min/max {mn_s}/{mx_s}，最差路徑 {wsum}，"
            f"緩判={'通過' if ok else '未通過'}"
        )
    if r["fail_count"] > 0:
        print(
            "  提示：ConnectTimeout 表示 TCP 連不上（非讀取逾時）。"
            "可試：--max-inflight-http 12～16（壓同時連線數）、"
            "--max-concurrent-http 略大於路數即可、"
            "或略增 --http-retry-jitter / --connect-timeout；並檢查後端 backlog／防火牆／負載。"
        )


def print_final_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("【階梯統整】峰值 = 滑動窗內最多完成段數/窗長；最差路 = 各路徑截止達成率之最小值（任一路落後即低）")
    print("=" * 100)
    hdr = (
        f"{'路數':>5} | {'平均段/s':>9} | {'穩態段/s':>9} | {'峰值10s窗段/s':>14} | "
        f"{'峰值1s窗段/s':>13} | {'P95 API':>8} | {'最差路':>7} | {'成功':>5}"
    )
    print(hdr)
    print("-" * 100)
    for r in rows:
        ss = f"{r['steady_sps']:.3f}" if not math.isnan(r["steady_sps"]) else "  N/A"
        wsr = r.get("rtsp_worst_stream_ratio")
        dr_s = f"{wsr:>6.0%}" if isinstance(wsr, float) and not math.isnan(wsr) else "    N/A"
        print(
            f"{r['streams']:>5} | {r['avg_sps']:>9.3f} | {ss:>9} | {r['peak10_sps']:>14.3f} | "
            f"{r['peak1_sps']:>13.3f} | {r['p95_backend_sec']:>7.2f}s | {dr_s:>7} | {r['ok_count']:>5}"
        )
    print("=" * 100)
    best_peak = max(rows, key=lambda x: x["peak10_sps"])
    print(
        f"本輪階梯中，峰值(10秒窗)最高：路數={best_peak['streams']} → {best_peak['peak10_sps']:.3f} 段/秒 "
        f"（該窗最多完成 {best_peak['peak10_count']} 段）"
    )


def row_feasible_for_auto(row: Dict[str, Any], args: Any) -> bool:
    """
    自動搜尋「通過」條件：
    - 必須全部 HTTP 成功（fail_count==0 且有紀錄）
    - 未加 --auto-skip-realtime-check 時：
      - 預設（--auto-deadline-check）：每路徑截止 — rtsp_worst_stream_ratio ≥ 門檻，且無缺段
      - --auto-legacy-rtsp-check：舊式（fixture 用穩態；RTSP 用峰值窗 + P95）
    """
    if row.get("fail_count", 0) > 0:
        return False
    if row.get("total_records", 0) <= 0:
        return False
    if getattr(args, "auto_skip_realtime_check", False):
        return True

    chunks = int(getattr(args, "chunks", 6))
    use_legacy = getattr(args, "auto_legacy_rtsp_check", False)
    use_deadline = not getattr(args, "no_auto_deadline_check", False)

    if use_deadline and not use_legacy and chunks >= 2:
        if row.get("rtsp_incomplete_streams") and not getattr(
            args, "auto_deadline_allow_incomplete", False
        ):
            return False
        pairs = int(row.get("rtsp_deadline_pairs", 0) or 0)
        if pairs <= 0:
            return False
        wsr = row.get("rtsp_worst_stream_ratio")
        if not isinstance(wsr, float) or math.isnan(wsr):
            return False
        min_r = float(getattr(args, "auto_deadline_min_ratio", 1.0))
        return wsr + 1e-9 >= min_r

    seg_d = float(getattr(args, "segment_duration", 10.0))
    need = row["streams"] / max(seg_d, 1e-9)
    steady = row["steady_sps"]
    eps = float(getattr(args, "auto_realtime_epsilon", 0.15))
    p95 = float(row.get("p95_backend_sec") or 0.0)
    peak10 = float(row.get("peak10_sps") or 0.0)
    lat_slack = float(getattr(args, "auto_latency_slack_sec", 0.5))
    latency_ok = p95 <= seg_d + lat_slack

    fixture = getattr(args, "fixture_mp4", None)
    if fixture is not None:
        if isinstance(steady, float) and math.isnan(steady):
            return False
        return float(steady) >= need - eps

    steady_ok = (
        isinstance(steady, float)
        and not math.isnan(steady)
        and float(steady) >= need - eps
    )
    peak_ok = peak10 >= need - eps
    return latency_ok and (steady_ok or peak_ok)


def run_auto_find_streams(args: Any) -> None:
    """
    在 [auto_low, auto_high] 內二分搜尋「最大」滿足 row_feasible_for_auto 的路數。
    假設：路數愈高愈容易失敗（單調）；網路若抖動可重跑或放寬 ε。
    """
    lo = max(1, int(getattr(args, "auto_low", 5)))
    hi = max(lo, int(getattr(args, "auto_high", 48)))
    probe_chunks = max(2, int(getattr(args, "auto_probe_chunks", 4)))

    saved_chunks = int(args.chunks)
    saved_w = float(getattr(args, "warmup_sec", 5.0))
    saved_c = float(getattr(args, "cooldown_sec", 5.0))

    args.chunks = probe_chunks
    args.warmup_sec = float(getattr(args, "auto_probe_warmup_sec", 3.0))
    args.cooldown_sec = float(getattr(args, "auto_probe_cooldown_sec", 3.0))

    try:
        rt_on = not bool(getattr(args, "auto_skip_realtime_check", False))
        eps = float(getattr(args, "auto_realtime_epsilon", 0.15))
        use_fix = getattr(args, "fixture_mp4", None) is not None
        no_dead = bool(getattr(args, "no_auto_deadline_check", False))
        use_leg = bool(getattr(args, "auto_legacy_rtsp_check", False))
        min_dr = float(getattr(args, "auto_deadline_min_ratio", 1.0))
        rt_desc = "關閉（僅要求全成功）"
        if rt_on:
            if no_dead or use_leg:
                if use_fix:
                    rt_desc = f"舊式｜夾具：穩態段/s ≥ 路數/segment_duration − {eps}"
                else:
                    rt_desc = (
                        f"舊式｜RTSP：峰值(10s)或穩態 ≥ 路數/segment_duration − {eps}，"
                        f"P95 ≤ segment_duration + {float(getattr(args, 'auto_latency_slack_sec', 0.5))}s"
                    )
            else:
                rt_desc = (
                    f"每路截止：最差一路比例 ≥ {min_dr:.2f}，且各路無缺段"
                )
        print(
            f"\n{'=' * 80}\n[自動搜尋] 二分搜尋路數 ∈ [{lo}, {hi}]，每路探測 {probe_chunks} 段；"
            f"即時檢查={rt_desc}\n{'=' * 80}\n",
            flush=True,
        )
        mh = int(getattr(args, "max_concurrent_http", 32))
        if mh < hi:
            print(
                f"[警告] --max-concurrent-http={mh} < --auto-high={hi}，"
                f"高路數時請考慮設為至少 {hi}。\n",
                flush=True,
            )

        print(f"[自動搜尋] ① 下界探針：路數={lo} …", flush=True)
        row_lo = run_one_round(args, lo)
        print_round_report(row_lo)
        if not row_feasible_for_auto(row_lo, args):
            print(
                "\n[自動搜尋] 中止：連下界路數都不滿足（全成功 + 即時條件）。"
                "可：降低 --auto-low、放寬 --auto-deadline-min-ratio、加 --auto-skip-realtime-check、"
                "或 --no-auto-deadline-check 改舊式判斷。",
                flush=True,
            )
            return

        best = lo
        best_row = row_lo
        l, r = lo + 1, hi
        while l <= r:
            mid = (l + r) // 2
            print(f"\n[自動搜尋] ② 探針：路數={mid} …", flush=True)
            row_m = run_one_round(args, mid)
            print_round_report(row_m)
            if row_feasible_for_auto(row_m, args):
                best = mid
                best_row = row_m
                l = mid + 1
            else:
                r = mid - 1

        need = best / max(float(getattr(args, "segment_duration", 10.0)), 1e-9)
        ss = best_row["steady_sps"]
        ss_s = f"{ss:.3f}" if not (isinstance(ss, float) and math.isnan(ss)) else "N/A"
        dr = best_row.get("rtsp_on_time_ratio")
        dr_s = f"{dr:.1%}" if isinstance(dr, float) and not math.isnan(dr) else "N/A"
        wsr = best_row.get("rtsp_worst_stream_ratio")
        wsr_s = f"{wsr:.1%}" if isinstance(wsr, float) and not math.isnan(wsr) else "N/A"
        dc = best_row.get("rtsp_on_time_count", 0)
        dp = best_row.get("rtsp_deadline_pairs", 0)
        print(
            f"\n{'=' * 80}\n"
            f"[自動搜尋] 結果：在目前條件下，最大可行路數 ≈ {best} 路\n"
            f"  （探測每路 {probe_chunks} 段；穩態 {ss_s} 段/s；"
            f"全體截止 {dc}/{dp}={dr_s}，最差一路 {wsr_s}；理論段/s 需求約 {need:.3f}）\n"
            f"{'=' * 80}\n",
            flush=True,
        )

        if getattr(args, "auto_confirm", False):
            cc = int(getattr(args, "auto_confirm_chunks", 0) or 0)
            cchunks = cc if cc > 0 else saved_chunks
            print(
                f"[自動搜尋] ③ 確認輪：路數={best}，每路 {cchunks} 段（較長）…",
                flush=True,
            )
            args.chunks = cchunks
            args.warmup_sec = saved_w
            args.cooldown_sec = saved_c
            row_c = run_one_round(args, best)
            print_round_report(row_c)
            if row_feasible_for_auto(row_c, args):
                print(
                    f"\n[自動搜尋] 確認輪通過：{best} 路在較長測試下仍滿足條件。\n",
                    flush=True,
                )
            else:
                print(
                    "\n[自動搜尋] 確認輪未通過：建議以略低路數重跑，或拉長 --auto-probe-chunks / 放寬 ε。\n",
                    flush=True,
                )
    finally:
        args.chunks = saved_chunks
        args.warmup_sec = saved_w
        args.cooldown_sec = saved_c


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM 多路 RTSP 吞吐：可單一路數或 --ladder 階梯；輸出平均/穩態/峰值吞吐。"
    )
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--rtsp", default=DEFAULT_RTSP)
    parser.add_argument(
        "--fixture-mp4",
        type=Path,
        default=None,
        help="若指定現有 mp4 路徑，每 chunk 改為複製此檔（略過 FFmpeg），專測後端 API／連線；與 RTSP 擇一",
    )
    parser.add_argument("--streams", type=int, default=23, help="單輪路數（未指定 --ladder 時使用）")
    parser.add_argument(
        "--ladder",
        default="",
        help="階梯路數，逗號分隔，例如 10,15,19,25,30；會依序多跑幾輪並最後統整",
    )
    parser.add_argument("--chunks", type=int, default=180, help="每路總段數（愈大愈易進入穩態，但耗時愈久）")
    parser.add_argument("--duration", type=float, default=10.0, help="FFmpeg 擷取秒數")
    parser.add_argument("--segment-duration", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--frames-per-segment", type=int, default=5)
    parser.add_argument("--target-short", type=int, default=432)
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=5.0,
        help="穩態區間：牆鐘起算略過前幾秒（避開 producer  stagger 與首段尖峰）",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=5.0,
        help="穩態區間：牆鐘結束前略過最後幾秒（避開尾段排空前的低負載）",
    )
    parser.add_argument(
        "--relaxed-api-sum-budget-sec",
        type=float,
        default=60.0,
        help="緩判：每路徑各段 API 秒數加總上限（秒）；0=不計算。與「即時(截止)」分開，可對照是否整路在門檻內",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=120.0,
        help="requests 建立 TCP 連線逾時（秒）；遠端忙或網路慢可略增",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=900.0,
        help="requests 等待伺服器回應本體逾時（秒）",
    )
    parser.add_argument(
        "--max-concurrent-http",
        type=int,
        default=32,
        help="ThreadPoolExecutor worker 數（同時 HTTP 上限）；每 worker 一個 Session 連線池；過高易塞滿對方 accept 佇列",
    )
    parser.add_argument(
        "--http-connect-retries",
        type=int,
        default=12,
        help="ConnectTimeout/ConnectionError 時額外重試次數（第 2 次起才重建 Session）",
    )
    parser.add_argument(
        "--http-retry-backoff-sec",
        type=float,
        default=0.75,
        help="連線失敗重試的指數退避基底秒數：第 n 次等待 基底×2^n",
    )
    parser.add_argument(
        "--http-retry-jitter",
        type=float,
        default=0.5,
        help="退避時間乘上 Uniform(1−j,1+j)，避免多 thread 同秒重試；0=固定退避、1=0～2 倍基底",
    )
    parser.add_argument(
        "--max-inflight-http",
        type=int,
        default=10,
        help="全域最多同時進行中的 multipart HTTP 數；0=不限制。"
        "預設 10 以降低對方 accept 塞爆；路數高仍連不上可再降到 6～8",
    )
    parser.add_argument(
        "--http-gate-jitter-sec",
        type=float,
        default=0.12,
        help="取得 inflight 許可後、真正 POST 前隨機等待 U(0,此值) 秒，打散同時連線",
    )
    parser.add_argument(
        "--print-slow-only",
        action="store_true",
        help="成功請求僅在 queue_to_done > 10s 時列印 [SLOW-DETECTED] 明細；失敗請求仍一律列印",
    )
    parser.add_argument(
        "--auto-find-streams",
        action="store_true",
        help="二分搜尋 [auto-low,auto-high] 內「最大可行」RTSP 路數（會多輪測試；建議搭配 --fixture-mp4）",
    )
    parser.add_argument("--auto-low", type=int, default=5, help="自動搜尋下界（路數）")
    parser.add_argument("--auto-high", type=int, default=48, help="自動搜尋上界（路數）")
    parser.add_argument(
        "--auto-probe-chunks",
        type=int,
        default=4,
        help="自動搜尋時每路段數（愈大愈準、愈久）",
    )
    parser.add_argument(
        "--auto-skip-realtime-check",
        action="store_true",
        help="自動搜尋只要求全部 HTTP 成功，不要求穩態吞吐達即時線",
    )
    parser.add_argument(
        "--auto-realtime-epsilon",
        type=float,
        default=0.15,
        help="即時條件容許誤差（段/s），探測輪次短時可略放大",
    )
    parser.add_argument(
        "--auto-latency-slack-sec",
        type=float,
        default=0.5,
        help="舊式 RTSP 即時檢查：P95 API 允許超過 segment-duration 的秒數",
    )
    parser.add_argument(
        "--no-auto-deadline-check",
        action="store_true",
        help="自動搜尋不用「同路截止」達成率，改走舊式（穩態/峰值+P95）",
    )
    parser.add_argument(
        "--auto-legacy-rtsp-check",
        action="store_true",
        help="強制舊式判斷（與截止模式二擇一；通常搭配 --no-auto-deadline-check）",
    )
    parser.add_argument(
        "--auto-deadline-min-ratio",
        type=float,
        default=1.0,
        help="自動搜尋：各路徑中「最差一路」的截止達成率下限（1.0=每一路每一對皆達標）",
    )
    parser.add_argument(
        "--auto-deadline-allow-incomplete",
        action="store_true",
        help="自動搜尋時允許部分路徑缺段仍繼續用（預設：有缺段則不通過）",
    )
    parser.add_argument(
        "--auto-probe-warmup-sec",
        type=float,
        default=3.0,
        help="自動搜尋探測時覆寫 --warmup-sec",
    )
    parser.add_argument(
        "--auto-probe-cooldown-sec",
        type=float,
        default=3.0,
        help="自動搜尋探測時覆寫 --cooldown-sec",
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="搜尋結束後以較長 chunks 再驗證一次最佳路數（用原 --chunks 或 --auto-confirm-chunks）",
    )
    parser.add_argument(
        "--auto-confirm-chunks",
        type=int,
        default=0,
        help="確認輪每路段數；0 表示用執行時原本的 --chunks（存於參數預設）",
    )
    args = parser.parse_args()

    if args.auto_find_streams:
        if args.ladder.strip():
            print("[警告] 已指定 --auto-find-streams，將忽略 --ladder。", flush=True)
        run_auto_find_streams(args)
        return

    if args.ladder.strip():
        levels = [int(x.strip()) for x in args.ladder.split(",") if x.strip()]
        if not levels:
            raise SystemExit("--ladder 無有效數字")
    else:
        levels = [args.streams]

    all_rows: List[Dict[str, Any]] = []
    for idx, n in enumerate(levels):
        print(f"\n########## 階梯 {idx + 1}/{len(levels)}：路數 = {n} ##########", flush=True)
        row = run_one_round(args, n)
        all_rows.append(row)
        print_round_report(row)

    if len(all_rows) > 1:
        print_final_table(all_rows)
    elif len(all_rows) == 1:
        r = all_rows[0]
        print(
            f"\n（單輪）峰值吞吐參考：10秒窗 {r['peak10_sps']:.3f} 段/s，1秒窗 {r['peak1_sps']:.3f} 段/s"
        )


if __name__ == "__main__":
    main()
