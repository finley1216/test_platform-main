#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 /v1/segment_pipeline_multipart 使用 model_type=vllm_qwen 的完整流程。
可選：本機影片檔（--video）或 RTSP 擷取一段（--rtsp --capture）後上傳，後端會走 infer_segment_vllm。

結果如何傳給你：
1) API 回應 body：後端跑完 pipeline 後，直接把完整結果放在 HTTP 回應的 JSON 裡
   （model_type, total_segments, success_segments, results, process_time_sec, total_time_sec, stem, diagnostics）。
2) 寫入 DB：若後端有連 PostgreSQL（HAS_DB），會自動呼叫 _save_results_to_postgres，把 results 寫入 DB
   （識別用 video_id 或 stem，上傳本機檔案時用 stem，例如檔名衍生的 upload_xxx）。
3) 後端本機 JSON：若 save_json=True（預設），後端還會在伺服器上的 segment 目錄存一份 JSON，路徑在回傳的 save_path。
4) 本腳本：收到 API 回傳後會印出摘要；加 --out 可把同一份 JSON 存到你指定的本機檔案。
"""
import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    print("請安裝 requests: pip install requests")
    sys.exit(1)


def _load_dotenv():
    """從 backend 目錄的 .env 載入環境變數。"""
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

# 與 test_segment_pipeline_rtsp.py 一致：.env 的 BACKEND_URL 或預設 140.117.176.42:3000
DEFAULT_BASE = os.environ.get("BACKEND_URL", "http://140.117.176.42:3000/api")
DEFAULT_RTSP = os.environ.get("RTSP_URL", "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")
API_KEY = os.environ.get("API_KEY") or os.environ.get("MY_API_KEY", "")

# vLLM 預設模型（與 docker-compose VLLM_MODEL 對應）
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"


def capture_rtsp_local(rtsp_url: str, duration_sec: float, video_id: str) -> Path:
    """從 RTSP 擷取一段影片存成暫存 .mp4，回傳路徑。"""
    import tempfile
    import subprocess

    fd, tmp = tempfile.mkstemp(prefix="rtsp_capture_", suffix=".mp4")
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
    filename: str = "video.mp4",
    model_type: str = "vllm_qwen",
    qwen_model: str = DEFAULT_VLLM_MODEL,
    segment_duration: float = 10.0,
    overlap: float = 0.0,
    frames_per_segment: int = 5,
    target_short: int = 432,
    sampling_fps: float = None,
    event_detection_prompt: str = "",
    summary_prompt: str = "",
    save_json: bool = True,
    max_retries: int = 3,
) -> dict:
    """上傳影片到 POST /v1/segment_pipeline_multipart，使用 model_type=vllm_qwen。"""
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_multipart"
    headers = {"X-API-Key": api_key} if api_key else {}
    print(f"[API 客戶端] 目標 URL: {url}")
    print(f"[API 客戶端] 檔案: {filename}, 大小: {file_path.stat().st_size} bytes")

    data = {
        "model_type": model_type,
        "segment_duration": str(segment_duration),
        "overlap": str(overlap),
        "qwen_model": qwen_model,
        "frames_per_segment": str(frames_per_segment),
        "target_short": str(target_short),
        "strict_segmentation": "False",
        "yolo_labels": "person,car",
        "yolo_every_sec": "2.0",
        "yolo_score_thr": "0.25",
        "event_detection_prompt": event_detection_prompt or "請根據提供的影格輸出事件 JSON。",
        "summary_prompt": summary_prompt or "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。",
        "save_json": "True" if save_json else "False",
    }
    if sampling_fps is not None:
        data["sampling_fps"] = str(sampling_fps)

    last_exc = None
    for attempt in range(max_retries):
        try:
            print(f"[API 客戶端] 發送 POST 請求 (嘗試 {attempt + 1}/{max_retries})...")
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "video/mp4")}
                r = requests.post(url, data=data, files=files, headers=headers, timeout=600)
            print(f"[API 客戶端] 收到回應: HTTP {r.status_code}")
            if r.status_code in (503, 429) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            body = r.json()
            print(f"[API 客戶端] 回應 JSON 鍵: {list(body.keys())}")
            return body
        except requests.HTTPError as e:
            last_exc = e
            if e.response is not None and e.response.status_code in (503, 429) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("upload_segment_to_api_vllm: max retries exceeded")


def run_one_video(
    base_url: str,
    video_path: Path,
    api_key: str,
    qwen_model: str = DEFAULT_VLLM_MODEL,
    segment_duration: float = 10.0,
    frames_per_segment: int = 5,
    target_short: int = 432,
    event_detection_prompt: str = "",
    summary_prompt: str = "",
) -> dict:
    """上傳單一影片檔並呼叫 vLLM pipeline，回傳 API 回應。"""
    name = video_path.name
    return upload_segment_to_api_vllm(
        base_url,
        video_path,
        api_key,
        filename=name,
        qwen_model=qwen_model,
        segment_duration=segment_duration,
        frames_per_segment=frames_per_segment,
        target_short=target_short,
        event_detection_prompt=event_detection_prompt,
        summary_prompt=summary_prompt,
    )


def run_one_rtsp(
    base_url: str,
    rtsp_url: str,
    capture_sec: float,
    api_key: str,
    qwen_model: str = DEFAULT_VLLM_MODEL,
    segment_duration: float = 10.0,
    frames_per_segment: int = 5,
    target_short: int = 432,
    event_detection_prompt: str = "",
    summary_prompt: str = "",
) -> dict:
    """從 RTSP 擷取一段後上傳並呼叫 vLLM pipeline。"""
    tmp_path = None
    try:
        print(f"  [本機] FFmpeg 擷取 RTSP {capture_sec}s...")
        tmp_path = capture_rtsp_local(rtsp_url, capture_sec, "vllm_rtsp_test")
        return upload_segment_to_api_vllm(
            base_url,
            tmp_path,
            api_key,
            filename=f"vllm_rtsp_{int(time.time())}.mp4",
            qwen_model=qwen_model,
            segment_duration=segment_duration,
            frames_per_segment=frames_per_segment,
            target_short=target_short,
            event_detection_prompt=event_detection_prompt,
            summary_prompt=summary_prompt,
        )
    finally:
        if tmp_path is not None and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Test /v1/segment_pipeline_multipart with model_type=vllm_qwen (vLLM 完整流程)"
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Backend base URL（與 RTSP 腳本相同，可於 .env 設 BACKEND_URL；不傳則用預設）")
    parser.add_argument("--api-key", default=API_KEY, help="X-API-Key（或設 API_KEY / MY_API_KEY）")
    parser.add_argument("--video", type=Path, default=None, help="本機影片檔路徑（與 --rtsp 二擇一）")
    parser.add_argument("--rtsp", default=None, help="RTSP URL（不給 --video 時使用）")
    parser.add_argument("--capture", type=float, default=10.0, help="RTSP 模式：擷取秒數")
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL, help="vLLM 模型名稱（如 Qwen/Qwen2.5-VL-7B-Instruct-AWQ）")
    parser.add_argument("--segment-duration", type=float, default=10.0, help="片段長度（秒）")
    parser.add_argument("--frames-per-segment", type=int, default=5, help="每段取幾張影格送 vLLM")
    parser.add_argument("--target-short", type=int, default=432, help="影格短邊 resize 長度")
    parser.add_argument("--out", type=Path, default=None, help="將 API 回傳 JSON 寫入此檔（可選）")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    api_key = args.api_key or API_KEY

    print("--- vLLM segment pipeline 測試 (model_type=vllm_qwen) ---")
    print(f"  Backend: {base}")
    print(f"  Model:   {args.model}")
    if args.video:
        print(f"  Video:   {args.video}")
        if not args.video.exists():
            print(f"  Error: 檔案不存在: {args.video}")
            sys.exit(1)
    else:
        rtsp = args.rtsp or DEFAULT_RTSP
        print(f"  RTSP:   {rtsp[:60]}...")
        print(f"  Capture: {args.capture}s")
    if not api_key:
        print("  Warning: 未設定 API Key（可於 backend/.env 設 MY_API_KEY 或傳 --api-key）")
    print()

    default_event = "請根據提供的影格輸出事件 JSON。"
    default_summary = "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。"

    t0 = time.time()
    try:
        if args.video:
            out = run_one_video(
                base,
                args.video,
                api_key,
                qwen_model=args.model,
                segment_duration=args.segment_duration,
                frames_per_segment=args.frames_per_segment,
                target_short=args.target_short,
                event_detection_prompt=default_event,
                summary_prompt=default_summary,
            )
        else:
            out = run_one_rtsp(
                base,
                args.rtsp or DEFAULT_RTSP,
                args.capture,
                api_key,
                qwen_model=args.model,
                segment_duration=args.segment_duration,
                frames_per_segment=args.frames_per_segment,
                target_short=args.target_short,
                event_detection_prompt=default_event,
                summary_prompt=default_summary,
            )
        elapsed = time.time() - t0

        print(f"OK 耗時: {elapsed:.1f}s")
        print(f"  結果已透過 API 回傳；後端若已連 DB 會自動寫入 PostgreSQL（識別: stem={out.get('stem')}）")
        print(f"  model_type: {out.get('model_type')}")
        print(f"  stem: {out.get('stem')}")
        print(f"  total_segments: {out.get('total_segments')}")
        print(f"  success_segments: {out.get('success_segments')}")
        print(f"  total_time_sec: {out.get('total_time_sec')}")
        if out.get("diagnostics"):
            print(f"  diagnostics: {out['diagnostics']}")

        if out.get("results"):
            for i, r in enumerate(out["results"][:3]):
                err = r.get("error")
                if err:
                    print(f"  [Segment {i}] error: {err}")
                parsed = r.get("parsed") or {}
                summary = parsed.get("summary_independent", "")
                frame_analysis = parsed.get("frame_analysis") or {}
                print(f"  [Segment {i}] summary: {summary[:80]}..." if len(summary) > 80 else f"  [Segment {i}] summary: {summary or '(空)'}")
                if frame_analysis:
                    print(f"           frame_analysis: {list(frame_analysis.keys())}")
            if len(out["results"]) > 3:
                print(f"  ... 共 {len(out['results'])} 段")

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  回傳 JSON 已寫入: {args.out}")

    except requests.exceptions.RequestException as e:
        print(f"FAIL: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  body: {e.response.text[:500]}")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: {e}")
        raise


if __name__ == "__main__":
    main()
