#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重跑指定 category/camera 的 segment_pipeline_multipart（與人員追蹤_20260528 批次相同參數）。"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("請安裝 requests: pip install requests")
    sys.exit(1)


def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
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

DEFAULT_BASE = os.environ.get("BACKEND_URL", "http://127.0.0.1:3000/api")
API_KEY = os.environ.get("API_KEY") or os.environ.get("MY_API_KEY", "")
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct-AWQ"


def run_one(base_url: str, category: str, camera: str, api_key: str, qwen_model: str, timeout: int) -> dict:
    video_id = f"{category}/{camera}"
    stem = f"{category}_{camera}"
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_multipart"
    headers = {"X-API-Key": api_key} if api_key else {}
    data = {
        "model_type": "vllm_qwen",
        "video_id": video_id,
        "segment_duration": "10.0",
        "overlap": "0.0",
        "qwen_model": qwen_model,
        "frames_per_segment": "5",
        "target_short": "432",
        "strict_segmentation": "False",
        "yolo_labels": "person,car",
        "yolo_every_sec": "2.0",
        "yolo_score_thr": "0.25",
        "save_json": "True",
        "save_basename": f"{stem}.json",
    }
    t0 = time.time()
    print(f"\n{'=' * 60}\n[{datetime.now().isoformat()}] START {video_id}\n{'=' * 60}", flush=True)
    try:
        r = requests.post(url, data=data, headers=headers, timeout=timeout)
        elapsed = time.time() - t0
        if r.status_code != 200:
            detail = (r.text or "")[:800]
            print(f"[FAIL] {stem} HTTP {r.status_code}: {detail}", flush=True)
            return {
                "video_id": stem,
                "camera": camera,
                "http_status": r.status_code,
                "ok": False,
                "detail": detail,
                "elapsed_sec": round(elapsed, 2),
            }
        body = r.json()
        ok = body.get("success_segments", 0) == body.get("total_segments", 0)
        print(
            f"[{'OK' if ok else 'WARN'}] {stem} segments={body.get('success_segments')}/{body.get('total_segments')} "
            f"process_time={body.get('process_time_sec')}s elapsed={elapsed:.1f}s",
            flush=True,
        )
        return {
            "video_id": stem,
            "camera": camera,
            "http_status": 200,
            "ok": ok,
            "success_segments": body.get("success_segments"),
            "total_segments": body.get("total_segments"),
            "process_time_sec": body.get("process_time_sec"),
            "total_time_sec": body.get("total_time_sec"),
            "save_path": body.get("save_path"),
            "elapsed_sec": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAIL] {stem} {type(e).__name__}: {e}", flush=True)
        return {
            "video_id": stem,
            "camera": camera,
            "ok": False,
            "detail": str(e),
            "elapsed_sec": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="重跑 segment_pipeline_multipart（指定鏡頭）")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--category", default="人員追蹤_20260528")
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--timeout", type=int, default=7200, help="單鏡頭 HTTP 逾時（秒）")
    parser.add_argument("--log", type=Path, default=None, help="附加寫入 log 檔")
    args = parser.parse_args()

    print(f"Backend: {args.base}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Cameras: {len(args.cameras)}", flush=True)

    results = []
    for i, cam in enumerate(args.cameras, 1):
        print(f"\n--- [{i}/{len(args.cameras)}] ---", flush=True)
        rec = run_one(args.base, args.category, cam, args.api_key, args.model, args.timeout)
        results.append(rec)

    summary = {
        "started_at": datetime.now().isoformat(),
        "category": args.category,
        "qwen_model": args.model,
        "results": results,
    }
    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"\n[DONE] {ok_n}/{len(results)} succeeded.", flush=True)

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
