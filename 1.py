"""
實際上傳影片到 API 並印出分析結果。

範例:
    python 1.py --video /path/to/demo.mp4
"""

import argparse
import json
import mimetypes
import os
from copy import deepcopy
from pathlib import Path

import requests

DEFAULT_VIDEO = (
    Path(__file__).resolve().parent.parent / "video/離開吸菸區吸菸/Video_非管制區吸.mp4"
)
DEFAULT_BASE_URL = os.environ.get("BACKEND_URL", "http://140.117.176.42:3000/api")
DEFAULT_API_KEY = os.environ.get("API_KEY") or os.environ.get("MY_API_KEY", "")
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="上傳影片到 /v1/segment_pipeline_multipart_vllm_video_direct 並顯示結果"
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="本機影片檔路徑")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="X-API-Key")
    parser.add_argument("--segment-duration", type=float, default=10.0, help="切片秒數")
    parser.add_argument("--overlap", type=float, default=0.0, help="切片重疊秒數")
    parser.add_argument("--qwen-model", default=DEFAULT_VLLM_MODEL, help="qwen model 名稱")
    parser.add_argument("--strict-segmentation", action="store_true", help="是否開 strict segmentation")
    parser.add_argument(
        "--qwen-inference-batch-size", type=int, default=None, help="每批 vLLM 推論片段數"
    )
    parser.add_argument("--save-json", action="store_true", help="要求 API 端存 json")
    parser.add_argument("--timeout", type=float, default=1200.0, help="HTTP timeout 秒數")
    return parser


def _drop_reid_embedding(obj):
    if isinstance(obj, dict):
        obj.pop("reid_embedding", None)
        for value in obj.values():
            _drop_reid_embedding(value)
    elif isinstance(obj, list):
        for item in obj:
            _drop_reid_embedding(item)


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"找不到影片檔: {video_path}")

    endpoint = f"{args.base_url.rstrip('/')}/v1/segment_pipeline_multipart_vllm_video_direct"
    data = {
        "segment_duration": str(args.segment_duration),
        "overlap": str(args.overlap),
        "qwen_model": args.qwen_model,
        "strict_segmentation": str(args.strict_segmentation).lower(),
        "save_json": str(args.save_json).lower(),
    }
    if args.qwen_inference_batch_size is not None:
        data["qwen_inference_batch_size"] = str(args.qwen_inference_batch_size)

    print("=== 開始呼叫 API ===")
    print(f"POST {endpoint}")
    print(f"video={video_path}")
    print(f"params={data}")

    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    def _post_once(url: str, payload: dict) -> requests.Response:
        with video_path.open("rb") as f:
            content_type, _ = mimetypes.guess_type(video_path.name)
            files = {"file": (video_path.name, f, content_type or "application/octet-stream")}
            return requests.post(
                url,
                data=payload,
                files=files,
                headers=headers,
                timeout=args.timeout,
            )

    response = _post_once(endpoint, data)

    print(f"\n=== HTTP {response.status_code} ===")
    try:
        payload = response.json()
    except ValueError:
        print(response.text)
        response.raise_for_status()
        return

    if response.status_code >= 400:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        response.raise_for_status()
        return

    print("=== 結果摘要 ===")
    print(f"model_type: {payload.get('model_type')}")
    print(f"total_segments: {payload.get('total_segments')}")
    print(f"success_segments: {payload.get('success_segments')}")
    print(f"total_time_sec: {payload.get('total_time_sec')}")
    if payload.get("save_path"):
        print(f"save_path: {payload.get('save_path')}")

    # 避免終端被 yolo/reid 大量向量淹沒，僅輸出精簡一行摘要。
    payload_for_print = deepcopy(payload)
    _drop_reid_embedding(payload_for_print)
    for result in payload_for_print.get("results", []):
        raw_detection = result.get("raw_detection")
        if not isinstance(raw_detection, dict):
            continue
        yolo_data = raw_detection.get("yolo")
        if not isinstance(yolo_data, dict):
            continue
        detections = yolo_data.get("detections", [])
        crop_paths = yolo_data.get("crop_paths", [])
        det_len = len(detections) if isinstance(detections, list) else 0
        crop_len = len(crop_paths) if isinstance(crop_paths, list) else 0
        yolo_data.clear()
        yolo_data["summary"] = (
            f"<omitted yolo detail: detections={det_len}, crop_paths={crop_len}, reid_embedding=removed>"
        )

    print("\n=== 完整回應 JSON ===")
    print(json.dumps(payload_for_print, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
