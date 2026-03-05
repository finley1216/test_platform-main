# -*- coding: utf-8 -*-
"""
RTSP 錄影產出的 10 秒短片 → 呼叫 test_platform 後端 API 並行處理。

流程：
1. 從 test_platform-main/backend/prompts 載入 event_detection_prompt / summary_prompt
2. 解析度與抽幀：target_short=432（對應 768×432）、每段 5 幀送 VLM（frames_per_segment=5）
3. 掃描 video/ 下各路的 .mp4，過濾已送過的片段
4. 並行 POST /v1/segment_pipeline_multipart 上傳每個片段

使用方式（在 test_platform-main 專案根目錄或 rtsp-recorder 目錄執行）：
  python rtsp-recorder/run_api_pipeline.py
  python rtsp-recorder/run_api_pipeline.py --once
  python rtsp-recorder/run_api_pipeline.py --workers 8 --interval 20

環境變數：
  BACKEND_URL    後端 API base（例 http://140.117.176.88:3000/api）
  API_KEY 或 MY_API_KEY
  VIDEO_LIB_DIR  錄影目錄（預設 PROJECT_ROOT/video 或 rtsp-recorder/video）
  SEGMENT_DURATION  每段秒數（預設 10）
"""
import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 專案根 = test_platform-main
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 載入 .env（rtsp-recorder 或 backend 的 .env）
for _env_dir in [_REPO_ROOT / "rtsp-recorder", _REPO_ROOT / "backend", _REPO_ROOT]:
    _env = _env_dir / ".env"
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env, override=False)
        except ImportError:
            pass
        break

# 後端 API
DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://140.117.176.88:3000/api")
API_KEY = os.environ.get("API_KEY") or os.environ.get("MY_API_KEY", "")
# 錄影目錄（與 recorder 的 output.directory 一致）
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(_REPO_ROOT)))
def _default_video_root() -> Path:
    if os.environ.get("VIDEO_LIB_DIR"):
        return Path(os.environ["VIDEO_LIB_DIR"])
    for candidate in [PROJECT_ROOT / "video", _REPO_ROOT / "rtsp-recorder" / "video", _REPO_ROOT / "video"]:
        if candidate.exists():
            return candidate
    return _REPO_ROOT / "rtsp-recorder" / "video"

# 預設參數（與 API 對接）
SEGMENT_DURATION = float(os.environ.get("SEGMENT_DURATION", "10.0"))
FRAMES_PER_SEGMENT = 5
TARGET_SHORT = 432   # 768×432 短邊
QWEN_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5vl:latest")


def load_prompts_from_backend() -> tuple:
    """從 test_platform-main/backend/prompts 讀取 frame_prompt.md、summary_prompt.md。"""
    prompts_dir = _REPO_ROOT / "backend" / "prompts"
    event_path = prompts_dir / "frame_prompt.md"
    summary_path = prompts_dir / "summary_prompt.md"
    event_prompt = ""
    summary_prompt = ""
    if event_path.exists():
        event_prompt = event_path.read_text(encoding="utf-8").strip()
    else:
        event_prompt = "請根據提供的影格輸出事件 JSON。"
    if summary_path.exists():
        summary_prompt = summary_path.read_text(encoding="utf-8").strip()
    else:
        summary_prompt = "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。"
    return event_prompt, summary_prompt


def get_video_root() -> Path:
    """錄影輸出路徑（與 recorder 的 output.directory 一致）。"""
    root = _default_video_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_processed_state(state_path: Path) -> set:
    """已送 API 的 (video_stem, segment_name) 集合。"""
    if not state_path.exists():
        return set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return set(tuple(x) for x in data.get("processed", []))
    except Exception:
        return set()


def save_processed_state(state_path: Path, processed: set) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"processed": [list(p) for p in sorted(processed)]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _stem_to_video_stem(segment_name: str) -> str:
    """從檔名 stem 推回 video_stem（例：Test_Site-Ch1-2025-02-22_12-00-00 → Test_Site-Ch1）。"""
    import re
    m = re.match(r"^(.+)-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", segment_name)
    return m.group(1) if m else "default"


def list_pending_clips(
    video_root: Path,
    processed: set,
    segment_duration: float = 10.0,
    min_age_sec: float = 2.0,
    min_size_bytes: int = 1,
    extensions: tuple = (".mp4", ".avi"),
) -> list:
    """列出尚未送 API 的短片。回傳 [(video_stem, clip_path, segment_name), ...]
    支援兩種目錄結構：
    - 子目錄：video/Test_Site-Ch1/xxx.avi → video_stem=Test_Site-Ch1
    - 扁平：video/Test_Site-Ch1-2025-02-22_12-00-00.avi（recorder 預設）→ video_stem 從檔名推得
    略過小於 min_size_bytes 的檔案（避免 0 位元或寫入中造成後端 400）。
    """
    if not video_root.exists():
        return []
    pending = []
    now = time.time()
    # 1) 子目錄結構：video/<stream>/file.avi
    subdirs = [d for d in video_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for stream_dir in sorted(subdirs):
        video_stem = stream_dir.name
        for p in sorted(stream_dir.iterdir()):
            if p.suffix.lower() not in extensions:
                continue
            try:
                if p.stat().st_size < min_size_bytes:
                    continue
            except OSError:
                continue
            if min_age_sec > 0:
                try:
                    if now - p.stat().st_mtime < min_age_sec:
                        continue
                except OSError:
                    pass
            segment_name = p.stem
            if (video_stem, segment_name) in processed:
                continue
            pending.append((video_stem, str(p), segment_name))
    # 2) 扁平結構：video/Test_Site-Ch1-2025-02-22_12-00-00.avi（與 recorder 一致）
    if not subdirs:
        for p in sorted(video_root.iterdir()):
            if not p.is_file() or p.suffix.lower() not in extensions:
                continue
            if p.name.startswith("_"):
                continue
            try:
                if p.stat().st_size < min_size_bytes:
                    continue
            except OSError:
                continue
            if min_age_sec > 0:
                try:
                    if now - p.stat().st_mtime < min_age_sec:
                        continue
                except OSError:
                    pass
            segment_name = p.stem
            video_stem = _stem_to_video_stem(segment_name)
            if (video_stem, segment_name) in processed:
                continue
            pending.append((video_stem, str(p), segment_name))
    return pending


def upload_segment_to_api(
    base_url: str,
    api_key: str,
    clip_path: str,
    video_stem: str,
    segment_name: str,
    event_prompt: str,
    summary_prompt: str,
    segment_duration: float = 10.0,
    frames_per_segment: int = 5,
    target_short: int = 432,
    qwen_model: str = "qwen2.5vl:latest",
    max_retries: int = 3,
) -> dict:
    """上傳單一 10s 短片到 POST /v1/segment_pipeline_multipart。503/429 時自動重試。"""
    url = f"{base_url.rstrip('/')}/v1/segment_pipeline_multipart"
    headers = {"X-API-Key": api_key} if api_key else {}
    p = Path(clip_path)
    filename = p.name if p.suffix else f"{segment_name}.mp4"
    mime = "video/mp4" if p.suffix.lower() == ".mp4" else "video/x-msvideo"
    data = {
        "model_type": "qwen",
        "segment_duration": str(segment_duration),
        "overlap": "0.0",
        "qwen_model": qwen_model,
        "frames_per_segment": str(frames_per_segment),
        "target_short": str(target_short),
        "strict_segmentation": "False",
        "yolo_labels": "person,car",
        "yolo_every_sec": "2.0",
        "yolo_score_thr": "0.25",
        "event_detection_prompt": event_prompt,
        "summary_prompt": summary_prompt,
        "save_json": "True",
    }
    last_exc = None
    for attempt in range(max_retries):
        try:
            with open(clip_path, "rb") as f:
                files = {"file": (filename, f, mime)}
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


def run_once(
    video_root: Path,
    state_path: Path,
    base_url: str,
    api_key: str,
    event_prompt: str,
    summary_prompt: str,
    segment_duration: float = 10.0,
    frames_per_segment: int = 5,
    target_short: int = 432,
    qwen_model: str = "qwen2.5vl:latest",
    min_age_sec: float = 2.0,
    max_workers: int = 4,
) -> None:
    processed = load_processed_state(state_path)
    pending = list_pending_clips(video_root, processed, segment_duration, min_age_sec)
    if not pending:
        print("--- [API Pipeline] 無待處理短片 ---", flush=True)
        return
    print(f"--- [API Pipeline] 待處理: {len(pending)} 支，並行數: {max_workers} ---", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for video_stem, clip_path, segment_name in pending:
            fut = ex.submit(
                upload_segment_to_api,
                base_url,
                api_key,
                clip_path,
                video_stem,
                segment_name,
                event_prompt,
                summary_prompt,
                segment_duration,
                frames_per_segment,
                target_short,
                qwen_model,
            )
            futures[fut] = (video_stem, segment_name)
        for fut in as_completed(futures):
            video_stem, segment_name = futures[fut]
            try:
                result = fut.result()
                processed.add((video_stem, segment_name))
                done += 1
                total = result.get("total_time_sec") or result.get("process_time_sec")
                print(f"  [OK] {video_stem} / {segment_name}  total_time_sec={total}")
            except Exception as e:
                print(f"  [FAIL] {video_stem} / {segment_name}  {e}")
    if done:
        save_processed_state(state_path, processed)
    print(f"--- [API Pipeline] 本輪完成: {done}/{len(pending)} ---", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="RTSP 錄影短片 → 並行呼叫 test_platform API 處理（prompt 來自 backend/prompts，768×432、每段 5 幀）"
    )
    parser.add_argument("--once", action="store_true", help="只處理一批後結束")
    parser.add_argument("--interval", type=float, default=1.0, help="監看間隔（秒）")
    parser.add_argument("--min-age", type=float, default=2.0, help="只處理發佈超過 N 秒的檔案")
    parser.add_argument("--workers", type=int, default=10, help="並行上傳數（過高易造成後端 503）")
    parser.add_argument("--segment-duration", type=float, default=SEGMENT_DURATION, help="每段秒數")
    parser.add_argument("--frames", type=int, default=FRAMES_PER_SEGMENT, help="每段送 VLM 幀數")
    parser.add_argument("--short", type=int, default=TARGET_SHORT, help="VLM 短邊像素（768×432 用 432）")
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL, help="後端 API base URL")
    args = parser.parse_args()

    video_root = get_video_root()
    state_path = video_root / "_api_pipeline_processed.json"
    event_prompt, summary_prompt = load_prompts_from_backend()
    print(f"--- [API Pipeline] 影片目錄: {video_root} ---", flush=True)
    print(f"--- [API Pipeline] 後端: {args.backend}, workers: {args.workers}, frames: {args.frames}, short: {args.short} ---", flush=True)
    if not API_KEY:
        print("--- [API Pipeline] 未設定 API_KEY/MY_API_KEY，部分後端可能拒絕請求 ---", flush=True)

    if args.once:
        run_once(
            video_root,
            state_path,
            args.backend,
            API_KEY,
            event_prompt,
            summary_prompt,
            segment_duration=args.segment_duration,
            frames_per_segment=args.frames,
            target_short=args.short,
            min_age_sec=args.min_age,
            max_workers=args.workers,
        )
        return
    print("--- [API Pipeline] 持續監看，Ctrl+C 結束 ---", flush=True)
    while True:
        run_once(
            video_root,
            state_path,
            args.backend,
            API_KEY,
            event_prompt,
            summary_prompt,
            segment_duration=args.segment_duration,
            frames_per_segment=args.frames,
            target_short=args.short,
            min_age_sec=args.min_age,
            max_workers=args.workers,
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
