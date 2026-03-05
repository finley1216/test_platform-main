#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 _ollama_chat 的 raw 回傳值長相（與 analysis_service 第 127 行相同呼叫方式）。
用法：
  cd backend && PYTHONPATH=. python scripts/test_ollama_chat_raw.py
  cd backend && PYTHONPATH=. python scripts/test_ollama_chat_raw.py /path/to/video.mp4   # 用真實影片取幀
  cd backend && PYTHONPATH=. python scripts/test_ollama_chat_raw.py --example   # 只印出 raw 範例，不呼叫 Ollama
"""
import sys
from pathlib import Path

# 讓 backend 為根，才能 import src
backend = Path(__file__).resolve().parent.parent
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from src.config import config
from src.utils.ollama_utils import _ollama_chat
from src.utils.image_utils import _pil_to_b64, _resize_short_side
from PIL import Image
import io


# 當 Ollama 未啟動時，可用 --example 看 raw 的典型長相（字串，一個 JSON）
RAW_EXAMPLE = '''{
  "events": {
    "reason": "畫面中未見明顯災害或人員異常，為一般室內場景。"
  },
  "persons": [],
  "summary": "室內環境，無人員走動，無異常狀況。"
}'''


def _make_dummy_images_b64(count: int = 3, target_short: int = 432) -> list:
    """產生幾張小圖的 base64，模擬影片取幀，方便無影片時也能跑通。"""
    out = []
    for _ in range(count):
        # 一張 432x320 的灰圖（與 target_short 同短邊）
        img = Image.new("RGB", (432, 320), color=(128, 128, 128))
        img = _resize_short_side(img, target_short)
        out.append(_pil_to_b64(img))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--example" in sys.argv:
        print("--- 僅顯示 raw 範例（未呼叫 Ollama）---")
        print("raw 為一字串，內容通常為「一個 JSON 物件」或前後帶少許文字：\n")
        print("========== raw 範例內容 ==========\n")
        print(RAW_EXAMPLE)
        print("\n========== 結束 ==========")
        print("\n實際執行時請先啟動 Ollama 並拉好模型，再執行：")
        print("  cd backend && PYTHONPATH=. python scripts/test_ollama_chat_raw.py")
        return

    video_path = args[0] if args else None
    target_short = 432
    frames_per_segment = 5

    if video_path and Path(video_path).exists():
        from src.utils.video_utils import _sample_frames_evenly_to_pil
        print(f"使用影片取幀: {video_path}")
        frames_pil = _sample_frames_evenly_to_pil(video_path, max_frames=frames_per_segment)
        images_b64 = [_pil_to_b64(_resize_short_side(img, target_short)) for img in frames_pil]
    else:
        print("未提供影片或檔案不存在，使用 3 張 dummy 圖")
        images_b64 = _make_dummy_images_b64(count=3, target_short=target_short)

    model_name = getattr(config, "OLLAMA_LLM_MODEL", "qwen2.5vl:latest")
    event_prompt = "請根據畫面判斷是否有災害或人員異常，若有請簡述原因。"
    summary_prompt = "請產出 50–100 字繁體中文畫面摘要。"

    combined_instruction = (
        f"{event_prompt}\n\n"
        f"{summary_prompt}\n\n"
        "請輸出「一個」JSON 物件，且必須包含以下欄位：\n"
        "- events: 事件描述物件（可含 reason 等）\n"
        "- persons: 人員相關陣列\n"
        "- summary: 50–100 字繁體中文畫面摘要（字串）\n"
        "只輸出該 JSON，不要其他文字。"
    )
    combined_msgs = [
        {"role": "system", "content": "你是災害與人員異常偵測器，並產出影片摘要。只輸出一個 JSON 物件，含 events、persons、summary。"},
        {"role": "user", "content": combined_instruction},
    ]

    print(f"模型: {model_name}, 圖片數: {len(images_b64)}")
    print("--- 呼叫 _ollama_chat (與 analysis_service 第 127 行相同) ---")
    raw = _ollama_chat(model_name, combined_msgs, images_b64=images_b64)
    print("--- raw 長度:", len(raw), "---")
    print("\n========== raw 完整內容 ==========\n")
    print(raw)
    print("\n========== 結束 ==========")
    return raw


if __name__ == "__main__":
    main()
