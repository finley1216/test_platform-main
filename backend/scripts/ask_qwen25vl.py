#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
對已下載的 Qwen2.5-VL-7B-Instruct-AWQ 提問（純文字或附一張圖）。
需安裝: pip install transformers torch pillow "autoawq>=0.1.8"
（使用 AWQ 模型時 autoawq 至少要 0.1.8）
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


def main():
    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    parser = argparse.ArgumentParser(description="對 Qwen2.5-VL-7B（AWQ）提問")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF 模型 ID 或本地快取路徑（預設 AWQ）",
    )
    parser.add_argument("--question", "-q", type=str, help="你的問題（文字）")
    parser.add_argument("--image", "-i", type=str, help="選填：圖片路徑，搭配問題一起送給模型")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 或 cpu")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--interactive", action="store_true", help="互動模式：重複輸入問題直到輸入 quit")
    args = parser.parse_args()

    if not args.question and not args.interactive:
        print("請用 --question \"你的問題\" 或 --interactive 進入互動模式。", file=sys.stderr)
        print("範例: python ask_qwen25vl.py -q \"什麼是遷移學習？\"")
        print("範例: python ask_qwen25vl.py -q \"圖裡有幾個人\" -i /path/to/image.jpg")
        sys.exit(1)

    # 使用 AWQ 模型時建議 autoawq >= 0.1.8（若 import 失敗仍嘗試載入，由 transformers 內部處理）
    if "AWQ" in args.model or "awq" in args.model.lower():
        try:
            import autoawq
            ver = getattr(autoawq, "__version__", "0.0.0")
            parts = []
            for x in ver.split(".")[:3]:
                parts.append(int(x) if x.isdigit() else 0)
            parts = tuple((parts + [0, 0])[:3])
            if parts < (0, 1, 8):
                print("使用 AWQ 模型需要 autoawq >= 0.1.8，請執行: pip install -U autoawq", file=sys.stderr)
                sys.exit(1)
        except ImportError as e:
            print("提示: 無法 import autoawq（可能為 CUDA/依賴問題），仍嘗試載入模型。若失敗可改用完整版: --model Qwen/Qwen2.5-VL-7B-Instruct", file=sys.stderr)

    # 載入模型與處理器（AWQ 與 full 皆可用；優先使用 VL 專用類別）
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText

    print(f"載入模型: {args.model} ...")
    model = model_cls.from_pretrained(
        args.model,
        dtype=torch.float16,
        trust_remote_code=True,
    ).eval()
    model = model.to(args.device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    def run_turn(question: str, image_path: Optional[str] = None) -> str:
        if image_path:
            img = Image.open(image_path).convert("RGB")
            content = [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]
        else:
            content = [{"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]
        text_in = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if image_path:
            inputs = processor(
                text=[text_in],
                images=[img],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = processor(
                text=[text_in],
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
        if args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
        decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        if "<|im_start|>assistant" in decoded:
            decoded = decoded.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in decoded:
            decoded = decoded.split("<|im_end|>")[0]
        return decoded.strip()

    if args.interactive:
        print("進入互動模式，輸入問題後按 Enter；輸入 quit 或 exit 結束。")
        image_path = args.image
        while True:
            try:
                q = input("\n你: ").strip()
            except EOFError:
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            print("模型: ", end="", flush=True)
            out = run_turn(q, image_path)
            print(out)
        return

    out = run_turn(args.question, args.image if args.image else None)
    print(out)


if __name__ == "__main__":
    main()
