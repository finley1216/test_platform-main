#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下載 Qwen2.5-VL-7B（Hugging Face 單一模型，不用 Ollama），並可查詢下載位置。

兩種版本差異：
- awq（預設）：Qwen2.5-VL-7B-Instruct-AWQ，4-bit 量化，推論較快、顯存較小，效益接近 Ollama 的 qwen2.5vl:7b。
- full：Qwen2.5-VL-7B-Instruct，完整精度（fp16/bf16），較慢、顯存較大。

需安裝: pip install huggingface_hub
"""
import argparse
import os
import sys
from pathlib import Path

# 對應 qwen2.5vl:7b 的量化版（AWQ），較快、省顯存
REPO_AWQ = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
# 完整精度版，較慢
REPO_FULL = "Qwen/Qwen2.5-VL-7B-Instruct"


def get_hf_cache_root():
    """取得 Hugging Face 快取根目錄（與 transformers / huggingface_hub 一致）。"""
    return os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")


def get_model_cache_dir(cache_root: str, repo_id: str) -> Path:
    """HF 標準快取下，該 repo 的目錄名稱（models--Org--Name）。"""
    safe = repo_id.replace("/", "--")
    return Path(cache_root) / f"models--{safe}"


def main():
    parser = argparse.ArgumentParser(
        description="下載 Qwen2.5-VL-7B（HF 單一模型）。預設為 AWQ 量化版（較快）。"
    )
    parser.add_argument(
        "--variant",
        choices=["awq", "full"],
        default="awq",
        help="awq=量化版（快、省顯存，預設）；full=完整精度（慢、顯存大）",
    )
    parser.add_argument("--repo", default=None, help="覆蓋為指定 HF repo id（若設則忽略 --variant）")
    parser.add_argument("--check-only", action="store_true", help="僅檢查並印出目前下載位置，不下載")
    parser.add_argument("--cache-dir", default=None, help="指定快取目錄（預設用 HF_HOME 或 ~/.cache/huggingface/hub）")
    args = parser.parse_args()

    repo_id = args.repo if args.repo else (REPO_AWQ if args.variant == "awq" else REPO_FULL)

    cache_root = args.cache_dir or get_hf_cache_root()
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    model_cache_dir = get_model_cache_dir(cache_root, repo_id)

    print(f"模型: {repo_id}")
    print(f"HF 快取根目錄: {cache_root}")
    print(f"此模型快取目錄: {model_cache_dir}")

    if args.check_only:
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_root,
                local_files_only=True,
            )
            print(f"狀態: 已下載")
            print(f"使用路徑: {path}")
        except Exception as e:
            if "Couldn't find" in str(e) or "does not exist" in str(e).lower() or "local_files_only" in str(e):
                print("狀態: 尚未下載（快取中無此模型）")
            else:
                print(f"狀態: 檢查時發生錯誤: {e}")
        return

    # 下載
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("請安裝: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"開始下載 {repo_id} ...")
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_root,
        local_dir=None,
        local_dir_use_symlinks=True,
    )
    print(f"下載完成。")
    print(f"使用路徑（snapshot）: {local_path}")
    print(f"快取根目錄: {cache_root}")
    print(f"此模型快取目錄: {model_cache_dir}")


if __name__ == "__main__":
    main()
