#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下載 moondream/starmie-v1 tokenizer 到 models/moondream3-preview/starmie-v1-tokenizer/。
僅需在有網路時執行一次，之後離線或 Docker 內即可使用 Moondream3。
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TARGET_DIR = REPO_ROOT / "models" / "moondream3-preview" / "starmie-v1-tokenizer"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("請先安裝: pip install huggingface_hub")
        sys.exit(1)

    if (TARGET_DIR / "tokenizer.json").exists():
        print(f"已存在 tokenizer，略過下載。目錄: {TARGET_DIR}")
        return

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("下載 moondream/starmie-v1 tokenizer ...")
    path = snapshot_download(
        repo_id="moondream/starmie-v1",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"完成。Tokenzier 已存至: {path}")


if __name__ == "__main__":
    main()
