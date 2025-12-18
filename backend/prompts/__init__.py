# prompts/__init__.py
# -*- coding: utf-8 -*-
from importlib.resources import files
from typing import Optional

def _read_text(name: str, encoding: str = "utf-8") -> str:
    """
    讀取與本套件同目錄下的文字檔，並做基本清理（移除 BOM、前後空白）。
    """
    data = files(__package__).joinpath(name).read_text(encoding=encoding)
    # 清理：去掉 BOM 與首尾空白
    if data and data[0] == "\ufeff":
        data = data[1:]
    return data.strip()

try:
    EVENT_DETECTION_PROMPT: str = _read_text("frame_prompt.md")
except Exception as e:
    raise RuntimeError(f"[prompts] 無法載入 frame_prompt.md：{e}")

try:
    SUMMARY_PROMPT: str = _read_text("summary_prompt.md")
except Exception as e:
    raise RuntimeError(f"[prompts] 無法載入 summary_prompt.md：{e}")

__all__ = ["EVENT_DETECTION_PROMPT", "SUMMARY_PROMPT"]
