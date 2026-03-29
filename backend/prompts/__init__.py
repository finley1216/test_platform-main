# prompts/__init__.py
# -*- coding: utf-8 -*-
"""預設 prompt：每次呼叫皆從與本套件同目錄的 .md 讀取，避免 worker 啟動後改檔仍沿用舊內容。"""
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

_FALLBACK_EVENT = "請根據提供的影格輸出事件 JSON。"
_FALLBACK_SUMMARY = (
    "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。"
)


def _read_prompt_file(name: str, encoding: str = "utf-8") -> str:
    p = _PROMPTS_DIR / name
    data = p.read_text(encoding=encoding)
    if data and data[0] == "\ufeff":
        data = data[1:]
    return data.strip()


def get_event_detection_prompt() -> str:
    try:
        return _read_prompt_file("frame_prompt.md")
    except Exception as e:
        print(f"[prompts] 讀取 frame_prompt.md 失敗，使用後備字串：{e}")
        return _FALLBACK_EVENT


def get_summary_prompt() -> str:
    try:
        return _read_prompt_file("summary_prompt.md")
    except Exception as e:
        print(f"[prompts] 讀取 summary_prompt.md 失敗，使用後備字串：{e}")
        return _FALLBACK_SUMMARY


__all__ = ["get_event_detection_prompt", "get_summary_prompt"]
