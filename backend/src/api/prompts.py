# -*- coding: utf-8 -*-
"""
Prompt 管理相關 API
"""

from fastapi import APIRouter
from pathlib import Path
from src.main import EVENT_DETECTION_PROMPT, SUMMARY_PROMPT

router = APIRouter(tags=["Prompt 管理"])

@router.get("/prompts/defaults")
def get_default_prompts():
    """回傳後端設定的預設 Prompts（動態讀取文件，無需重啟服務）"""
    # 動態讀取 prompt 文件，而不是使用啟動時緩存的變數
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    
    def _read_prompt_file(filename: str) -> str:
        """讀取 prompt 文件"""
        file_path = prompts_dir / filename
        try:
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                # 清理：去掉 BOM 與首尾空白
                if content and content[0] == "\ufeff":
                    content = content[1:]
                return content.strip()
            else:
                # 如果文件不存在，回退到緩存的變數
                return EVENT_DETECTION_PROMPT if filename == "frame_prompt.md" else SUMMARY_PROMPT
        except Exception as e:
            # 讀取失敗時回退到緩存的變數
            print(f"[警告] 無法讀取 {filename}：{e}，使用緩存值")
            return EVENT_DETECTION_PROMPT if filename == "frame_prompt.md" else SUMMARY_PROMPT
    
    return {
        "event_prompt": _read_prompt_file("frame_prompt.md"),
        "summary_prompt": _read_prompt_file("summary_prompt.md")
    }

