# -*- coding: utf-8 -*-
"""
Prompt 管理相關 API
"""

from fastapi import APIRouter
from prompts import get_event_detection_prompt, get_summary_prompt

router = APIRouter(tags=["Prompt 管理"])


@router.get("/prompts/defaults")
def get_default_prompts():
    """回傳目前磁碟上的預設 Prompts（每次請求讀取 prompts/*.md，與 pipeline 一致）"""
    return {
        "event_prompt": get_event_detection_prompt(),
        "summary_prompt": get_summary_prompt(),
    }
