# -*- coding: utf-8 -*-
import json
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from src.config import config

def _ollama_chat(model_name: str, messages: List[Dict], images_b64: Optional[List[str]] = None, stream: bool = False) -> str:
    """與 Ollama API 進行對話"""
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 1024
        }
    }
    if images_b64:
        for msg in messages:
            if msg["role"] == "user":
                msg["images"] = images_b64
                break
    
    url = f"{config.OLLAMA_BASE.rstrip('/')}/api/chat"
    try:
        r = requests.post(url, json=payload, timeout=3600)
        r.raise_for_status()
        if stream:
            # 這裡簡單處理串流，實際使用可能需要更複雜的邏輯
            full_txt = ""
            for line in r.iter_lines():
                if line:
                    j = json.loads(line)
                    full_txt += j.get("message", {}).get("content", "")
                    if j.get("done"): break
            return full_txt
        else:
            return r.json().get("message", {}).get("content", "")
    except Exception as e:
        print(f"--- [Ollama] ✗ 請求失敗: {e} ---")
        raise

def _safe_parse_json(text: str) -> Optional[Dict]:
    """嘗試安全地解析 JSON"""
    if not text: return None
    try:
        return json.loads(text)
    except:
        return None

def _extract_first_json(text: str) -> Optional[Dict]:
    """從文字中提取第一個 JSON 物件"""
    if not text: return None
    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except:
        pass
    return None

def _extract_first_json_and_tail(text: str) -> Tuple[Dict, str]:
    """提取 JSON 及其後的文字（用於解析多重輸出）"""
    if not text: return {}, ""
    try:
        match = re.search(r'(\{.*\})(.*)', text, re.DOTALL)
        if match:
            return json.loads(match.group(1)), match.group(2).strip()
    except:
        pass
    return {}, text

def _clean_summary_text(s: str) -> str:
    """清理摘要文字"""
    if not s: return ""
    s = re.sub(r'```[a-z]*\n?', '', s)
    s = s.replace('```', '')
    s = re.sub(r'\{.*\}', '', s, flags=re.DOTALL)
    return s.strip()
