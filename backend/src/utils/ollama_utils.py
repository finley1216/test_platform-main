# -*- coding: utf-8 -*-
import json
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from src.config import config

def _ollama_chat(model_name: str, messages: List[Dict], images_b64: Optional[List[str]] = None, stream: bool = False) -> str:
    
    # 構建發送給 Ollama API 的 JSON 資料主體。
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            # qwen3 等有 thinking 的模型會先把 token 用在 reasoning，需留足空間給實際 content（JSON+摘要）
            "num_predict": 2048
        }
    }

    # 如果傳入的圖片 Base64 字串列表存在，則將其附加到最後一個 user message 的 images 欄位。
    if images_b64:
        for msg in messages:
            if msg["role"] == "user":
                msg["images"] = images_b64
                break
    
    # 構建 Ollama API 的 URL。
    url = f"{config.OLLAMA_BASE.rstrip('/')}/api/chat"
    try:

        # 設定 HTTP 請求超時時間，預設為 600 秒（10 分鐘）。
        timeout = getattr(config, "OLLAMA_REQUEST_TIMEOUT", 600)

        # 發送 HTTP POST 請求到 Ollama API。
        r = requests.post(url, json=payload, timeout=timeout)

        # 檢查 HTTP 請求是否成功。
        r.raise_for_status()

        # 如果 stream 為 True，則處理串流回應。
        if stream:

            # 初始化一個空字串，用於累積串流回應的文字。
            full_txt = ""
            for line in r.iter_lines():
                if line:
                    j = json.loads(line)
                    full_txt += j.get("message", {}).get("content", "")
                    if j.get("done"): break
            return full_txt

        # 非串流，直接等待模型全部跑完，一次性拿回 JSON 結果並取出文字內容。
        else:
            data = r.json()
            msg = data.get("message", {}) or {}
            content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
            # qwen3 等模型會把整段輸出放在 thinking，content 為空；改為用 thinking 當作可解析文字
            if not content and thinking:
                print("--- [Ollama] content 為空，改使用 thinking 作為回應內容以供解析 ---")
                content = thinking
            # 若內容仍為空，印出完整 API 回應以便除錯
            elif not content and data:
                print("--- [Ollama] 回應內容為空，完整 API 回應如下 ---")
                try:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:
                    print(data)
                print("--- [Ollama] 以上為空回應時的完整 body ---")
            return content
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
    """清理摘要文字，處理 Markdown code blocks 及可能的 JSON 格式"""
    if not s: return ""
    
    # 1. 移除 Markdown code blocks 標籤
    s = re.sub(r'```[a-z]*\n?', '', s)
    s = s.replace('```', '')
    
    # 2. 檢查是否為純 JSON (有些模型會把摘要包在 JSON 裡)
    try:
        # 尋找第一個 { 和最後一個 }
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_str = s[start:end+1]
            data = json.loads(json_str)
            if isinstance(data, dict):
                # 嘗試提取常見的摘要欄位
                for key in ['summary', 'summary_independent', 'description', 'content']:
                    if key in data and data[key]:
                        return str(data[key]).strip()
                # 如果是 JSON 但沒找到關鍵欄位，且 JSON 之外沒文字，則可能需要保留 JSON 或清理掉
                # 這裡我們先不清除 JSON，交給後續步驟
    except:
        pass

    # 3. 移除任何剩餘的 JSON 區塊（如果 JSON 之外還有文字）
    # 但要注意不要誤刪包含大括號的正常文字
    # 只有當大括號內看起來像 JSON 時才移除
    s = re.sub(r'\{[^{}]*"[^{}]*":.*\}', '', s, flags=re.DOTALL)
    
    return s.strip()
