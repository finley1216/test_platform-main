# -*- coding: utf-8 -*-
"""
健康檢查和認證相關 API
"""

import requests
from fastapi import APIRouter, Depends, Security
from fastapi.security import APIKeyHeader
from src.main import get_api_key, _now_ts, ADMIN_TOKEN, OLLAMA_BASE

router = APIRouter(tags=["健康檢查和認證"])

@router.get("/health")
def health():
    """Ping 的功能，確認 API 還有在運行"""
    return {"ok": True, "time": _now_ts()}

@router.get("/auth/verify")
def auth_verify(api_key: str = Depends(get_api_key)):
    """
    驗證 Key 是否有效 (前端 checkAuth 用)
    簡單回傳 200 OK，代表 Header 裡的 Key 是正確的。
    如果 Key 錯誤，get_api_key 會直接拋出 401 異常，根本進不到這裡。
    """
    is_admin = api_key == ADMIN_TOKEN
    return {"ok": True, "message": "Key is valid", "is_admin": is_admin}

@router.get("/health/ollama")
def check_ollama_status():
    """
    檢查 Ollama 服務狀態
    返回 Ollama 服務是否可用、可用模型列表等資訊
    """
    result = {
        "ollama_base": OLLAMA_BASE,
        "status": "unknown",
        "available": False,
        "models": [],
        "error": None,
        "timestamp": _now_ts()
    }
    
    try:
        # 檢查 Ollama 服務是否可用
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        # 使用較短的 timeout 並確保是同步調用，FastAPI 會在 threadpool 中執行此函式
        response = requests.get(url, timeout=5.0)
        
        if response.status_code == 200:
            data = response.json()
            result["status"] = "available"
            result["available"] = True
            result["models"] = [model.get("name", "") for model in data.get("models", [])]
            result["model_count"] = len(result["models"])
        else:
            result["status"] = f"error_{response.status_code}"
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.ConnectionError as e:
        result["status"] = "connection_error"
        result["error"] = f"無法連接到 Ollama 服務 ({OLLAMA_BASE})"
    except requests.exceptions.Timeout as e:
        result["status"] = "timeout"
        result["error"] = f"連接 Ollama 服務超時"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"檢查 Ollama 狀態時發生錯誤：{str(e)}"
    
    return result

