# -*- coding: utf-8 -*-
"""
健康檢查和認證相關 API
"""

from fastapi import APIRouter, Depends, Security
from fastapi.security import APIKeyHeader
from src.main import get_api_key, _now_ts, ADMIN_TOKEN

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

