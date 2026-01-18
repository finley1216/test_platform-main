# -*- coding: utf-8 -*-
"""
RAG 相關 API
包含：搜索、回答、索引、統計
"""

import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, select
from datetime import datetime, timedelta

# 先定義 router，避免循環導入問題
router = APIRouter(tags=["RAG 相關 API"])

# 延遲導入以避免循環導入
def _get_main_imports():
    from src.main import (
        get_api_key, HAS_DB, get_db,
        _parse_query_filters, get_embedding_model, EMBEDDING_MODEL_NAME,
        _event_cn_name, _derive_video_and_folder, _results_to_docs,
        _ollama_chat, Summary
    )
    return {
        "get_api_key": get_api_key,
        "HAS_DB": HAS_DB,
        "get_db": get_db,
        "_parse_query_filters": _parse_query_filters,
        "get_embedding_model": get_embedding_model,
        "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
        "_event_cn_name": _event_cn_name,
        "_derive_video_and_folder": _derive_video_and_folder,
        "_results_to_docs": _results_to_docs,
        "_ollama_chat": _ollama_chat,
        "Summary": Summary
    }

@router.post("/rag/index")
async def rag_index(request: Request):
    """
    將指定的分析結果（JSON）寫入 PostgreSQL
    注意：數據已經通過 _save_results_to_postgres 自動保存，此 API 主要用於兼容性
    """
    imports = _get_main_imports()
    HAS_DB = imports["HAS_DB"]
    get_db = imports["get_db"]
    Summary = imports["Summary"]

    # 手動獲取 db session
    db = None
    if HAS_DB:
        try:
            db = next(get_db())
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Database not available: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Database not available")
    
    payload = await request.json()
    src_resp = payload.get("results")
    save_path = payload.get("save_path")

    if not src_resp and save_path:
        try:
            src_resp = json.loads(Path(save_path).read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"load save_path failed: {e}")
    if not src_resp:
        raise HTTPException(status_code=422, detail="missing results or save_path")

    video_stem = "unknown"
    if save_path:
        parts = Path(save_path).parts
        if "segment" in parts:
            idx = parts.index("segment")
            if idx + 1 < len(parts):
                video_stem = parts[idx + 1]
    elif isinstance(src_resp, dict) and "results" in src_resp:
        first_result = src_resp.get("results", [{}])[0]
        segment = first_result.get("segment", "")
        if segment:
            video_stem = Path(segment).parent.name if "/" in segment else "unknown"
    
    if isinstance(src_resp, dict) and "results" in src_resp:
        results = src_resp["results"]
    elif isinstance(src_resp, list):
        results = src_resp
    else:
        results = []

    saved_count = 0
    if results:
        try:
            from src.main import _save_results_to_postgres
            _save_results_to_postgres(db, results, video_stem)
            saved_count = len([r for r in results if r.get("success", False)])
        except Exception as e:
            print(f"--- [WARNING] 保存到 PostgreSQL 失敗: {e} ---")

    total = 0
    try:
        total = db.query(func.count(Summary.id)).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)
        ).scalar()
    except: pass

    return {
        "success": True, 
        "message": f"成功索引 {saved_count} 筆片段", 
        "total_in_db": total
    }

@router.get("/rag/stats")
async def rag_stats(api_key: str = Depends(lambda: _get_main_imports()["get_api_key"])):
    """獲取資料庫中的總記錄數"""
    imports = _get_main_imports()
    HAS_DB = imports["HAS_DB"]
    get_db = imports["get_db"]
    Summary = imports["Summary"]

    if not HAS_DB:
        return {"count": 0, "status": "no_db"}
    
    try:
        db = next(get_db())
        count = db.query(func.count(Summary.id)).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)
        ).scalar()
        return {"count": count, "status": "ok"}
    except Exception as e:
        return {"count": 0, "status": "error", "error": str(e)}

@router.post("/rag/search")
async def rag_search(
    request: Request, 
    api_key: str = Depends(lambda: _get_main_imports()["get_api_key"])
):
    """在 PostgreSQL 中執行語義搜尋"""
    imports = _get_main_imports()
    HAS_DB = imports["HAS_DB"]
    get_db = imports["get_db"]
    _parse_query_filters = imports["_parse_query_filters"]
    get_embedding_model = imports["get_embedding_model"]
    Summary = imports["Summary"]

    if not HAS_DB:
        raise HTTPException(status_code=503, detail="Database not available")
    
    payload = await request.json()
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 5))
    score_threshold = float(payload.get("score_threshold", 0.35))
    
    if not query:
        return {"hits": [], "count": 0}

    # 1. 解析查詢中的過濾條件（日期、事件類型、關鍵字）
    query_filters = _parse_query_filters(query)
    message_keywords = query_filters.get("message_keywords", [])
    
    # 2. 生成查詢向量
    try:
        model = get_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")

    # 3. 執行資料庫查詢
    try:
        db = next(get_db())
        
        # 使用 pgvector 的 <-> 運算符（L2 距離）或 <=>（餘弦相似度）
        # 這裡我們使用 1 - (embedding <=> query_vector) 來獲得餘弦相似度分數
        similarity = (1 - Summary.embedding.cosine_distance(query_vector)).label("similarity")
        
        # 建立基本查詢
        stmt = select(Summary, similarity).filter(
            Summary.embedding.isnot(None)
        )
        
        # 應用過濾條件
        # 日期/時間範圍過濾
        time_start = query_filters.get("time_start")
        time_end = query_filters.get("time_end")
        
        if time_start:
            # 轉換為 datetime 如果是字串
            if isinstance(time_start, str):
                t0 = datetime.fromisoformat(time_start.replace("Z", "+00:00")).replace(tzinfo=None)
            else:
                t0 = time_start
            stmt = stmt.filter(Summary.start_timestamp >= t0)
            
        if time_end:
            # 轉換為 datetime 如果是字串
            if isinstance(time_end, str):
                t1 = datetime.fromisoformat(time_end.replace("Z", "+00:00")).replace(tzinfo=None)
            else:
                t1 = time_end
            stmt = stmt.filter(Summary.start_timestamp < t1)
            
        # 事件類型過濾
        event_types = query_filters.get("event_types", [])
        if event_types:
            event_filters = []
            for event in event_types:
                if hasattr(Summary, event):
                    event_filters.append(getattr(Summary, event) == True)
            if event_filters:
                stmt = stmt.filter(or_(*event_filters))
        
        # 4. 關鍵字模糊匹配（如果有的話，與語義搜尋並行）
        # 語義搜尋分數門檻
        stmt = stmt.filter(similarity >= score_threshold)
        
        # 排序與限制數量
        stmt = stmt.order_by(similarity.desc()).limit(top_k * 2) # 多取一點以便後續排序
        
        results = db.execute(stmt).all()
        
        # 5. 格式化結果並應用關鍵字加分
        hits = []
        for summary, score in results:
            final_score = float(score)
            
            # [加分邏輯] 如果摘要中包含搜尋關鍵字，給予加分
            if message_keywords:
                msg_lower = (summary.message or "").lower()
                bonus = 0.0
                for kw in message_keywords:
                    if kw.lower() in msg_lower:
                        bonus += 0.2
                final_score = min(1.0, final_score + bonus)
            
            # [加分邏輯] 如果日期範圍精準匹配（代表日期過濾生效），給予基礎加分
            if time_start:
                final_score = min(1.0, final_score + 0.3)

            hits.append({
                "id": summary.id,
                "video": summary.video,
                "segment": summary.segment,
                "time_range": summary.time_range,
                "summary": summary.message,
                "score": final_score,
                "timestamp": summary.start_timestamp.isoformat() if summary.start_timestamp else None,
                "events": {
                    "fire": summary.fire,
                    "water_flood": summary.water_flood,
                    "person_fallen": summary.person_fallen_unmoving,
                    "double_parking": summary.double_parking_lane_block,
                    "smoking": summary.smoking_outside_zone,
                    "crowd": summary.crowd_loitering,
                    "security_door": summary.security_door_tamper,
                    "abnormal_attire": summary.abnormal_attire_face_cover_at_entry
                }
            })
            
        # 重新按分數排序並限制數量為 top_k
        hits.sort(key=lambda x: x["score"], reverse=True)
        hits = hits[:top_k]
            
        return {
            "hits": hits, 
            "count": len(hits),
            "date_parsed": {
                "mode": query_filters.get("date_mode"),
                "time_start": query_filters.get("time_start").isoformat() if query_filters.get("time_start") and hasattr(query_filters.get("time_start"), "isoformat") else str(query_filters.get("time_start")),
                "time_end": query_filters.get("time_end").isoformat() if query_filters.get("time_end") and hasattr(query_filters.get("time_end"), "isoformat") else str(query_filters.get("time_end"))
            } if query_filters.get("time_start") else None,
            "keywords_found": message_keywords,
            "event_types_found": [imports["_event_cn_name"](e) for e in query_filters.get("event_types", [])]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@router.post("/rag/answer")
async def rag_answer(
    request: Request, 
    api_key: str = Depends(lambda: _get_main_imports()["get_api_key"])
):
    """執行 RAG 流程：搜索相關片段 + 使用 LLM 總結回答"""
    imports = _get_main_imports()
    _ollama_chat = imports["_ollama_chat"]
    _results_to_docs = imports["_results_to_docs"]

    # 1. 執行搜索
    search_resp = await rag_search(request, api_key)
    hits = search_resp.get("hits", [])
    
    if not hits:
        return {
            "answer": "抱歉，在資料庫中找不到與您查詢相關的片段。",
            "hits": [],
            "success": True
        }
        
    # 2. 構建上下文
    context = _results_to_docs(hits)
    payload = await request.json()
    query = payload.get("query", "")
    
    prompt = f"""
    你是一個專業的安控系統助手。請根據以下從影片分析資料庫中檢索到的片段資訊，回答使用者的問題。
    
    檢索到的相關片段：
    {context}
    
    使用者問題：{query}
    
    請注意：
    1. 如果資訊不足以回答問題，請誠實說明。
    2. 請用繁體中文回答。
    3. 盡量提及具體的影片名稱和時間範圍。
    """
    
    # 3. 呼叫 Ollama 生成回答
    try:
        from src.config import config
        messages = [{"role": "user", "content": prompt}]
        answer = _ollama_chat(config.OLLAMA_EMBED_MODEL, messages)
        
        return {
            "answer": answer,
            "hits": hits,
            "success": True,
            "date_parsed": search_resp.get("date_parsed"),
            "keywords_found": search_resp.get("keywords_found"),
            "event_types_found": search_resp.get("event_types_found")
        }
    except Exception as e:
        return {
            "answer": f"生成回答時發生錯誤：{str(e)}",
            "hits": hits,
            "success": False
        }
