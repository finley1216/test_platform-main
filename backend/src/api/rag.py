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

# 從 main.py 導入必要的函數和變數
# 注意：_save_results_to_postgres 在函數中使用時再導入，避免循環導入
from src.main import (
    get_api_key, HAS_DB, get_db,
    _parse_query_filters, get_embedding_model, EMBEDDING_MODEL_NAME,
    _event_cn_name, _derive_video_and_folder, _results_to_docs,
    _ollama_chat, Summary
)

@router.post("/rag/index")
async def rag_index(request: Request):
    """
    將指定的分析結果（JSON）寫入 PostgreSQL
    注意：數據已經通過 _save_results_to_postgres 自動保存，此 API 主要用於兼容性
    """
    # 手動獲取 db session（避免條件表達式導致 FastAPI 路由失敗）
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

    # 提取 video_stem（從 save_path 或 results 中）
    video_stem = "unknown"
    if save_path:
        # 從 save_path 提取：segment/video_name/video_name.json
        parts = Path(save_path).parts
        if "segment" in parts:
            idx = parts.index("segment")
            if idx + 1 < len(parts):
                video_stem = parts[idx + 1]
    elif isinstance(src_resp, dict) and "results" in src_resp:
        # 從 results 中提取第一個 segment 的 video 信息
        first_result = src_resp.get("results", [{}])[0]
        segment = first_result.get("segment", "")
        if segment:
            # 嘗試從 segment 文件名提取 video_stem
            video_stem = Path(segment).parent.name if "/" in segment else "unknown"
    
    # 轉換為 results 格式（如果是完整響應，提取 results）
    if isinstance(src_resp, dict) and "results" in src_resp:
        results = src_resp["results"]
    elif isinstance(src_resp, list):
        results = src_resp
    else:
        results = []

    # 保存到 PostgreSQL（數據已經包含 embedding）
    # 延遲導入 _save_results_to_postgres 以避免循環導入
    saved_count = 0
    if results:
        try:
            from src.main import _save_results_to_postgres
            _save_results_to_postgres(db, results, video_stem)
            saved_count = len([r for r in results if r.get("success", False)])
        except Exception as e:
            print(f"--- [WARNING] 保存到 PostgreSQL 失敗: {e} ---")

    # 從 PostgreSQL 查詢總數
    total = 0
    try:
        total = db.query(func.count(Summary.id)).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)
        ).scalar() or 0
    except Exception as e:
        print(f"--- [WARNING] 查詢總數失敗: {e} ---")

    return {
        "ok": True,
        "backend": EMBEDDING_MODEL_NAME,
        "removed_old": 0,  # PostgreSQL 使用更新或新增邏輯，不需要刪除
        "added": saved_count,
        "total": total
    }

@router.get("/rag/stats", dependencies=[Depends(get_api_key)])
def rag_stats():
    """
    計算目前 RAG 資料庫裡有多少筆資料。
    從 PostgreSQL 查詢有 embedding 的記錄數量。
    """
    # 手動獲取 db session（避免條件表達式導致 FastAPI 路由失敗）
    db = None
    if HAS_DB:
        try:
            db = next(get_db())
        except Exception as e:
            return {"count": 0, "error": str(e)}
    
    if not HAS_DB or not db:
        return {
            "count": 0,
            "path": "PostgreSQL (not available)"
        }
    
    try:
        count = db.query(func.count(Summary.id)).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)
        ).scalar() or 0
    except Exception as e:
        print(f"--- [WARNING] 查詢統計失敗: {e} ---")
        count = 0

    return {
        "count": count,
        "path": "PostgreSQL (summaries table)"
    }

@router.post("/rag/search")
async def rag_search(request: Request):
    """
    使用 PostgreSQL + pgvector 進行混合搜索
    - Filter 1 (Hard Filter): 時間範圍、事件類型、關鍵字過濾
    - Filter 2 (Vector Search): 使用 embedding 的 cosine_distance 進行語義搜索
    """
    # 手動獲取 db session（避免條件表達式導致 FastAPI 路由失敗）
    db = None
    if HAS_DB:
        try:
            db = next(get_db())
        except Exception as e:
            return {"hits": [], "error": f"Database not available: {str(e)}"}
    
    try:
        payload = await request.json()

        # query: 你要找什麼？
        query = (payload.get("query") or "").strip()
        top_k = int(payload.get("top_k") or 5)

        # [NEW] 新增分數門檻參數 (預設 0.0 代表不過濾，前端可傳 0.6 代表 60%)
        score_threshold = float(payload.get("score_threshold") or 0.0)

        if not query:
            raise HTTPException(status_code=422, detail="missing query")

        if not HAS_DB or not db:
            raise HTTPException(status_code=503, detail="Database not available")

        # 步驟 1: 解析查詢條件
        query_filters = {}
        date_info = None
        
        try:
            query_filters = _parse_query_filters(query)
        
            # 提取日期解析資訊，用於返回給前端
            if query_filters.get("date_filter") or query_filters.get("time_start"):
                date_filter = query_filters.get("date_filter")
                if date_filter:
                    if hasattr(date_filter, 'isoformat'):
                        picked_date_str = date_filter.isoformat()
                    else:
                        picked_date_str = str(date_filter)
                else:
                    picked_date_str = None
                
                date_info = {
                    "mode": query_filters.get("date_mode", "NONE"),
                    "picked_date": picked_date_str,
                    "time_start": query_filters.get("time_start"),
                    "time_end": query_filters.get("time_end"),
                }
                print(f"\n{'='*60}")
                print(f"[日期解析] 查詢: {query}")
                print(f"[日期解析] 模式: {date_info['mode']}")
                print(f"[日期解析] 解析到的日期: {date_info['picked_date']}")
                if date_info['time_start']:
                    print(f"[日期解析] 時間範圍: {date_info['time_start'][:19]} ~ {date_info['time_end'][:19]}")
                print(f"{'='*60}\n")
        except Exception as e:
            print(f"--- [WARNING] 查詢解析失敗: {e} ---")
            import traceback
            traceback.print_exc()
            query_filters = {}

        # 步驟 2: 生成查詢向量（去除日期後的 clean query）
        clean_query = query
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}',
            r'今天|昨天|明天|本週|上週|下週',
            r'\d{4}年\d{1,2}月\d{1,2}日',
        ]
        for pattern in date_patterns:
            clean_query = re.sub(pattern, '', clean_query)
        clean_query = clean_query.strip()
        
        if not clean_query:
            clean_query = query

        # 生成 embedding
        embedding_model = get_embedding_model()
        if not embedding_model:
            raise HTTPException(status_code=503, detail="Embedding model not available")
        
        query_embedding = embedding_model.encode(clean_query, normalize_embeddings=True)
        print(f"--- [DEBUG] 查詢向量生成完成 (維度: {len(query_embedding)}) ---")

        # 步驟 3: 構建 PostgreSQL + pgvector 混合查詢
        stmt = select(Summary).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)
        )

        # 時間範圍過濾
        if query_filters.get("time_start") and query_filters.get("time_end"):
            try:
                time_start_str = query_filters["time_start"]
                time_end_str = query_filters["time_end"]
                
                if "T" in time_start_str:
                    t0 = datetime.fromisoformat(time_start_str.replace("Z", "+00:00"))
                else:
                    t0 = datetime.fromisoformat(time_start_str)
                
                if "T" in time_end_str:
                    t1 = datetime.fromisoformat(time_end_str.replace("Z", "+00:00"))
                else:
                    t1 = datetime.fromisoformat(time_end_str)
                
                # 修正時區處理：資料庫中的時間是無時區的本地時間
                # 如果查詢時間帶時區，直接移除時區信息（保持時間值不變）
                # 這樣可以避免時區轉換導致的時間偏移
                if t0.tzinfo:
                    t0 = t0.replace(tzinfo=None)
                if t1.tzinfo:
                    t1 = t1.replace(tzinfo=None)
                
                stmt = stmt.filter(
                    Summary.start_timestamp >= t0,
                    Summary.start_timestamp < t1
                )
                print(f"--- [DEBUG] 應用時間範圍過濾: {t0} ~ {t1} ---")
            except Exception as e:
                print(f"--- [WARNING] 時間範圍解析失敗: {e} ---")
        elif query_filters.get("date_filter"):
            target_date = query_filters["date_filter"]
            t0 = datetime.combine(target_date, datetime.min.time())
            next_day = target_date + timedelta(days=1)
            t1 = datetime.combine(next_day, datetime.min.time())
            stmt = stmt.filter(
                Summary.start_timestamp >= t0,
                Summary.start_timestamp < t1
            )
            print(f"--- [DEBUG] 應用日期過濾: {target_date} ({t0} ~ {t1}) ---")

        # 事件類型過濾
        if query_filters.get("event_types"):
            event_conditions = []
            for event_type in query_filters["event_types"]:
                if event_type == "fire":
                    event_conditions.append(Summary.fire == True)
                elif event_type == "water_flood":
                    event_conditions.append(Summary.water_flood == True)
                elif event_type == "abnormal_attire_face_cover_at_entry":
                    event_conditions.append(Summary.abnormal_attire_face_cover_at_entry == True)
                elif event_type == "person_fallen_unmoving":
                    event_conditions.append(Summary.person_fallen_unmoving == True)
                elif event_type == "double_parking_lane_block":
                    event_conditions.append(Summary.double_parking_lane_block == True)
                elif event_type == "smoking_outside_zone":
                    event_conditions.append(Summary.smoking_outside_zone == True)
                elif event_type == "crowd_loitering":
                    event_conditions.append(Summary.crowd_loitering == True)
                elif event_type == "security_door_tamper":
                    event_conditions.append(Summary.security_door_tamper == True)
            
            if event_conditions:
                stmt = stmt.filter(or_(*event_conditions))
                print(f"--- [DEBUG] 應用事件過濾: {query_filters['event_types']} ---")

        # 關鍵字過濾（只在沒有事件類型過濾時使用，避免過度過濾）
        # 如果已經有事件類型過濾，關鍵字過濾可能會導致結果為 0（因為 message 可能不包含關鍵字）
        if query_filters.get("message_keywords") and not query_filters.get("event_types"):
            message_conditions = []
            for keyword in query_filters["message_keywords"]:
                message_conditions.append(Summary.message.ilike(f"%{keyword}%"))
            if message_conditions:
                stmt = stmt.filter(or_(*message_conditions))
                print(f"--- [DEBUG] 應用關鍵字過濾: {query_filters['message_keywords']} ---")
        elif query_filters.get("message_keywords") and query_filters.get("event_types"):
            print(f"--- [DEBUG] 跳過關鍵字過濾（已有事件類型過濾）: {query_filters['message_keywords']} ---")

        # Filter 2: Vector search - 使用 cosine_distance 排序
        try:
            from pgvector.sqlalchemy import Vector
            distance_expr = Summary.embedding.cosine_distance(query_embedding)
            
            stmt = stmt.add_columns(
                distance_expr.label('cosine_distance')
            ).order_by(
                distance_expr
            ).limit(top_k * 3)
            
            print(f"--- [DEBUG] 執行 PostgreSQL + pgvector 混合查詢 ---")
            results = db.execute(stmt).all()
            print(f"--- [DEBUG] 查詢返回 {len(results)} 筆結果 ---")
        except ImportError:
            raise HTTPException(status_code=503, detail="pgvector not available")
        except Exception as e:
            print(f"--- [ERROR] pgvector 查詢失敗: {e} ---")
            import traceback
            traceback.print_exc()
            results = []

        # 步驟 4: 計算相似度分數並格式化結果
        norm_hits = []
        for row in results:
            result = row[0]  # Summary 對象
            cosine_distance = row[1]  # cosine_distance 值
            
            # 將 cosine_distance 轉換為相似度分數
            score = max(0.0, min(1.0, 1.0 - (cosine_distance / 2.0)))

            # 過濾低於門檻的結果
            if score < score_threshold:
                continue

            # 構建事件列表
            events_true = []
            if result.fire:
                events_true.append("fire")
            if result.water_flood:
                events_true.append("water_flood")
            if result.abnormal_attire_face_cover_at_entry:
                events_true.append("abnormal_attire_face_cover_at_entry")
            if result.person_fallen_unmoving:
                events_true.append("person_fallen_unmoving")
            if result.double_parking_lane_block:
                events_true.append("double_parking_lane_block")
            if result.smoking_outside_zone:
                events_true.append("smoking_outside_zone")
            if result.crowd_loitering:
                events_true.append("crowd_loitering")
            if result.security_door_tamper:
                events_true.append("security_door_tamper")

            norm_hits.append({
                "score": round(score, 4),
                "video": result.video or "",
                "segment": result.segment or "",
                "time_range": result.time_range or "",
                "events_true": events_true,
                "summary": result.message or "",
                "reason": result.event_reason or "",
                "doc_id": None,
            })

        # 取 top_k
        norm_hits = norm_hits[:top_k]
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")

        # 構建響應
        response = {"backend": EMBEDDING_MODEL_NAME, "hits": norm_hits}
        if date_info:
            response["date_parsed"] = date_info
        if query_filters.get("message_keywords"):
            response["keywords_found"] = query_filters["message_keywords"]
        if query_filters.get("event_types"):
            response["event_types_found"] = query_filters["event_types"]
        response["embedding_query"] = clean_query
        print(f"--- [DEBUG] Embedding 查詢文本: '{clean_query}' ---")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"--- [ERROR] 搜索失敗: {e} ---")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/rag/answer")
async def rag_answer(request: Request):
    """
    使用 PostgreSQL + pgvector 進行混合搜索，然後使用 LLM 生成回答
    搜索邏輯與 /rag/search 完全相同
    """
    # 手動獲取 db session（避免條件表達式導致 FastAPI 路由失敗）
    db = None
    if HAS_DB:
        try:
            db = next(get_db())
        except Exception as e:
            return {"answer": "資料庫連接失敗", "error": str(e)}
    
    try:
        payload = await request.json()
        question = (payload.get("query") or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail="missing query")

        top_k = int(payload.get("top_k") or 5)
        score_threshold = float(payload.get("score_threshold") or 0.0)
        llm_model = (payload.get("model") or "qwen2.5vl:latest").strip()

        if not HAS_DB or not db:
            raise HTTPException(status_code=503, detail="Database not available")

        # 步驟 1: 解析查詢條件（與 /rag/search 統一）
        query_filters = {}
        date_info = None
        
        try:
            query_filters = _parse_query_filters(question)
        
            # 提取日期解析資訊，用於返回給前端
            if query_filters.get("date_filter") or query_filters.get("time_start"):
                date_filter = query_filters.get("date_filter")
                if date_filter:
                    if hasattr(date_filter, 'isoformat'):
                        picked_date_str = date_filter.isoformat()
                    else:
                        picked_date_str = str(date_filter)
                else:
                    picked_date_str = None
                
                date_info = {
                    "mode": query_filters.get("date_mode", "NONE"),
                    "picked_date": picked_date_str,
                    "time_start": query_filters.get("time_start"),
                    "time_end": query_filters.get("time_end"),
                }
                
                if date_info:
                    print(f"\n{'='*60}")
                    print(f"[日期解析] 查詢: {question}")
                    print(f"[日期解析] 模式: {date_info['mode']}")
                    print(f"[日期解析] 解析到的日期: {date_info['picked_date']}")
                    if date_info.get('time_start'):
                        print(f"[日期解析] 時間範圍: {date_info['time_start'][:19]} ~ {date_info.get('time_end', '')[:19]}")
                    print(f"{'='*60}\n")
        except Exception as e:
            print(f"--- [WARNING] 查詢解析失敗: {e} ---")
            import traceback
            traceback.print_exc()
            query_filters = {}

        # 步驟 2: 生成查詢向量（與 /rag/search 相同）
        clean_query = question
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}',
            r'今天|昨天|明天|本週|上週|下週',
            r'\d{4}年\d{1,2}月\d{1,2}日',
        ]
        for pattern in date_patterns:
            clean_query = re.sub(pattern, '', clean_query)
        clean_query = clean_query.strip()
        
        if not clean_query:
            clean_query = question

        embedding_model = get_embedding_model()
        if not embedding_model:
            raise HTTPException(status_code=503, detail="Embedding model not available")
        
        query_embedding = embedding_model.encode(clean_query, normalize_embeddings=True)
        print(f"--- [DEBUG] 查詢向量生成完成 (維度: {len(query_embedding)}) ---")

        # 步驟 3: 構建 PostgreSQL + pgvector 混合查詢（與 /rag/search 相同）
        stmt = select(Summary).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)
        )

        # 時間範圍過濾
        if query_filters.get("time_start") and query_filters.get("time_end"):
            try:
                time_start_str = query_filters["time_start"]
                time_end_str = query_filters["time_end"]
                
                if "T" in time_start_str:
                    t0 = datetime.fromisoformat(time_start_str.replace("Z", "+00:00"))
                else:
                    t0 = datetime.fromisoformat(time_start_str)
                
                if "T" in time_end_str:
                    t1 = datetime.fromisoformat(time_end_str.replace("Z", "+00:00"))
                else:
                    t1 = datetime.fromisoformat(time_end_str)
                
                # 修正時區處理：資料庫中的時間是無時區的本地時間
                # 如果查詢時間帶時區，直接移除時區信息（保持時間值不變）
                # 這樣可以避免時區轉換導致的時間偏移
                if t0.tzinfo:
                    t0 = t0.replace(tzinfo=None)
                if t1.tzinfo:
                    t1 = t1.replace(tzinfo=None)
                
                stmt = stmt.filter(
                    Summary.start_timestamp >= t0,
                    Summary.start_timestamp < t1
                )
                print(f"--- [DEBUG] 應用時間範圍過濾: {t0} ~ {t1} ---")
            except Exception as e:
                print(f"--- [WARNING] 時間範圍解析失敗: {e} ---")
        elif query_filters.get("date_filter"):
            target_date = query_filters["date_filter"]
            t0 = datetime.combine(target_date, datetime.min.time())
            next_day = target_date + timedelta(days=1)
            t1 = datetime.combine(next_day, datetime.min.time())
            stmt = stmt.filter(
                Summary.start_timestamp >= t0,
                Summary.start_timestamp < t1
            )
            print(f"--- [DEBUG] 應用日期過濾: {target_date} ({t0} ~ {t1}) ---")

        # 事件類型過濾
        if query_filters.get("event_types"):
            event_conditions = []
            for event_type in query_filters["event_types"]:
                if event_type == "fire":
                    event_conditions.append(Summary.fire == True)
                elif event_type == "water_flood":
                    event_conditions.append(Summary.water_flood == True)
                elif event_type == "abnormal_attire_face_cover_at_entry":
                    event_conditions.append(Summary.abnormal_attire_face_cover_at_entry == True)
                elif event_type == "person_fallen_unmoving":
                    event_conditions.append(Summary.person_fallen_unmoving == True)
                elif event_type == "double_parking_lane_block":
                    event_conditions.append(Summary.double_parking_lane_block == True)
                elif event_type == "smoking_outside_zone":
                    event_conditions.append(Summary.smoking_outside_zone == True)
                elif event_type == "crowd_loitering":
                    event_conditions.append(Summary.crowd_loitering == True)
                elif event_type == "security_door_tamper":
                    event_conditions.append(Summary.security_door_tamper == True)
            
            if event_conditions:
                stmt = stmt.filter(or_(*event_conditions))
                print(f"--- [DEBUG] 應用事件過濾: {query_filters['event_types']} ---")

        # 關鍵字過濾（只在沒有事件類型過濾時使用，避免過度過濾）
        # 如果已經有事件類型過濾，關鍵字過濾可能會導致結果為 0（因為 message 可能不包含關鍵字）
        if query_filters.get("message_keywords") and not query_filters.get("event_types"):
            message_conditions = []
            for keyword in query_filters["message_keywords"]:
                message_conditions.append(Summary.message.ilike(f"%{keyword}%"))
            if message_conditions:
                stmt = stmt.filter(or_(*message_conditions))
                print(f"--- [DEBUG] 應用關鍵字過濾: {query_filters['message_keywords']} ---")
        elif query_filters.get("message_keywords") and query_filters.get("event_types"):
            print(f"--- [DEBUG] 跳過關鍵字過濾（已有事件類型過濾）: {query_filters['message_keywords']} ---")

        # Vector search
        try:
            from pgvector.sqlalchemy import Vector
            distance_expr = Summary.embedding.cosine_distance(query_embedding)
            
            stmt = stmt.add_columns(
                distance_expr.label('cosine_distance')
            ).order_by(
                distance_expr
            ).limit(top_k * 3)
            
            print(f"--- [DEBUG] 執行 PostgreSQL + pgvector 混合查詢 ---")
            results = db.execute(stmt).all()
            print(f"--- [DEBUG] 查詢返回 {len(results)} 筆結果 ---")
        except ImportError:
            raise HTTPException(status_code=503, detail="pgvector not available")
        except Exception as e:
            print(f"--- [ERROR] pgvector 查詢失敗: {e} ---")
            import traceback
            traceback.print_exc()
            results = []

        # 步驟 4: 計算相似度分數並格式化結果（與 /rag/search 相同）
        norm_hits = []
        for row in results:
            result = row[0]
            cosine_distance = row[1]
            
            score = max(0.0, min(1.0, 1.0 - (cosine_distance / 2.0)))

            if score < score_threshold:
                continue

            events_true = []
            if result.fire:
                events_true.append("fire")
            if result.water_flood:
                events_true.append("water_flood")
            if result.abnormal_attire_face_cover_at_entry:
                events_true.append("abnormal_attire_face_cover_at_entry")
            if result.person_fallen_unmoving:
                events_true.append("person_fallen_unmoving")
            if result.double_parking_lane_block:
                events_true.append("double_parking_lane_block")
            if result.smoking_outside_zone:
                events_true.append("smoking_outside_zone")
            if result.crowd_loitering:
                events_true.append("crowd_loitering")
            if result.security_door_tamper:
                events_true.append("security_door_tamper")

            norm_hits.append({
                "score": round(score, 4),
                "video": result.video or "",
                "segment": result.segment or "",
                "time_range": result.time_range or "",
                "events_true": events_true,
                "summary": result.message or "",
                "reason": result.event_reason or "",
                "doc_id": None,
            })

        norm_hits = norm_hits[:top_k]
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")
        
        # 2. 組裝 Context (A) - 使用 norm_hits 作為上下文
        if not norm_hits:
            try:
                msgs = [
                    {"role": "system", "content": "你只能根據系統提供的資料回答。現在沒有資料可用，請直接說你找不到答案。"},
                    {"role": "user", "content": question},
                ]
                rj = _ollama_chat(llm_model, msgs, timeout=1800)
                msg = ""
                if isinstance(rj, dict):
                    msg = (rj.get("message") or {}).get("content", "").strip()
                elif isinstance(rj, str):
                    msg = rj
            except Exception as ollama_error:
                print(f"--- [WARNING] Ollama 失敗，返回空結果: {ollama_error} ---")
                msg = "目前索引到的片段裡找不到答案（或是相似度過低）。LLM 服務暫時無法使用。"

            response = {
                "backend": {"embed_model": EMBEDDING_MODEL_NAME, "llm": llm_model},
                "hits": [],
                "answer": msg or "目前索引到的片段裡找不到答案（或是相似度過低）。",
            }
            if date_info:
                response["date_parsed"] = date_info
            if query_filters.get("message_keywords"):
                response["keywords_found"] = query_filters["message_keywords"]
            if query_filters.get("event_types"):
                response["event_types_found"] = query_filters["event_types"]
            response["embedding_query"] = clean_query
            return response

        # 組裝 Context (A) - 使用已經格式化好的 norm_hits
        context_blocks = []
        for i, hit in enumerate(norm_hits, start=1):
            summary = hit.get("summary", "")
            video = hit.get("video")
            time_range = hit.get("time_range")
            
            context_blocks.append(
                f"[{i}] 影片: {video}  時間: {time_range}\n摘要: {summary}"
            )

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "你是工廠監控影片說明助理，必須嚴格根據提供的片段摘要回答問題。"
            "如果資料中沒有答案，就回答「我在目前索引到的片段裡找不到相關資訊」。"
        )
        user_prompt = (
            f"使用下面這些片段摘要回答問題：\n\n{context_text}\n\n"
            f"問題：{question}\n\n"
            "請用繁體中文回答，並在回答中附上你參考的片段編號（例如 [1]、[2]）。"
        )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 3. 呼叫 LLM (G) - 如果失敗，返回搜尋結果
        try:
            ans_content = _ollama_chat(llm_model, msgs, timeout=1800)
            if isinstance(ans_content, dict):
                answer = (ans_content.get("message") or {}).get("content", "").strip()
            else:
                answer = str(ans_content).strip()
        except Exception as ollama_error:
            print(f"--- [WARNING] Ollama 失敗，返回搜尋結果: {ollama_error} ---")
            answer = f"抱歉，LLM 服務暫時無法使用（錯誤：{str(ollama_error)[:100]}）。以下是根據您的查詢找到的相關片段：\n\n"
            for i, hit in enumerate(norm_hits, 1):
                answer += f"[{i}] 影片: {hit.get('video', 'N/A')}  時間: {hit.get('time_range', 'N/A')}\n"
                answer += f"    摘要: {hit.get('summary', 'N/A')[:100]}...\n\n"

        # [NEW] 在響應中包含日期解析資訊和關鍵字資訊
        response = {
            "backend": {"embed_model": EMBEDDING_MODEL_NAME, "llm": llm_model},
            "hits": norm_hits,
            "answer": answer,
        }
        if date_info:
            response["date_parsed"] = date_info
        if query_filters.get("message_keywords"):
            response["keywords_found"] = query_filters["message_keywords"]
        if query_filters.get("event_types"):
            response["event_types_found"] = query_filters["event_types"]
        response["embedding_query"] = clean_query
        return response

    except Exception as e:
        print(f"--- [RAG Answer Error] ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"RAG Answer Failed: {str(e)}", "detail": str(e)}
        )

