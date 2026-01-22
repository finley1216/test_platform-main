# -*- coding: utf-8 -*-
"""
影片分析相關 API
"""
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import psutil
import logging

from src.config import config
from src.services.video_service import VideoService
from src.services.analysis_service import AnalysisService
from src.utils.video_utils import _fmt_hms

# 設定日誌
logger = logging.getLogger(__name__)

# 全域計數器：正在執行的請求數量
active_requests_counter = 0

def get_active_requests():
    return active_requests_counter

# 延遲導入以解決循環引用
def _get_db_and_models():
    from src.database import get_db
    from src.main import get_api_key, _save_results_to_postgres, HAS_DB, SegmentAnalysisRequest, EVENT_DETECTION_PROMPT, SUMMARY_PROMPT
    return get_db, get_api_key, _save_results_to_postgres, HAS_DB, SegmentAnalysisRequest, EVENT_DETECTION_PROMPT, SUMMARY_PROMPT

router = APIRouter(tags=["影片分析"])

@router.post("/v1/analyze_segment_result")
def analyze_segment_result_api(req: dict, api_key: str = Depends(lambda: _get_db_and_models()[1])):
    """分析單一片段的 API 進入點"""
    # 這裡將 req 轉為 Service 需要的格式
    from src.main import SegmentAnalysisRequest
    request_obj = SegmentAnalysisRequest(**req)
    return AnalysisService.analyze_segment(request_obj)

@router.post("/v1/segment_pipeline_multipart")
def segment_pipeline_multipart(
    request: Request,
    model_type: str = Form(...),
    file: UploadFile = File(None),
    video_url: str = Form(None),
    video_id: str = Form(None),
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    qwen_model: str = Form("qwen3-vl:8b"),
    frames_per_segment: int = Form(8),
    target_short: int = Form(720),
    sampling_fps: Optional[float] = Form(None),
    strict_segmentation: bool = Form(False),
    yolo_labels: Optional[str] = Form(None),
    yolo_every_sec: float = Form(2.0),
    yolo_score_thr: float = Form(0.25),
    event_detection_prompt: str = Form(""),
    summary_prompt: str = Form(""),
    save_json: bool = Form(True),
    save_basename: str = Form(None),
):
    global active_requests_counter
    active_requests_counter += 1
    t0 = time.time()
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    
    logger.info(f"--- [Pipeline Start] Model: {model_type}, Qwen: {qwen_model}, Strict: {strict_segmentation} ---")
    if strict_segmentation:
        logger.warning("--- [Diagnostics] Performing Re-encoding (CPU Intensive) ---")

    get_db, get_api_key, _save_results_to_postgres, HAS_DB, _, DEF_EVENT_PROMPT, DEF_SUMMARY_PROMPT = _get_db_and_models()
    
    # 如果 prompt 為空，則使用預設值
    if not event_detection_prompt or not event_detection_prompt.strip():
        event_detection_prompt = DEF_EVENT_PROMPT
    if not summary_prompt or not summary_prompt.strip():
        summary_prompt = DEF_SUMMARY_PROMPT

    # 1. 處理上傳與下載
    target_filename = "unknown_video"
    local_path = None
    cleanup = False
    
    try:
        if video_id and video_id.strip():
            local_path = None
        elif file is not None:
            # 檢查檔案大小 (Point 5, 9)
            file_content = file.file.read()
            file_size = len(file_content)
            logger.info(f"--- [Upload Check] Filename: {file.filename}, Size: {file_size} bytes ---")
            
            if file_size == 0:
                logger.error(f"--- [Upload Error] File is empty (0 bytes). Upload blocked by firewall? ---")
                raise HTTPException(status_code=400, detail="File is empty (0 bytes). Upload blocked by firewall?")
                
            target_filename = file.filename or "video.mp4"
            fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(target_filename).suffix)
            with os.fdopen(fd, "wb") as f: 
                f.write(file_content)
            local_path, cleanup = tmp, True
        elif video_url:
            target_filename = Path(video_url).name or "video_url.mp4"
            local_path, cleanup = VideoService.download_to_temp(video_url), True
        else:
            raise HTTPException(status_code=422, detail="需要 file、video_url 或 video_id")

        # 2. 準備片段
        try:
            seg_dir, seg_files, stem, total_duration = VideoService.prepare_segments(
                local_path, video_id, target_filename, segment_duration, overlap, target_short, strict_segmentation
            )
        except ValueError as ve:
            logger.error(f"--- [FFmpeg Error] {ve} ---")
            raise HTTPException(status_code=500, detail=f"FFmpeg Processing Failed: {ve}")

        t1 = time.time()
        # 3. 執行 Pipeline
        try:
            results = AnalysisService.run_full_pipeline(
                seg_files, total_duration, segment_duration, overlap,
                model_type, qwen_model, frames_per_segment, target_short, sampling_fps,
                event_detection_prompt, summary_prompt, yolo_labels, yolo_every_sec, yolo_score_thr
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"--- [CUDA OOM] {e} ---")
            raise HTTPException(status_code=500, detail="CUDA Out of Memory: GPU is overloaded. Please try a smaller model or fewer frames.")
        except MemoryError as e:
            logger.error(f"--- [OOM] {e} ---")
            raise HTTPException(status_code=500, detail="System Out of Memory: CPU/RAM is overloaded.")
        
        t2 = time.time()
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info(f"--- [Pipeline Success] Time: {round(t2-t0, 2)}s, Mem Delta: {round(mem_after - mem_before, 2)}MB ---")

        # 4. 存檔與資料庫
        resp = {
            "model_type": model_type,
            "total_segments": len(results),
            "success_segments": sum(1 for r in results if r.get("success")),
            "results": results,
            "process_time_sec": round(t2 - t1, 2),
            "total_time_sec": round(t2 - t0, 2),
            "stem": stem,
            "diagnostics": {
                "mem_delta_mb": round(mem_after - mem_before, 2),
                "strict_mode": strict_segmentation
            }
        }

        if save_json:
            save_path_obj = seg_dir / (save_basename or f"{stem}.json")
            with open(save_path_obj, "w", encoding="utf-8") as f:
                json.dump(resp, f, ensure_ascii=False, indent=2)
            resp["save_path"] = str(save_path_obj)

        if HAS_DB:
            from src.database import SessionLocal
            db = SessionLocal()
            try:
                _save_results_to_postgres(db, results, stem)
            finally:
                db.close()

        return JSONResponse(resp)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"--- [Unexpected Error] {type(e).__name__}: {e} ---", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {type(e).__name__} - {str(e)}")
    finally:
        active_requests_counter -= 1
        if cleanup and local_path and os.path.exists(local_path):
            os.remove(local_path)

@router.post("/v1/search/image", dependencies=[Depends(lambda: _get_db_and_models()[1])])
def search_by_image(
    request: Request,
    api_key: str = Depends(lambda: _get_db_and_models()[1]),
    file: UploadFile = File(None),  # 上傳的查詢圖片
    top_k: int = Form(10),  # 返回前 K 個最相似的結果
    threshold: float = Form(0.7),  # 相似度閾值（0.0-1.0，越高越嚴格）
    label_filter: Optional[str] = Form(None),  # 可選：過濾特定類別（例如 "person"）
):
    """
    以圖搜圖 API：根據上傳的圖片，找到外表相似的物件 crops
    """
    get_db, _, _, HAS_DB, _, _, _ = _get_db_and_models()
    
    if not HAS_DB:
        raise HTTPException(status_code=503, detail="資料庫未連接，無法執行搜索")
    
    from src.database import SessionLocal
    db = SessionLocal()
    
    try:
        from src.models import ObjectCrop, Summary, HAS_PGVECTOR
        from src.main import generate_reid_embedding
        from sqlalchemy import text
        import numpy as np
        
        # 1. 生成查詢 embedding
        if file is None:
            raise HTTPException(status_code=422, detail="需要提供 file（圖片）")
            
        # 保存上傳的圖片到臨時文件
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(file.file.read())
            
            # 使用 ReID 生成 embedding
            query_embedding, embedding_type = generate_reid_embedding(tmp_path)
            
            if query_embedding is None:
                raise HTTPException(status_code=500, detail="無法生成圖片 ReID embedding")
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        # 2. 在資料庫中搜索相似的 crops
        if not HAS_PGVECTOR:
            raise HTTPException(status_code=500, detail="資料庫未安裝 pgvector，無法執行向量搜索")
            
        # 將 query_embedding 轉換為 PostgreSQL 向量格式
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # 構建 SQL 查詢
        cosine_distance_threshold = 1.0 - threshold
        embedding_column = "reid_embedding"
        
        sql_query = f"""
            SELECT 
                oc.id,
                oc.summary_id,
                oc.crop_path,
                oc.label,
                oc.score,
                oc.timestamp,
                oc.frame,
                oc.box,
                s.video,
                s.segment,
                s.time_range,
                1 - (oc.{embedding_column} <=> '{query_embedding_str}'::vector) as similarity
            FROM object_crops oc
            JOIN summaries s ON oc.summary_id = s.id
            WHERE oc.{embedding_column} IS NOT NULL
              AND (oc.{embedding_column} <=> '{query_embedding_str}'::vector) <= {cosine_distance_threshold}
        """
        
        if label_filter and label_filter.strip():
            label_filter_escaped = label_filter.strip().replace("'", "''")
            sql_query += f" AND oc.label = '{label_filter_escaped}'"
            
        sql_query += f" ORDER BY oc.{embedding_column} <=> '{query_embedding_str}'::vector ASC LIMIT {top_k}"
        
        # 執行查詢
        result = db.execute(text(sql_query))
        rows = result.fetchall()
        
        # 3. 格式化結果
        search_results = []
        for row in rows:
            (crop_id, summary_id, crop_path, label, score, timestamp, 
             frame, box, video, segment, time_range, similarity) = row
            
            search_results.append({
                "crop_id": crop_id,
                "summary_id": summary_id,
                "crop_path": crop_path,
                "label": label,
                "score": float(score) if score is not None else 0.0,
                "timestamp": float(timestamp) if timestamp is not None else 0.0,
                "frame": frame,
                "box": json.loads(box) if box else [],
                "video": video,
                "segment": segment,
                "time_range": time_range,
                "similarity": float(similarity)
            })
            
        return JSONResponse({
            "query_type": "image",
            "query_info": {"filename": file.filename},
            "results": search_results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"--- [Image Search] ✗ 搜索發生異常: {e} ---")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()