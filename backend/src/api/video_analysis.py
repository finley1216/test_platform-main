# -*- coding: utf-8 -*-
"""
影片分析相關 API
"""
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.config import config
from src.services.video_service import VideoService
from src.services.analysis_service import AnalysisService
from src.utils.video_utils import _fmt_hms

# 延遲導入以解決循環引用
def _get_db_and_models():
    from src.database import get_db
    from src.main import get_api_key, _save_results_to_postgres, HAS_DB, SegmentAnalysisRequest
    return get_db, get_api_key, _save_results_to_postgres, HAS_DB, SegmentAnalysisRequest

router = APIRouter(tags=["影片分析"])

@router.post("/v1/analyze_segment_result")
def analyze_segment_result_api(req: dict, api_key: str = Depends(lambda: _get_db_and_models()[1]())):
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
    get_db, get_api_key, _save_results_to_postgres, HAS_DB, _ = _get_db_and_models()
    
    # 1. 處理上傳與下載
    target_filename = "unknown_video"
    local_path = None
    cleanup = False
    
    if video_id and video_id.strip():
        local_path = None
    elif file is not None:
        target_filename = file.filename or "video.mp4"
        fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(target_filename).suffix)
        with os.fdopen(fd, "wb") as f: f.write(file.file.read())
        local_path, cleanup = tmp, True
    elif video_url:
        target_filename = Path(video_url).name or "video_url.mp4"
        local_path, cleanup = VideoService.download_to_temp(video_url), True
    else:
        raise HTTPException(status_code=422, detail="需要 file、video_url 或 video_id")

    try:
        # 2. 準備片段
        seg_dir, seg_files, stem, total_duration = VideoService.prepare_segments(
            local_path, video_id, target_filename, segment_duration, overlap, target_short, strict_segmentation
        )

        # 3. 執行 Pipeline
        results = AnalysisService.run_full_pipeline(
            seg_files, total_duration, segment_duration, overlap,
            model_type, qwen_model, frames_per_segment, target_short, sampling_fps,
            event_detection_prompt, summary_prompt, yolo_labels, yolo_every_sec, yolo_score_thr
        )

        # 4. 存檔與資料庫
    resp = {
        "model_type": model_type,
        "total_segments": len(results),
            "success_segments": sum(1 for r in results if r.get("success")),
        "results": results,
    }

        if save_json:
            save_path = seg_dir / (save_basename or f"{stem}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(resp, f, ensure_ascii=False, indent=2)

    if HAS_DB:
            db = next(get_db())
            _save_results_to_postgres(db, results, stem)

        return JSONResponse(resp)

            finally:
        if cleanup and local_path and os.path.exists(local_path):
            os.remove(local_path)
