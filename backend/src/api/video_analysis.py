# -*- coding: utf-8 -*-
"""
影片分析相關 API
包含：片段分析、影片切割、完整分析流程
"""

import os
import re
import json
import tempfile
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

# 從 main.py 導入必要的函數和變數
# 注意：由於循環導入，這些導入會在運行時解析
import sys
from pathlib import Path
_main_module_path = Path(__file__).parent.parent / "main.py"
if str(_main_module_path.parent) not in sys.path:
    sys.path.insert(0, str(_main_module_path.parent))

# 延遲導入以避免循環導入
def _get_main_imports():
    from src.main import (
        get_api_key, _fmt_hms, _probe_duration_seconds, _split_one_video,
        _download_to_temp, infer_segment_qwen, infer_segment_gemini, infer_segment_owl,
        EVENT_DETECTION_PROMPT, SUMMARY_PROMPT, AUTO_RAG_INDEX, VIDEO_LIB_DIR,
        _auto_index_to_rag, _save_results_to_postgres, HAS_DB, get_db
    )
    return {
        'get_api_key': get_api_key, '_fmt_hms': _fmt_hms,
        '_probe_duration_seconds': _probe_duration_seconds, '_split_one_video': _split_one_video,
        '_download_to_temp': _download_to_temp, 'infer_segment_qwen': infer_segment_qwen,
        'infer_segment_gemini': infer_segment_gemini, 'infer_segment_owl': infer_segment_owl,
        'EVENT_DETECTION_PROMPT': EVENT_DETECTION_PROMPT, 'SUMMARY_PROMPT': SUMMARY_PROMPT,
        'AUTO_RAG_INDEX': AUTO_RAG_INDEX, 'VIDEO_LIB_DIR': VIDEO_LIB_DIR,
        '_auto_index_to_rag': _auto_index_to_rag, '_save_results_to_postgres': _save_results_to_postgres,
        'HAS_DB': HAS_DB, 'get_db': get_db
    }

# 直接導入（Python 的導入機制會處理循環導入）
from src.main import (
    get_api_key, _fmt_hms, _probe_duration_seconds, _split_one_video,
    _download_to_temp, infer_segment_qwen, infer_segment_gemini, infer_segment_owl,
    EVENT_DETECTION_PROMPT, SUMMARY_PROMPT, AUTO_RAG_INDEX, VIDEO_LIB_DIR,
    _auto_index_to_rag, _save_results_to_postgres, HAS_DB, get_db
)

router = APIRouter(tags=["影片分析"])

# 從 main.py 導入 SegmentAnalysisRequest
from src.main import SegmentAnalysisRequest

@router.post("/v1/analyze_segment_result", dependencies=[Depends(get_api_key)])
def analyze_segment_result(req: SegmentAnalysisRequest):
    """
    這就是您要求的 API：輸入單一片段資訊，輸出該片段的分析結果 (JSON + Summary)
    """
    p = req.segment_path
    tr = f"{_fmt_hms(req.start_time)} - {_fmt_hms(req.end_time)}"
    t1 = time.time()

    # 回傳結構初始化
    result = {
        "segment": Path(p).name,
        "time_range": tr,
        "duration_sec": round(req.end_time - req.start_time, 2),
        "success": False,
        "time_sec": 0.0,
        "parsed": {},
        "raw_detection": None,
        "error": None
    }

    try:
        # ==================== Qwen / Gemini 邏輯 ====================
        if req.model_type in ("qwen", "gemini"):
            # 1. 執行推論
            if req.model_type == "qwen":
                frame_obj, summary_txt = infer_segment_qwen(
                    req.qwen_model, p, req.event_detection_prompt, req.summary_prompt,
                    target_short=req.target_short, frames_per_segment=req.frames_per_segment,
                    sampling_fps=req.sampling_fps
                )
            else: # gemini
                # 自動判斷模型名稱
                g_model = req.qwen_model if req.qwen_model.startswith("gemini") else "gemini-2.5-flash"
                frame_obj, summary_txt = infer_segment_gemini(
                    g_model, p, req.event_detection_prompt, req.summary_prompt,
                    req.target_short, req.frames_per_segment, sampling_fps=req.sampling_fps
                )

            # 2. 資料清洗與標準化 (Normalization)
            frame_norm = {
                "events": {
                    "water_flood": False, "fire": False,
                    "abnormal_attire_face_cover_at_entry": False,
                    "person_fallen_unmoving": False,
                    "double_parking_lane_block": False,
                    "smoking_outside_zone": False,
                    "crowd_loitering": False,
                    "security_door_tamper": False,
                    "reason": ""
                }
            }

            if isinstance(frame_obj, dict) and "error" not in frame_obj:
                ev = frame_obj.get("events") or {}
                defaults = frame_norm["events"]

                # [動態欄位更新] 支援使用者自訂 Prompt
                for k, v in ev.items():
                    if k == "reason": continue
                    try:
                        defaults[k] = bool(v)
                    except: pass

                # [Reason 排序修正] 刪除再新增，確保排在最後
                reason_text = str(ev.get("reason", "") or "")
                if "reason" in defaults: del defaults["reason"]
                defaults["reason"] = reason_text

            # 填寫成功結果
            result["success"] = ("error" not in (frame_obj or {})) and \
                                (not req.summary_prompt.strip() or len((summary_txt or "").strip()) > 0)
            result["parsed"] = {
                "frame_analysis": frame_norm,
                "summary_independent": (summary_txt or "").strip()
            }

        # ==================== OWL 邏輯 ====================
        elif req.model_type == "owl":
            j = infer_segment_owl(p, labels=req.owl_labels, every_sec=req.owl_every_sec, score_thr=req.owl_score_thr)
            result["success"] = True
            result["raw_detection"] = j

        else:
            raise ValueError("model_type must be qwen, gemini, or owl")

    except Exception as ex:
        result["error"] = str(ex)

    result["time_sec"] = round(time.time() - t1, 2)
    return result

@router.post("/v1/segment_video", dependencies=[Depends(get_api_key)])
def segment_video(
    request: Request,
    api_key: str = Depends(get_api_key),
    file: UploadFile = File(None),
    video_url: str = Form(None),
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    sampling_fps: float = Form(0.5),
    resolution: int = Form(720),
    output_dir: Optional[str] = Form(None),  # 可選：自訂輸出目錄
):
    """
    專門用於切割影片的 API，嚴格遵循所有參數：
    
    1. Segment Duration (s): 嚴格每 N 秒切割一段（除了最後一段可能較短）
    2. Overlap (s): 嚴格重疊設定的秒數
    3. Sampling Rate (FPS): 嚴格遵循，例如 0.5 fps = 每2秒取1 frame
    4. Resolution (px): 強制設定影片長邊解析度
    
    參數:
    - file: 上傳的影片檔案
    - video_url: 影片 URL（與 file 二選一）
    - segment_duration: 每段秒數（嚴格遵循）
    - overlap: 重疊秒數（嚴格遵循）
    - sampling_fps: 取樣 FPS（嚴格遵循，例如 0.5 = 每2秒取1 frame）
    - resolution: 長邊解析度（px，嚴格遵循）
    
    返回:
    - segments: 切割後的影片片段列表
    - segment_info: 每個片段的詳細資訊（路徑、時間範圍、取樣幀數等）
    """
    # 1. 下載或獲取影片
    if file is not None:
        target_filename = file.filename or "video.mp4"
        fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(file.filename or "video.mp4").suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(file.file.read())
        local_path, cleanup = tmp, True
    elif video_url:
        target_filename = Path(video_url).name or "video_url.mp4"
        local_path, cleanup = _download_to_temp(video_url), True
    else:
        raise HTTPException(status_code=422, detail="需要 file 或 video_url")
    
    # 2. 驗證參數
    if segment_duration <= 0:
        raise HTTPException(status_code=422, detail="segment_duration 必須大於 0")
    if overlap < 0:
        raise HTTPException(status_code=422, detail="overlap 不能為負數")
    if overlap >= segment_duration:
        raise HTTPException(status_code=422, detail="overlap 必須小於 segment_duration")
    if sampling_fps <= 0:
        raise HTTPException(status_code=422, detail="sampling_fps 必須大於 0")
    if resolution <= 0:
        raise HTTPException(status_code=422, detail="resolution 必須大於 0")
    
    try:
        # 3. 切割影片（使用嚴格模式）
        stem = Path(target_filename).stem
        if output_dir:
            # 使用自訂輸出目錄
            seg_dir = Path(output_dir)
        else:
            # 預設輸出目錄
            seg_dir = Path("segment") / f"{stem}_strict"
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"--- [INFO] 開始切割影片（嚴格模式）---")
        print(f"  - Segment Duration: {segment_duration} 秒")
        print(f"  - Overlap: {overlap} 秒")
        print(f"  - Sampling FPS: {sampling_fps} (每 {1.0/sampling_fps:.1f} 秒取1 frame)")
        print(f"  - Resolution: {resolution} px (長邊)")
        
        seg_files = _split_one_video(
            local_path, 
            seg_dir, 
            segment_duration, 
            overlap, 
            prefix="segment",
            resolution=resolution,
            strict_mode=True  # 使用嚴格模式
        )
        
        print(f"--- [INFO] 切割完成，共 {len(seg_files)} 個片段 ---")
        
        # 4. 為每個片段生成詳細資訊
        segment_info = []
        total_duration = _probe_duration_seconds(local_path)
        
        for i, seg_file in enumerate(seg_files):
            # 計算時間範圍
            start = i * (segment_duration - overlap)
            end = min(start + segment_duration, total_duration)
            
            # 驗證實際片段時長
            actual_duration = _probe_duration_seconds(seg_file)
            
            # 計算應該取樣的幀數（根據 FPS 設定）
            expected_frames = max(1, int(round(sampling_fps * actual_duration)))
            
            segment_info.append({
                "index": i,
                "path": str(seg_file),
                "relative_path": f"/segment/{Path(seg_file).parent.name}/{Path(seg_file).name}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "expected_duration": round(segment_duration, 3),
                "actual_duration": round(actual_duration, 3),
                "expected_frames": expected_frames,
                "sampling_fps": sampling_fps,
                "resolution": resolution,
            })
        
        # 5. 返回結果
        return {
            "status": "success",
            "message": f"成功切割 {len(seg_files)} 個片段",
            "segments": [str(f) for f in seg_files],
            "segment_info": segment_info,
            "total_segments": len(seg_files),
            "total_duration": round(total_duration, 3),
            "parameters": {
                "segment_duration": segment_duration,
                "overlap": overlap,
                "sampling_fps": sampling_fps,
                "resolution": resolution,
            }
        }
    
    except Exception as e:
        if cleanup and os.path.exists(local_path):
            os.remove(local_path)
        raise HTTPException(status_code=500, detail=f"切割失敗：{str(e)}")
    finally:
        if cleanup and os.path.exists(local_path):
            os.remove(local_path)

@router.post("/v1/segment_pipeline_multipart", dependencies=[Depends(get_api_key)])
def segment_pipeline_multipart(
    request: Request,
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db) if HAS_DB else None,
    model_type: str = Form(...),
    file: UploadFile = File(None),
    video_url: str = Form(None),
    video_id: str = Form(None),  # 新增：重新分析已存在的影片
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    qwen_model: str = Form("qwen3-vl:8b"),
    frames_per_segment: int = Form(8),
    target_short: int = Form(720),
    sampling_fps: Optional[float] = Form(None),  # 新增：取樣 FPS（如果提供則嚴格遵循）
    strict_segmentation: bool = Form(False),  # 新增：是否使用嚴格切割模式
    owl_labels: str = Form("person,pedestrian,motorcycle,car,bus,scooter,truck"),
    owl_every_sec: float = Form(2.0),
    owl_score_thr: float = Form(0.15),
    event_detection_prompt: str = Form(EVENT_DETECTION_PROMPT),
    summary_prompt: str = Form(SUMMARY_PROMPT),
    save_json: bool = Form(True),
    save_basename: str = Form(None),
):
    """
    它不親自做分析，而是負責調度資源與流程控制。影片，切割，片段影片填入標準格式，片段 API 處理，打包成大的 JSON
    """
    target_filename = "unknown_video"

    # 1. 下載與儲存 (維持原樣)
    # 如果提供了 video_id，表示要重新分析已存在的影片，跳過下載和切割
    if video_id and video_id.strip():
        # 使用已存在的影片，不需要下載或切割
        local_path = None
        cleanup = False
    elif file is not None:
        # [修正 1] 抓取原始檔名 (例如 "my_video.mp4")
        target_filename = file.filename or "video.mp4"
        fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(file.filename or "video.mp4").suffix)
        with os.fdopen(fd, "wb") as f: f.write(file.file.read())
        local_path, cleanup = tmp, True
    elif video_url:
        # [修正 2] 如果是 URL，從網址抓檔名
        target_filename = Path(video_url).name or "video_url.mp4"
        local_path, cleanup = _download_to_temp(video_url), True
    else:
        raise HTTPException(status_code=422, detail="需要 file、video_url 或 video_id")

    # 2. 切割影片 (如果沒有使用已存在的影片)
    # [修正 3] 使用 "原始檔名" 來當作 ID，而不是用 local_path 的亂碼檔名
    if video_id and video_id.strip():
        video_id_clean = video_id.strip()
        
        # 檢查是否為 video_lib 格式 (category/video_name)
        if "/" in video_id_clean:
            # 從 video 資料夾讀取原始影片
            category, video_name = video_id_clean.split("/", 1)
            
            # 檢查是否已經在 segment 中處理過（使用 {category}_{video_name} 作為 ID）
            stem = f"{category}_{video_name}"  # 使用分類和影片名作為 ID
            seg_dir = Path("segment") / stem
            
            if seg_dir.exists() and list(seg_dir.glob("segment_*.mp4")):
                # 已經處理過，直接使用現有的片段（不從 video 資料夾複製）
                seg_files = sorted(seg_dir.glob("segment_*.mp4"))
                try:
                    json_files = list(seg_dir.glob("*.json"))
                    if json_files:
                        with open(max(json_files, key=lambda p: p.stat().st_mtime), "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                            total_duration = sum(r.get("duration_sec", segment_duration) for r in old_data.get("results", []))
                    else:
                        total_duration = len(seg_files) * segment_duration
                except:
                    total_duration = len(seg_files) * segment_duration
            else:
                # 尚未處理過，需要從 video 資料夾讀取原始影片並切割
                video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
                
                # 嘗試其他擴展名
                if not video_path.exists():
                    for ext in ['.avi', '.mov', '.mkv', '.flv']:
                        video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                        if video_path.exists():
                            break
                
                if not video_path.exists():
                    raise HTTPException(status_code=404, detail=f"Video {video_id_clean} not found in video library")
                
                # 尚未處理過，需要切割影片
                seg_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # 複製影片到 segment 資料夾進行處理
                    import shutil
                    temp_video = seg_dir / video_path.name
                    shutil.copy2(video_path, temp_video)
                    seg_files = _split_one_video(
                        temp_video, 
                        seg_dir, 
                        segment_duration, 
                        overlap, 
                        prefix="segment",
                        resolution=target_short if strict_segmentation else None,
                        strict_mode=strict_segmentation
                    )
                    total_duration = _probe_duration_seconds(temp_video)
                    # 處理完後可以選擇刪除臨時副本（保留原始文件在 video 資料夾）
                    # os.remove(temp_video)  # 可選：刪除臨時副本
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"切割失敗：{e}")
        else:
            # 傳統的 segment 中的影片
            stem = video_id_clean
            seg_dir = Path("segment") / stem
            if not seg_dir.exists():
                raise HTTPException(status_code=404, detail=f"Video {video_id_clean} not found")
            
            # 查找已存在的片段影片
            seg_files_existing = sorted(seg_dir.glob("segment_*.mp4"))
            if not seg_files_existing:
                raise HTTPException(status_code=404, detail=f"No segment files found for video {video_id_clean}")
            
            seg_files = seg_files_existing
            # 估算總時長（從片段數量推斷，或從 JSON 讀取）
            try:
                json_files = list(seg_dir.glob("*.json"))
                if json_files:
                    with open(max(json_files, key=lambda p: p.stat().st_mtime), "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        total_duration = sum(r.get("duration_sec", segment_duration) for r in old_data.get("results", []))
                else:
                    total_duration = len(seg_files) * segment_duration
            except:
                total_duration = len(seg_files) * segment_duration
    else:
        stem = Path(target_filename).stem
        # 建立固定的資料夾 segment/video_1/
        seg_dir = Path("segment") / stem
        seg_dir.mkdir(parents=True, exist_ok=True)
        try:
            seg_files = _split_one_video(
                local_path, 
                seg_dir, 
                segment_duration, 
                overlap, 
                prefix="segment",
                resolution=target_short if strict_segmentation else None,
                strict_mode=strict_segmentation
            )
            total_duration = _probe_duration_seconds(local_path)
        except Exception as e:
            if cleanup and os.path.exists(local_path): os.remove(local_path)
            raise HTTPException(status_code=500, detail=f"切割失敗：{e}")

    # 3. 迴圈：Call API 取得結果
    results = []
    t0 = time.time()

    print(f"--- 開始處理 {len(seg_files)} 個片段，呼叫分析 API ---")

    for p in seg_files:
        # 3.1 計算時間區段資訊
        m = re.search(r"(\d+)", Path(p).name)
        idx = int(m.group(1)) if m else 0
        start = idx * (segment_duration - overlap)
        end = min(start + segment_duration, total_duration)

        # 3.2 準備參數 (Request Body)
        req_data = SegmentAnalysisRequest(
            segment_path=str(p), # 傳遞絕對路徑
            segment_index=idx,
            start_time=start,
            end_time=end,
            model_type=model_type,
            qwen_model=qwen_model,
            frames_per_segment=frames_per_segment,
            target_short=target_short,
            sampling_fps=sampling_fps,  # 傳遞取樣 FPS
            event_detection_prompt=event_detection_prompt,
            summary_prompt=summary_prompt,
            owl_labels=owl_labels,
            owl_every_sec=owl_every_sec,
            owl_score_thr=owl_score_thr
        )

        # 3.3 【關鍵步驟】Call API
        # 這裡直接呼叫函式，這等同於透過內部網路呼叫該 API，但更快
        res = analyze_segment_result(req_data)
        results.append(res)

    # 4. 統計與存檔 (維持原樣)
    total_time = time.time() - t0
    ok_count = sum(1 for r in results if r.get("success"))

    resp = {
        "model_type": model_type,
        "total_segments": len(results),
        "success_segments": ok_count,
        "total_time_sec": round(total_time, 2),
        "results": results,
    }

    try:
        if save_json:
            filename = save_basename or f"{stem}.json"

            save_path = seg_dir / filename
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(resp, f, ensure_ascii=False, indent=2)
            resp["save_path"] = str(save_path)

            if AUTO_RAG_INDEX:
                resp["rag_auto_indexed"] = _auto_index_to_rag(resp)
    except Exception: pass

    # 5. 保存分析結果到 PostgreSQL（與 RAG 同步）
    if HAS_DB and db:
        try:
            _save_results_to_postgres(db, results, stem)
        except Exception as e:
            print(f"--- [WARNING] 保存到 PostgreSQL 失敗: {e} ---")
            # 不中斷流程，只記錄警告

    if cleanup and os.path.exists(local_path):
        try: os.remove(local_path)
        except: pass

    return JSONResponse(resp, media_type="application/json; charset=utf-8")

