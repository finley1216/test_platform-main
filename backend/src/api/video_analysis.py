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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# 延遲導入 YOLO 函數以避免循環導入
def _get_infer_segment_yolo():
    from src.main import infer_segment_yolo
    return infer_segment_yolo

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
        "raw_detection": {},  # 改為空字典，避免 None 錯誤
        "error": None
    }

    try:
        # ==================== Qwen / Gemini 邏輯 ====================
        if req.model_type in ("qwen", "gemini"):
            # 1. 執行推論
            frame_obj = None
            summary_txt = None
            try:
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
            except Exception as infer_e:
                # 如果推論失敗，設置預設值
                frame_obj = {"error": str(infer_e)}
                summary_txt = ""
            
            # 確保 frame_obj 和 summary_txt 不是 None
            if frame_obj is None:
                frame_obj = {"error": "推論返回 None"}
            if summary_txt is None:
                summary_txt = ""

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

            # 確保 frame_norm["events"] 是字典，避免 None 錯誤
            if not isinstance(frame_norm.get("events"), dict):
                frame_norm["events"] = {
                    "water_flood": False, "fire": False,
                    "abnormal_attire_face_cover_at_entry": False,
                    "person_fallen_unmoving": False,
                    "double_parking_lane_block": False,
                    "smoking_outside_zone": False,
                    "crowd_loitering": False,
                    "security_door_tamper": False,
                    "reason": ""
                }

            if isinstance(frame_obj, dict) and "error" not in frame_obj:
                ev = frame_obj.get("events") or {}
                # 確保 ev 是字典
                if not isinstance(ev, dict):
                    ev = {}
                
                defaults = frame_norm.get("events")
                # 確保 defaults 是字典，避免 None 錯誤
                if not isinstance(defaults, dict):
                    defaults = {
                        "water_flood": False, "fire": False,
                        "abnormal_attire_face_cover_at_entry": False,
                        "person_fallen_unmoving": False,
                        "double_parking_lane_block": False,
                        "smoking_outside_zone": False,
                        "crowd_loitering": False,
                        "security_door_tamper": False,
                        "reason": ""
                    }
                    frame_norm["events"] = defaults

                # [動態欄位更新] 支援使用者自訂 Prompt
                for k, v in ev.items():
                    if k == "reason": continue
                    try:
                        if isinstance(defaults, dict):
                            defaults[k] = bool(v)
                    except Exception as e:
                        print(f"--- [WARNING] 設置欄位 {k} 失敗: {e} ---")
                        pass

                # [Reason 排序修正] 刪除再新增，確保排在最後
                reason_text = str(ev.get("reason", "") or "")
                if isinstance(defaults, dict):
                    if "reason" in defaults: 
                        del defaults["reason"]
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

        # ==================== YOLO 邏輯 ====================
        elif req.model_type == "yolo":
            infer_segment_yolo = _get_infer_segment_yolo()
            j = infer_segment_yolo(
                p, 
                labels=req.yolo_labels or "person,pedestrian,motorcycle,car,bus,scooter,truck",
                every_sec=req.yolo_every_sec,
                score_thr=req.yolo_score_thr
            )
            result["success"] = True
            result["raw_detection"] = j

        else:
            raise ValueError("model_type must be qwen, gemini, owl, or yolo")

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
    yolo_labels: Optional[str] = Form(None),  # YOLO 偵測類別
    yolo_every_sec: float = Form(2.0),  # YOLO 取樣頻率
    yolo_score_thr: float = Form(0.25),  # YOLO 信心門檻
    # [DEPRECATED] enable_yolo_parallel 已移除，現在每個片段都會自動執行 YOLO
    event_detection_prompt: str = Form(EVENT_DETECTION_PROMPT),
    summary_prompt: str = Form(SUMMARY_PROMPT),
    save_json: bool = Form(True),
    save_basename: str = Form(None),
):
    """
    它不親自做分析，而是負責調度資源與流程控制。影片，切割，片段影片填入標準格式，片段 API 處理，打包成大的 JSON
    """
    # 記錄總開始時間（包括上傳、下載、切割、處理）
    t0_total = time.time()
    
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
    # t0 用於計算處理時間（不包括上傳、切割）
    t0 = time.time()

    # VLM 和 YOLO 都已啟用
    SKIP_VLM = False  # 設為 True 時跳過 VLM 分析，只執行 YOLO
    SKIP_YOLO = False  # 設為 True 時跳過 YOLO 偵測，只執行 VLM
    
    print(f"--- 開始處理 {len(seg_files)} 個片段，呼叫分析 API ---")
    if SKIP_VLM:
        print(f"--- [注意] VLM (Ollama) 已暫時停用，只執行 YOLO 偵測 ---")
    elif SKIP_YOLO:
        print(f"--- [測試模式] YOLO 已停用，只執行 VLM (Ollama) 分析 ---")
    else:
        print(f"--- 處理模式: VLM ({model_type}) + YOLO (每個片段都會執行，並行處理) ---")

    def process_segment(seg_path, seg_idx):
        """處理單個片段：VLM 分析 + YOLO 偵測（優化版本：內存處理、批量推理）"""
        # 計算基本資訊（用於錯誤處理）
        m = re.search(r"(\d+)", Path(seg_path).name)
        idx = int(m.group(1)) if m else seg_idx
        start = idx * (segment_duration - overlap)
        end = min(start + segment_duration, total_duration)
        time_range_str = f"{_fmt_hms(start)} - {_fmt_hms(end)}"
        
        # 使用優化版本（內存處理、批量推理、共享解碼）
        try:
            from src.main_optimized import infer_segment_yolo_optimized
            from src.main import _yolo_world_model, get_reid_model
            
            # 獲取全局模型
            yolo_model = _yolo_world_model
            reid_model, reid_device = get_reid_model()
            
            # 3.1 計算時間區段資訊
            m = re.search(r"(\d+)", Path(seg_path).name)
            idx = int(m.group(1)) if m else seg_idx
            start = idx * (segment_duration - overlap)
            end = min(start + segment_duration, total_duration)
            time_range_str = f"{_fmt_hms(start)} - {_fmt_hms(end)}"

            # 3.2 準備 VLM 分析參數（如果未跳過 VLM）
            vlm_result = None  # 初始化 vlm_result，避免未定義錯誤
            if not SKIP_VLM:
                try:
                    req_data = SegmentAnalysisRequest(
                        segment_path=str(seg_path),
                        segment_index=idx,
                        start_time=start,
                        end_time=end,
                        model_type=model_type,
                        qwen_model=qwen_model,
                        frames_per_segment=frames_per_segment,
                        target_short=target_short,
                        sampling_fps=sampling_fps,
                        event_detection_prompt=event_detection_prompt,
                        summary_prompt=summary_prompt,
                        owl_labels=owl_labels,
                        owl_every_sec=owl_every_sec,
                        owl_score_thr=owl_score_thr,
                        yolo_labels=yolo_labels,
                        yolo_every_sec=yolo_every_sec,
                        yolo_score_thr=yolo_score_thr
                    )

                    # 3.3 執行 VLM 分析
                    vlm_result = analyze_segment_result(req_data)
                except Exception as vlm_e:
                    # 如果 VLM 分析失敗，創建錯誤結果
                    print(f"--- [ERROR] VLM 分析失敗 (segment {idx}): {vlm_e} ---")
                    import traceback
                    traceback.print_exc()
                    vlm_result = {
                        "segment": Path(seg_path).name,
                        "time_range": time_range_str,
                        "duration_sec": end - start,
                        "success": False,
                        "time_sec": 0.0,
                        "parsed": {},
                        "raw_detection": {},  # 改為空字典，避免 None 錯誤
                        "error": f"VLM 分析失敗: {str(vlm_e)}"
                    }
            else:
                # [跳過 VLM] 創建基本的結果結構，不執行 Ollama 分析
                print(f"--- [跳過 VLM] 片段 {idx}：只執行 YOLO 偵測 ---")
                vlm_result = {
                    "segment": Path(seg_path).name,
                    "time_range": time_range_str,
                    "duration_sec": end - start,
                    "success": True,  # YOLO 成功就算成功
                    "time_sec": 0.0,
                    "parsed": {
                        "frame_analysis": {
                            "events": {
                                "water_flood": False,
                                "fire": False,
                                "abnormal_attire_face_cover_at_entry": False,
                                "person_fallen_unmoving": False,
                                "double_parking_lane_block": False,
                                "smoking_outside_zone": False,
                                "crowd_loitering": False,
                                "security_door_tamper": False,
                                "reason": "VLM 分析已暫時停用（GPU 滿載）"
                            },
                            "persons": []
                        },
                        "summary_independent": "VLM 分析已暫時停用"
                    },
                    "raw_detection": {},
                    "error": None
                }
            
            # 確保 vlm_result 是有效的字典
            if vlm_result is None or not isinstance(vlm_result, dict):
                print(f"--- [ERROR] VLM 分析返回無效結果: {vlm_result} ---")
                import traceback
                traceback.print_exc()
                vlm_result = {
                    "segment": Path(seg_path).name,
                    "time_range": time_range_str,
                    "duration_sec": end - start,
                    "success": False,
                    "time_sec": 0.0,
                    "parsed": {
                        "frame_analysis": {
                            "events": {
                                "water_flood": False,
                                "fire": False,
                                "abnormal_attire_face_cover_at_entry": False,
                                "person_fallen_unmoving": False,
                                "double_parking_lane_block": False,
                                "smoking_outside_zone": False,
                                "crowd_loitering": False,
                                "security_door_tamper": False,
                                "reason": ""
                            }
                        },
                        "summary_independent": ""
                    },
                    "raw_detection": {},
                    "error": f"VLM 分析返回無效結果: {vlm_result}"
                }
            
            # 確保 vlm_result 有必要的欄位
            if "parsed" not in vlm_result:
                vlm_result["parsed"] = {
                    "frame_analysis": {
                        "events": {
                            "water_flood": False,
                            "fire": False,
                            "abnormal_attire_face_cover_at_entry": False,
                            "person_fallen_unmoving": False,
                            "double_parking_lane_block": False,
                            "smoking_outside_zone": False,
                            "crowd_loitering": False,
                            "security_door_tamper": False,
                            "reason": ""
                        }
                    },
                    "summary_independent": ""
                }
            # 確保 raw_detection 是字典，不是 None
            if "raw_detection" not in vlm_result or vlm_result["raw_detection"] is None:
                vlm_result["raw_detection"] = {}
            
            # 3.4 【可選執行】每個片段都要執行 YOLO 偵測（優化版本：內存處理、批量推理）
            yolo_result = None
            if SKIP_YOLO:
                print(f"--- [跳過 YOLO] 片段 {idx}：只執行 VLM 分析 ---")
            else:
                try:
                    # 如果全局模型未初始化，先初始化它（避免回退到慢速的舊版本）
                    if yolo_model is None:
                        print("--- [YOLO] 全局模型未初始化，先初始化模型（使用優化版本）---")
                        # 先調用一次舊版本來初始化模型（但我們會立即使用優化版本）
                        infer_segment_yolo = _get_infer_segment_yolo()
                        # 只初始化模型，不執行完整處理
                        from src.main import _yolo_world_model
                        if _yolo_world_model is None:
                            # 觸發模型初始化（通過調用一次，但只初始化，不處理）
                            try:
                                # 快速初始化：只載入模型，不處理視頻
                                from ultralytics import YOLOWorld
                                import os
                                local_model_path = '/app/models/yolov8s-world.pt'
                                if os.path.exists(local_model_path):
                                    _yolo_world_model = YOLOWorld(local_model_path)
                                else:
                                    _yolo_world_model = YOLOWorld('yolov8s-world.pt')
                                print("--- [YOLO] 模型初始化完成（優化版本）---")
                                yolo_model = _yolo_world_model
                            except Exception as init_e:
                                print(f"--- [WARNING] 模型初始化失敗: {init_e}，使用舊版本 ---")
                                # 回退到舊版本
                                yolo_result = infer_segment_yolo(
                                    str(seg_path),
                                    labels=yolo_labels or "person,pedestrian,motorcycle,car,bus,scooter,truck",
                                    every_sec=yolo_every_sec,
                                    score_thr=yolo_score_thr
                                )
                                yolo_model = None  # 標記為已處理
                    
                    # 使用優化版本（如果模型已初始化）
                    if yolo_model is not None:
                        yolo_result = infer_segment_yolo_optimized(
                            str(seg_path),
                            labels=yolo_labels or "person,pedestrian,motorcycle,car,bus,scooter,truck",
                            every_sec=yolo_every_sec,
                            score_thr=yolo_score_thr,
                            yolo_model=yolo_model,
                            reid_model=reid_model,
                            reid_device=reid_device
                        )
                    elif yolo_result is None:
                        # 如果還是沒有結果，使用舊版本（最後備用）
                        print("--- [WARNING] 使用舊版本 YOLO（較慢）---")
                        infer_segment_yolo = _get_infer_segment_yolo()
                        yolo_result = infer_segment_yolo(
                            str(seg_path),
                            labels=yolo_labels or "person,pedestrian,motorcycle,car,bus,scooter,truck",
                            every_sec=yolo_every_sec,
                            score_thr=yolo_score_thr
                        )
                    
                    # 將 YOLO 結果添加到 VLM 結果中（確保 vlm_result 是字典）
                    if isinstance(vlm_result, dict):
                        # 確保 raw_detection 是字典，不是 None
                        if "raw_detection" not in vlm_result or vlm_result["raw_detection"] is None:
                            vlm_result["raw_detection"] = {}
                        vlm_result["raw_detection"]["yolo"] = yolo_result
                        print(f"--- [YOLO Optimized] 片段 {idx} 完成：偵測到 {yolo_result.get('total_detections', 0) if yolo_result else 0} 個物件 ---")
                    else:
                        print(f"--- [ERROR] vlm_result 不是字典，無法添加 YOLO 結果 ---")
                except Exception as e:
                    print(f"--- [WARNING] YOLO 處理失敗 (segment {idx}): {e} ---")
                    import traceback
                    traceback.print_exc()
                    # 確保 vlm_result 是字典後再設置錯誤
                    if isinstance(vlm_result, dict):
                        vlm_result["yolo_error"] = str(e)
                        # 即使失敗也設置空結果，確保結構一致
                        if "raw_detection" not in vlm_result or vlm_result["raw_detection"] is None:
                            vlm_result["raw_detection"] = {}
                        vlm_result["raw_detection"]["yolo"] = None
                    else:
                        print(f"--- [ERROR] vlm_result 不是字典，無法設置 YOLO 錯誤 ---")
            
            # 如果跳過 YOLO，確保 raw_detection 欄位存在但為空
            if SKIP_YOLO:
                if isinstance(vlm_result, dict):
                    # 確保 raw_detection 是字典，不是 None
                    if "raw_detection" not in vlm_result or vlm_result["raw_detection"] is None:
                        vlm_result["raw_detection"] = {}
                    vlm_result["raw_detection"]["yolo"] = None
                    vlm_result["raw_detection"]["yolo_skipped"] = True
            
            # 確保返回的是有效的字典
            if vlm_result is None or not isinstance(vlm_result, dict):
                print(f"--- [ERROR] process_segment 返回無效結果: {vlm_result} ---")
                vlm_result = {
                    "segment": Path(seg_path).name,
                    "time_range": time_range_str,
                    "duration_sec": end - start,
                    "success": False,
                    "time_sec": 0.0,
                    "parsed": {},
                    "raw_detection": {},  # 改為空字典，避免 None 錯誤
                    "error": "process_segment 返回無效結果"
                }
            return vlm_result
        except Exception as outer_e:
            # 捕獲所有其他異常（包括 ImportError 和其他錯誤）
            print(f"--- [ERROR] process_segment 發生異常: {outer_e} ---")
            import traceback
            traceback.print_exc()
            # 返回一個有效的錯誤結果
            return {
                "segment": Path(seg_path).name,
                "time_range": time_range_str,
                "duration_sec": end - start,
                "success": False,
                "time_sec": 0.0,
                "parsed": {},
                "raw_detection": {},  # 改為空字典，避免 None 錯誤
                "error": str(outer_e)
            }
        except ImportError:
            # 如果優化版本不可用，回退到舊版本
            print("--- [WARNING] 優化版本不可用，使用舊版本 ---")
            # 使用舊的 process_segment 邏輯
            m = re.search(r"(\d+)", Path(seg_path).name)
            idx = int(m.group(1)) if m else seg_idx
            start = idx * (segment_duration - overlap)
            end = min(start + segment_duration, total_duration)
            time_range_str = f"{_fmt_hms(start)} - {_fmt_hms(end)}"

            vlm_result = None  # 初始化 vlm_result，避免未定義錯誤
            if not SKIP_VLM:
                try:
                    req_data = SegmentAnalysisRequest(
                        segment_path=str(seg_path),
                        segment_index=idx,
                        start_time=start,
                        end_time=end,
                        model_type=model_type,
                        qwen_model=qwen_model,
                        frames_per_segment=frames_per_segment,
                        target_short=target_short,
                        sampling_fps=sampling_fps,
                        event_detection_prompt=event_detection_prompt,
                        summary_prompt=summary_prompt,
                        owl_labels=owl_labels,
                        owl_every_sec=owl_every_sec,
                        owl_score_thr=owl_score_thr,
                        yolo_labels=yolo_labels,
                        yolo_every_sec=yolo_every_sec,
                        yolo_score_thr=yolo_score_thr
                    )

                    vlm_result = analyze_segment_result(req_data)
                except Exception as vlm_e:
                    # 如果 VLM 分析失敗，創建錯誤結果
                    print(f"--- [ERROR] VLM 分析失敗 (segment {idx}, 舊版本回退): {vlm_e} ---")
                    import traceback
                    traceback.print_exc()
                    vlm_result = {
                        "segment": Path(seg_path).name,
                        "time_range": time_range_str,
                        "duration_sec": end - start,
                        "success": False,
                        "time_sec": 0.0,
                        "parsed": {
                            "frame_analysis": {
                                "events": {
                                    "water_flood": False,
                                    "fire": False,
                                    "abnormal_attire_face_cover_at_entry": False,
                                    "person_fallen_unmoving": False,
                                    "double_parking_lane_block": False,
                                    "smoking_outside_zone": False,
                                    "crowd_loitering": False,
                                    "security_door_tamper": False,
                                    "reason": ""
                                }
                            },
                            "summary_independent": ""
                        },
                        "raw_detection": {},  # 確保是空字典，不是 None
                        "error": f"VLM 分析失敗: {str(vlm_e)}"
                    }
            else:
                # [跳過 VLM] 創建基本的結果結構
                vlm_result = {
                    "segment": Path(seg_path).name,
                    "time_range": time_range_str,
                    "duration_sec": end - start,
                    "success": True,
                    "time_sec": 0.0,
                    "parsed": {
                        "frame_analysis": {
                            "events": {
                                "water_flood": False,
                                "fire": False,
                                "abnormal_attire_face_cover_at_entry": False,
                                "person_fallen_unmoving": False,
                                "double_parking_lane_block": False,
                                "smoking_outside_zone": False,
                                "crowd_loitering": False,
                                "security_door_tamper": False,
                                "reason": "VLM 分析已暫時停用（GPU 滿載）"
                            },
                            "persons": []
                        },
                        "summary_independent": "VLM 分析已暫時停用"
                    },
                    "raw_detection": {},
                    "error": None
                }
            
            # 確保 vlm_result 是有效的字典（舊版本回退路徑）
            if vlm_result is None or not isinstance(vlm_result, dict):
                print(f"--- [ERROR] VLM 分析返回無效結果（舊版本回退）: {vlm_result} ---")
                import traceback
                traceback.print_exc()
                vlm_result = {
                    "segment": Path(seg_path).name,
                    "time_range": time_range_str,
                    "duration_sec": end - start,
                    "success": False,
                    "time_sec": 0.0,
                    "parsed": {
                        "frame_analysis": {
                            "events": {
                                "water_flood": False,
                                "fire": False,
                                "abnormal_attire_face_cover_at_entry": False,
                                "person_fallen_unmoving": False,
                                "double_parking_lane_block": False,
                                "smoking_outside_zone": False,
                                "crowd_loitering": False,
                                "security_door_tamper": False,
                                "reason": ""
                            }
                        },
                        "summary_independent": ""
                    },
                    "raw_detection": {},
                    "error": f"VLM 分析返回無效結果: {vlm_result}"
                }
            
            # 確保 vlm_result 有必要的欄位（舊版本回退路徑）
            if "parsed" not in vlm_result:
                vlm_result["parsed"] = {
                    "frame_analysis": {
                        "events": {
                            "water_flood": False,
                            "fire": False,
                            "abnormal_attire_face_cover_at_entry": False,
                            "person_fallen_unmoving": False,
                            "double_parking_lane_block": False,
                            "smoking_outside_zone": False,
                            "crowd_loitering": False,
                            "security_door_tamper": False,
                            "reason": ""
                        }
                    },
                    "summary_independent": ""
                }
            # 確保 raw_detection 是字典，不是 None
            if "raw_detection" not in vlm_result or vlm_result["raw_detection"] is None:
                vlm_result["raw_detection"] = {}
            
            # 舊版本回退路徑：也檢查是否跳過 YOLO
            if SKIP_YOLO:
                print(f"--- [跳過 YOLO] 片段 {idx}（舊版本回退）：只執行 VLM 分析 ---")
                if isinstance(vlm_result, dict):
                    # 確保 raw_detection 是字典，不是 None
                    if "raw_detection" not in vlm_result or vlm_result["raw_detection"] is None:
                        vlm_result["raw_detection"] = {}
                    vlm_result["raw_detection"]["yolo"] = None
                    vlm_result["raw_detection"]["yolo_skipped"] = True
            else:
                yolo_result = None
                try:
                    infer_segment_yolo = _get_infer_segment_yolo()
                    yolo_result = infer_segment_yolo(
                        str(seg_path),
                        labels=yolo_labels or "person,pedestrian,motorcycle,car,bus,scooter,truck",
                        every_sec=yolo_every_sec,
                        score_thr=yolo_score_thr
                    )
                    # 確保 vlm_result 是字典後再賦值
                    if isinstance(vlm_result, dict):
                        # 確保 raw_detection 是字典，不是 None
                        if "raw_detection" not in vlm_result or vlm_result["raw_detection"] is None:
                            vlm_result["raw_detection"] = {}
                        vlm_result["raw_detection"]["yolo"] = yolo_result
                        print(f"--- [YOLO] 片段 {idx} 完成：偵測到 {yolo_result.get('total_detections', 0) if yolo_result else 0} 個物件 ---")
                    else:
                        print(f"--- [ERROR] vlm_result 不是字典（舊版本回退），無法添加 YOLO 結果 ---")
                except Exception as e:
                    print(f"--- [WARNING] YOLO 處理失敗 (segment {idx}): {e} ---")
                    import traceback
                    traceback.print_exc()
                    # 確保 vlm_result 是字典後再設置錯誤
                    if isinstance(vlm_result, dict):
                        vlm_result["yolo_error"] = str(e)
                        if "raw_detection" not in vlm_result:
                            vlm_result["raw_detection"] = {}
                        vlm_result["raw_detection"]["yolo"] = None
                    else:
                        print(f"--- [ERROR] vlm_result 不是字典（舊版本回退），無法設置 YOLO 錯誤 ---")
            
            return vlm_result

    # 使用線程池實現併行處理（增加並行度以加速 VLM 處理）
    # max_workers 設為 4，可以同時處理 4 個片段（VLM + YOLO）
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, p in enumerate(seg_files):
            future = executor.submit(process_segment, p, i)
            futures.append((future, i))
        
        # 收集結果（按順序）
        results_dict = {}
        for future, idx in futures:
            try:
                res = future.result()
                results_dict[idx] = res
            except Exception as e:
                print(f"--- [ERROR] 片段處理失敗 (index {idx}): {e} ---")
                results_dict[idx] = {
                    "success": False,
                    "error": str(e),
                    "time_sec": 0
                }
        
        # 按索引排序
        results = [results_dict[i] for i in sorted(results_dict.keys())]

    # 4. 統計與存檔 (維持原樣)
    process_time = time.time() - t0  # 處理時間（不包括上傳、切割）
    total_time = time.time() - t0_total  # 總時間（包括上傳、切割、處理）
    ok_count = sum(1 for r in results if r.get("success"))

    resp = {
        "model_type": model_type,
        "total_segments": len(results),
        "success_segments": ok_count,
        "total_time_sec": round(total_time, 2),  # 總時間（包括所有操作）
        "process_time_sec": round(process_time, 2),  # 處理時間（不包括上傳、切割）
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

