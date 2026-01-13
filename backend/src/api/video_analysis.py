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
    # 手動獲取 db session（避免條件表達式導致 FastAPI 路由失敗）
    db = None
    if HAS_DB:
        try:
            db = next(get_db())
        except Exception as e:
            print(f"--- [WARNING] 無法獲取資料庫連接: {e} ---")
    
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
        finally:
            # 關閉 db session
            try:
                db.close()
            except:
                pass

    if cleanup and os.path.exists(local_path):
        try: os.remove(local_path)
        except: pass

    return JSONResponse(resp, media_type="application/json; charset=utf-8")

@router.post("/v1/search/image", dependencies=[Depends(get_api_key)])
def search_by_image(
    request: Request,
    api_key: str = Depends(get_api_key),
    file: UploadFile = File(None),  # 上傳的查詢圖片
    text_query: Optional[str] = Form(None),  # 文字描述（例如 "藍色衣服的人"）
    top_k: int = Form(10),  # 返回前 K 個最相似的結果
    threshold: float = Form(0.7),  # 相似度閾值（0.0-1.0，越高越嚴格）
    label_filter: Optional[str] = Form(None),  # 可選：過濾特定類別（例如 "person"）
):
    """
    以圖搜圖 API：根據上傳的圖片或文字描述，找到外表相似的物件 crops
    
    支持兩種查詢方式：
    1. 圖片上傳：上傳一張圖片，找到相似的物件
    2. 文字描述：輸入文字描述（例如 "藍色衣服的人"），找到符合描述的物件
    
    參數:
    - file: 上傳的查詢圖片（與 text_query 二選一）
    - text_query: 文字描述（與 file 二選一）
    - top_k: 返回前 K 個最相似的結果（預設 10）
    - threshold: 相似度閾值（0.0-1.0，預設 0.7，越高越嚴格）
    - label_filter: 可選，過濾特定類別（例如 "person", "car"）
    
    返回:
    - query_type: 查詢類型（"image" 或 "text"）
    - query_info: 查詢信息
    - results: 相似物件列表，每個包含：
      - crop_id: ObjectCrop ID
      - crop_path: 物件切片圖片路徑
      - label: 物件類別
      - score: 偵測信心分數
      - timestamp: 時間戳
      - similarity: 相似度分數（0.0-1.0）
      - summary_id: 關聯的 Summary ID
      - video: 影片名稱
      - segment: 片段名稱
      - time_range: 時間範圍
    """
    # 手動獲取 db session（避免條件表達式導致 FastAPI 路由失敗）
    db = None
    if HAS_DB:
        try:
            db = next(get_db())
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"無法獲取資料庫連接: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="資料庫未連接，無法執行搜索")
    
    try:
        from src.main import generate_image_embedding, generate_text_embedding
        from src.models import ObjectCrop, Summary
        import numpy as np
        import tempfile
        
        print("=" * 80)
        print("--- [Image Search] 開始處理搜索請求 ---")
        print(f"--- [Image Search] 參數: top_k={top_k}, threshold={threshold}, label_filter={label_filter} ---")
        print("=" * 80)
        
        import sys
        sys.stdout.flush()  # 強制刷新輸出
        
        # 1. 生成查詢 embedding
        query_embedding = None
        query_type = None
        query_info = {}
        embedding_type = None  # 實際使用的 embedding 類型（"reid" 或 "clip"）
        uploaded_file_content = None  # 保存上傳的文件內容，以便後續可能需要重新生成 CLIP embedding
        
        if file is not None:
            # 圖片查詢
            query_type = "image"
            query_info = {"filename": file.filename}
            print(f"--- [Image Search] 步驟 1/3: 處理圖片上傳 ({file.filename}) ---")
            sys.stdout.flush()
            
            # 保存上傳的圖片到臨時文件
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            try:
                with os.fdopen(fd, "wb") as f:
                    file_content = file.file.read()
                    uploaded_file_content = file_content  # 保存文件內容以便後續使用
                    f.write(file_content)
                print(f"--- [Image Search] 圖片已保存到臨時文件: {tmp_path}, 大小: {len(file_content)} bytes ---")
                sys.stdout.flush()
                
                print(f"--- [Image Search] 步驟 2/3: 生成圖片 ReID embedding（全面使用 ReID）---")
                print(f"--- [Image Search] 臨時文件路徑: {tmp_path} ---")
                print(f"--- [Image Search] 檢查文件是否存在: {os.path.exists(tmp_path)} ---")
                if os.path.exists(tmp_path):
                    file_size = os.path.getsize(tmp_path)
                    print(f"--- [Image Search] 文件大小: {file_size} bytes ---")
                sys.stdout.flush()
                
                try:
                    from src.main import generate_reid_embedding
                    query_embedding, embedding_type = generate_reid_embedding(tmp_path)
                    print(f"--- [Image Search] generate_reid_embedding 返回: embedding_type={embedding_type}, 是否為 None: {query_embedding is None} ---")
                    if query_embedding is not None:
                        print(f"--- [Image Search] {embedding_type.upper()} embedding 長度: {len(query_embedding)} ---")
                        # 根據 embedding 類型檢查維度
                        expected_dim = 2048 if embedding_type == "reid" else 512
                        if len(query_embedding) != expected_dim:
                            error_msg = f"{embedding_type.upper()} embedding 維度錯誤: 預期 {expected_dim} 維，實際 {len(query_embedding)} 維"
                            print(f"--- [Image Search] ✗ {error_msg} ---")
                            sys.stdout.flush()
                            raise HTTPException(
                                status_code=500,
                                detail=error_msg
                            )
                        # 如果使用了 CLIP 回退，記錄警告
                        if embedding_type == "clip":
                            print("--- [Image Search] ⚠️  警告：使用了 CLIP embedding（512 維）而非 ReID（2048 維），搜索結果可能不準確 ---")
                            print("--- [Image Search] ⚠️  建議：安裝 ReID 模型以獲得更好的搜索效果 ---")
                except HTTPException:
                    # 重新拋出 HTTPException
                    raise
                except Exception as emb_error:
                    print(f"--- [Image Search] ✗ 生成 ReID embedding 時發生異常: {emb_error} ---")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise HTTPException(
                        status_code=500,
                        detail=f"生成圖片 ReID embedding 失敗: {str(emb_error)}。請檢查：1) ReID 模型是否正確加載（是否安裝 torchreid: pip install torchreid）2) 圖片格式是否正確 3) 後端日誌中的詳細錯誤信息"
                    )
                sys.stdout.flush()
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
            
            if query_embedding is None:
                print("--- [Image Search] ✗ 無法生成圖片 ReID embedding ---")
                sys.stdout.flush()
                # 檢查 ReID 模型狀態
                try:
                    from src.main import get_reid_model
                    reid_model, reid_device = get_reid_model()
                    model_status = "已載入" if reid_model is not None else "未載入"
                    device_status = reid_device if reid_device else "未設置"
                    
                    if reid_model is None:
                        error_detail = f"無法生成查詢圖片的 ReID embedding。ReID 模型未載入（可能是 torchreid 未安裝或模型載入失敗）。請檢查：1) 是否安裝 torchreid: pip install torchreid 2) 後端日誌中的詳細錯誤信息。模型狀態: {model_status}, 設備: {device_status}"
                    else:
                        error_detail = f"無法生成查詢圖片的 ReID embedding。ReID 模型已載入但生成失敗。請檢查：1) 圖片格式是否正確 2) 圖片文件是否損壞 3) 後端日誌中的詳細錯誤信息。"
                except Exception as e:
                    error_detail = f"無法生成查詢圖片的 ReID embedding。ReID 模型載入檢查失敗: {str(e)}。請檢查後端日誌以獲取詳細錯誤信息。"
                raise HTTPException(status_code=500, detail=error_detail)
            # 使用實際的 embedding 類型來顯示訊息
            embedding_type_display = embedding_type.upper() if embedding_type else "UNKNOWN"
            print(f"--- [Image Search] ✓ 圖片 {embedding_type_display} embedding 生成完成 (維度: {len(query_embedding)}) ---")
            print(f"--- [Image Search] 查詢向量前5個值: {query_embedding[:5] if len(query_embedding) >= 5 else query_embedding} ---")
            sys.stdout.flush()
                
        elif text_query and text_query.strip():
            # 文字查詢（ReID 不支持文字，使用 CLIP）
            query_type = "text"
            embedding_type = "clip"  # 文字查詢固定使用 CLIP
            query_info = {"text": text_query.strip()}
            print(f"--- [Image Search] 步驟 1/3: 生成文字 CLIP embedding (文字: \"{text_query.strip()}\") ---")
            print(f"--- [Image Search] 注意：文字查詢使用 CLIP（ReID 不支持文字輸入）---")
            query_embedding = generate_text_embedding(text_query.strip())
            
            if query_embedding is None:
                print("--- [Image Search] ✗ 無法生成文字 embedding ---")
                raise HTTPException(status_code=500, detail="無法生成文字描述的 embedding")
            print(f"--- [Image Search] ✓ 文字 embedding 生成完成 (維度: {len(query_embedding)}) ---")
            print(f"--- [Image Search] 查詢向量前5個值: {query_embedding[:5] if len(query_embedding) >= 5 else query_embedding} ---")
        else:
            raise HTTPException(status_code=422, detail="需要提供 file（圖片）或 text_query（文字描述）")
        
        # 2. 在資料庫中搜索相似的 crops（使用 PostgreSQL 向量搜索）
        print(f"--- [Image Search] 步驟 3/3: 在資料庫中搜索相似物件 (threshold: {threshold}, top_k: {top_k}) ---")
        sys.stdout.flush()
        
        results_raw = []
        try:
            from sqlalchemy import text
            from src.models import HAS_PGVECTOR
            print(f"--- [Image Search] HAS_PGVECTOR: {HAS_PGVECTOR} ---")
            sys.stdout.flush()
            
            # 檢查是否有 pgvector 支持
            if not HAS_PGVECTOR:
                print("--- [Image Search] ⚠️  pgvector 未安裝，回退到 Python 計算模式 ---")
                # 回退到舊的 Python 計算方式（這裡可以保留舊代碼作為備用）
                raise NotImplementedError("pgvector not available, please install pgvector")
            
            # 判斷使用哪種 embedding
            # 如果圖片查詢返回的是 CLIP（回退情況），使用 CLIP
            if query_type == "image":
                # 檢查實際使用的 embedding 類型（可能因為回退而使用 CLIP）
                use_reid = embedding_type == "reid"
                embedding_column = "reid_embedding" if use_reid else "clip_embedding"
                expected_dim = 2048 if use_reid else 512
            else:
                # 文字查詢使用 CLIP
                use_reid = False
                embedding_column = "clip_embedding"
                expected_dim = 512
            
            # 檢查資料庫中是否有對應的 embedding
            reid_count = db.query(ObjectCrop).filter(
                ObjectCrop.reid_embedding.isnot(None)
            ).count()
            clip_count = db.query(ObjectCrop).filter(
                ObjectCrop.clip_embedding.isnot(None)
            ).count()
            
            print(f"--- [Image Search] 資料庫統計: ReID embedding={reid_count} 筆, CLIP embedding={clip_count} 筆 ---")
            
            # 如果查詢使用 ReID 但資料庫中沒有 ReID embedding，自動回退到 CLIP
            if use_reid and reid_count == 0 and clip_count > 0:
                print("--- [Image Search] ⚠️  警告：查詢使用 ReID（2048 維），但資料庫中沒有 ReID embedding ---")
                print(f"--- [Image Search] ⚠️  自動回退到 CLIP embedding（資料庫中有 {clip_count} 筆 CLIP embedding）---")
                print("--- [Image Search] ⚠️  注意：查詢向量是 2048 維（ReID），但資料庫中是 512 維（CLIP），無法直接比較 ---")
                print("--- [Image Search] ⚠️  建議：重新生成查詢圖片的 CLIP embedding 以匹配資料庫 ---")
                
                # 重新生成查詢圖片的 CLIP embedding
                if uploaded_file_content is not None:
                    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                    try:
                        with os.fdopen(fd, "wb") as f:
                            f.write(uploaded_file_content)
                        
                        from src.main import generate_image_embedding
                        query_embedding = generate_image_embedding(tmp_path)
                        embedding_type = "clip"
                        if query_embedding is None:
                            raise HTTPException(status_code=500, detail="無法生成 CLIP embedding")
                        print(f"--- [Image Search] ✓ 已重新生成 CLIP embedding（維度: {len(query_embedding)}）---")
                    finally:
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail="無法回退到 CLIP：缺少上傳的圖片文件。請重新上傳圖片。"
                    )
                
                # 更新為使用 CLIP
                use_reid = False
                embedding_column = "clip_embedding"
                expected_dim = 512
            
            print(f"--- [Image Search] 使用 {embedding_column} 搜索（{'ReID' if use_reid else 'CLIP'}，{expected_dim} 維）---")
            
            # 統計要使用的 embedding 記錄數
            if use_reid:
                total_count = reid_count
            else:
                total_count = clip_count
            
            if total_count == 0:
                embedding_type_name = "ReID" if use_reid else "CLIP"
                print(f"--- [Image Search] ⚠️  資料庫中沒有 {embedding_type_name} embedding 記錄，返回空結果 ---")
                results_raw = []
            else:
                # 將 query_embedding 轉換為 PostgreSQL 向量格式
                # pgvector 需要格式：'[0.1, 0.2, 0.3, ...]'
                # 確保所有值都是有效的浮點數
                try:
                    query_embedding_clean = [float(x) for x in query_embedding]
                    if len(query_embedding_clean) != expected_dim:
                        raise ValueError(f"Invalid embedding dimension: expected {expected_dim}, got {len(query_embedding_clean)}")
                    query_embedding_str = '[' + ','.join(map(str, query_embedding_clean)) + ']'
                except (ValueError, TypeError) as e:
                    raise HTTPException(status_code=400, detail=f"Invalid query embedding format: {e}")
                
                # 構建 SQL 查詢，使用 pgvector 的 <=> 運算符（cosine 距離）
                # cosine 距離 = 1 - cosine similarity
                # 所以 similarity = 1 - distance
                # 我們要 similarity >= threshold，即 1 - distance >= threshold，即 distance <= 1 - threshold
                cosine_distance_threshold = 1.0 - threshold
                
                # 當 threshold = 0 時，cosine_distance_threshold = 1.0，這會查詢所有記錄
                # 為了性能，我們設置一個合理的上限（例如 0.99，對應 similarity >= 0.01）
                if threshold == 0:
                    print("--- [Image Search] ⚠️  threshold = 0，將使用較寬鬆的距離閾值（0.99）以提升性能 ---")
                    cosine_distance_threshold = 0.99  # 對應 similarity >= 0.01
                
                print(f"--- [Image Search] 使用 PostgreSQL 向量搜索（threshold: {threshold:.4f}, cosine 距離閾值: {cosine_distance_threshold:.4f}）---")
                
                # 構建 SQL 查詢，使用 pgvector 的 <=> 運算符（cosine 距離）
                # 注意：直接使用字符串插值來構建向量，因為參數綁定對向量類型支持有限
                # 但我們會對所有輸入進行驗證，確保安全性
                
                # 驗證 query_embedding 格式（應該是數字列表）
                if not isinstance(query_embedding, list) or len(query_embedding) != expected_dim:
                    raise ValueError(f"Invalid query_embedding: expected list of {expected_dim} floats, got {type(query_embedding)} with length {len(query_embedding) if isinstance(query_embedding, list) else 'N/A'}")
                
                # 構建基礎 SQL（使用字符串格式化，因為向量類型需要特殊處理）
                # 注意：列名不能使用 f-string，需要手動拼接
                base_sql = f"""
                    SELECT 
                        oc.id,
                        oc.summary_id,
                        oc.crop_path,
                        oc.label,
                        oc.score,
                        oc.timestamp,
                        oc.frame,
                        oc.box,
                        oc.{embedding_column},
                        s.video,
                        s.segment,
                        s.time_range,
                        s.location,
                        s.camera,
                        1 - (oc.{embedding_column} <=> '{query_embedding_str}'::vector) as similarity
                    FROM object_crops oc
                    JOIN summaries s ON oc.summary_id = s.id
                    WHERE oc.{embedding_column} IS NOT NULL
                """
                
                # 添加相似度過濾（當 threshold > 0 時，或 threshold = 0 但我們設置了上限時）
                # 注意：當 threshold = 0 時，我們已經將 cosine_distance_threshold 設置為 0.99
                base_sql += f" AND (oc.{embedding_column} <=> '{query_embedding_str}'::vector) <= {cosine_distance_threshold}"
                
                # 添加類別過濾
                if label_filter and label_filter.strip():
                    # 使用參數綁定來防止 SQL 注入
                    label_filter_escaped = label_filter.strip().replace("'", "''")
                    base_sql += f" AND oc.label = '{label_filter_escaped}'"
                    if use_reid:
                        filtered_count = db.query(ObjectCrop).filter(
                            ObjectCrop.reid_embedding.isnot(None),
                            ObjectCrop.label == label_filter.strip()
                        ).count()
                    else:
                        filtered_count = db.query(ObjectCrop).filter(
                            ObjectCrop.clip_embedding.isnot(None),
                            ObjectCrop.label == label_filter.strip()
                        ).count()
                    print(f"--- [Image Search] 過濾類別 \"{label_filter.strip()}\" 後剩餘 {filtered_count} 筆記錄 ---")
                
                # 添加排序和限制
                sql_query = base_sql + f"""
                    ORDER BY oc.{embedding_column} <=> '{query_embedding_str}'::vector ASC
                    LIMIT {top_k}
                """
                
                print(f"--- [Image Search] SQL 查詢預覽（前 200 字）: {sql_query[:200]}... ---")
                
                # 執行查詢
                query_start = time.time()
                print("--- [Image Search] 執行 PostgreSQL 向量搜索... ---")
                
                try:
                    # 執行查詢（不需要參數，因為已經在 SQL 中直接插值）
                    result = db.execute(text(sql_query))
                    query_time = time.time() - query_start
                    
                    rows = result.fetchall()
                    print(f"--- [Image Search] ✓ 向量搜索完成（耗時: {query_time:.3f} 秒），找到 {len(rows)} 筆結果 ---")
                    
                    # 處理結果
                    for row in rows:
                        crop_id, summary_id, crop_path, label, score, timestamp, frame, box, emb, video, segment, time_range, location, camera, similarity = row
                        
                        # 創建簡化的對象結構
                        crop_obj = type('Crop', (), {
                            'id': crop_id,
                            'summary_id': summary_id,
                            'crop_path': crop_path,
                            'label': label,
                            'score': score,
                            'timestamp': timestamp,
                            'frame': frame,
                            'box': box,
                            'clip_embedding': None,  # 不再使用（全面使用 ReID）
                            'reid_embedding': emb if use_reid else None,
                            'similarity': float(similarity)
                        })()
                        
                        summary_obj = type('Summary', (), {
                            'id': summary_id,
                            'video': video,
                            'segment': segment,
                            'time_range': time_range,
                            'location': location,
                            'camera': camera
                        })()
                        
                        results_raw.append((crop_obj, summary_obj))
                    
                    print(f"--- [Image Search] ✓ 返回 {len(results_raw)} 筆結果 ---")
                    
                except Exception as query_error:
                    query_time = time.time() - query_start
                    error_msg = str(query_error)
                    print(f"--- [Image Search] ✗ 向量搜索失敗 (耗時: {query_time:.3f} 秒): {error_msg} ---")
                    import traceback
                    traceback.print_exc()
                    
                    # 如果查詢失敗，嘗試回退到 Python 計算模式（僅當資料量不大時）
                    print("--- [Image Search] 嘗試回退到 Python 計算模式... ---")
                    try:
                        # 限制回退模式的資料量（最多 1000 筆）
                        if use_reid:
                            fallback_query = db.query(
                                ObjectCrop.id,
                                ObjectCrop.summary_id,
                                ObjectCrop.crop_path,
                                ObjectCrop.label,
                                ObjectCrop.score,
                                ObjectCrop.timestamp,
                                ObjectCrop.frame,
                                ObjectCrop.box,
                                ObjectCrop.reid_embedding,
                                Summary.video,
                                Summary.segment,
                                Summary.time_range,
                                Summary.location,
                                Summary.camera
                            ).join(
                                Summary, ObjectCrop.summary_id == Summary.id
                            ).filter(
                                ObjectCrop.reid_embedding.isnot(None)
                            )
                        else:
                            fallback_query = db.query(
                                ObjectCrop.id,
                                ObjectCrop.summary_id,
                                ObjectCrop.crop_path,
                                ObjectCrop.label,
                                ObjectCrop.score,
                                ObjectCrop.timestamp,
                                ObjectCrop.frame,
                                ObjectCrop.box,
                                ObjectCrop.clip_embedding,
                                Summary.video,
                                Summary.segment,
                                Summary.time_range,
                                Summary.location,
                                Summary.camera
                            ).join(
                                Summary, ObjectCrop.summary_id == Summary.id
                            ).filter(
                                ObjectCrop.clip_embedding.isnot(None)
                            )
                        
                        if label_filter and label_filter.strip():
                            fallback_query = fallback_query.filter(ObjectCrop.label == label_filter.strip())
                        
                        # 限制數量
                        all_crops_data = fallback_query.limit(1000).all()
                        
                        if len(all_crops_data) > 1000:
                            raise Exception("資料量過大，無法使用回退模式")
                        
                        print(f"--- [Image Search] 回退模式：載入 {len(all_crops_data)} 筆記錄，開始計算... ---")
                        
                        # 使用 Python 計算相似度
                        query_embedding_array = np.array(query_embedding, dtype=np.float32)
                        similarities = []
                        
                        for row in all_crops_data:
                            if use_reid:
                                crop_id, summary_id, crop_path, label, score, timestamp, frame, box, emb, video, segment, time_range, location, camera = row
                            else:
                                crop_id, summary_id, crop_path, label, score, timestamp, frame, box, emb, video, segment, time_range, location, camera = row
                            
                            if emb:
                                try:
                                    # 處理 embedding
                                    crop_emb_data = emb
                                    if isinstance(crop_emb_data, str):
                                        crop_emb_data = json.loads(crop_emb_data)
                                    
                                    crop_embedding = np.array(crop_emb_data, dtype=np.float32)
                                    
                                    if len(crop_embedding.shape) == 0 or crop_embedding.size == 0:
                                        continue
                                    
                                    # 計算 cosine similarity
                                    similarity = float(np.dot(query_embedding_array, crop_embedding) / (
                                        np.linalg.norm(query_embedding_array) * np.linalg.norm(crop_embedding) + 1e-12
                                    ))
                                    
                                    if similarity >= threshold:
                                        crop_obj = type('Crop', (), {
                                            'id': crop_id, 'summary_id': summary_id, 'crop_path': crop_path,
                                            'label': label, 'score': score, 'timestamp': timestamp,
                                            'frame': frame, 'box': box, 'clip_embedding': None,
                                            'reid_embedding': emb if use_reid else None,
                                            'similarity': similarity
                                        })()
                                        summary_obj = type('Summary', (), {
                                            'id': summary_id, 'video': video, 'segment': segment,
                                            'time_range': time_range, 'location': location, 'camera': camera
                                        })()
                                        similarities.append((similarity, crop_obj, summary_obj))
                                except Exception as e:
                                    continue
                        
                        # 排序並返回
                        similarities.sort(key=lambda x: x[0], reverse=True)
                        results_raw = [(crop, summary) for _, crop, summary in similarities[:top_k]]
                        print(f"--- [Image Search] ✓ 回退模式完成，返回 {len(results_raw)} 筆結果 ---")
                        
                    except Exception as fallback_error:
                        print(f"--- [Image Search] ✗ 回退模式也失敗: {fallback_error} ---")
                        try:
                            db.rollback()
                        except:
                            pass
                        results_raw = []
                        raise HTTPException(
                            status_code=500,
                            detail=f"向量搜索失敗: {error_msg}。回退模式也失敗: {str(fallback_error)}"
                        )
            
        except Exception as e:
            # 錯誤處理
            print(f"--- [Image Search] ✗ 搜索失敗: {e} ---")
            import traceback
            traceback.print_exc()
            
            # 回滾事務（重要！）
            try:
                db.rollback()
                print("--- [Image Search] 已回滾事務 ---")
            except Exception as rollback_error:
                print(f"--- [Image Search] 回滾失敗: {rollback_error} ---")
            
            # 返回空結果，避免再次錯誤
            results_raw = []
        
        # 3. 獲取第一筆資料的向量（用於調試）
        first_crop_embedding = None
        first_crop_info = None
        try:
            print("--- [Image Search] 嘗試獲取第一筆資料的向量... ---")
            if use_reid:
                first_crop = db.query(ObjectCrop).filter(
                    ObjectCrop.reid_embedding.isnot(None)
                ).first()
            else:
                first_crop = db.query(ObjectCrop).filter(
                    ObjectCrop.clip_embedding.isnot(None)
                ).first()
            if first_crop:
                print(f"--- [Image Search] 找到第一筆資料: ID={first_crop.id}, label={first_crop.label} ---")
                first_crop_info = {
                    "id": first_crop.id,
                    "label": first_crop.label,
                    "crop_path": first_crop.crop_path
                }
                # 處理 embedding（根據查詢類型選擇對應的欄位）
                if use_reid:
                    first_emb = first_crop.reid_embedding
                else:
                    first_emb = first_crop.clip_embedding
                print(f"--- [Image Search] 第一筆資料 embedding 類型: {type(first_emb)} ---")
                
                if isinstance(first_emb, str):
                    try:
                        first_emb = json.loads(first_emb)
                        print("--- [Image Search] 成功解析字符串格式的 embedding ---")
                    except Exception as e:
                        print(f"--- [Image Search] 解析字符串 embedding 失敗: {e} ---")
                        pass
                
                if first_emb is not None:
                    try:
                        if hasattr(first_emb, 'tolist'):
                            first_crop_embedding = first_emb.tolist()
                        elif isinstance(first_emb, list):
                            first_crop_embedding = first_emb
                        elif hasattr(first_emb, '__iter__') and not isinstance(first_emb, str):
                            first_crop_embedding = list(first_emb)
                        else:
                            first_crop_embedding = None
                        
                        if first_crop_embedding:
                            print(f"--- [Image Search] 第一筆資料向量維度: {len(first_crop_embedding)} ---")
                            print(f"--- [Image Search] 第一筆資料向量前5個值: {first_crop_embedding[:5] if len(first_crop_embedding) >= 5 else first_crop_embedding} ---")
                    except Exception as e:
                        print(f"--- [Image Search] 轉換 embedding 格式失敗: {e} ---")
                else:
                    print("--- [Image Search] 第一筆資料的 embedding 為 None ---")
            else:
                embedding_type = "ReID" if use_reid else "CLIP"
                print(f"--- [Image Search] 資料庫中沒有找到有 {embedding_type} embedding 的記錄 ---")
        except Exception as e:
            print(f"--- [Image Search] 獲取第一筆資料向量失敗: {e} ---")
            import traceback
            traceback.print_exc()
        
        # 3. 組裝結果
        print("--- [Image Search] 組裝搜索結果... ---")
        results = []
        for crop, summary in results_raw:
            # 計算相似度（如果還沒計算）
            if hasattr(crop, 'similarity'):
                similarity = crop.similarity
            else:
                # 處理 embedding（可能是 list、str 或其他格式）
                # 根據查詢類型選擇對應的 embedding 欄位
                if use_reid:
                    crop_emb = crop.reid_embedding if hasattr(crop, 'reid_embedding') else None
                else:
                    crop_emb = crop.clip_embedding if hasattr(crop, 'clip_embedding') else None
                
                if isinstance(crop_emb, str):
                    try:
                        crop_emb = json.loads(crop_emb)
                    except:
                        crop_emb = None
                
                if crop_emb is None:
                    similarity = 0.0
                else:
                    crop_embedding = np.array(crop_emb, dtype=np.float32)
                    query_embedding_array = np.array(query_embedding, dtype=np.float32)
                    similarity = float(np.dot(query_embedding_array, crop_embedding) / (
                        np.linalg.norm(query_embedding_array) * np.linalg.norm(crop_embedding) + 1e-12
                    ))
            
            results.append({
                "crop_id": crop.id,
                "crop_path": crop.crop_path,
                "label": crop.label,
                "score": crop.score,
                "timestamp": crop.timestamp,
                "frame": crop.frame,
                "box": json.loads(crop.box) if crop.box else None,
                "similarity": round(similarity, 4),
                "summary_id": crop.summary_id,
                "video": summary.video,
                "segment": summary.segment,
                "time_range": summary.time_range,
                "location": summary.location,
                "camera": summary.camera,
            })
        
        print(f"--- [Image Search] ✓ 搜索完成，返回 {len(results)} 筆結果 ---")
        
        # 準備調試信息
        # 獲取實際使用的 embedding 類型（如果圖片查詢回退到 CLIP，這裡會是 "clip"）
        actual_embedding_type = embedding_type if query_type == "image" else "clip"
        debug_info = {
            "query_embedding": query_embedding,  # 查詢向量（ReID: 2048 維，CLIP: 512 維）
            "query_embedding_dim": len(query_embedding) if query_embedding else 0,
            "query_embedding_sample": query_embedding[:10] if query_embedding and len(query_embedding) >= 10 else query_embedding,  # 前10個值
            "first_crop_info": first_crop_info,
            "first_crop_embedding": first_crop_embedding,  # 第一筆資料的向量
            "first_crop_embedding_dim": len(first_crop_embedding) if first_crop_embedding else 0,
            "first_crop_embedding_sample": first_crop_embedding[:10] if first_crop_embedding and len(first_crop_embedding) >= 10 else first_crop_embedding,  # 前10個值
            "expected_embedding_dim": expected_dim,  # 預期的向量維度
            "embedding_type": actual_embedding_type.upper(),  # 實際使用的 embedding 類型（REID 或 CLIP）
            "is_fallback": actual_embedding_type == "clip" and query_type == "image",  # 是否為回退情況
        }
        
        print(f"--- [Image Search] 準備返回結果，包含 {len(results)} 筆結果和調試信息 ---")
        print(f"--- [Image Search] 調試信息: query_embedding_dim={debug_info['query_embedding_dim']}, first_crop_embedding_dim={debug_info['first_crop_embedding_dim']} ---")
        
        response_data = {
            "query_type": query_type,
            "query_info": query_info,
            "top_k": top_k,
            "threshold": threshold,
            "label_filter": label_filter,
            "total_results": len(results),
            "results": results,
            "debug": debug_info  # 調試信息
        }
        
        print(f"--- [Image Search] 返回數據結構: {list(response_data.keys())} ---")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"--- [ERROR] 以圖搜圖失敗: {e} ---")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索失敗: {str(e)}")

