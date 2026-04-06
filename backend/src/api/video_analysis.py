# -*- coding: utf-8 -*-
"""
影片分析相關 API
"""
import os
import gc
import subprocess
import time
import json
import tempfile
import uuid
import traceback
import threading
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import psutil
import logging
import torch

from src.config import config
from src.services.video_service import VideoService
from src.services.analysis_service import AnalysisService, resolve_vllm_video_direct_num_frames
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
    from src.main import get_api_key, _save_results_to_postgres, HAS_DB, SegmentAnalysisRequest
    from prompts import get_event_detection_prompt, get_summary_prompt

    return (
        get_db,
        get_api_key,
        _save_results_to_postgres,
        HAS_DB,
        SegmentAnalysisRequest,
        get_event_detection_prompt,
        get_summary_prompt,
    )

router = APIRouter(tags=["影片分析"])


def _print_ram_diagnosis(stage: str) -> None:
    """即時列印系統 / 進程 RAM 與 Swap，供診斷 vm_pct 飆高（segment_pipeline 與 YOLO batch 內延遲匯入呼叫）。"""
    try:
        vm = psutil.virtual_memory()
        vm_pct = float(vm.percent)
        available_mib = vm.available / (1024 * 1024)
        rss = psutil.Process(os.getpid()).memory_info().rss
        rss_mb = rss / (1024 * 1024)
        sw = psutil.swap_memory()
        swap_used = int(sw.used)
        swap_used_mib = swap_used / (1024 * 1024)
        print(
            f"--- [RAM-DIAG] {stage} vm_pct={vm_pct:.1f}% available={available_mib:.1f}MiB "
            f"process_rss={rss_mb:.1f}MiB swap_used={swap_used_mib:.1f}MiB ({swap_used} bytes) ---",
            flush=True,
        )
        if vm_pct > 90.0:
            print(
                "--- [RAM-CRITICAL] 記憶體即將耗盡，系統可能觸發 Swap 導致 I/O 鎖死！ ---",
                flush=True,
            )
    except Exception as ex:
        print(f"--- [RAM-DIAG] {stage} 無法讀取 psutil: {type(ex).__name__}: {ex} ---", flush=True)


# 對每個片段做 VLM + YOLO，含偵測與切割
@router.post("/v1/analyze_segment_result")
def analyze_segment_result_api(req: dict, api_key: str = Depends(lambda: _get_db_and_models()[1])):
    """分析單一片段的 API 進入點"""
    # 這裡將 req 轉為 Service 需要的格式
    from src.main import SegmentAnalysisRequest
    request_obj = SegmentAnalysisRequest(**req)
    return AnalysisService.analyze_segment(request_obj)


class AnalyzeSingleSegmentBody(BaseModel):
    """單段即時分析請求 body（video_id + 時間區間，不傳整部影片）"""
    video_id: str
    start_time: float  # 秒
    duration: float = 10.0  # 秒


@router.post("/v1/analyze_single_segment")
def analyze_single_segment(
    body: AnalyzeSingleSegmentBody = Body(...),
    api_key: str = Depends(lambda: _get_db_and_models()[1]),
):
    """
    依 video_id 與時間區間切出單一段落，做 YOLO + Qwen 推論後直接回傳 JSON。
    不寫入資料庫，供前端播放同步即時顯示用。
    """
    get_db, get_api_key, _save_results_to_postgres, HAS_DB, SegmentAnalysisRequest, get_event_prompt, get_summary_prompt = _get_db_and_models()
    video_id = (body.video_id or "").strip()
    start_time = float(body.start_time)
    duration = float(body.duration)
    if duration <= 0 or duration > 60:
        raise HTTPException(status_code=400, detail="duration 需介於 0～60 秒")
    # 1. 解析 video_id 取得原始影片路徑（與 VideoService.prepare_segments 一致）
    source_path = None
    if "/" in video_id:
        category, video_name = video_id.split("/", 1)
        potential = config.VIDEO_LIB_DIR / category / f"{video_name}.mp4"
        if not potential.exists():
            for ext in [".avi", ".mov", ".mkv", ".flv"]:
                p = config.VIDEO_LIB_DIR / category / f"{Path(video_name).stem}{ext}"
                if p.exists():
                    potential = p
                    break
        if potential.exists():
            source_path = str(potential)
    if not source_path or not Path(source_path).exists():
        raise HTTPException(status_code=404, detail=f"找不到影片: {video_id}")
    # 2. 使用 FFmpeg -ss 快速切出該區間暫存檔
    fd, tmp_path = tempfile.mkstemp(prefix="single_seg_", suffix=".mp4")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}", "-i", source_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-strict", "experimental",
            tmp_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=f"FFmpeg 切片失敗: {(proc.stderr or proc.stdout or '')[-500:]}")
        end_time = start_time + duration
        time_range_str = f"{_fmt_hms(start_time)} - {_fmt_hms(end_time)}"
        # 3. 組單段分析請求（與 run_full_pipeline 內格式一致）
        req_data = SegmentAnalysisRequest(
            segment_path=tmp_path,
            segment_index=0,
            start_time=start_time,
            end_time=end_time,
            model_type="qwen",
            qwen_model="qwen2.5vl:latest",
            frames_per_segment=5,
            target_short=432,
            event_detection_prompt=get_event_prompt(),
            summary_prompt=get_summary_prompt(),
            yolo_labels="person,car",
            yolo_every_sec=2.0,
            yolo_score_thr=0.25,
        )
        # 4. 即時推論，不存 DB
        result = AnalysisService.analyze_segment(req_data)
        return result
    finally:
        try:
            if Path(tmp_path).exists():
                os.remove(tmp_path)
        except OSError:
            pass


# POST 端點，接收上傳檔案 (file)、URL (video_url)、或已存在影片 (video_id)
@router.post("/v1/segment_pipeline_multipart")
def segment_pipeline_multipart(
    request: Request,
    model_type: str = Form(...),
    file: UploadFile = File(None),
    video_url: str = Form(None),
    video_id: str = Form(None),
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    qwen_model: str = Form("qwen2.5vl:latest"),
    frames_per_segment: int = Form(5),
    target_short: int = Form(432),
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
    req_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    t1 = None  # pipeline 開始時刻（供 api_process_time 與慢請求診斷）
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    client_host = request.client.host if request.client else "?"
    try:
        vm_pct = psutil.virtual_memory().percent
    except Exception:
        vm_pct = -1.0

    print(
        f"--- [API] 收到 POST /v1/segment_pipeline_multipart | req_id={req_id} | active_same_worker={active_requests_counter} "
        f"| client={client_host} | python_threads≈{threading.active_count()} | rss_mb={mem_before:.1f} | vm_pct={vm_pct} ---",
        flush=True,
    )
    print(
        "--- [診斷] 若客戶端僅見 503 且本行未出現，請查 Nginx backlog / worker 數；若見 500 請對照下方 traceback 與 [GPU_LOCK] 等待時間 ---",
        flush=True,
    )
    logger.info(f"--- [Pipeline Start] Model: {model_type}, Qwen: {qwen_model}, Strict: {strict_segmentation} ---")
    if strict_segmentation:
        logger.warning("--- [Diagnostics] Performing Re-encoding (CPU Intensive) ---")

    get_db, get_api_key, _save_results_to_postgres, HAS_DB, _, get_event_prompt, get_summary_prompt = _get_db_and_models()

    # 若前端未送或送空字串，則每次請求從 prompts/*.md 重新讀取（無需重啟 worker）
    if not event_detection_prompt or not event_detection_prompt.strip():
        event_detection_prompt = get_event_prompt()
    if not summary_prompt or not summary_prompt.strip():
        summary_prompt = get_summary_prompt()

    _print_ram_diagnosis(f"[START] req_id={req_id} segment_pipeline_multipart（讀取影片前）")

    # 1. 處理上傳與下載
    target_filename = "unknown_video"
    local_path = None
    cleanup = False
    
    try:
        if video_id and video_id.strip():
            print(f"--- [API] 使用已存在 video_id: {video_id} ---")
            local_path = None
        elif file is not None:
            # 檢查檔案大小 (Point 5, 9)
            file_content = file.file.read()
            file_size = len(file_content)
            logger.info(f"--- [Upload Check] Filename: {file.filename}, Size: {file_size} bytes ---")
            
            if file_size == 0:
                logger.error(f"--- [Upload Error] File is empty (0 bytes). Upload blocked by firewall? ---")
                raise HTTPException(status_code=400, detail="File is empty (0 bytes). Upload blocked by firewall?")
            print(f"--- [API] 已收到上傳檔案: {file.filename or 'video.mp4'}, 大小: {file_size} bytes ---")
            target_filename = file.filename or "video.mp4"
            fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(target_filename).suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(file_content)
            local_path, cleanup = tmp, True
            del file_content
        elif video_url:
            print(f"--- [API] 使用 video_url 下載: {video_url[:80]}... ---")
            target_filename = Path(video_url).name or "video_url.mp4"
            local_path, cleanup = VideoService.download_to_temp(video_url), True
        else:
            raise HTTPException(status_code=422, detail="需要 file、video_url 或 video_id")

        # 2. 準備片段（上傳檔案時不傳 video_id 給 prepare_segments，讓它用 local_path）
        try:
            seg_dir, seg_files, stem, total_duration = VideoService.prepare_segments(
                local_path, None if file else video_id, target_filename, segment_duration, overlap, target_short, strict_segmentation
            )
        except ValueError as ve:
            logger.error(f"--- [FFmpeg Error] {ve} ---")
            raise HTTPException(status_code=500, detail=f"FFmpeg Processing Failed: {ve}")

        print(f"--- [API] req_id={req_id} 片段準備完成: 共 {len(seg_files)} 個片段, stem={stem} ---", flush=True)
        # 3. 執行 Pipeline（t1：僅 pipeline，不含上傳/FFmpeg 切片）
        print(f"--- [API] req_id={req_id} 開始執行分析 pipeline (model_type={model_type}), 共 {len(seg_files)} 段 ---", flush=True)
        t1 = time.time()
        try:
            results = AnalysisService.run_full_pipeline(
                seg_files, total_duration, segment_duration, overlap,
                model_type, qwen_model, frames_per_segment, target_short, sampling_fps,
                event_detection_prompt, summary_prompt, yolo_labels, yolo_every_sec, yolo_score_thr
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"--- [API] req_id={req_id} CUDA OOM ---", flush=True)
            traceback.print_exc()
            logger.error(f"--- [CUDA OOM] req_id={req_id} {e} ---", exc_info=True)
            raise HTTPException(status_code=500, detail="CUDA Out of Memory: GPU is overloaded. Please try a smaller model or fewer frames.")
        except MemoryError as e:
            print(f"--- [API] req_id={req_id} MemoryError ---", flush=True)
            traceback.print_exc()
            logger.error(f"--- [OOM] req_id={req_id} {e} ---", exc_info=True)
            raise HTTPException(status_code=500, detail="System Out of Memory: CPU/RAM is overloaded.")
        finally:
            # 整次請求結束後釋放 GPU 暫存，模型保留在 worker 內
            AnalysisService.release_gpu_memory()
        
        t2 = time.time()
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        ok_count = sum(1 for r in results if r.get("success"))
        wall_total = t2 - t0
        if wall_total > 10.0 and t1 is not None:
            proc_only = t2 - t1
            print(
                f"--- [API][SlowRequest] req_id={req_id} api_process_time={round(proc_only, 2)}s (pipeline 起訖) "
                f"api_total_time={round(wall_total, 2)}s (含上傳/切片) | "
                f"差額≈{round(wall_total - proc_only, 2)}s → 大差額多為 I/O/切片；pipeline 內慢多為 AI/GPU ---",
                flush=True,
            )
        print(
            f"--- [API] Pipeline 完成: req_id={req_id} 耗時 {round(t2-t0, 2)}s, success_segments={ok_count}/{len(results)} "
            f"active_same_worker={active_requests_counter} ---",
            flush=True,
        )
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
                _save_results_to_postgres(db, results, video_id or stem)
            finally:
                db.close()

        if config.CLEANUP_YOLO_CROPS:
            try:
                AnalysisService.cleanup_yolo_object_crop_files(results, Path(seg_dir))
            except Exception as crop_clean_err:
                logger.warning(
                    "--- [Pipeline] object_crops 清理失敗: %s ---",
                    crop_clean_err,
                    exc_info=True,
                )

        response_obj = JSONResponse(resp)
        try:
            del results
            del resp
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return response_obj

    except HTTPException:
        raise
    except Exception as e:
        print(f"--- [API] req_id={req_id} segment_pipeline_multipart 未預期錯誤 ---", flush=True)
        traceback.print_exc()
        logger.error(f"--- [Unexpected Error] req_id={req_id} {type(e).__name__}: {e} ---", exc_info=True)
        try:
            te = time.time()
            if t1 is not None and (te - t0) > 10.0:
                print(
                    f"--- [API][SlowRequest][錯誤路徑] req_id={req_id} "
                    f"自 pipeline 起算≈{round(te - t1, 2)}s 總耗時≈{round(te - t0, 2)}s ---",
                    flush=True,
                )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {type(e).__name__} - {str(e)}")
    finally:
        active_requests_counter -= 1
        if cleanup and local_path:
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except OSError:
                pass
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

# 批次端點：配合 test_segment_pipeline_rtsp_batch.py 使用
@router.post("/v1/segment_pipeline_batch")
def segment_pipeline_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    segment_duration: float = Form(10.0),
    qwen_model: str = Form("Qwen/Qwen2.5-VL-7B-Instruct"),
    frames_per_segment: int = Form(5),
    target_short: int = Form(432),
    sampling_fps: Optional[float] = Form(None),
    yolo_labels: Optional[str] = Form(None),
    yolo_every_sec: float = Form(2.0),
    yolo_score_thr: float = Form(0.25),
    event_detection_prompt: str = Form(""),
    summary_prompt: str = Form(""),
    save_json: bool = Form(False),
    qwen_inference_batch_size: Optional[int] = Form(None),
    yolo_batch_size: Optional[int] = Form(None),
):

    # 全域計數器，用來監控目前有多少個請求正在處理中。
    global active_requests_counter
    active_requests_counter += 1
    t0 = time.time()

    # 記錄處理前的系統記憶體使用量（MB），用於後續診斷。
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)

    # 限制單次 API 請求最多只能處理的影片片段數，防止過載。
    MAX_BATCH = 16
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"最多上傳 {MAX_BATCH} 個檔案")

    # 取得資料庫連線與 API 金鑰，以及預設的 event_detection_prompt 和 summary_prompt。
    get_db, get_api_key, _save_results_to_postgres, HAS_DB, _, get_event_prompt, get_summary_prompt = _get_db_and_models()
    if not event_detection_prompt or not event_detection_prompt.strip():
        event_detection_prompt = get_event_prompt()
    if not summary_prompt or not summary_prompt.strip():
        summary_prompt = get_summary_prompt()

    temp_dir = None
    seg_files: List[Path] = []
    try:

        # 在系統中建立一個唯一的臨時資料夾，存放上傳的影片片段。
        temp_dir = Path(tempfile.mkdtemp(prefix="segment_batch_"))

        # 遍歷所有上傳的檔案，讀取二進位內容並寫入磁碟。這是為了讓後續的 OpenCV 或 FFmpeg 能透過路徑讀取影片。
        for i, uf in enumerate(files):
            content = uf.file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail=f"檔案 {i+1} 為空")
            seg_path = temp_dir / f"segment_{i:03d}.mp4"
            seg_path.write_bytes(content)
            seg_files.append(seg_path)

        if not seg_files:
            raise HTTPException(status_code=400, detail="無有效影片檔")

        num_segments = len(seg_files)
       
        # 計算 VLM 一次需要處理的段數（未傳 qwen_inference_batch_size 則使用 MAX_BATCH）。
        qwen_batch = qwen_inference_batch_size if qwen_inference_batch_size is not None else MAX_BATCH

        # 計算每段影片需要處理的幀數（每 segment_duration 秒處理一幀）。
        frames_per_seg = max(1, int(segment_duration / yolo_every_sec))

        # 計算 YOLO 一次需要處理的幀數（num_segments * frames_per_seg）。
        yolo_frame_batch = yolo_batch_size if yolo_batch_size is not None else (num_segments * frames_per_seg)

        # 計算總影片長度（num_segments * segment_duration）。
        total_duration = num_segments * segment_duration

        # 記錄開始時間，用於計算整個處理過程的耗時。
        t1 = time.time()
        try:
            results = AnalysisService.run_full_pipeline(
                seg_files,
                total_duration,
                segment_duration,
                0.0,
                "qwen_hf",
                qwen_model,
                frames_per_segment,
                target_short,
                sampling_fps,
                event_detection_prompt,
                summary_prompt,
                yolo_labels,
                yolo_every_sec,
                yolo_score_thr,
                worker_count=4,
                qwen_inference_batch_size=qwen_batch,
                yolo_batch_size=yolo_frame_batch,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"--- [Batch Pipeline] CUDA OOM: {e} ---")
            raise HTTPException(status_code=500, detail="GPU 記憶體不足")
        except MemoryError as e:
            logger.error(f"--- [Batch Pipeline] OOM: {e} ---")
            raise HTTPException(status_code=500, detail="系統記憶體不足")
        finally:
            AnalysisService.release_gpu_memory()

        t2 = time.time()
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info(f"--- [Batch Pipeline] Time: {round(t2-t0, 2)}s, Mem Delta: {round(mem_after - mem_before, 2)}MB ---")

        stem = temp_dir.name

        # 組合回傳結果
        resp = {
            "model_type": "qwen_hf",
            "total_segments": len(results),
            "success_segments": sum(1 for r in results if r.get("success")),
            "results": results,
            "process_time_sec": round(t2 - t1, 2),
            "total_time_sec": round(t2 - t0, 2),
            "stem": stem,
            "diagnostics": {
                "mem_delta_mb": round(mem_after - mem_before, 2),
                "batch_size": len(seg_files),
                "qwen_inference_batch_size": qwen_batch,
                "yolo_batch_size": yolo_frame_batch,
            },
        }
        if save_json:
            save_path = temp_dir / f"{stem}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(resp, f, ensure_ascii=False, indent=2)
            resp["save_path"] = str(save_path)

        # 如果資料庫連線成功，則將結果存入資料庫。
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
        logger.error(f"--- [Batch Pipeline Error] {type(e).__name__}: {e} ---", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    finally:
        active_requests_counter -= 1
        if temp_dir and temp_dir.exists():
            try:
                for p in seg_files:
                    if p.exists():
                        p.unlink()
                temp_dir.rmdir()
            except OSError:
                pass


@router.post("/v1/segment_pipeline_multipart_vllm_video_direct")
def segment_pipeline_multipart_vllm_video_direct(
    request: Request,
    file: UploadFile = File(None),
    video_url: str = Form(None),
    video_id: str = Form(None),
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    qwen_model: str = Form("qwen3-vl"),
    target_short: int = Form(432),  # 保留參數型態相容，實際不使用
    sampling_fps: Optional[float] = Form(None),  # 保留參數型態相容，實際不使用
    strict_segmentation: bool = Form(False),
    event_detection_prompt: str = Form(""),
    summary_prompt: str = Form(""),
    save_json: bool = Form(True),
    save_basename: str = Form(None),
    qwen_inference_batch_size: Optional[int] = Form(None),
    video_num_frames: Optional[int] = Form(None),
    video_sample_fps: Optional[float] = Form(None),
):
    """
    新流程：
    1) segment_pipeline_multipart 同樣的輸入與切片流程
    2) 僅跑 vLLM（不跑 YOLO）
    3) vLLM 走影片直送（不在後端先截圖）；可附帶 video_url.num_frames 或依每秒取樣幀數換算。
    """
    global active_requests_counter
    active_requests_counter += 1
    req_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    t1 = None
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    client_host = request.client.host if request.client else "?"
    try:
        vm_pct = psutil.virtual_memory().percent
    except Exception:
        vm_pct = -1.0

    print(
        f"--- [API][VideoDirect] 收到 POST /v1/segment_pipeline_multipart_vllm_video_direct | req_id={req_id} "
        f"| active_same_worker={active_requests_counter} | client={client_host} | rss_mb={mem_before:.1f} | vm_pct={vm_pct} ---",
        flush=True,
    )

    get_db, get_api_key, _save_results_to_postgres, HAS_DB, _, get_event_prompt, get_summary_prompt = _get_db_and_models()

    if not event_detection_prompt or not event_detection_prompt.strip():
        event_detection_prompt = get_event_prompt()
    if not summary_prompt or not summary_prompt.strip():
        summary_prompt = get_summary_prompt()

    target_filename = "unknown_video"
    local_path = None
    cleanup = False

    try:
        if video_id and video_id.strip():
            print(f"--- [API][VideoDirect] 使用已存在 video_id: {video_id} ---", flush=True)
            local_path = None
        elif file is not None:
            file_content = file.file.read()
            file_size = len(file_content)
            if file_size == 0:
                raise HTTPException(status_code=400, detail="File is empty (0 bytes).")
            target_filename = file.filename or "video.mp4"
            fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(target_filename).suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(file_content)
            local_path, cleanup = tmp, True
            del file_content
        elif video_url:
            target_filename = Path(video_url).name or "video_url.mp4"
            local_path, cleanup = VideoService.download_to_temp(video_url), True
        else:
            raise HTTPException(status_code=422, detail="需要 file、video_url 或 video_id")

        try:
            seg_dir, seg_files, stem, total_duration = VideoService.prepare_segments(
                local_path,
                None if file else video_id,
                target_filename,
                segment_duration,
                overlap,
                target_short,
                strict_segmentation,
            )
        except ValueError as ve:
            raise HTTPException(status_code=500, detail=f"FFmpeg Processing Failed: {ve}")

        t1 = time.time()
        results = AnalysisService.run_vllm_video_direct_pipeline(
            seg_files=seg_files,
            total_duration=total_duration,
            segment_duration=segment_duration,
            overlap=overlap,
            qwen_model=qwen_model,
            event_detection_prompt=event_detection_prompt,
            summary_prompt=summary_prompt,
            qwen_inference_batch_size=qwen_inference_batch_size,
            video_num_frames=video_num_frames,
            video_sample_fps=video_sample_fps,
        )

        t2 = time.time()
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        _vd_nf = resolve_vllm_video_direct_num_frames(
            segment_duration,
            request_num_frames=video_num_frames,
            request_sample_fps=video_sample_fps,
        )
        resp = {
            "model_type": "vllm_qwen3_video_direct",
            "total_segments": len(results),
            "success_segments": sum(1 for r in results if r.get("success")),
            "results": results,
            "process_time_sec": round(t2 - t1, 2),
            "total_time_sec": round(t2 - t0, 2),
            "stem": stem,
            "diagnostics": {
                "mem_delta_mb": round(mem_after - mem_before, 2),
                "strict_mode": strict_segmentation,
                "video_direct": True,
                "sampling_fps_ignored": sampling_fps,
                "video_direct_num_frames": _vd_nf,
                "video_direct_sample_fps_request": video_sample_fps,
                "video_direct_num_frames_request": video_num_frames,
            },
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
                _save_results_to_postgres(db, results, video_id or stem)
            finally:
                db.close()

        return JSONResponse(resp)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {type(e).__name__} - {str(e)}")
    finally:
        active_requests_counter -= 1
        if cleanup and local_path:
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except OSError:
                pass
        try:
            AnalysisService.release_gpu_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

# POST 端點：來源為 RTSP，其餘流程與 /v1/segment_pipeline_multipart 完全相同
@router.post("/v1/segment_pipeline_rtsp")
def segment_pipeline_rtsp(
    request: Request,
    rtsp_url: str = Form(...),
    video_id: str = Form(None),
    capture_duration: float = Form(60.0),
    model_type: str = Form(...),
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    qwen_model: str = Form("qwen2.5vl:latest"),
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

    logger.info(f"--- [Pipeline RTSP] Model: {model_type}, Qwen: {qwen_model}, Strict: {strict_segmentation} ---")

    get_db, get_api_key, _save_results_to_postgres, HAS_DB, _, get_event_prompt, get_summary_prompt = _get_db_and_models()
    if not event_detection_prompt or not event_detection_prompt.strip():
        event_detection_prompt = get_event_prompt()
    if not summary_prompt or not summary_prompt.strip():
        summary_prompt = get_summary_prompt()

    local_path = None
    cleanup = False
    stem = None

    try:
        # 1. 從 RTSP 擷取到暫存檔
        rtsp_url = (rtsp_url or "").strip()
        if not rtsp_url or not rtsp_url.lower().startswith("rtsp://"):
            raise HTTPException(status_code=422, detail="需要有效的 rtsp_url（以 rtsp:// 開頭）")
        logger.info(f"--- [Pipeline RTSP] Capturing {capture_duration}s from {rtsp_url[:60]}... ---")
        local_path = VideoService.capture_rtsp_to_temp(rtsp_url, capture_duration)
        cleanup = True
        target_filename = (video_id.strip() if video_id and video_id.strip() else "rtsp_capture") + ".mp4"

        # 2. 準備片段（與 multipart 相同：同一支 prepare_segments）
        try:
            seg_dir, seg_files, stem, total_duration = VideoService.prepare_segments(
                local_path, None, target_filename, segment_duration, overlap, target_short, strict_segmentation
            )
        except ValueError as ve:
            logger.error(f"--- [FFmpeg Error] {ve} ---")
            raise HTTPException(status_code=500, detail=f"FFmpeg Processing Failed: {ve}")

        t1 = time.time()
        # 3. 執行 Pipeline（與 multipart 相同）
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
        finally:
            AnalysisService.release_gpu_memory()

        t2 = time.time()
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info(f"--- [Pipeline RTSP Success] Time: {round(t2-t0, 2)}s, Mem Delta: {round(mem_after - mem_before, 2)}MB ---")

        time_ollama_sec = sum(r.get("time_ollama_sec", 0) for r in results)
        time_yolo_reid_sec = sum(r.get("time_yolo_reid_sec", 0) for r in results)
        resp = {
            "model_type": model_type,
            "total_segments": len(results),
            "success_segments": sum(1 for r in results if r.get("success")),
            "results": results,
            "process_time_sec": round(t2 - t1, 2),
            "total_time_sec": round(t2 - t0, 2),
            "stem": stem,
            "time_ollama_sec": round(time_ollama_sec, 2),
            "time_yolo_reid_sec": round(time_yolo_reid_sec, 2),
            "diagnostics": {
                "mem_delta_mb": round(mem_after - mem_before, 2),
                "strict_mode": strict_segmentation,
                "source": "rtsp",
                "capture_duration_sec": capture_duration,
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
        logger.error(f"--- [Pipeline RTSP Error] {type(e).__name__}: {e} ---", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {type(e).__name__} - {str(e)}")
    finally:
        active_requests_counter -= 1
        if cleanup and local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass

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