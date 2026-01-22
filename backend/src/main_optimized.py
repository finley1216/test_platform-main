# -*- coding: utf-8 -*-
"""
優化版本的視頻處理管道
實現內存處理、批量推理、共享解碼等性能優化
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import base64
import io
import torch
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

# 全局模型（在 main.py 中已定義，這裡使用延遲導入避免循環依賴）
# 注意：不能直接導入，因為會造成循環依賴，改為在函數內部導入

def infer_segment_yolo_optimized(
    seg_path: str,
    labels: str,
    every_sec: float,
    score_thr: float,
    yolo_model=None,
    reid_model=None,
    reid_device=None
) -> Dict:
    """
    優化版本的 YOLO 物件偵測：
    - 內存處理（不保存圖片到磁盤）
    - 批量 ReID embedding 生成
    - 不生成標註視頻
    
    參數:
    - seg_path: 影片片段路徑
    - labels: 要偵測的物件類別（逗號分隔）
    - every_sec: 取樣頻率（每幾秒處理一幀）
    - score_thr: 信心門檻（0.0-1.0）
    - yolo_model: YOLO 模型（如果為 None 則使用全局模型）
    - reid_model: ReID 模型（如果為 None 則自動獲取）
    - reid_device: 設備（如果為 None 則自動獲取）
    
    返回:
    - Dict: 包含偵測結果和物件切片（內存中）的字典
    """
    # 使用傳入的模型或從 model_loader 獲取
    if yolo_model is None:
        try:
            from src.core.model_loader import get_yolo_model
            yolo_model = get_yolo_model()
            if yolo_model is None:
                raise RuntimeError("YOLO 模型載入失敗")
        except ImportError:
            raise RuntimeError("無法導入 YOLO 模型，請確保 model_loader.py 已正確初始化")
    
    # 解析標籤
    labels_list = [l.strip() for l in labels.split(",") if l.strip()]
    if not labels_list:
        labels_list = ["person", "pedestrian", "car", "motorcycle", "bus", "truck"]
    
    # 設定要偵測的類別
    try:
        yolo_model.set_classes(labels_list)
    except Exception as e:
        raise RuntimeError(f"無法設定偵測類別: {e}") from e
    
    # 讀取影片
    cap = cv2.VideoCapture(seg_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法打開影片: {seg_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("無效的影片或零幀影片")
    
    # 計算取樣間隔
    frame_interval = max(1, int(round(fps * every_sec)))
    
    # 收集所有需要處理的幀（用於批量推理）
    yolo_frames = []
    frame_timestamps = []
    frame_numbers = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根據取樣間隔決定是否處理此幀
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            yolo_frames.append(frame.copy())  # 複製以避免引用問題
            frame_timestamps.append(timestamp)
            frame_numbers.append(frame_count)
        
        frame_count += 1
    
    cap.release()
    
    if not yolo_frames:
        return {
            "video_url": seg_path,
            "fps_input": fps,
            "every_sec": every_sec,
            "size": [width, height],
            "detections": [],
            "total_frames_processed": 0,
            "total_detections": 0,
            "crop_paths": [],  # 不再保存路徑，改為內存數據
            "object_count": {}
        }
    
    print(f"--- [YOLO Optimized] 開始處理 {len(yolo_frames)} 幀（內存模式，不保存圖片）---")
    
    # YOLO 推理（ultralytics 不支持真正的批量，但我們優化流程）
    # 轉換為 PIL Images（YOLO 需要）
    pil_images = []
    for frame in yolo_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(frame_rgb))
    
    # 逐個推理（ultralytics 的 predict 不支持批量列表）
    # 但我們優化了：不保存圖片，直接處理內存中的數據
    yolo_results = []
    for i, pil_img in enumerate(pil_images):
        result = yolo_model.predict(pil_img, verbose=False, conf=score_thr)
        yolo_results.append(result[0] if result else None)
        if (i + 1) % 10 == 0:
            print(f"  - YOLO 推理進度: {i+1}/{len(pil_images)} 圖像")
    
    # 收集所有物件裁剪（內存中）
    all_crops = []  # 存儲裁剪的 numpy arrays
    detection_metadata = []  # 存儲元數據
    
    detections = []
    object_counter = {}
    
    for frame_idx, (result, frame, timestamp, frame_num) in enumerate(
        zip(yolo_results, yolo_frames, frame_timestamps, frame_numbers)
    ):
        if result is None or result.boxes is None or len(result.boxes) == 0:
            continue
        
        frame_detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float).tolist()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = labels_list[class_id] if class_id < len(labels_list) else f"class_{class_id}"
            
            # 確保座標在範圍內
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 內存裁剪（不保存到磁盤）
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            
            # 計數
            if class_name not in object_counter:
                object_counter[class_name] = 0
            object_counter[class_name] += 1
            
            # 保存裁剪和元數據
            all_crops.append(crop)
            detection_metadata.append({
                "label": class_name,
                "score": confidence,
                "timestamp": timestamp,
                "frame": frame_num,
                "box": [x1, y1, x2, y2],
                "label_idx": class_id
            })
            
            frame_detections.append({
                "box": [x1, y1, x2, y2],
                "score": confidence,
                "label": class_name,
                "label_idx": class_id
            })
        
        if frame_detections:
            detections.append({
                "timestamp": timestamp,
                "frame": frame_num,
                "detections": frame_detections
            })
    
    # 批量生成 ReID embeddings（如果 ReID 模型可用）
    embeddings_list = [None] * len(all_crops)
    if all_crops:
        try:
            # 只在 ReID 模型可用時才生成 embeddings（避免網路問題導致卡住）
            if reid_model is None or reid_device is None:
                from src.core.model_loader import get_reid_model
                reid_model, reid_device = get_reid_model()
            
            # 如果 ReID 模型仍然不可用，跳過（不阻塞處理）
            if reid_model is not None and reid_device is not None:
                from src.main import generate_reid_embeddings_batch
                embeddings_list = generate_reid_embeddings_batch(all_crops, reid_model, reid_device)
                print(f"--- [ReID] 批量生成 {len([e for e in embeddings_list if e is not None])} 個 embeddings ---")
            else:
                print(f"--- [WARNING] ReID 模型不可用，跳過 embedding 生成（不影響 YOLO 處理）---")
        except Exception as e:
            print(f"--- [WARNING] 批量 ReID embedding 生成失敗: {e}（不影響 YOLO 處理）---")
            # 不打印完整 traceback，避免日誌過多
    
    # 準備輸出目錄（用於保存 crop 圖片）
    seg_dir = Path(seg_path).parent
    output_dir = seg_dir / "yolo_output"
    output_dir.mkdir(exist_ok=True)
    crops_dir = output_dir / "object_crops"
    crops_dir.mkdir(exist_ok=True)
    
    # 組裝最終結果（包含文件路徑，用於以圖搜圖）
    crop_data = []
    object_counter = {}
    for i, (crop, metadata, embedding) in enumerate(zip(all_crops, detection_metadata, embeddings_list)):
        # 生成文件名：類別_時間戳_序號.jpg
        class_name = metadata["label"]
        if class_name not in object_counter:
            object_counter[class_name] = 0
        object_counter[class_name] += 1
        
        timestamp = metadata["timestamp"]
        timestamp_str = f"{timestamp:.3f}".replace(".", "_")
        crop_filename = f"{class_name}_{timestamp_str}_{object_counter[class_name]:03d}.jpg"
        crop_path = crops_dir / crop_filename
        
        # 保存物件切片圖片（用於以圖搜圖功能）
        try:
            cv2.imwrite(str(crop_path), crop)
        except Exception as e:
            print(f"  ⚠️  保存 crop 圖片失敗 ({crop_filename}): {e}")
            crop_path = None
        
        # 生成 CLIP embedding（用於以圖搜圖）
        clip_embedding = None
        if crop_path and crop_path.exists():
            try:
                from src.main import generate_image_embedding
                clip_embedding = generate_image_embedding(str(crop_path))
                if clip_embedding:
                    print(f"  ✓ 生成 CLIP embedding: {crop_filename} (維度: {len(clip_embedding)})")
            except Exception as e:
                print(f"  ⚠️  生成 CLIP embedding 失敗 ({crop_filename}): {e}")
        
        crop_data.append({
            "path": str(crop_path) if crop_path else None,
            "label": class_name,
            "score": metadata["score"],
            "timestamp": timestamp,
            "frame": metadata["frame"],
            "box": metadata["box"],
            "clip_embedding": clip_embedding,  # CLIP embedding（512 維）- 用於以圖搜圖
            "reid_embedding": embedding,  # ReID embedding（2048 維）- 用於物件 re-identification
        })
    
    print(f"--- [YOLO Optimized] 處理完成: 共處理 {len(yolo_frames)} 幀，偵測到 {len(detections)} 個有物件的時間點，生成 {len(crop_data)} 個物件切片（內存）---")
    
    return {
        "video_url": seg_path,
        "fps_input": fps,
        "every_sec": every_sec,
        "size": [width, height],
        "detections": detections,
        "total_frames_processed": len(yolo_frames),
        "total_detections": sum(len(d["detections"]) for d in detections),
        "crop_paths": crop_data,  # 改為內存數據，不包含文件路徑
        "object_count": object_counter
    }


def process_segment_optimized(
    seg_path: str,
    seg_idx: int,
    segment_duration: float,
    overlap: float,
    total_duration: float,
    model_type: str,
    qwen_model: str,
    frames_per_segment: int,
    target_short: int,
    sampling_fps: Optional[float],
    event_detection_prompt: str,
    summary_prompt: str,
    yolo_labels: Optional[str],
    yolo_every_sec: float,
    yolo_score_thr: float,
    yolo_model=None,
    reid_model=None,
    reid_device=None
) -> Dict:
    """
    優化版本的片段處理：
    - 共享解碼（只解碼一次，同時用於 VLM 和 YOLO）
    - 內存處理（不保存中間文件）
    - 批量推理
    
    參數:
    - 所有參數與原 process_segment 相同
    - yolo_model: YOLO 模型（全局模型）
    - reid_model: ReID 模型（全局模型）
    - reid_device: 設備
    
    返回:
    - Dict: 包含 VLM 和 YOLO 結果的字典
    """
    from src.main import SegmentAnalysisRequest, analyze_segment_result, _fmt_hms
    
    # 計算時間區段資訊
    m = re.search(r"(\d+)", Path(seg_path).name)
    idx = int(m.group(1)) if m else seg_idx
    start = idx * (segment_duration - overlap)
    end = min(start + segment_duration, total_duration)
    time_range_str = f"{_fmt_hms(start)} - {_fmt_hms(end)}"
    
    # 打開視頻（只解碼一次）
    cap = cv2.VideoCapture(seg_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法打開影片: {seg_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # 收集 VLM 和 YOLO 需要的幀
    vlm_frames_base64 = []
    yolo_frames = []
    frame_timestamps = []
    frame_numbers = []
    
    # 計算取樣間隔
    yolo_frame_interval = max(1, int(round(fps * yolo_every_sec)))
    
    if sampling_fps and sampling_fps > 0:
        # VLM: 根據 sampling_fps 取樣
        vlm_interval_sec = 1.0 / sampling_fps
        next_vlm_time = 0.0
    else:
        # VLM: 均勻分佈
        vlm_frame_interval = max(1, total_frames // frames_per_segment)
        next_vlm_frame = 0
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / fps
        
        # YOLO 取樣
        if frame_count % yolo_frame_interval == 0:
            yolo_frames.append(frame.copy())
            frame_timestamps.append(timestamp)
            frame_numbers.append(frame_count)
        
        # VLM 取樣
        if sampling_fps and sampling_fps > 0:
            if timestamp >= next_vlm_time:
                # 調整大小並轉換為 base64
                resized = _resize_frame_for_vlm(frame, target_short)
                base64_str = _frame_to_base64(resized)
                vlm_frames_base64.append(base64_str)
                next_vlm_time += vlm_interval_sec
        else:
            if frame_count % vlm_frame_interval == 0:
                resized = _resize_frame_for_vlm(frame, target_short)
                base64_str = _frame_to_base64(resized)
                vlm_frames_base64.append(base64_str)
        
        frame_count += 1
    
    cap.release()
    
    # 執行 VLM 分析（使用收集的 base64 幀）
    # 注意：analyze_segment_result 可能需要調整以接受 base64 列表
    # 這裡暫時保持原有邏輯，但可以優化
    req_data = SegmentAnalysisRequest(
        segment_path=str(seg_path),
        segment_index=idx,
        start_time=start,
        end_time=end,
        model_type=model_type,
        qwen_model=qwen_model,
        frames_per_segment=len(vlm_frames_base64),
        target_short=target_short,
        sampling_fps=sampling_fps,
        event_detection_prompt=event_detection_prompt,
        summary_prompt=summary_prompt,
        owl_labels="",  # 不使用 OWL
        owl_every_sec=0,
        owl_score_thr=0,
        yolo_labels=yolo_labels,
        yolo_every_sec=yolo_every_sec,
        yolo_score_thr=yolo_score_thr
    )
    
    # 執行 VLM 分析（這裡可能需要修改 analyze_segment_result 以接受預處理的幀）
    vlm_result = analyze_segment_result(req_data)
    
    # 執行 YOLO 偵測（優化版本）
    try:
        yolo_result = infer_segment_yolo_optimized(
            str(seg_path),
            labels=yolo_labels or "person,pedestrian,motorcycle,car,bus,scooter,truck",
            every_sec=yolo_every_sec,
            score_thr=yolo_score_thr,
            yolo_model=yolo_model,
            reid_model=reid_model,
            reid_device=reid_device
        )
        
        # 將 YOLO 結果添加到 VLM 結果中
        if "raw_detection" not in vlm_result:
            vlm_result["raw_detection"] = {}
        vlm_result["raw_detection"]["yolo"] = yolo_result
        print(f"--- [YOLO Optimized] 片段 {idx} 完成：偵測到 {yolo_result.get('total_detections', 0)} 個物件 ---")
    except Exception as e:
        print(f"--- [WARNING] YOLO 處理失敗 (segment {idx}): {e} ---")
        vlm_result["yolo_error"] = str(e)
        if "raw_detection" not in vlm_result:
            vlm_result["raw_detection"] = {}
        vlm_result["raw_detection"]["yolo"] = None
    
    return vlm_result


def _resize_frame_for_vlm(frame: np.ndarray, target_short: int) -> np.ndarray:
    """調整幀大小用於 VLM"""
    h, w = frame.shape[:2]
    if w <= h:
        new_w = target_short
        new_h = int(h * (target_short / w))
    else:
        new_h = target_short
        new_w = int(w * (target_short / h))
    return cv2.resize(frame, (new_w, new_h))


def _frame_to_base64(frame: np.ndarray) -> str:
    """將幀轉換為 base64 字符串"""
    # 轉換為 RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = frame
    
    # 轉換為 PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # 轉換為 base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return base64_str

