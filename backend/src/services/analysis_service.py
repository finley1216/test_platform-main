# -*- coding: utf-8 -*-
import os
import time
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.model_loader import get_yolo_model, get_reid_model, get_clip_model
from src.utils.video_utils import _sample_frames_evenly_to_pil, _fmt_hms
from src.utils.image_utils import _resize_short_side, _pil_to_b64
from src.utils.ollama_utils import (
    _ollama_chat, _safe_parse_json, _extract_first_json, 
    _clean_summary_text, _extract_first_json_and_tail
)
from src.config import config

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# [全局修復] 確保所有 ReID 模型都使用 FP32
def _ensure_reid_model_fp32(reid_model, reid_device):
    """強制將 ReID 模型轉換為 FP32，避免 c10::Half != float 錯誤"""
    if reid_model is None:
        return None
    try:
        first_param = next(reid_model.parameters())
        if first_param.dtype == torch.float16:
            print("--- [AnalysisService] 檢測到 ReID 模型為 FP16，強制轉換為 FP32 ---")
            reid_model = reid_model.float()
            if reid_device:
                reid_model = reid_model.to(reid_device)
            reid_model.eval()
        else:
            # 即使已經是 FP32，也確保在正確設備上
            if reid_device:
                reid_model = reid_model.to(reid_device)
            reid_model.eval()
    except (StopIteration, AttributeError) as e:
        # 如果無法檢查，直接轉換為 FP32
        print(f"--- [AnalysisService] 無法檢查 ReID 模型類型，強制轉換為 FP32: {e} ---")
        reid_model = reid_model.float()
        if reid_device:
            reid_model = reid_model.to(reid_device)
        reid_model.eval()
    return reid_model

class AnalysisService:
    @staticmethod
    def infer_segment_qwen(
        model_name: str,
        video_path: str,
        event_prompt: str,
        summary_prompt: str,
        target_short: int = 720,
        frames_per_segment: int = 8,
        sampling_fps: Optional[float] = None
    ) -> Tuple[Dict, str]:
        """使用 Qwen 模型分析影片片段"""
        try:
            frames_pil = _sample_frames_evenly_to_pil(video_path, max_frames=frames_per_segment, sampling_fps=sampling_fps)
            images_b64 = [_pil_to_b64(_resize_short_side(img, target_short)) for img in frames_pil]
        except Exception as e:
            print(f"--- [Qwen] ✗ 處理失敗: {e} ---")
            return {"error": f"影格擷取失敗: {e}"}, ""

        event_msgs = [
            {"role": "system", "content": "你是『嚴格的災害與人員異常偵測器』。只輸出純 JSON 物件。"},
            {"role": "user", "content": f"{event_prompt}\n\n強制規則：只輸出一個 JSON 物件。"}
        ]
        
        event_error = None
        try:
            print(f"--- [Qwen] 呼叫 Ollama 進行事件偵測 (模型: {model_name})... ---")
            event_txt = _ollama_chat(model_name, event_msgs, images_b64=images_b64)
            print(f"--- [Qwen] 事件偵測完成，長度: {len(event_txt)} ---")
        except Exception as e:
            event_txt, event_error = "", f"Ollama 失敗: {e}"
            print(f"--- [Qwen] ✗ 事件偵測失敗: {e} ---")

        frame_obj = _safe_parse_json(event_txt) or _extract_first_json(event_txt) or {"events": {"reason": ""}, "persons": []}
        if event_error: frame_obj["error"] = event_error

        summary_txt = ""
        if summary_prompt.strip():
            summary_msgs = [
                {"role": "system", "content": "你是影片小結產生器。只能輸出 50–100 個中文字的摘要。"},
                {"role": "user", "content": f"{summary_prompt}\n\n強制規則：只輸出 50–100 字中文。"}
            ]
            try:
                print(f"--- [Qwen] 呼叫 Ollama 進行摘要產生 (模型: {model_name})... ---")
                summary_raw = _ollama_chat(model_name, summary_msgs, images_b64=images_b64)
                summary_txt = _clean_summary_text(summary_raw)
                print(f"--- [Qwen] 摘要產生完成，長度: {len(summary_txt)} ---")
            except Exception as e:
                print(f"--- [Qwen] ✗ 摘要產生失敗: {e} ---")
                summary_txt = ""
        else:
            print("--- [Qwen] 跳過摘要產生 (summary_prompt 為空) ---")

        return frame_obj, summary_txt

    @staticmethod
    def infer_segment_gemini(
        model_name: str,
        video_path: str,
        event_prompt: str,
        summary_prompt: str,
        target_short: int = 720,
        frames_per_segment: int = 8,
        sampling_fps: Optional[float] = None
    ) -> Tuple[Dict, str]:
        """使用 Gemini 模型分析影片片段"""
        if not HAS_GEMINI or not config.GEMINI_API_KEY:
            return {"error": "Gemini 未設定"}, ""

        try:
            frames = _sample_frames_evenly_to_pil(video_path, max_frames=frames_per_segment, sampling_fps=sampling_fps)
            prompt_content = list(frames)
            
            text_instruction = f"""
            你是一個專業的影像分析 AI。請分析附帶的連續影格。
            任務 1 (Event Detection): {event_prompt}
            任務 2 (Summary): {summary_prompt}
            請直接輸出 JSON 物件，不要使用 Markdown。在 JSON 結束後，換行輸出中文摘要。
            """
            prompt_content.append(text_instruction)

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_content,
                generation_config=genai.types.GenerationConfig(temperature=0.2),
                safety_settings=safety_settings
            )
            
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return _extract_first_json_and_tail(clean_text)
        except Exception as e:
            return {"error": f"Gemini 錯誤: {e}"}, ""

    @staticmethod
    def generate_image_embedding(image_path: str) -> Optional[List[float]]:
        """為圖像生成 CLIP embedding"""
        model, processor = get_clip_model()
        if not model or not processor: return None
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
                return feat[0].cpu().tolist()
        except Exception as e:
            print(f"--- [CLIP] ✗ 處理失敗: {e} ---")
            return None

    @staticmethod
    def generate_reid_embeddings_batch(crop_images: List[np.ndarray], reid_model=None, reid_device=None) -> List[Optional[List[float]]]:
        """
        批量生成 ReID embeddings (強制 FP32 版本)
        確保模型與輸入都使用 FP32，徹底解決 c10::Half != float 錯誤。
        """
        if not crop_images:
            return []

        # 1. 確保模型與設備存在
        if reid_model is None:
            from src.core.model_loader import get_reid_model
            reid_model, device_str = get_reid_model()
            if reid_device is None:
                reid_device = device_str
        
        if reid_model is None:
            print("--- [AnalysisService] Warning: ReID model is None ---")
            return [None] * len(crop_images)

        try:
            import torch
            import cv2
            import numpy as np
            
            # 2. 【關鍵】強制模型為 FP32
            reid_model = reid_model.float()
            if reid_device:
                reid_model = reid_model.to(reid_device)
            reid_model.eval()

            # 3. 批量預處理 (ImageNet 正規化)
            processed_tensors = []
            valid_indices = []

            for i, img in enumerate(crop_images):
                if img is None or img.size == 0:
                    continue
                try:
                    # BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize 到 ReID 標準大小 (W=128, H=256)
                    img = cv2.resize(img, (128, 256))
                    # 轉為 Float32 並歸一化到 [0, 1]
                    img = img.astype(np.float32) / 255.0
                    # ImageNet Mean/Std Normalize
                    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = (img - mean) / std
                    # HWC -> CHW (PyTorch 格式)
                    img = img.transpose(2, 0, 1)
                    
                    # 轉為 Tensor
                    tensor = torch.from_numpy(img)
                    processed_tensors.append(tensor)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"--- [ReID] Preprocessing error: {e} ---")

            if not processed_tensors:
                return [None] * len(crop_images)

            # 4. 堆疊並轉型為 FP32
            batch_tensor = torch.stack(processed_tensors)
            if reid_device:
                batch_tensor = batch_tensor.to(reid_device, dtype=torch.float32)
            else:
                batch_tensor = batch_tensor.type(torch.float32)

            # 5. 推論
            with torch.no_grad():
                features = reid_model(batch_tensor)
                # L2 Normalize
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                embeddings = features.cpu().numpy()

            # 6. 整理結果
            result = [None] * len(crop_images)
            for idx, valid_idx in enumerate(valid_indices):
                result[valid_idx] = embeddings[idx].tolist()
            
            return result

        except Exception as e:
            print(f"--- [AnalysisService] Batch ReID failed: {e} ---")
            import traceback
            traceback.print_exc()
            return [None] * len(crop_images)

    @staticmethod
    def analyze_segment(req) -> Dict:
        """分析單一片段的完整邏輯 (VLM + YOLO + ReID)"""
        p = req.segment_path
        tr = f"{_fmt_hms(req.start_time)} - {_fmt_hms(req.end_time)}"
        print(f"--- [AnalysisService] 開始分析片段: {p} ({tr}) ---")
        print(f"--- [AnalysisService] Model: {req.model_type}, Qwen Model: {getattr(req, 'qwen_model', 'N/A')} ---")
        print(f"--- [AnalysisService] Event Prompt Len: {len(req.event_detection_prompt)}, Summary Prompt Len: {len(req.summary_prompt)} ---")
        t1 = time.time()

        result = {
            "segment": Path(p).name,
            "time_range": tr,
            "duration_sec": round(req.end_time - req.start_time, 2),
            "success": False,
            "time_sec": 0.0,
            "parsed": {},
            "raw_detection": {},
            "error": None
        }

        try:
            # 1. 執行 VLM 分析 (Qwen 或 Gemini)
            if req.model_type in ("qwen", "gemini"):
                if req.model_type == "qwen":
                    frame_obj, summary_txt = AnalysisService.infer_segment_qwen(
                        req.qwen_model, p, req.event_detection_prompt, req.summary_prompt,
                        target_short=req.target_short, frames_per_segment=req.frames_per_segment,
                        sampling_fps=req.sampling_fps
                    )
                else:
                    g_model = req.qwen_model if req.qwen_model.startswith("gemini") else "gemini-2.5-flash"
                    frame_obj, summary_txt = AnalysisService.infer_segment_gemini(
                        g_model, p, req.event_detection_prompt, req.summary_prompt,
                        req.target_short, req.frames_per_segment, sampling_fps=req.sampling_fps
                    )

                frame_norm = AnalysisService._normalize_vlm_output(frame_obj)
                
                result["success"] = "error" not in frame_obj
                result["error"] = frame_obj.get("error")
                result["parsed"] = {
                    "frame_analysis": frame_norm,
                    "summary_independent": summary_txt
                }

            # 2. 強制執行 YOLO-World + FastReID (作為正常流程的一部分)
            # 即使是 Qwen 模型，也要跑 YOLO 以獲取物件追蹤資訊
            yolo_labels = req.yolo_labels if hasattr(req, 'yolo_labels') and req.yolo_labels else "person,car"
            yolo_every_sec = req.yolo_every_sec if hasattr(req, 'yolo_every_sec') else 2.0
            yolo_score_thr = req.yolo_score_thr if hasattr(req, 'yolo_score_thr') else 0.25
            
            print(f"--- [AnalysisService] 執行 YOLO+ReID 正常流程 (Labels: {yolo_labels}) ---")
            try:
                yolo_res = AnalysisService.infer_segment_yolo(
                    p, 
                    labels=yolo_labels,
                    every_sec=yolo_every_sec,
                    score_thr=yolo_score_thr
                )
                # [修正] 必須包裝在 "yolo" key 下，後端資料庫保存邏輯才能抓到
                result["raw_detection"] = {"yolo": yolo_res}
            except Exception as yolo_err:
                error_msg = str(yolo_err)
                print(f"--- [AnalysisService] YOLO 處理失敗: {error_msg} ---")
                result["raw_detection"] = {"yolo": None}
                result["error"] = error_msg  # 保留原有錯誤，同時記錄 YOLO 錯誤
            
            # 如果原本不是 Qwen/Gemini，則根據 YOLO 結果決定 success
            if req.model_type == "yolo":
                result["success"] = True

        except Exception as ex:
            print(f"--- [AnalysisService] ✗ 分析發生異常: {ex} ---")
            result["error"] = str(ex)

        result["time_sec"] = round(time.time() - t1, 2)
        return result

    @staticmethod
    def run_full_pipeline(
        seg_files: List[Path],
        total_duration: float,
        segment_duration: float,
        overlap: float,
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
        worker_count: int = 4
    ) -> List[Dict]:
        """執行完整分析流程"""
        results = []
        
        class Req:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)

        def process_one(seg_path, idx):
            start = idx * (segment_duration - overlap)
            end = min(start + segment_duration, total_duration)
            req = Req(
                segment_path=str(seg_path),
                start_time=start,
                end_time=end,
                model_type=model_type,
                qwen_model=qwen_model,
                frames_per_segment=frames_per_segment,
                target_short=target_short,
                sampling_fps=sampling_fps,
                event_detection_prompt=event_detection_prompt,
                summary_prompt=summary_prompt,
                yolo_labels=yolo_labels,
                yolo_every_sec=yolo_every_sec,
                yolo_score_thr=yolo_score_thr
            )
            return AnalysisService.analyze_segment(req)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(process_one, p, i): i for i, p in enumerate(seg_files)}
            for future in as_completed(futures):
                results.append(future.result())
        
        results.sort(key=lambda x: x["segment"])
        return results

    @staticmethod
    def _normalize_vlm_output(frame_obj: Dict) -> Dict:
        """標準化 VLM 輸出的 JSON 格式"""
        default_events = {
            "water_flood": False, "fire": False,
            "abnormal_attire_face_cover_at_entry": False,
            "person_fallen_unmoving": False,
            "double_parking_lane_block": False,
            "smoking_outside_zone": False,
            "crowd_loitering": False,
            "security_door_tamper": False,
            "reason": ""
        }
        
        if not isinstance(frame_obj, dict): return {"events": default_events}
        
        ev = frame_obj.get("events") or {}
        if not isinstance(ev, dict): ev = {}
        
        norm_events = default_events.copy()
        for k, v in ev.items():
            if k in norm_events and k != "reason":
                norm_events[k] = bool(v)
        
        norm_events["reason"] = str(ev.get("reason", ""))
        return {"events": norm_events, "persons": frame_obj.get("persons", [])}

    @staticmethod
    def infer_segment_yolo(seg_path: str, labels: str, every_sec: float, score_thr: float, batch_size: int = 16) -> Dict:
        """使用 YOLO-World 進行物件偵測並執行 FastReID 提取特徵，同時儲存物件切片"""
        
        # 1. 獲取模型
        model = get_yolo_model()
        if not model: 
            # 嘗試強制重載一次
            from src.core.model_loader import get_yolo_model as retry_get_yolo
            print("--- [AnalysisService] YOLO model not found, retrying... ---")
            model = retry_get_yolo()
            if not model:
                return {"error": "YOLO model failed to load", "detections": []}
        
        reid_model, reid_device = get_reid_model()
        
        # 準備目錄
        seg_dir = Path(seg_path).parent
        crops_dir = seg_dir / "yolo_output" / "object_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        labels_list = [l.strip() for l in labels.split(",") if l.strip()] or ["person", "car"]
        
        # 2. 安全設定類別
        try:
            model.set_classes(labels_list)
        except AttributeError as e:
            error_msg = str(e)
            # 捕獲 'NoneType' object has no attribute 'names' 錯誤
            if "'NoneType' object has no attribute 'names'" in error_msg or "'NoneType' object has no attribute" in error_msg:
                print(f"--- [YOLO Error] 模型載入成功但內部結構損壞 ({error_msg})，強制重載... ---")
                # 清空全域變數並重載
                import src.core.model_loader
                src.core.model_loader._yolo_world_model = None
                model = src.core.model_loader.get_yolo_model()
                if model:
                    try:
                        model.set_classes(labels_list)
                        print("--- [YOLO] 模型重載成功，set_classes 執行完成 ---")
                    except Exception as e2:
                        print(f"--- [YOLO Error] 重載後仍然失敗: {e2} ---")
                        return {"error": "YOLO model broken and reload failed", "detections": []}
                else:
                    print("--- [YOLO Error] 模型重載失敗，無法獲取模型實例 ---")
                    return {"error": "YOLO model broken and reload failed", "detections": []}
            else:
                # 其他 AttributeError，直接返回錯誤
                print(f"--- [YOLO Error] AttributeError in set_classes: {e} ---")
                return {"error": f"YOLO set_classes failed: {str(e)}", "detections": []}
        except Exception as e:
            # 捕獲其他可能的錯誤
            print(f"--- [YOLO Error] Unexpected error in set_classes: {e} ---")
            return {"error": f"YOLO set_classes failed: {str(e)}", "detections": []}
        
        cap = cv2.VideoCapture(seg_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        interval = max(1, int(round(fps * every_sec)))
        all_results = []
        crop_paths = []
        
        # 收集需要處理的幀索引
        frame_indices = list(range(0, total_frames, interval))
        
        print(f"--- [YOLO] 開始處理片段: {seg_path}, 共 {len(frame_indices)} 幀 ---")
        
        # 批次處理
        for i in range(0, len(frame_indices), batch_size):
            batch_idxs = frame_indices[i : i + batch_size]
            batch_frames = []
            valid_idxs = []
            
            for fi in batch_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if ok:
                    batch_frames.append(frame)
                    valid_idxs.append(fi)
            
            if not batch_frames:
                continue
                
            # YOLO 批次推論 (加入錯誤處理)
            try:
                batch_predict_res = model.predict(batch_frames, conf=score_thr, verbose=False)
            except Exception as predict_err:
                error_msg = str(predict_err)
                print(f"--- [YOLO Error] Predict failed: {error_msg} ---")
                # 如果是 'names' 相關錯誤，嘗試重載模型
                if "names" in error_msg.lower():
                    print("--- [YOLO] Attempting model reload due to names error... ---")
                    import src.core.model_loader
                    src.core.model_loader._yolo_world_model = None
                    model = src.core.model_loader.get_yolo_model()
                    if model:
                        try:
                            model.set_classes(labels_list)
                            # 重試一次推論
                            batch_predict_res = model.predict(batch_frames, conf=score_thr, verbose=False)
                        except Exception as retry_err:
                            print(f"--- [YOLO Error] Retry failed: {retry_err} ---")
                            continue  # 跳過這個批次
                    else:
                        print("--- [YOLO Error] Model reload failed, skipping batch ---")
                        continue
                else:
                    # 其他類型的錯誤，跳過這個批次
                    continue
            
            for idx, res in enumerate(batch_predict_res):
                fi = valid_idxs[idx]
                timestamp = round(fi / fps, 2)
                frame = batch_frames[idx]
                
                crops_imgs = []
                temp_detections = []
                
                for b_idx, box in enumerate(res.boxes):
                    cls_id = int(box.cls)
                    label = labels_list[cls_id]
                    conf = round(float(box.conf), 3)
                    xyxy = [int(round(float(c))) for i, c in enumerate(box.xyxy[0].tolist())]
                    
                    # 提取切片
                    x1, y1, x2, y2 = xyxy
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        # 儲存切片圖片
                        crop_filename = f"crop_{fi}_{b_idx}_{label}.jpg"
                        crop_path = crops_dir / crop_filename
                        cv2.imwrite(str(crop_path), crop)
                        
                        rel_crop_path = str(crop_path.relative_to(Path(".")))
                        
                        crops_imgs.append(crop)
                        temp_detections.append({
                            "timestamp": timestamp,
                            "label": label,
                            "score": conf,
                            "box": xyxy,
                            "frame": fi,
                            "path": rel_crop_path
                        })
                
                # 如果有切片且 ReID 模型可用，執行批次特徵提取
                if crops_imgs and reid_model:
                    embeddings = AnalysisService.generate_reid_embeddings_batch(crops_imgs, reid_model, reid_device)
                    for d_idx, emb in enumerate(embeddings):
                        if emb:
                            temp_detections[d_idx]["reid_embedding"] = emb
                
                all_results.extend(temp_detections)
                crop_paths.extend(temp_detections) # 這裡 crop_paths 內容與 detections 類似，但為了相容 main.py
        
        cap.release()
        
        # 計算物件統計
        object_count = {}
        for d in all_results:
            label = d["label"]
            object_count[label] = object_count.get(label, 0) + 1
            
        print(f"--- [YOLO] 處理完成，偵測到 {len(all_results)} 個物件，統計: {object_count} ---")
        return {
            "total_detections": len(all_results), 
            "detections": all_results,
            "crop_paths": crop_paths,
            "object_count": object_count,
            "total_frames_processed": len(frame_indices)
        }
