# -*- coding: utf-8 -*-
import os
import gc
import time
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading

from src.core.model_loader import get_yolo_model, get_reid_model, get_clip_model, gpu_inference_lock_traced
from src.utils.video_utils import _sample_frames_evenly_to_pil, _fmt_hms
from src.utils.image_utils import _resize_short_side, _pil_to_b64
from src.utils.ollama_utils import (
    _ollama_chat, _safe_parse_json, _extract_first_json,
    _clean_summary_text, _extract_first_json_and_tail
)
from src.config import config

try:
    from src.utils.vllm_utils import _vllm_chat
    from src.utils.vllm_utils import _vllm_chat_batch
    HAS_VLLM = True
except Exception as _e:
    _vllm_chat = None
    HAS_VLLM = False

# 本機 HF Qwen2.5-VL（batch 推論用）；未安裝 transformers 時為 None
try:
    from src.utils import qwen_hf_utils
    HAS_QWEN_HF = True
except Exception as _e:
    qwen_hf_utils = None
    HAS_QWEN_HF = False
    print("--- [AnalysisService] Qwen HF 模組載入失敗，qwen_hf batch 不可用，改走單段 analyze_segment ---")
    print("--- [AnalysisService] 錯誤:", type(_e).__name__, str(_e), "---")

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


def _release_gpu_memory():
    """釋放 GPU 上推理產生的暫存張量與快取，不卸載模型。供每個片段/請求結束後呼叫。"""
    try:
        if torch.cuda.is_available():
            # 先同步，避免仍在排隊的 CUDA kernel 尚未結束導致 empty_cache 釋放不完全
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 釋放跨進程共享記憶體（CUDA IPC）殘留
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()
    except Exception:
        pass


def release_gpu_memory():
    """對外介面：釋放目前 process 的 GPU 暫存（API 層在整次請求結束後可呼叫）。"""
    _release_gpu_memory()


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
    def release_gpu_memory():
        """釋放 GPU 暫存（不卸載模型），供 API 請求結束後呼叫。"""
        _release_gpu_memory()

    @staticmethod
    def _safe_crop_path_for_storage(crop_path: Path, seg_dir: Path) -> str:
        """
        取得供儲存/DB 使用的 crop 路徑字串。
        若 crop 在專案目錄下則回傳相對專案根的路徑；
        否則（例如片段在 /tmp）回傳相對 seg_dir 或絕對路徑，避免 relative_to 拋錯。
        """
        try:
            return str(crop_path.relative_to(Path(".").resolve()))
        except ValueError:
            try:
                return str(crop_path.relative_to(seg_dir))
            except ValueError:
                return str(crop_path)

    @staticmethod
    def cleanup_yolo_object_crop_files(
        results: Optional[List[Dict]],
        segments_parent: Optional[Path] = None,
    ) -> None:
        """
        ReID embedding 已寫入結果、且 DB/JSON 持久化完成後，刪除 yolo_output/object_crops 暫存圖檔，
        降低磁碟佔用與大量小檔 I/O 造成的壓力。
        segments_parent: 片段檔所在目錄（與 prepare_segments 的 seg_dir 一致），供相對 path 還原實際檔案。
        """
        if not results:
            return
        seg_base = Path(segments_parent).resolve() if segments_parent else None
        for r in results:
            if not isinstance(r, dict):
                continue
            yolo = (r.get("raw_detection") or {}).get("yolo") or {}
            dets = yolo.get("detections") or yolo.get("crop_paths") or []
            if not isinstance(dets, list):
                continue
            for d in dets:
                if not isinstance(d, dict):
                    continue
                p = d.get("path")
                if not p or not isinstance(p, str):
                    continue
                if "object_crops" not in p.replace("\\", "/"):
                    continue
                try:
                    ap = Path(p)
                    if not ap.is_absolute() and seg_base is not None:
                        ap = (seg_base / p).resolve()
                    elif not ap.is_absolute():
                        ap = Path(p).resolve()
                    if ap.is_file():
                        ap.unlink()
                except OSError:
                    pass

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
        """使用 Qwen 模型分析影片片段。改為單一 Ollama 請求同時取得事件 JSON + 摘要，約可減半 VLM 耗時（與 rtsp-recorder 一致）。"""
        try:
            frames_pil = _sample_frames_evenly_to_pil(video_path, max_frames=frames_per_segment, sampling_fps=sampling_fps)
            images_b64 = [_pil_to_b64(_resize_short_side(img, target_short)) for img in frames_pil]
        except Exception as e:
            print(f"--- [Qwen] ✗ 處理失敗: {e} ---")
            return {"error": f"影格擷取失敗: {e}"}, ""

        want_summary = bool(summary_prompt.strip())

        # 有摘要時
        if want_summary:
            # 單一請求：同時要事件 JSON + 摘要，減少一次 round-trip（與 rtsp-recorder 一致）
            combined_instruction = (
                f"{event_prompt}\n\n"
                f"{summary_prompt}\n\n"
                "請輸出「一個」JSON 物件，且必須包含以下欄位：\n"
                "- events: 事件描述物件（可含 reason 等）\n"
                "- persons: 人員相關陣列\n"
                "- summary: 50–100 字繁體中文畫面摘要（字串）\n"
                "只輸出該 JSON，不要其他文字。"
            )
            combined_msgs = [
                {"role": "system", "content": "你是災害與人員異常偵測器，並產出影片摘要。只輸出一個 JSON 物件，含 events、persons、summary。"},
                {"role": "user", "content": combined_instruction}
            ]
            raw = ""
            combined_error = None
            try:
                print(f"--- [Qwen] 呼叫 Ollama 進行事件偵測+摘要 (單一請求, 模型: {model_name})... ---")
                raw = _ollama_chat(model_name, combined_msgs, images_b64=images_b64)
                print(f"--- [Qwen] 事件+摘要完成，回應長度: {len(raw)} ---")
            except Exception as e:
                combined_error = f"Ollama 失敗: {e}"
                print(f"--- [Qwen] ✗ 單一請求失敗: {e} ---")
            combined = _safe_parse_json(raw) or _extract_first_json(raw) or {}
            events_in = combined.get("events") if isinstance(combined.get("events"), dict) else {}
            reason = combined.get("event_reason") or combined.get("reason") or (events_in.get("reason", "") if isinstance(events_in, dict) else "")
            if reason and isinstance(events_in, dict) and "reason" not in events_in:
                events_in = {**events_in, "reason": reason}
            frame_obj = {
                "events": events_in or {"reason": ""},
                "persons": combined.get("persons", []),
            }
            if not isinstance(frame_obj.get("events"), dict):
                frame_obj["events"] = {"reason": ""}
            if combined_error:
                frame_obj["error"] = combined_error
            summary_txt = _clean_summary_text(combined.get("summary") or raw or "")
            if not summary_txt and raw:
                summary_txt = _clean_summary_text(raw)
            return frame_obj, summary_txt

        # 僅事件偵測（無摘要）
        event_msgs = [
            {"role": "system", "content": "你是『嚴格的災害與人員異常偵測器』。只輸出純 JSON 物件。"},
            {"role": "user", "content": f"{event_prompt}\n\n強制規則：只輸出一個 JSON 物件。"}
        ]
        try:
            print(f"--- [Qwen] 呼叫 Ollama 進行事件偵測 (模型: {model_name})... ---")
            event_txt = _ollama_chat(model_name, event_msgs, images_b64=images_b64)
            print(f"--- [Qwen] 事件偵測完成，長度: {len(event_txt)} ---")
        except Exception as e:
            event_txt = ""
            print(f"--- [Qwen] ✗ 事件偵測失敗: {e} ---")
        frame_obj = _safe_parse_json(event_txt) or _extract_first_json(event_txt) or {"events": {"reason": ""}, "persons": []}
        if not isinstance(frame_obj.get("events"), dict):
            frame_obj["events"] = {"reason": ""}
        return frame_obj, ""

    @staticmethod
    def infer_segment_vllm(
        model_name: str,
        video_path: str,
        event_prompt: str,
        summary_prompt: str,
        target_short: int = 720,              # 先保留，雖然 video 直傳時通常不再用到
        frames_per_segment: int = 8,          # 先保留，給 fallback 模式用
        sampling_fps: Optional[float] = None
    ) -> Tuple[Dict, str]:
        """使用 vLLM 分析影片片段。優先走原生 video 輸入；若失敗可 fallback 成 frame 模式。"""
        print(f"--- [infer_segment_vllm] 進入, model={model_name}, video_path={video_path!r}, HAS_VLLM={HAS_VLLM}, _vllm_chat={_vllm_chat is not None} ---", flush=True)
        if not HAS_VLLM or _vllm_chat is None:
            print("--- [infer_segment_vllm] 離開: vLLM 不可用 ---")
            return {"error": "vLLM 不可用（未實作或載入失敗）"}, ""

        if not os.path.exists(video_path):
            return {"error": f"找不到影片檔案: {video_path}"}, ""

        want_summary = bool(summary_prompt.strip())
        qwen3_disable_thinking = ("qwen3" in (model_name or "").lower())

        # 有摘要時
        if want_summary:
            combined_instruction = (
                f"{event_prompt}\n\n"
                f"{summary_prompt}\n\n"
                "請輸出「一個」JSON 物件，且必須包含以下欄位：\n"
                "- events: 事件描述物件（可含 reason 等）\n"
                "- persons: 人員相關陣列\n"
                "- summary: 50–100 字繁體中文畫面摘要（字串）\n"
                "只輸出該 JSON，不要其他文字。"
            )

            combined_msgs = [
                {
                    "role": "system",
                    "content": "你是災害與人員異常偵測器，並產出影片摘要。只輸出一個 JSON 物件，含 events、persons、summary。"
                },
                {
                    "role": "user",
                    "content": combined_instruction
                }
            ]

            raw = ""
            combined_error = None

            try:
                print(f"--- [vLLM] 呼叫 vLLM 進行事件偵測+摘要（影片直傳, 模型: {model_name}）... ---")
                raw = _vllm_chat(
                    model_name=model_name,
                    messages=combined_msgs,
                    video_path=video_path,
                    sampling_fps=sampling_fps,
                    frames_per_segment=frames_per_segment,
                    enable_thinking=False if qwen3_disable_thinking else None,
                )
                print(f"--- [vLLM] 事件+摘要完成，回應長度: {len(raw)} ---")
                print("========== [vLLM RAW 回應] 模型: {} ==========".format(model_name))
                print(raw if raw else "(空)")
                print("========== [vLLM RAW 結束] ==========")

            except Exception as e:
                combined_error = f"vLLM 失敗: {e}"
                print(f"--- [vLLM] ✗ 單一請求失敗: {e} ---")

            combined = _safe_parse_json(raw) or _extract_first_json(raw) or {}
            print(f"--- [vLLM 解析] 解析後 combined 鍵: {list(combined.keys()) if isinstance(combined, dict) else type(combined)}, summary 長度: {len(str(combined.get('summary') or ''))} ---")
            events_in = combined.get("events") if isinstance(combined.get("events"), dict) else {}
            reason = combined.get("event_reason") or combined.get("reason") or (events_in.get("reason", "") if isinstance(events_in, dict) else "")

            if reason and isinstance(events_in, dict) and "reason" not in events_in:
                events_in = {**events_in, "reason": reason}

            frame_obj = {
                "events": events_in or {"reason": ""},
                "persons": combined.get("persons", []),
            }

            if not isinstance(frame_obj.get("events"), dict):
                frame_obj["events"] = {"reason": ""}

            if combined_error:
                frame_obj["error"] = combined_error

            summary_txt = _clean_summary_text(combined.get("summary") or raw or "")
            if not summary_txt and raw:
                summary_txt = _clean_summary_text(raw)

            return frame_obj, summary_txt

        # 僅事件偵測（無摘要）
        event_msgs = [
            {
                "role": "system",
                "content": "你是『嚴格的災害與人員異常偵測器』。只輸出純 JSON 物件。"
            },
            {
                "role": "user",
                "content": f"{event_prompt}\n\n強制規則：只輸出一個 JSON 物件。"
            }
        ]

        try:
            print(f"--- [vLLM] 呼叫 vLLM 進行事件偵測（影片直傳, 模型: {model_name}）... ---")
            event_txt = _vllm_chat(
                model_name=model_name,
                messages=event_msgs,
                video_path=video_path,
                sampling_fps=sampling_fps,
                frames_per_segment=frames_per_segment,
                enable_thinking=False if qwen3_disable_thinking else None,
            )
            print(f"--- [vLLM] 事件偵測完成，長度: {len(event_txt)} ---")
            print("========== [vLLM RAW 回應-僅事件] 模型: {} ==========".format(model_name))
            print(event_txt if event_txt else "(空)")
            print("========== [vLLM RAW 結束] ==========")

        except Exception as e:
            event_txt = ""
            print(f"--- [vLLM] ✗ 事件偵測失敗: {e} ---")

        frame_obj = _safe_parse_json(event_txt) or _extract_first_json(event_txt) or {
            "events": {"reason": ""},
            "persons": []
        }
        print(f"--- [vLLM 解析] 僅事件模式 解析後 frame_obj 鍵: {list(frame_obj.keys()) if isinstance(frame_obj, dict) else type(frame_obj)} ---")

        if not isinstance(frame_obj.get("events"), dict):
            frame_obj["events"] = {"reason": ""}

        return frame_obj, ""

    @staticmethod
    def infer_segment_vllm_batch(
        model_name: str,
        path_strs: List[str],
        event_prompt: str,
        summary_prompt: str,
        target_short: int = 432,
        frames_per_segment: int = 5,
        sampling_fps: Optional[float] = None
    ) -> List[Tuple[Dict, str]]:
        """
        使用 vLLM 批次分析多個影片片段。
        優勢：利用 Continuous Batching 與 Prefix Caching 提升 3-5 倍吞吐量。
        """
        print(f"--- [vLLM Batch] 開始處理 {len(path_strs)} 個片段, 模型={model_name} ---", flush=True)
        qwen3_disable_thinking = ("qwen3" in (model_name or "").lower())
        
        if not HAS_VLLM or _vllm_chat is None:
            print("--- [vLLM Batch] 錯誤: vLLM 環境未就緒 ---")
            return [({"error": "vLLM 不可用"}, "")] * len(path_strs)

        # 預設回傳容器
        all_parsed: List[Tuple[Dict, str]] = [({"error": "未處理"}, "")] * len(path_strs)
        valid_indices: List[int] = []
        batch_requests = []

        # 1. 準備批次資料 (Data Preparation)
        # 由於 Prompt 都一樣，vLLM 會自動進行 Prefix Caching 優化
        instruction = (
            f"{event_prompt}\n\n"
            f"{summary_prompt}\n\n"
            "請輸出「一個」JSON 物件，且必須包含以下欄位：\n"
            "- events: 事件描述物件（可含 reason 等）\n"
            "- persons: 人員相關陣列\n"
            "- summary: 50–100 字繁體中文畫面摘要（字串）\n"
            "只輸出該 JSON，不要其他文字。"
        )

        combined_msgs = [
            {
                "role": "system",
                "content": "你是災害與人員異常偵測器，並產出影片摘要。只輸出一個 JSON 物件，含 events、persons、summary。"
            },
            {
                "role": "user",
                "content": instruction
            }
        ]

        for i, path in enumerate(path_strs):
            if not os.path.exists(path):
                print(f"--- [vLLM Batch] 找不到檔案: {path} ---")
                all_parsed[i] = ({"error": f"找不到影片檔案: {path}"}, "")
                continue
            
            # 封裝成 vLLM 批次請求格式
            batch_requests.append({
                "model_name": model_name,
                "messages": combined_msgs,
                "video_path": path,
                "sampling_fps": sampling_fps,
                "frames_per_segment": frames_per_segment,
                "enable_thinking": False if qwen3_disable_thinking else True,
            })
            valid_indices.append(i)

        if not batch_requests:
            return all_parsed

        # 2. 執行 vLLM 批次推論 (Inference)
        # 這裡呼叫支援 List 輸入的 _vllm_chat 封裝，啟動 Continuous Batching
        raw_outputs = []
        try:
            _mw = max(1, min(int(getattr(config, "VLLM_BATCH_MAX_WORKERS", 16)), 32, len(batch_requests)))
            print(
                f"--- [vLLM Batch] 正在呼叫 vLLM 引擎進行 {len(batch_requests)} 段並行分析 (HTTP threads={_mw}) ---",
                flush=True,
            )
            raw_outputs = _vllm_chat_batch(
                model_name=model_name,
                batch_requests=batch_requests,
                max_workers=_mw,
                enable_thinking=False if qwen3_disable_thinking else None,
            )
            print(f"--- [vLLM Batch] 推論完成，取得 {len(raw_outputs)} 筆回應 ---")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"--- [vLLM Batch] ✗ 批次請求崩潰: {e} | active_threads={threading.active_count()} ---", flush=True)
            for idx in valid_indices:
                all_parsed[idx] = ({"error": f"vLLM Batch 失敗: {e}"}, "")
            return all_parsed

        # 3. 解析與結果歸位 (Parsing & Post-processing)
        for j, raw in enumerate(raw_outputs):
            orig_idx = valid_indices[j]
            try:
                # 沿用您原本精密的解析與清理邏輯
                combined = _safe_parse_json(raw) or _extract_first_json(raw) or {}
                
                events_in = combined.get("events") if isinstance(combined.get("events"), dict) else {}
                reason = (combined.get("event_reason") or 
                        combined.get("reason") or 
                        (events_in.get("reason", "") if isinstance(events_in, dict) else ""))

                if reason and isinstance(events_in, dict) and "reason" not in events_in:
                    events_in = {**events_in, "reason": reason}

                frame_obj = {
                    "events": events_in or {"reason": ""},
                    "persons": combined.get("persons", []),
                }

                if not isinstance(frame_obj.get("events"), dict):
                    frame_obj["events"] = {"reason": ""}

                summary_txt = _clean_summary_text(combined.get("summary") or raw or "")
                if not summary_txt and raw:
                    summary_txt = _clean_summary_text(raw)

                all_parsed[orig_idx] = (frame_obj, summary_txt)

            except Exception as parse_err:
                print(f"--- [vLLM Batch] 解析失敗 (Index {orig_idx}): {parse_err} ---")
                all_parsed[orig_idx] = ({"error": f"解析失敗: {parse_err}", "raw": raw[:100]}, "")

        return all_parsed

    @staticmethod
    def infer_segment_qwen_hf(
        model_name: str,
        video_path: str,
        event_prompt: str,
        summary_prompt: str,
        target_short: int = 432,
        frames_per_segment: int = 5,
        sampling_fps: Optional[float] = None
    ) -> Tuple[Dict, str]:
        """使用本機 Hugging Face Qwen2.5-VL 分析影片片段（僅有摘要，與 infer_segment_qwen 同格式）。"""
        if not HAS_QWEN_HF or qwen_hf_utils is None:
            return {"error": "未安裝 transformers 或 qwen_hf_utils 不可用"}, ""
        return qwen_hf_utils.infer_one(
            model_name,
            video_path,
            event_prompt,
            summary_prompt,
            target_short=target_short,
            frames_per_segment=frames_per_segment,
            sampling_fps=sampling_fps,
            sample_frames_fn=_sample_frames_evenly_to_pil,
            resize_fn=_resize_short_side,
        )

    @staticmethod
    def infer_segment_qwen_hf_batch(
        model_name: str,
        video_paths: List[str],
        event_prompt: str,
        summary_prompt: str,
        target_short: int = 432,
        frames_per_segment: int = 5,
        sampling_fps: Optional[float] = None,
        max_inference_batch_size: int = 4,
    ) -> List[Tuple[Dict, str]]:
        """Batch：多段影片一次送入（模型只載入一次），回傳 list of (frame_obj, summary_txt)。max_inference_batch_size 為 None 時一次處理全部。"""
        if not HAS_QWEN_HF or qwen_hf_utils is None:
            return [({"error": "未安裝 transformers 或 qwen_hf_utils 不可用"}, "")] * len(video_paths)
        return qwen_hf_utils.infer_batch(
            model_name,
            video_paths,
            event_prompt,
            summary_prompt,
            target_short=target_short,
            frames_per_segment=frames_per_segment,
            sampling_fps=sampling_fps,
            sample_frames_fn=_sample_frames_evenly_to_pil,
            resize_fn=_resize_short_side,
            max_inference_batch_size=max_inference_batch_size or len(video_paths),
        )

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
            # 釋放 GPU 上的批次張量與 ReID 輸出，僅保留回傳的 list
            del batch_tensor, features, embeddings, processed_tensors
            _release_gpu_memory()
            return result

        except Exception as e:
            print(f"--- [AnalysisService] Batch ReID failed: {e} ---")
            import traceback
            traceback.print_exc()
            return [None] * len(crop_images)

    @staticmethod
    def analyze_segment(req) -> Dict:
        """
        分析單一片段的完整邏輯：VLM（事件偵測+摘要）+ YOLO（物件偵測+裁剪+ReID）
        流程：依 model_type 選擇 VLM → 強制執行 YOLO → 合併結果回傳
        """
        p = req.segment_path  # 影片片段檔案路徑（如 .../video_stem/segment_000.mp4）
        tr = f"{_fmt_hms(req.start_time)} - {_fmt_hms(req.end_time)}"  # 時間區段字串（HH:MM:SS 格式）
        print(f"--- [AnalysisService] 開始分析片段: {p} ({tr}) ---")
        print(f"--- [AnalysisService] Model: {req.model_type}, Qwen Model: {getattr(req, 'qwen_model', 'N/A')} ---")
        print(f"--- [AnalysisService] Event Prompt Len: {len(req.event_detection_prompt)}, Summary Prompt Len: {len(req.summary_prompt)} ---")
        t1 = time.time()  # 記錄開始時間，用於計算處理耗時

        # 初始化回傳結構，後續依分析結果填充
        result = {
            "segment": Path(p).name,           # 片段檔名（如 segment_000.mp4）
            "time_range": tr,                  # 時間區段字串
            "duration_sec": round(req.end_time - req.start_time, 2),
            "success": False,                  # 是否成功，依 VLM 或 YOLO 結果設定
            "time_sec": 0.0,                   # 處理耗時（秒，相容 DB）
            "vlm_time": 0.0,                   # 呼叫 Ollama 推論的純耗時（秒）
            "yolo_reid_time": 0.0,             # YOLO+ReID 總耗時（秒）
            "total_api_time": 0.0,             # 從進入 analyze_segment 到結束的總耗時（秒）
            "parsed": {},                      # VLM 解析結果（frame_analysis, summary_independent）
            "raw_detection": {},               # 原始偵測結果（含 yolo 等）
            "error": None                     # 錯誤訊息，若有
        }

        try:
            # ========== 階段 1：執行 VLM 分析（依 model_type 選擇模型） ==========
            t_vlm_start = time.time()
            if req.model_type in ("qwen", "gemini"):
                if req.model_type == "qwen":
                    # 使用 Qwen-VL 透過 Ollama 進行事件偵測與摘要
                    frame_obj, summary_txt = AnalysisService.infer_segment_qwen(
                        req.qwen_model, p, req.event_detection_prompt, req.summary_prompt,
                        target_short=req.target_short, frames_per_segment=req.frames_per_segment,
                        sampling_fps=req.sampling_fps
                    )
                else:
                    # 使用 Gemini 模型進行分析
                    g_model = req.qwen_model if req.qwen_model.startswith("gemini") else "gemini-2.5-flash"
                    frame_obj, summary_txt = AnalysisService.infer_segment_gemini(
                        g_model, p, req.event_detection_prompt, req.summary_prompt,
                        req.target_short, req.frames_per_segment, sampling_fps=req.sampling_fps
                    )

                frame_norm = AnalysisService._normalize_vlm_output(frame_obj)  # 將 VLM 輸出標準化為固定格式

                result["success"] = "error" not in frame_obj  # 若 frame_obj 無 error key 視為成功
                result["error"] = frame_obj.get("error")       # 若有錯誤則記錄
                result["parsed"] = {
                    "frame_analysis": frame_norm,   # 標準化後的事件分析結果
                    "summary_independent": summary_txt  # VLM 產生的文字摘要
                }

            elif req.model_type == "vllm_qwen":
                # 使用 vLLM（Qwen-VL 等）進行事件偵測與摘要
                print(f"--- [AnalysisService] 即將呼叫 infer_segment_vllm (segment_path={p}) ---", flush=True)
                try:
                    frame_obj, summary_txt = AnalysisService.infer_segment_vllm(
                        req.qwen_model, p, req.event_detection_prompt, req.summary_prompt,
                        target_short=req.target_short, frames_per_segment=req.frames_per_segment,
                        sampling_fps=req.sampling_fps
                    )
                    print(f"--- [AnalysisService] infer_segment_vllm 返回, has_error={('error' in frame_obj)}, summary_len={len(summary_txt or '')} ---", flush=True)
                except Exception as vllm_ex:
                    print(f"--- [AnalysisService] infer_segment_vllm 拋出異常: {vllm_ex} ---", flush=True)
                    frame_obj, summary_txt = {"error": str(vllm_ex)}, ""
                frame_norm = AnalysisService._normalize_vlm_output(frame_obj)
                result["success"] = "error" not in frame_obj
                result["error"] = frame_obj.get("error")
                result["parsed"] = {
                    "frame_analysis": frame_norm,
                    "summary_independent": summary_txt
                }

            elif req.model_type == "moondream":
                try:
                    from src.utils.moondream_utils import infer_segment_moondream
                    moondream_version = getattr(req, "qwen_model", "moondream3-preview") or "moondream3-preview"
                    if moondream_version not in ("moondream-2b-2025-04-14", "moondream3-preview"):
                        moondream_version = "moondream3-preview"  # 只支援這兩種版本
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    frame_obj, summary_txt = infer_segment_moondream(
                        model_version=moondream_version,
                        segment_path=p,
                        event_detection_prompt=req.event_detection_prompt or "",
                        summary_prompt=req.summary_prompt or "",
                        frames_per_segment=getattr(req, "frames_per_segment", 8),
                        sampling_fps=getattr(req, "sampling_fps", None),
                        device=device,
                    )
                    frame_norm = AnalysisService._normalize_vlm_output(frame_obj)
                    result["success"] = "error" not in frame_obj
                    result["error"] = frame_obj.get("error")
                    result["parsed"] = {
                        "frame_analysis": frame_norm,
                        "summary_independent": summary_txt or ""
                    }
                except Exception as moondream_ex:
                    print(f"--- [Moondream] ✗ 異常（含 import/載入/推論）: {moondream_ex} ---")
                    result["success"] = False
                    result["error"] = str(moondream_ex)
                    result["parsed"] = {
                        "frame_analysis": AnalysisService._normalize_vlm_output({}),
                        "summary_independent": ""
                    }

            result["vlm_time"] = round(time.time() - t_vlm_start, 2)

            # ========== 階段 2：強制執行 YOLO-World + FastReID ==========
            # 不論 VLM 類型為何（qwen/gemini/moondream），皆執行 YOLO，以取得物件偵測、裁剪、ReID 特徵（供以圖搜圖、物件追蹤等）

            # 從 req 讀取 YOLO 參數，若無該屬性或值為空則用預設值
            yolo_labels = req.yolo_labels if hasattr(req, 'yolo_labels') and req.yolo_labels else "person,car"
            yolo_every_sec = req.yolo_every_sec if hasattr(req, 'yolo_every_sec') else 2.0
            yolo_score_thr = req.yolo_score_thr if hasattr(req, 'yolo_score_thr') else 0.25

            print(f"--- [AnalysisService] 執行 YOLO+ReID 正常流程 (Labels: {yolo_labels}) ---")
            t_yolo_start = time.time()
            try:
                # 呼叫 YOLO 物件偵測：讀取影片片段、依 every_sec 取樣幀、推論、裁剪、存檔、產生 ReID 特徵
                yolo_res = AnalysisService.infer_segment_yolo(
                    p,  # 片段路徑（如 segment_000.mp4）
                    labels=yolo_labels,      # 要偵測的類別，逗號分隔字串
                    every_sec=yolo_every_sec,  # 每幾秒取一幀（2.0 表示每 2 秒一幀）
                    score_thr=yolo_score_thr   # 信心門檻，低於此值的偵測會被過濾
                )
                # 必須包在 "yolo" key 下：_save_results_to_postgres 會依 result["raw_detection"]["yolo"] 讀取 crop_paths 並寫入 object_crops 表
                result["raw_detection"] = {"yolo": yolo_res}
            except Exception as yolo_err:
                error_msg = str(yolo_err)
                print(f"--- [AnalysisService] YOLO 處理失敗: {error_msg} ---")
                result["raw_detection"] = {"yolo": None}  # 失敗時設為 None，避免 DB 邏輯報錯
                prev = (result.get("error") or "").strip()  # 取得之前 VLM 的錯誤（若有）
                result["error"] = f"{prev}; YOLO: {error_msg}".strip().lstrip(";").strip() if prev else error_msg
            finally:
                result["yolo_reid_time"] = round(time.time() - t_yolo_start, 2)

            # 若 model_type 為 "yolo"（純物件偵測模式，無 VLM），則 YOLO 完成即視為整個片段分析成功
            if req.model_type == "yolo":
                result["success"] = True

        except Exception as ex:
            # 捕捉階段 1 或階段 2 中未處理的異常（如 VLM 拋錯、其他邏輯錯誤）
            print(f"--- [AnalysisService] ✗ 分析發生異常: {ex} ---")
            result["error"] = str(ex)

        result["total_api_time"] = round(time.time() - t1, 2)  # 從進入到結束的總耗時（秒）
        result["time_sec"] = result["total_api_time"]  # 相容 DB
        _release_gpu_memory()  # 釋放本片段推理產生的 GPU 暫存，模型保留
        return result  # 回傳完整分析結果（含 parsed、raw_detection、success、error 等）

    @staticmethod
    def _analyze_segment_yolo_only(req, frame_obj: Dict, summary_txt: str) -> Dict:
        """僅執行 YOLO+ReID，VLM 結果由外部提供（供 qwen_hf batch 流程使用）。"""
        p = req.segment_path
        tr = f"{_fmt_hms(req.start_time)} - {_fmt_hms(req.end_time)}"
        t1 = time.time()
        result = {
            "segment": Path(p).name,
            "time_range": tr,
            "duration_sec": round(req.end_time - req.start_time, 2),
            "success": "error" not in frame_obj,
            "time_sec": 0.0,
            "vlm_time": 0.0,
            "yolo_reid_time": 0.0,
            "total_api_time": 0.0,
            "parsed": {
                "frame_analysis": AnalysisService._normalize_vlm_output(frame_obj),
                "summary_independent": summary_txt or "",
            },
            "raw_detection": {},
            "error": frame_obj.get("error"),
        }
        yolo_labels = getattr(req, "yolo_labels", None) or "person,car"
        yolo_every_sec = getattr(req, "yolo_every_sec", 2.0)
        yolo_score_thr = getattr(req, "yolo_score_thr", 0.25)
        t_yolo_start = time.time()
        try:
            yolo_res = AnalysisService.infer_segment_yolo(p, labels=yolo_labels, every_sec=yolo_every_sec, score_thr=yolo_score_thr)
            result["raw_detection"] = {"yolo": yolo_res}
        except Exception as yolo_err:
            result["raw_detection"] = {"yolo": None}
            prev = (result.get("error") or "").strip()
            result["error"] = f"{prev}; YOLO: {yolo_err}".strip().lstrip(";").strip() if prev else str(yolo_err)
        result["yolo_reid_time"] = round(time.time() - t_yolo_start, 2)
        result["total_api_time"] = round(time.time() - t1, 2)
        result["time_sec"] = result["total_api_time"]
        _release_gpu_memory()
        return result

    # 多段影片「一次」送入 YOLO（跨段湊成幀 batch）→ 再用 YOLO 偵測結果做 crop → 所有 crops 一次送入 ReID 做 embedding；回傳與 seg_paths 同序的結果列表。
    @staticmethod
    def infer_segment_yolo_batch(
        seg_paths: List[str],
        labels: str = "person,car",
        every_sec: float = 2.0,
        score_thr: float = 0.25,
        yolo_batch_size: int = 20,
    ) -> List[Dict]:
        # print(
        #     "[API][infer_segment_yolo_batch][INPUT] "
        #     + json.dumps(
        #         {
        #             "seg_paths": seg_paths or [],
        #             "labels": labels,
        #             "every_sec": every_sec,
        #             "score_thr": score_thr,
        #             "yolo_batch_size": yolo_batch_size,
        #         },
        #         ensure_ascii=False,
        #         default=str,
        #     ),
        #     flush=True,
        # )

        # 同一時間只允許一個執行緒做推論，避免多執行緒同時用 GPU 造成設備不一致或 CUDA 錯誤。
        with gpu_inference_lock_traced():
            return AnalysisService._infer_segment_yolo_batch_impl(
                seg_paths, labels, every_sec, score_thr, yolo_batch_size)

    # 物件偵測 (YOLO) 與 行人重識別 (ReID) 的核心實作。它的精髓在於「打平 (Flattening)」了多段影片的影格，將其整合成一個巨大的批次進行運算，以極大化 GPU 的吞吐量。
    @staticmethod
    def _infer_segment_yolo_batch_impl(
        seg_paths: List[str],
        labels: str,
        every_sec: float,
        score_thr: float,
        batch_size: int = 20,
    ) -> List[Dict]:

        if not seg_paths:
            return []

        # 1. 獲取模型與基礎參數
        model = get_yolo_model()
        reid_model, reid_device = get_reid_model()
        labels_list = [l.strip() for l in (labels or "person,car").split(",") if l.strip()]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. 設備初始化與類別設定 (僅處理設備邏輯，不在此處 print 尚未宣告的變數)
        try:
            model.to(device)
            model.set_classes(labels_list)
            model.to(device)
        except RuntimeError as e:
            if "same device" in str(e):
                print(f"--- [YOLO Batch] 偵測到張量設備不一致，嘗試 CPU 修復再回傳 {device} ---", flush=True)
                try:
                    model.cpu()
                    model.set_classes(labels_list)
                    model.to(device)
                    print("--- [YOLO Batch] 設備同步修復完成 ---", flush=True)
                except Exception as e2:
                    return [{"error": f"Device Fix Failed: {str(e2)}", "detections": [], "crop_paths": [], "object_count": {}, "total_frames_processed": 0}] * len(seg_paths)
            else:
                return [{"error": f"RuntimeError: {str(e)}", "detections": [], "crop_paths": [], "object_count": {}, "total_frames_processed": 0}] * len(seg_paths)
        except Exception as e:
            return [{"error": f"Init Error: {str(e)}", "detections": [], "crop_paths": [], "object_count": {}, "total_frames_processed": 0}] * len(seg_paths)

        # 3. 收集所有影格 (Flat Frames)
        flat_frames: List[tuple] = []
        frames_per_seg: List[int] = [0] * len(seg_paths)

        for seg_idx, p in enumerate(seg_paths):
            seg_dir = Path(p).parent
            crops_dir = seg_dir / "yolo_output" / "object_crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            interval = max(1, int(round(fps * every_sec)))
            frame_indices = list(range(0, total_frames, interval))

            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if ok:
                    # (段索引, 影格索引, 影像陣列, FPS, 儲存目錄)
                    flat_frames.append((seg_idx, fi, frame.copy(), fps, crops_dir))

            frames_per_seg[seg_idx] = len(frame_indices)
            cap.release()

        try:
            from src.api.video_analysis import _print_ram_diagnosis

            _print_ram_diagnosis(
                f"[AFTER_CV2_EXTRACT] flat_frames={len(flat_frames)} segs={len(seg_paths)}"
            )
        except Exception:
            pass

        # 4. 執行 YOLO Batch 推論
        flat_for_reid: List[tuple] = []
        seg_detections: List[List[Dict]] = [[] for _ in range(len(seg_paths))]

        for start in range(0, len(flat_frames), batch_size):
            chunk = flat_frames[start : start + batch_size]
            batch_frames = [x[2] for x in chunk]

            batch_predict_res: List[Any] = []
            try:
                batch_predict_res = list(model.predict(batch_frames, conf=score_thr, verbose=False))
            except Exception as e:
                print(
                    f"--- [YOLO Batch] 批次 Predict 失敗，改單幀重試: {type(e).__name__}: {e} ---",
                    flush=True,
                )
                for j, single_frame in enumerate(batch_frames):
                    try:
                        one = model.predict([single_frame], conf=score_thr, verbose=False)
                        if one is not None and len(one) > 0:
                            batch_predict_res.append(one[0])
                        else:
                            batch_predict_res.append(None)
                    except Exception as e2:
                        fi_hint = chunk[j][1] if j < len(chunk) else None
                        print(
                            f"--- [YOLO Batch] 單幀略過 start={start} j={j} fi={fi_hint}: "
                            f"{type(e2).__name__}: {e2} ---",
                            flush=True,
                        )
                        batch_predict_res.append(None)

            while len(batch_predict_res) < len(chunk):
                batch_predict_res.append(None)

            for idx, res in enumerate(batch_predict_res):
                if idx >= len(chunk):
                    break
                if res is None:
                    continue
                try:
                    _ = res.boxes
                except Exception as e_parse:
                    print(f"--- [YOLO Batch] 結果物件無效 idx={idx}: {type(e_parse).__name__}: {e_parse} ---", flush=True)
                    continue

                seg_idx, fi, frame, fps, crops_dir = chunk[idx]
                timestamp = round(fi / fps, 2)
                temp_detections = []
                crops_imgs = []

                for b_idx, box in enumerate(res.boxes):
                    cls_id = int(box.cls)
                    label = labels_list[cls_id]
                    conf = round(float(box.conf), 3)
                    xyxy = [int(round(float(c))) for c in box.xyxy[0].tolist()]

                    # 裁切物件影像
                    crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    if crop.size > 0:
                        crop_filename = f"crop_{fi}_{b_idx}_{label}.jpg"
                        crop_path = crops_dir / crop_filename
                        cv2.imwrite(str(crop_path), crop)
                        
                        rel_crop_path = AnalysisService._safe_crop_path_for_storage(crop_path, crops_dir.parent.parent)
                        
                        det_info = {
                            "timestamp": timestamp, "label": label, "score": conf,
                            "box": xyxy, "frame": fi, "path": rel_crop_path,
                        }
                        temp_detections.append(det_info)
                        crops_imgs.append(crop)

                # 將結果歸類到對應的段落
                for d in temp_detections:
                    seg_detections[seg_idx].append(d)

                # 準備傳遞給 ReID 的資料包 (段索引, 偵測字典, 像素陣列)
                for d, img in zip(temp_detections, crops_imgs):
                    flat_for_reid.append((seg_idx, d, img))

            del batch_frames
            del chunk
            del batch_predict_res
            _release_gpu_memory()

        # 5. DEBUG 輸出：列印完整的 flat_for_reid 物件內容
        # 此處已經確保 flat_for_reid 變數存在且已填充資料
        # print(
        #     "[API][_infer_segment_yolo_batch_impl][DEBUG][flat_for_reid_content] "
        #     + str(flat_for_reid),
        #     flush=True,
        # )

        # 6. 執行 ReID Embedding Batch
        if flat_for_reid and reid_model:
            all_crops = [x[2] for x in flat_for_reid]
            # 一次將所有裁切圖送入 ReID 提取特徵向量
            embeddings = AnalysisService.generate_reid_embeddings_batch(all_crops, reid_model, reid_device)
            
            for i, emb in enumerate(embeddings):
                if i < len(flat_for_reid) and emb is not None:
                    # 將 Embedding 回填至對應的偵測字典中 (透過引用修改)
                    flat_for_reid[i][1]["reid_embedding"] = emb
            del all_crops
            del embeddings

        _release_gpu_memory()

        # 7. 組合最終回傳格式
        final_output = []
        for seg_idx in range(len(seg_paths)):
            dets = seg_detections[seg_idx]
            
            # 計算該段物件分布
            obj_summary = {}
            for d in dets:
                lbl = d["label"]
                obj_summary[lbl] = obj_summary.get(lbl, 0) + 1
            
            final_output.append({
                "total_detections": len(dets),
                "detections": dets,
                "crop_paths": dets, # 與舊版欄位兼容
                "object_count": obj_summary,
                "total_frames_processed": frames_per_seg[seg_idx],
            })

        print(
            f"--- [YOLO Batch] 完成 {len(seg_paths)} 段分析，處理 {len(flat_frames)} 幀，產生 {len(flat_for_reid)} 個 ReID 特徵 ---",
            flush=True,
        )
        del flat_frames
        del flat_for_reid
        del seg_detections
        del frames_per_seg
        gc.collect()
        try:
            from src.api.video_analysis import _print_ram_diagnosis

            _print_ram_diagnosis("[AFTER_GC_CLEAN] YOLO batch 結束，已 del 大 list 並 gc.collect()")
        except Exception:
            pass
        return final_output

    @staticmethod
    def _merge_vlm_and_yolo_result(req, frame_obj: Dict, summary_txt: str, yolo_res: Dict, batch_elapsed_sec: Optional[float] = None) -> Dict:
        """合併已算好的 VLM 結果與 YOLO 結果為單一 result 字典（不再呼叫 YOLO）。
        batch_elapsed_sec: 若為 batch 路徑呼叫，可傳入該 batch 的總耗時（秒），會寫入 time_sec / total_api_time。"""
        p = req.segment_path
        tr = f"{_fmt_hms(req.start_time)} - {_fmt_hms(req.end_time)}"
        vlm_ok = "error" not in frame_obj
        yolo_ok = yolo_res and isinstance(yolo_res, dict) and "error" not in yolo_res
        has_yolo_data = yolo_ok and (yolo_res.get("detections") or yolo_res.get("total_detections", 0) > 0)
        # 只要 VLM 成功，或 YOLO 有結果且無錯誤，就視為可寫入 DB
        success = vlm_ok or has_yolo_data
        elapsed = batch_elapsed_sec if batch_elapsed_sec is not None else 0.0
        result = {
            "segment": Path(p).name,
            "time_range": tr,
            "duration_sec": round(req.end_time - req.start_time, 2),
            "success": success,
            "time_sec": round(elapsed, 2),
            "vlm_time": 0.0,
            "yolo_reid_time": 0.0,
            "total_api_time": round(elapsed, 2),
            "parsed": {
                "frame_analysis": AnalysisService._normalize_vlm_output(frame_obj),
                "summary_independent": summary_txt or "",
            },
            "raw_detection": {"yolo": yolo_res},
            "error": frame_obj.get("error") or (yolo_res.get("error") if isinstance(yolo_res, dict) else None),
        }
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
        worker_count: int = 4,
        qwen_inference_batch_size: Optional[int] = None,
        yolo_batch_size: Optional[int] = None,
    ) -> List[Dict]:
        
        results = []
        QWEN_HF_BATCH_SIZE = 10
        # VLM 單次 forward 的段數上限（等同 n_batch），可由環境變數 QWEN_HF_MAX_INFERENCE_BATCH_SIZE 覆寫
        try:
            QWEN_HF_MAX_INFERENCE_BATCH_SIZE = int(os.environ.get("QWEN_HF_MAX_INFERENCE_BATCH_SIZE", "4"))
        except (TypeError, ValueError):
            QWEN_HF_MAX_INFERENCE_BATCH_SIZE = 4

        class Req:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)

        # 單執行緒備案：定義如何處理單一影片片段。它計算該片段在原始影片中的開始/結束時間，並呼叫 analyze_segment。
        def process_one(seg_path, idx):
            from pathlib import Path
            print(f"--- [Pipeline] 開始處理片段 {idx + 1}/{len(seg_files)}: {Path(seg_path).name} ---")
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
            res = AnalysisService.analyze_segment(req)
            print(f"--- [Pipeline] 片段 {idx + 1}/{len(seg_files)} 完成 (success={res.get('success')}) ---")
            return res

        # 在 run_full_pipeline 內部新增 vllm 判斷
        if model_type == "vllm_qwen":
            # vLLM 支援更高的並行，batch_size 可以設大 (例如 64)
            # 為了讓每個 worker 的 GPU RAM 使用量在請求結束後更接近「未執行前」的基線，
            # 這裡預設不要用太大的 batch，避免觸發一次性較大的 CUDA/allocator 保留記憶體。
            vlm_batch_size = qwen_inference_batch_size or 10
            yolo_frame_batch = yolo_batch_size or 10

            print(
                f"--- [Pipeline vllm_qwen] 共 {len(seg_files)} 段 | VLM batch={vlm_batch_size} YOLO 幀batch={yolo_frame_batch} "
                f"| active_threads={threading.active_count()}（單請求內 YOLO 與 VLM 各一條 worker 並行，跨請求會爭奪 GPU_LOCK）---",
                flush=True,
            )

            for batch_start in range(0, len(seg_files), vlm_batch_size):
                batch_paths = seg_files[batch_start : batch_start + vlm_batch_size]
                path_strs = [str(p) for p in batch_paths]

                # 定義 VLM 批次任務
                def run_vlm_vllm_batch():
                    return AnalysisService.infer_segment_vllm_batch(
                        qwen_model,
                        path_strs,
                        event_detection_prompt,
                        summary_prompt,
                        frames_per_segment=frames_per_segment,
                        sampling_fps=sampling_fps,
                    )

                # 定義 YOLO 批次任務
                def run_yolo_batch():
                    return AnalysisService.infer_segment_yolo_batch(
                        path_strs,
                        labels=yolo_labels,
                        every_sec=yolo_every_sec,
                        yolo_batch_size=yolo_frame_batch,
                    )

                # 兩者並行，最大化 GPU 吞吐量
                # [修正] 在 ThreadPoolExecutor 啟動前記錄時間
                t_batch_start = time.time() 
                with ThreadPoolExecutor(max_workers=2) as ex:
                    future_vlm = ex.submit(run_vlm_vllm_batch)
                    future_yolo = ex.submit(run_yolo_batch)
                    vlm_list = future_vlm.result()
                    yolo_list = future_yolo.result()
                # [修正] 計算該批次總耗時
                batch_elapsed_sec = round(time.time() - t_batch_start, 2) 

                # 後續合併結果時，batch_elapsed_sec 就不會噴 NameError 了
                
                # 合併 VLM 與 YOLO 的結果，並生成最終的結果列表。
                for i, seg_path in enumerate(batch_paths):
                    idx = batch_start + i
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

                    # 從各自的批次結果清單中，根據索引取出對應這段影片的 AI 分析結果。
                    frame_obj, summary_txt = vlm_list[i] if i < len(vlm_list) else ({"error": "batch 長度不符"}, "")
                    yolo_res = yolo_list[i] if i < len(yolo_list) else {}
                    results.append(AnalysisService._merge_vlm_and_yolo_result(req, frame_obj, summary_txt, yolo_res, batch_elapsed_sec=batch_elapsed_sec))

            # 最後將所有結果按片段順序排序，確保結果的順序與原始影片片段的順序一致。
            results.sort(key=lambda x: x["segment"])
            return results

       
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
            if k == "reason":
                continue
            # 保留既有白名單鍵，也保留新加入的事件鍵（如 violence、dangerous_items）
            # 避免前端統計異常時被「正規化」掉。
            norm_events[k] = bool(v)
        
        norm_events["reason"] = str(ev.get("reason", ""))
        return {"events": norm_events, "persons": frame_obj.get("persons", [])}

    @staticmethod
    def infer_segment_yolo(seg_path: str, labels: str, every_sec: float, score_thr: float, batch_size: int = 16) -> Dict:
        """
        使用 YOLO-World 進行物件偵測並執行 FastReID 提取特徵，同時儲存物件切片
        流程：讀取影片片段 → 依 every_sec 取樣幀 → YOLO 批次推論 → 裁剪 bbox → 存檔 → ReID 特徵
        YOLO/ReID 非 thread-safe，必須在 _gpu_inference_lock 內執行，避免多 thread 並行導致設備不一致與 CUDA 錯誤。
        """
        with gpu_inference_lock_traced():
            return AnalysisService._infer_segment_yolo_impl(seg_path, labels, every_sec, score_thr, batch_size)

    @staticmethod
    def _infer_segment_yolo_impl(seg_path: str, labels: str, every_sec: float, score_thr: float, batch_size: int = 16) -> Dict:
        """infer_segment_yolo 的實際實作（在鎖內呼叫）"""

        # ========== 階段 1：獲取 YOLO 模型 ==========
        model = get_yolo_model()  # 從 model_loader 取得全域單例 YOLO-World 模型
        if not model:
            from src.core.model_loader import get_yolo_model as retry_get_yolo
            print("--- [AnalysisService] YOLO model not found, retrying... ---")
            model = retry_get_yolo()  # 再嘗試一次載入（可能前次載入未完成）
            if not model:
                return {"error": "YOLO model failed to load", "detections": []}  # 失敗則回傳錯誤結構

        # 取得 ReID 模型與設備（用於後續物件 re-identification 特徵提取）
        reid_model, reid_device = get_reid_model()

        # ========== 階段 2：準備輸出目錄 ==========
        seg_dir = Path(seg_path).parent  # 片段檔案的父目錄（如 .../video_stem/）
        crops_dir = seg_dir / "yolo_output" / "object_crops"  # 物件裁剪圖存放路徑
        crops_dir.mkdir(parents=True, exist_ok=True)  # 遞迴建立目錄，若已存在則不覆蓋

        # 解析 labels 字串（逗號分隔）為列表，若為空則預設 person, car
        labels_list = [l.strip() for l in labels.split(",") if l.strip()] or ["person", "car"]

        # ========== 階段 3：設定 YOLO 偵測類別與設備 ==========
        device = "cuda" if torch.cuda.is_available() else "cpu"  # 依環境決定推理設備
        try:
            model.to(device)  # 確保模型在目標設備上（避免 CPU/GPU 張量混用）
            model.set_classes(labels_list)  # YOLO-World 支援開放詞彙，動態設定要偵測的類別
            model.to(device)  # 設定類別後再次同步設備（部分內部 buffer 可能被重建）
            print(f"--- [YOLO] 成功在 {device} 設定類別: {labels_list} ---")

        except AttributeError as e:
            error_msg = str(e)
            if "'NoneType' object has no attribute 'names'" in error_msg:
                print(f"--- [YOLO Error] 偵測到殭屍模型，強制重載... ---")
                import src.core.model_loader
                src.core.model_loader._yolo_world_model = None  # 清空全域變數，強制重新載入
                model = src.core.model_loader.get_yolo_model()
                if model:
                    model.to(device)
                    model.set_classes(labels_list)
                    model.to(device)
                else:
                    return {"error": f"YOLO AttributeError: {error_msg}", "detections": []}
            else:
                return {"error": f"YOLO AttributeError: {error_msg}", "detections": []}

        except RuntimeError as e:
            if "same device" in str(e):  # 處理「張量不在同一設備」錯誤
                print(f"--- [YOLO] 偵測到設備不一致，執行降級同步修復... ---")
                try:
                    model.cpu()  # 先將模型全部移回 CPU
                    model.set_classes(labels_list)  # 在 CPU 環境下設定類別（避免 GPU 張量混用）
                    model.to(device)  # 再將模型移回目標設備（GPU）
                    print("--- [YOLO] 設備同步修復完成 ---")
                except Exception as e2:
                    return {"error": f"YOLO Device Sync Failed: {str(e2)}", "detections": []}
            else:
                return {"error": f"YOLO RuntimeError: {str(e)}", "detections": []}

        # ========== 階段 4：開啟影片並計算取樣參數 ==========
        cap = cv2.VideoCapture(seg_path)  # 用 OpenCV 開啟影片檔案
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 取得幀率，若為 0 則預設 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 總幀數

        interval = max(1, int(round(fps * every_sec)))  # 取樣間隔（幀數），每 every_sec 秒取一幀，至少 1
        all_results = []  # 累積所有偵測結果
        crop_paths = []   # 累積所有 crop 的元數據（供 main.py 寫入 DB）

        frame_indices = list(range(0, total_frames, interval))  # 要處理的幀索引：[0, interval, 2*interval, ...]
        print(f"--- [YOLO] 開始處理片段: {seg_path}, 共 {len(frame_indices)} 幀 ---")

        # ========== 階段 5：批次讀取、推論、裁剪 ==========
        for i in range(0, len(frame_indices), batch_size):
            batch_idxs = frame_indices[i : i + batch_size]  # 本批次要處理的幀索引
            batch_frames = []  # 本批次的 BGR 影像
            valid_idxs = []    # 實際成功讀取的幀索引（可能有讀取失敗）

            for fi in batch_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)  # 將讀取位置跳至第 fi 幀
                ok, frame = cap.read()  # 讀取該幀（BGR numpy array）
                if ok:
                    batch_frames.append(frame)
                    valid_idxs.append(fi)

            if not batch_frames:
                continue  # 本批次無有效幀，跳過

            # --- YOLO 批次推論 ---
            try:
                batch_predict_res = model.predict(batch_frames, conf=score_thr, verbose=False)
            except Exception as predict_err:
                error_msg = str(predict_err)
                print(f"--- [YOLO Error] Predict failed: {error_msg} ---")
                # 模型損壞時強制重載：names 錯誤、NoneType、'Conv' object 等
                need_reload = (
                    "names" in error_msg.lower()
                    or "nonetype" in error_msg.lower()
                    or "'conv' object" in error_msg.lower()
                    or "not callable" in error_msg.lower()
                )
                if need_reload:
                    print("--- [YOLO] Attempting model reload due to corruption... ---")
                    import src.core.model_loader
                    src.core.model_loader._yolo_world_model = None
                    model = src.core.model_loader.get_yolo_model()
                    if model:
                        try:
                            model.to(device)
                            model.set_classes(labels_list)
                            model.to(device)
                            batch_predict_res = model.predict(batch_frames, conf=score_thr, verbose=False)
                        except Exception as retry_err:
                            print(f"--- [YOLO Error] Retry failed: {retry_err} ---")
                            continue
                    else:
                        print("--- [YOLO Error] Model reload failed, skipping batch ---")
                        continue
                else:
                    continue  # 其他錯誤，跳過本批次

            # --- 遍歷本批次中每一幀的推論結果 ---
            for idx, res in enumerate(batch_predict_res):
                fi = valid_idxs[idx]  # 對應的幀索引
                timestamp = round(fi / fps, 2)  # 該幀對應的時間戳（秒）
                frame = batch_frames[idx]  # 該幀的原始影像

                crops_imgs = []       # 本幀的 crop 影像（numpy array），供 ReID 批次用
                temp_detections = []  # 本幀的偵測記錄

                for b_idx, box in enumerate(res.boxes):
                    cls_id = int(box.cls)  # 類別 ID（對應 labels_list 索引）
                    label = labels_list[cls_id]  # 類別名稱（如 "person"）
                    conf = round(float(box.conf), 3)  # 信心分數，保留 3 位小數
                    xyxy = [int(round(float(c))) for i, c in enumerate(box.xyxy[0].tolist())]  # 邊界框 [x1,y1,x2,y2]

                    x1, y1, x2, y2 = xyxy
                    crop = frame[y1:y2, x1:x2]  # numpy 切片：從原幀裁出 bbox 區域
                    if crop.size > 0:  # 若裁切區域有效（非空）
                        crop_filename = f"crop_{fi}_{b_idx}_{label}.jpg"
                        crop_path = crops_dir / crop_filename
                        cv2.imwrite(str(crop_path), crop)  # 將裁剪圖存成 jpg

                        rel_crop_path = AnalysisService._safe_crop_path_for_storage(crop_path, seg_dir)

                        crops_imgs.append(crop)
                        temp_detections.append({
                            "timestamp": timestamp,
                            "label": label,
                            "score": conf,
                            "box": xyxy,
                            "frame": fi,
                            "path": rel_crop_path
                        })

                # 若本幀有 crop 且 ReID 模型可用，批次產生 ReID 特徵向量
                if crops_imgs and reid_model:
                    embeddings = AnalysisService.generate_reid_embeddings_batch(crops_imgs, reid_model, reid_device)
                    for d_idx, emb in enumerate(embeddings):
                        if emb:
                            temp_detections[d_idx]["reid_embedding"] = emb  # 附加 2048 維 embedding

                all_results.extend(temp_detections)
                crop_paths.extend(temp_detections)  # 相容 main.py 的 _save_results_to_postgres

            # 每批結束釋放該批 GPU 暫存，避免長片段累積
            del batch_frames, batch_predict_res
            _release_gpu_memory()

        cap.release()  # 釋放影片資源

        # 釋放 YOLO/ReID 推理過程產生的 GPU 暫存（模型保留在 GPU）
        _release_gpu_memory()

        # ========== 階段 6：統計與回傳 ==========
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
