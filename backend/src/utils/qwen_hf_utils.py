# -*- coding: utf-8 -*-
"""
本機 Hugging Face Qwen2.5-VL 推論工具。
支援單段與 batch（多段一次送入，模型只載入一次），與 infer_segment_qwen 相同輸出格式（events + summary）。
"""
from __future__ import annotations

import re
import json
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# 延遲 import，避免未安裝 transformers 時影響其他模組
_cached_model = None
_cached_processor = None
_cached_model_name: Optional[str] = None
_qwen_load_lock = threading.Lock()


def _get_qwen_local_snapshot_path(model_name: str) -> Optional[str]:
    """
    解析 HF cache 中的 snapshot 目錄路徑，供離線時直接 from_pretrained(path)。
    與 download_qwen25vl_7b.py / ask_qwen25vl 一致：先看環境變數，再掃 hub/models--Org--Name/snapshots/。
    """
    import os
    # 1) 明確指定 snapshot 路徑時直接使用（容器內路徑，例：/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/xxx）
    explicit = os.environ.get("QWEN_25VL_SNAPSHOT_PATH", "").strip()
    if explicit and Path(explicit).is_dir():
        return explicit
    # 2) 從 HF 快取根目錄找（與 huggingface_hub 一致）
    cache_root = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache_root:
        cache_root = os.path.expanduser("~/.cache/huggingface")
    hub_dir = Path(cache_root) / "hub"
    safe = model_name.replace("/", "--")
    snap_dir = hub_dir / f"models--{safe}" / "snapshots"
    if not snap_dir.is_dir():
        return None
    # 取第一個（或唯一）revision 目錄
    revs = sorted(p for p in snap_dir.iterdir() if p.is_dir())
    if not revs:
        return None
    return str(revs[0])


def _get_prompts_dir() -> Path:
    """後端 prompts 目錄（frame_prompt.md, summary_prompt.md）。"""
    backend = Path(__file__).resolve().parent.parent.parent
    return backend / "prompts"


def _read_prompt_file(filename: str) -> str:
    """讀取 prompts 目錄下的檔案。"""
    path = _get_prompts_dir() / filename
    if path.exists():
        t = path.read_text(encoding="utf-8").strip()
        if t and t[0] == "\ufeff":
            t = t[1:]
        return t
    return ""


def get_model_and_processor(model_name: str):
    """取得 HF 模型與 processor（單例 + 載入鎖，同 model_name 才重用；多執行緒同時呼叫時僅一個執行載入，避免並發載入導致 OOM）。"""
    global _cached_model, _cached_processor, _cached_model_name
    # 快速路徑：已快取且名稱符合則直接回傳（無需取鎖）
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model, _cached_processor

    with _qwen_load_lock:
        # 雙重檢查：可能其他 thread 已載入完成
        if _cached_model is not None and _cached_model_name == model_name:
            return _cached_model, _cached_processor

        import os
        if os.environ.get("TRANSFORMERS_OFFLINE", "").strip() in ("1", "true", "True"):
            os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        local_only = os.environ.get("TRANSFORMERS_OFFLINE", "").strip() in ("1", "true", "True")
        load_path = model_name
        if local_only:
            snapshot_path = _get_qwen_local_snapshot_path(model_name)
            if snapshot_path:
                load_path = snapshot_path
                print(f"--- [QwenHF] 離線載入: {load_path} ---")
            else:
                print("--- [QwenHF] 離線模式但未找到 snapshot 路徑，仍用 model_name 嘗試 ---")

        processor = AutoProcessor.from_pretrained(
            load_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
            local_files_only=local_only,
        )
        # 使用 device_map 時由 transformers/accelerate 負責放置設備，切勿再呼叫 model.to(device)，否則會觸發 Meta Tensor 報錯。
        _device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            load_path,
            torch_dtype=torch.float16,
            device_map=_device_map,
            attn_implementation="sdpa",
            local_files_only=local_only,
            low_cpu_mem_usage=False,
        )
        model.eval()
        _cached_model, _cached_processor, _cached_model_name = model, processor, model_name
    return _cached_model, _cached_processor


def _safe_parse_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_first_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    try:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return None


def _clean_summary(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"```[a-z]*\n?", "", s)
    s = s.replace("```", "").strip()
    return s


def _build_combined_instruction(event_prompt: str, summary_prompt: str) -> str:
    """與 analysis_service infer_segment_qwen 一致：單一請求要 events + summary。"""
    return (
        f"{event_prompt}\n\n"
        f"{summary_prompt}\n\n"
        "請輸出「一個」JSON 物件，且必須包含以下欄位：\n"
        "- events: 事件描述物件（可含 reason 等）\n"
        "- persons: 人員相關陣列\n"
        "- summary: 50–100 字繁體中文畫面摘要（字串）\n"
        "只輸出該 JSON，不要其他文字。"
    )


def infer_one(
    model_name: str,
    video_path: str,
    event_prompt: str,
    summary_prompt: str,
    target_short: int = 432,
    frames_per_segment: int = 5,
    sampling_fps: Optional[float] = None,
    sample_frames_fn=None,
    resize_fn=None,
) -> Tuple[Dict, str]:
    """
    單一段影片：取樣影格 → 組 message → HF generate → 解析 JSON。
    回傳 (frame_obj, summary_txt)，格式與 infer_segment_qwen 一致。
    """
    if sample_frames_fn is None or resize_fn is None:
        raise ValueError("infer_one 需要傳入 sample_frames_fn 與 resize_fn")

    try:
        frames_pil = sample_frames_fn(video_path, max_frames=frames_per_segment, sampling_fps=sampling_fps)
        frames_pil = [resize_fn(img, target_short) for img in frames_pil]
    except Exception as e:
        print(f"--- [QwenHF] ✗ 影格擷取失敗: {e} ---")
        return {"error": f"影格擷取失敗: {e}"}, ""

    model, processor = get_model_and_processor(model_name)
    import torch

    content: List[Dict[str, Any]] = []
    for img in frames_pil:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": _build_combined_instruction(event_prompt, summary_prompt)})

    messages = [
        {"role": "system", "content": "你是災害與人員異常偵測器，並產出影片摘要。只輸出一個 JSON 物件，含 events、persons、summary。"},
        {"role": "user", "content": content},
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if not isinstance(inputs, dict):
            inputs = dict(inputs) if hasattr(inputs, "items") else {}
        device = next(model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated[:, input_len:]
        output_text = processor.decode(generated_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as e:
        print(f"--- [QwenHF] ✗ 推論失敗: {e} ---")
        return {"error": f"HF 推論失敗: {e}"}, ""

    try:
        combined = _safe_parse_json(output_text) or _extract_first_json(output_text) or {}
        if isinstance(combined, list) and len(combined) > 0 and isinstance(combined[0], dict):
            combined = combined[0]
        if not isinstance(combined, dict):
            combined = {}
        events_in = combined.get("events") if isinstance(combined.get("events"), dict) else {}
        reason = (
            combined.get("event_reason")
            or combined.get("reason")
            or (events_in.get("reason", "") if isinstance(events_in, dict) else "")
        )
        if reason and isinstance(events_in, dict) and "reason" not in events_in:
            events_in = {**events_in, "reason": reason}
        frame_obj = {
            "events": events_in or {"reason": ""},
            "persons": combined.get("persons", []) if isinstance(combined.get("persons"), list) else [],
        }
        if not isinstance(frame_obj.get("events"), dict):
            frame_obj["events"] = {"reason": ""}
        summary_txt = _clean_summary(combined.get("summary") or output_text or "")
        if not summary_txt and output_text:
            summary_txt = _clean_summary(output_text)
        return frame_obj, summary_txt
    except (TypeError, AttributeError, KeyError) as e:
        if "string indices" in str(e) or "must be integers" in str(e):
            return {"events": {"reason": ""}, "persons": []}, _clean_summary(output_text or "")
        return {"error": f"解析失敗: {e}"}, _clean_summary(output_text or "")


def _parse_one_output(output_text: str) -> Tuple[Dict, str]:
    """將單一 generate 輸出的文字解析為 (frame_obj, summary_txt)。支援頂層為 dict 或 list 的 JSON，避免 string indices must be integers, not 'str'。"""
    if not isinstance(output_text, str):
        output_text = str(output_text) if output_text is not None else ""
    try:
        combined = _safe_parse_json(output_text) or _extract_first_json(output_text) or {}
        if isinstance(combined, list) and len(combined) > 0 and isinstance(combined[0], dict):
            combined = combined[0]
        if not isinstance(combined, dict):
            combined = {}
        events_in = combined.get("events") if isinstance(combined.get("events"), dict) else {}
        reason = (
            combined.get("event_reason")
            or combined.get("reason")
            or (events_in.get("reason", "") if isinstance(events_in, dict) else "")
        )
        if reason and isinstance(events_in, dict) and "reason" not in events_in:
            events_in = {**events_in, "reason": reason}
        frame_obj = {
            "events": events_in or {"reason": ""},
            "persons": combined.get("persons", []) if isinstance(combined.get("persons"), list) else [],
        }
        if not isinstance(frame_obj.get("events"), dict):
            frame_obj["events"] = {"reason": ""}
        summary_txt = _clean_summary(combined.get("summary") or output_text or "")
        if not summary_txt and output_text:
            summary_txt = _clean_summary(output_text)
        return frame_obj, summary_txt
    except (TypeError, AttributeError, KeyError) as e:
        if "string indices" in str(e) or "must be integers" in str(e) or "get" in str(e).lower():
            return {"events": {"reason": ""}, "persons": []}, _clean_summary(output_text or "")
        raise


def infer_batch(
    model_name: str,
    video_paths: List[str],
    event_prompt: str,
    summary_prompt: str,
    target_short: int = 432,
    frames_per_segment: int = 5,
    sampling_fps: Optional[float] = None,
    sample_frames_fn=None,
    resize_fn=None,
    max_inference_batch_size: Optional[int] = None,
) -> List[Tuple[Dict, str]]:
    
    if not video_paths:
        return []
    if sample_frames_fn is None or resize_fn is None:
        raise ValueError("infer_batch 需要傳入 sample_frames_fn 與 resize_fn")

    import torch

    # 將事件偵測與摘要的 Prompt 合併成一段給模型的指令。
    instruction = _build_combined_instruction(event_prompt, summary_prompt)
    system_content = "你是災害與人員異常偵測器，並產出影片摘要。只輸出一個 JSON 物件，含 events、persons、summary。"

    # 初始化結果列表，預設狀態為「未處理」，確保即便後續出錯也能回傳對應數量的結果。
    all_parsed: List[Tuple[Dict, str]] = [({"error": "未處理"}, "")] * len(video_paths)

    # 1) 資料準備：為每段影片取樣影格並建 messages
    list_of_messages: List[List[Dict]] = []

    # 初始化有效索引列表，用來追蹤哪些影片成功取樣，哪些失敗。
    valid_indices: List[int] = []

    # 開始遍歷每一段影片。
    for i, video_path in enumerate(video_paths):
        try:

            # 呼叫 sample_frames_fn 取樣影格，並調用 resize_fn 縮放成指定大小。
            frames_pil = sample_frames_fn(
                video_path, max_frames=frames_per_segment, sampling_fps=sampling_fps)
            frames_pil = [resize_fn(img, target_short) for img in frames_pil]
        except Exception as e:
            print(f"--- [QwenHF] 影格擷取失敗 [{i}]: {e} ---")
            all_parsed[i] = ({"error": f"影格擷取失敗: {e}"}, "")
            continue

        # 建立 content 列表，包含所有影格的 image 字典。
        content: List[Dict[str, Any]] = []
        for img in frames_pil:
            content.append({"type": "image", "image": img})

        # 將 instruction 指令加入 content 列表。
        content.append({"type": "text", "text": instruction})

        # 建立 messages 列表，包含 system 和 user 角色。組裝成 Chat 格式，包含 system 指令（規範輸出為 JSON）與 user 的多模態內容。
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": content},
        ]

        # 將 messages 列表加入 list_of_messages 列表。
        list_of_messages.append(messages)

        # 將有效索引 i 加入 valid_indices 列表。
        valid_indices.append(i)

    # 如果 list_of_messages 列表為空，表示所有影片都取樣失敗，直接回傳預設的「未處理」結果。
    if not list_of_messages:
        return all_parsed

    # 取得模型與 processor。
    model, processor = get_model_and_processor(model_name)

    # 取得 device。
    device = next(model.parameters()).device

    # 2) 依 max_inference_batch_size 分批，避免 5090 VRAM 溢出
    batch_size = min(max_inference_batch_size, len(list_of_messages))
    for start in range(0, len(list_of_messages), batch_size):
        batch_messages = list_of_messages[start : start + batch_size]
        batch_valid_idx = valid_indices[start : start + batch_size]

        try:
            # 3) 使用 HuggingFace Processor 將多組訊息轉換為 PyTorch 張量。padding=True 會自動補齊不同長度的輸入，讓它們能整合成一個 Batch。
            inputs = processor.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            # 如果 apply_chat_template 回傳 list，表示無法批次推論，改依序處理。
            if isinstance(inputs, list):
                raise TypeError("apply_chat_template 回傳 list，無法批次推論，改依序處理")

            # 如果 apply_chat_template 回傳 dict，表示可以批次推論。
            if not isinstance(inputs, dict):

                # 將 inputs 轉換為 dict，如果 inputs 有 items 屬性，則將其轉換為 dict。
                inputs = dict(inputs) if hasattr(inputs, "items") else {}

            # 如果 inputs 為空或沒有 input_ids，表示無法推論，直接跳過。
            if not inputs or "input_ids" not in inputs:
                raise ValueError("apply_chat_template 未回傳有效 input_ids")

            # 將 inputs 轉換為 dict，如果 inputs 有 items 屬性，則將其轉換為 dict。
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):

                    # 將所有輸入資料（圖片特徵、Token IDs 等）搬移到 GPU 記憶體中。
                    inputs[k] = v.to(device)

            # 4) 一次 model.generate，關閉梯度計算以節省記憶體。max_new_tokens=512 限制模型回傳的字數，避免模型「話太多」導致逾時或 OOM。
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,

                    # 設定 pad_token_id，用於填充短序列，確保所有序列長度相同。
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                )

            # 如果 generated 為 1 維，則將其轉換為 2 維。
            if generated.dim() == 1:
                generated = generated.unsqueeze(0)

            # model.generate 回傳的是「原始輸入 + 新生成的文字」。這裡我們切掉前面的輸入部分，只留下模型新產生的回答。
            input_len = inputs["input_ids"].shape[1]
            generated_trimmed = [
                generated[b, input_len:].clone()
                for b in range(generated.size(0))]

            # 5) 只解碼新生成的 token，並 batch_decode
            output_texts = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # 依序解析每個 batch 的輸出，並將結果存入 all_parsed 列表。
            for j, text in enumerate(output_texts):
                if j < len(batch_valid_idx):
                    orig_i = batch_valid_idx[j]

                    # 解析單一輸出，並將結果存入 all_parsed 列表。
                    all_parsed[orig_i] = _parse_one_output(text if isinstance(text, str) else str(text or ""))
        except Exception as e:
            print(f"--- [QwenHF] Batch 編碼/推論失敗 (start={start}), 改為依序推論: {e} ---")
            for j, orig_i in enumerate(batch_valid_idx):
                try:
                    out = infer_one(
                        model_name,
                        video_paths[orig_i],
                        event_prompt,
                        summary_prompt,
                        target_short=target_short,
                        frames_per_segment=frames_per_segment,
                        sampling_fps=sampling_fps,
                        sample_frames_fn=sample_frames_fn,
                        resize_fn=resize_fn,
                    )
                    all_parsed[orig_i] = out
                except Exception as e2:
                    all_parsed[orig_i] = ({"error": f"HF 推論失敗: {e2}"}, "")

    # 未出現在 valid_indices 的影片（取樣失敗）已維持預設錯誤
    return all_parsed
