# -*- coding: utf-8 -*-
"""
本機 Hugging Face Qwen2.5-VL 推論工具。
支援單段與 batch（多段一次送入，模型只載入一次），與 infer_segment_qwen 相同輸出格式（events + summary）。
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# 延遲 import，避免未安裝 transformers 時影響其他模組
_cached_model = None
_cached_processor = None
_cached_model_name: Optional[str] = None


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
    """取得 HF 模型與 processor（單例，同 model_name 才重用）。"""
    global _cached_model, _cached_processor, _cached_model_name
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model, _cached_processor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    _cached_model, _cached_processor, _cached_model_name = model, processor, model_name
    return model, processor


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
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated[:, input_len:]
        output_text = processor.decode(generated_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as e:
        print(f"--- [QwenHF] ✗ 推論失敗: {e} ---")
        return {"error": f"HF 推論失敗: {e}"}, ""

    combined = _safe_parse_json(output_text) or _extract_first_json(output_text) or {}
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
        "persons": combined.get("persons", []),
    }
    if not isinstance(frame_obj.get("events"), dict):
        frame_obj["events"] = {"reason": ""}
    summary_txt = _clean_summary(combined.get("summary") or output_text or "")
    if not summary_txt and output_text:
        summary_txt = _clean_summary(output_text)
    return frame_obj, summary_txt


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
) -> List[Tuple[Dict, str]]:
    """
    多段影片「一次」送入模型：先為每段組 message 並取得 inputs，pad/stack 成一個 batch 後單次 model.generate，
    再 batch_decode 得到 N 段輸出，順序與 video_paths 一致。
    """
    if not video_paths:
        return []
    if sample_frames_fn is None or resize_fn is None:
        raise ValueError("infer_batch 需要傳入 sample_frames_fn 與 resize_fn")

    import torch

    model, processor = get_model_and_processor(model_name)
    instruction = _build_combined_instruction(event_prompt, summary_prompt)
    system_msg = "你是災害與人員異常偵測器，並產出影片摘要。只輸出一個 JSON 物件，含 events、persons、summary。"

    # 為每段取樣影格並呼叫 processor，得到 list of input dicts（每個 batch_size=1）
    inputs_list: List[Dict[str, Any]] = []
    valid_indices: List[int] = []  # 成功處理的 video 索引

    for i, video_path in enumerate(video_paths):
        try:
            frames_pil = sample_frames_fn(video_path, max_frames=frames_per_segment, sampling_fps=sampling_fps)
            frames_pil = [resize_fn(img, target_short) for img in frames_pil]
        except Exception as e:
            print(f"--- [QwenHF] Batch 影格擷取失敗 [{i}]: {e} ---")
            continue
        content: List[Dict[str, Any]] = []
        for img in frames_pil:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": instruction})
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content},
        ]
        try:
            inp = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs_list.append(inp)
            valid_indices.append(i)
        except Exception as e:
            print(f"--- [QwenHF] Batch apply_chat_template 失敗 [{i}]: {e} ---")

    if not inputs_list:
        return [({"error": "所有片段影格或 template 失敗"}, "")] * len(video_paths)

    # 將多筆 inputs pad/stack 成一個 batch
    pad_token_id = getattr(processor, "pad_token_id", None) or getattr(model.config, "pad_token_id", 0)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    max_len = max(inp["input_ids"].shape[1] for inp in inputs_list)
    device = next(model.parameters()).device
    batch_size = len(inputs_list)

    padded_input_ids = []
    padded_attention_mask = []
    for inp in inputs_list:
        seq_len = inp["input_ids"].shape[1]
        pad_len = max_len - seq_len
        ids = inp["input_ids"].squeeze(0)
        mask = inp.get("attention_mask")
        if mask is not None:
            mask = mask.squeeze(0)
        else:
            mask = torch.ones_like(ids, dtype=torch.long)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)], dim=0)
        padded_input_ids.append(ids)
        padded_attention_mask.append(mask)

    batched = {
        "input_ids": torch.stack(padded_input_ids).to(device),
        "attention_mask": torch.stack(padded_attention_mask).to(device),
    }

    # 其餘 tensor 鍵（如 pixel_values, image_grid_thw）若存在且可沿 dim=0 疊加則加入
    first = inputs_list[0]
    for key in first.keys():
        if key in ("input_ids", "attention_mask"):
            continue
        if not hasattr(first[key], "to") or not hasattr(first[key], "shape"):
            continue
        try:
            tensors = [inp[key].to(device) for inp in inputs_list]
            batched[key] = torch.cat(tensors, dim=0)
        except Exception:
            pass

    # 一次 generate，再 batch_decode
    try:
        with torch.no_grad():
            generated = model.generate(**batched, max_new_tokens=512, do_sample=False, pad_token_id=pad_token_id)
    except Exception as e:
        print(f"--- [QwenHF] Batch generate 失敗（可能 pixel 結構不支援 stack），改依序推論: {e} ---")
        return [
            infer_one(
                model_name, video_paths[i], event_prompt, summary_prompt,
                target_short=target_short, frames_per_segment=frames_per_segment,
                sampling_fps=sampling_fps, sample_frames_fn=sample_frames_fn, resize_fn=resize_fn,
            )
            for i in range(len(video_paths))
        ]

    input_len = batched["input_ids"].shape[1]
    generated_trimmed = generated[:, input_len:]
    output_texts = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # 依 output_texts 解析 JSON，並對應回原本 video_paths 順序（含失敗的補預設）
    def parse_one(text: str) -> Tuple[Dict, str]:
        combined = _safe_parse_json(text) or _extract_first_json(text) or {}
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
            "persons": combined.get("persons", []),
        }
        if not isinstance(frame_obj.get("events"), dict):
            frame_obj["events"] = {"reason": ""}
        summary_txt = _clean_summary(combined.get("summary") or text or "")
        if not summary_txt and text:
            summary_txt = _clean_summary(text)
        return frame_obj, summary_txt

    result_by_valid_idx = [parse_one(t) for t in output_texts]
    results: List[Tuple[Dict, str]] = []
    j = 0
    for i in range(len(video_paths)):
        if j < len(valid_indices) and valid_indices[j] == i:
            results.append(result_by_valid_idx[j])
            j += 1
        else:
            results.append(({"error": "影格擷取或 template 失敗"}, ""))
    print(f"--- [QwenHF] Batch 一次推論 {len(inputs_list)} 段完成 ---")
    return results
