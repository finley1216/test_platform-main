# -*- coding: utf-8 -*-
"""vLLM 推論工具。提供與 ollama_utils 相容的 _vllm_chat 介面，供 analysis_service.infer_segment_vllm 使用。"""
import json
import copy
import requests
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.config import config


def _resolve_vllm_base(model_name: str) -> str:
    """依模型名稱選擇對應的 vLLM endpoint。"""
    model_lower = (model_name or "").lower()
    if "qwen3-vl" in model_lower and "awq" in model_lower:
        return (
            getattr(config, "QWEN3_AWQ_VLLM_BASE", None)
            or getattr(config, "QWEN3_VLLM_BASE", None)
            or config.VLLM_BASE
            or ""
        ).rstrip("/")
    if "qwen3-vl" in model_lower:
        return (getattr(config, "QWEN3_VLLM_BASE", None) or config.VLLM_BASE or "").rstrip("/")
    return (config.VLLM_BASE or "").rstrip("/")


def _is_qwen3_model(model_name: str) -> bool:
    """判斷是否為 Qwen3 系列模型（如 Qwen3-VL）。"""
    model_lower = (model_name or "").lower()
    return "qwen3" in model_lower


def _sanitize_openai_payload_for_log(payload: Dict) -> Dict:
    """輸出日誌用：保留 OpenAI 結構，避免 base64 圖片塞滿 log。"""
    masked = copy.deepcopy(payload)
    msgs = masked.get("messages")
    if not isinstance(msgs, list):
        return masked

    for m in msgs:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            image_obj = part.get("image_url")
            if not isinstance(image_obj, dict):
                continue
            url = image_obj.get("url")
            if isinstance(url, str) and url.startswith("data:image"):
                image_obj["url"] = "<data:image;base64,...(omitted)>"
    return masked


def _vllm_chat(
    model_name: str,
    messages: List[Dict],
    images_b64: Optional[List[str]] = None,
    stream: bool = False,
    video_path: Optional[str] = None,
    sampling_fps: Optional[float] = None,
    frames_per_segment: int = 5,
    enable_thinking: Optional[bool] = None,
    **kwargs,
) -> str:
    """呼叫 vLLM OpenAI-compatible API 進行多模態對話（影像+文字或影片）。"""
    # 若傳入 video_path，從影片取樣影格轉成 images_b64
    if video_path and (images_b64 is None or len(images_b64) == 0):
        try:
            from pathlib import Path
            from src.utils.video_utils import _sample_frames_evenly_to_pil
            from src.utils.image_utils import _pil_to_b64
            if not Path(video_path).exists():
                print(f"--- [vLLM] video_path 不存在: {video_path} ---")
            else:
                frames_pil = _sample_frames_evenly_to_pil(
                    video_path, max_frames=frames_per_segment, sampling_fps=sampling_fps
                )
                images_b64 = [_pil_to_b64(img) for img in frames_pil]
                print(f"--- [vLLM] 從影片取樣 {len(images_b64)} 張影格送 vLLM ---")
        except Exception as e:
            print(f"--- [vLLM] 影片取樣失敗: {e} ---")
            images_b64 = []

    # 深拷貝 messages，避免改動呼叫端傳入的物件
    payload_messages = copy.deepcopy(messages)

    # 若有圖片，將最後一則 user message 的 content 改為 OpenAI 多模態格式：
    # content = [ {"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}, ... ]
    if images_b64:
        for msg in reversed(payload_messages):
            if msg.get("role") == "user":
                text = msg.get("content") or ""
                content_parts = [{"type": "text", "text": text}]
                for b64 in images_b64:
                    url = f"data:image/jpeg;base64,{b64}" if not b64.startswith("data:") else b64
                    content_parts.append({"type": "image_url", "image_url": {"url": url}})
                msg["content"] = content_parts
                break

    payload = {
        "model": model_name,
        "messages": payload_messages,
        "stream": stream,
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    if enable_thinking is not None and _is_qwen3_model(model_name):
        payload["enable_thinking"] = bool(enable_thinking)

    base = _resolve_vllm_base(model_name)
    url = f"{base}/v1/chat/completions"
    timeout = getattr(config, "VLLM_REQUEST_TIMEOUT", 600)
    headers = {"Content-Type": "application/json"}
    if getattr(config, "VLLM_API_KEY", None):
        headers["Authorization"] = f"Bearer {config.VLLM_API_KEY}"

    print(f"--- [vLLM] 請求 URL: {url} (model={model_name}, timeout={timeout}s) ---", flush=True)
    if not base:
        print("--- [vLLM] 警告: VLLM_BASE 為空，請在 .env 設定 VLLM_BASE（例如 http://vllm:8440 或 http://127.0.0.1:8440）---", flush=True)

    try:
        payload_for_log = _sanitize_openai_payload_for_log(payload)
        print(
            "--- [vLLM][OpenAI API][Request] ---\n"
            + json.dumps(payload_for_log, ensure_ascii=False, indent=2)[:6000]
            + "\n--- [vLLM][OpenAI API][Request End] ---",
            flush=True,
        )
        print("--- [vLLM] 發送 POST 至 vLLM... ---", flush=True)
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        print(f"--- [vLLM] 收到 HTTP 狀態: {r.status_code} ---", flush=True)
        r.raise_for_status()
        try:
            resp_preview = r.text[:6000]
            print(
                f"--- [vLLM][OpenAI API][Response Raw] ---\n{resp_preview}\n--- [vLLM][OpenAI API][Response Raw End] ---",
                flush=True,
            )
        except Exception:
            pass

        if stream:
            full_txt = ""
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.strip().startswith("data: "):
                    s = line.strip()[6:]
                    if s == "[DONE]":
                        break
                    try:
                        j = json.loads(s)
                        delta = (j.get("choices") or [{}])[0].get("delta", {}) or {}
                        full_txt += delta.get("content", "")
                        if j.get("choices") and (j["choices"][0].get("finish_reason") is not None):
                            break
                    except json.JSONDecodeError:
                        pass
            return full_txt

        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            print("--- [vLLM] 回應無 choices ---")
            try:
                print(json.dumps(data, ensure_ascii=False, indent=2)[:1500])
            except Exception:
                print(data)
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        print(
            "--- [vLLM][OpenAI API][Response Parsed] "
            f"choices={len(choices)} content_len={len(content)} ---",
            flush=True,
        )
        if content:
            print(f"--- [vLLM] 回應 content 長度: {len(content)} 字，前 200 字: {content[:200]!r} ---")
        else:
            print("--- [vLLM] 回應 content 為空，完整 API 回應如下 ---")
            try:
                print(json.dumps(data, ensure_ascii=False, indent=2)[:1500])
            except Exception:
                print(data)
            print("--- [vLLM] 以上為空回應時的 body ---")
        return content
    except requests.exceptions.ConnectionError as e:
        print(f"--- [vLLM] ✗ 連線失敗（vLLM 服務可能未啟動或 URL 錯誤）: {e} ---")
        print(f"--- [vLLM] 請確認: 1) VLLM_BASE={base}  2) vLLM 容器是否運行 3) 若後端在 Docker 內，是否用服務名如 http://vllm:8440 ---")
        raise
    except requests.exceptions.Timeout as e:
        print(f"--- [vLLM] ✗ 請求逾時 (timeout={timeout}s): {e} ---")
        raise
    except requests.exceptions.HTTPError as e:
        err_body = ""
        if e.response is not None:
            try:
                err_body = e.response.text[:800]
            except Exception:
                pass
        print(f"--- [vLLM] ✗ HTTP 錯誤: {e} ---")
        if err_body:
            print(f"--- [vLLM] 回應 body: {err_body} ---")
        raise
    except Exception as e:
        print(f"--- [vLLM] ✗ 請求失敗: {type(e).__name__}: {e} ---")
        raise




def _vllm_chat_batch(
    model_name: str,
    batch_requests: List[Dict],
    max_workers: int = 8,
    enable_thinking: Optional[bool] = None,
) -> List[str]:
    """
    批次呼叫 vLLM API。
    透過 ThreadPoolExecutor 並行發送請求，觸發伺服器端的 Continuous Batching。
    
    batch_requests 格式範例:
    [
        {
            "messages": [...], 
            "video_path": "...", 
            "sampling_fps": 1.0, 
            "frames_per_segment": 8
        },
        ...
    ]
    """
    nreq = len(batch_requests)
    mw = max(1, min(max_workers, nreq or 1))
    print(f"--- [vLLM Batch] 開始並行處理 {nreq} 個任務 (Threads: {mw}) ---", flush=True)

    # 定義一個內部 Worker，負責執行單一請求
    # 這裡直接呼叫您現有的 _vllm_chat 函式，完整繼承其取樣、b64 轉換與錯誤處理邏輯
    def _single_worker(req_item: Dict) -> str:
        try:
            req_enable_thinking = req_item.get("enable_thinking", enable_thinking)
            return _vllm_chat(
                model_name=model_name,
                messages=req_item.get("messages", []),
                video_path=req_item.get("video_path"),
                sampling_fps=req_item.get("sampling_fps"),
                frames_per_segment=req_item.get("frames_per_segment", 5),
                enable_thinking=req_enable_thinking,
                stream=False  # 批次模式通常不建議使用串流
            )
        except Exception as e:
            print(f"--- [vLLM Batch Worker] 請求出錯: {e} ---")
            return ""

    # 使用執行緒池並行執行
    results = ["" for _ in range(len(batch_requests))]
    
    # 使用 ThreadPoolExecutor 同時進行影像取樣 (CPU) 與 API 請求 (I/O)
    with concurrent.futures.ThreadPoolExecutor(max_workers=mw) as executor:
        # 建立 Future 物件與索引的對照，確保回傳順序一致
        future_to_index = {
            executor.submit(_single_worker, item): i 
            for i, item in enumerate(batch_requests)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                data = future.result()
                results[index] = data
            except Exception as exc:
                print(f"--- [vLLM Batch] 片段索引 {index} 產生異常: {exc} ---")
                results[index] = ""

    print(f"--- [vLLM Batch] 所有任務執行完畢，成功取得 {len(results)} 筆結果 ---", flush=True)
    return results


def _vllm_chat_video_direct(
    model_name: str,
    messages: List[Dict],
    video_path: str,
    stream: bool = False,
    enable_thinking: Optional[bool] = None,
) -> str:
    """
    影片直送 vLLM（不做本地截圖取幀）。
    會把最後一則 user 訊息轉成多模態 content，附上 video_url。
    """
    payload_messages = copy.deepcopy(messages)

    # 使用標準 file URI，避免 file://segment/... 這種非標準路徑造成 vLLM 400。
    vp = Path(video_path).expanduser()
    if not vp.is_absolute():
        vp = (Path.cwd() / vp).resolve()
    video_uri = vp.as_uri()

    for msg in reversed(payload_messages):
        if msg.get("role") != "user":
            continue
        text = msg.get("content") or ""
        if isinstance(text, list):
            # 呼叫端已經給多模態內容時，不覆蓋
            break
        msg["content"] = [
            {"type": "text", "text": str(text)},
            {"type": "video_url", "video_url": {"url": video_uri}},
        ]
        break

    payload = {
        "model": model_name,
        "messages": payload_messages,
        "stream": stream,
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    if enable_thinking is not None and _is_qwen3_model(model_name):
        payload["enable_thinking"] = bool(enable_thinking)

    base = _resolve_vllm_base(model_name)
    url = f"{base}/v1/chat/completions"
    timeout = getattr(config, "VLLM_REQUEST_TIMEOUT", 600)
    headers = {"Content-Type": "application/json"}
    if getattr(config, "VLLM_API_KEY", None):
        headers["Authorization"] = f"Bearer {config.VLLM_API_KEY}"

    print(
        f"--- [vLLM-VideoDirect] POST {url} (model={model_name}, video_path={video_path}) ---",
        flush=True,
    )
    try:
        payload_for_log = _sanitize_openai_payload_for_log(payload)
        print(
            "--- [vLLM-VideoDirect][Request] ---\n"
            + json.dumps(payload_for_log, ensure_ascii=False, indent=2)[:6000]
            + "\n--- [vLLM-VideoDirect][Request End] ---",
            flush=True,
        )
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        print(f"--- [vLLM-VideoDirect] HTTP {r.status_code} ---", flush=True)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""
    except Exception as e:
        if isinstance(e, requests.HTTPError) and e.response is not None:
            body = (e.response.text or "").strip()
            if body:
                print(f"--- [vLLM-VideoDirect][ErrorBody] {body[:2000]} ---", flush=True)
        print(f"--- [vLLM-VideoDirect] 請求失敗: {type(e).__name__}: {e} ---", flush=True)
        raise


def _vllm_chat_batch_video_direct(
    model_name: str,
    batch_requests: List[Dict],
    max_workers: int = 8,
    enable_thinking: Optional[bool] = None,
) -> List[str]:
    """
    批次影片直送 vLLM（不經過 _vllm_chat，避免截圖路徑）。
    batch_requests 每筆需包含 messages 與 video_path。
    """
    nreq = len(batch_requests)
    mw = max(1, min(max_workers, nreq or 1))
    results = ["" for _ in range(nreq)]
    print(f"--- [vLLM-VideoDirect-Batch] 任務數={nreq}, threads={mw} ---", flush=True)

    def _single_worker(req_item: Dict) -> str:
        req_enable_thinking = req_item.get("enable_thinking", enable_thinking)
        return _vllm_chat_video_direct(
            model_name=model_name,
            messages=req_item.get("messages", []),
            video_path=req_item.get("video_path", ""),
            stream=False,
            enable_thinking=req_enable_thinking,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=mw) as executor:
        future_to_index = {
            executor.submit(_single_worker, item): i
            for i, item in enumerate(batch_requests)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"--- [vLLM-VideoDirect-Batch] idx={idx} 異常: {exc} ---", flush=True)
                results[idx] = ""
    return results