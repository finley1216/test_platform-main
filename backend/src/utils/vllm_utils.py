# -*- coding: utf-8 -*-
"""vLLM 推論工具。提供與 ollama_utils 相容的 _vllm_chat 介面，供 analysis_service.infer_segment_vllm 使用。"""
import json
import copy
import re
import subprocess
import requests
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from src.config import config


def _parse_fraction_fps(s: Optional[str]) -> Optional[float]:
    if not s or s in ("0/0", "N/A"):
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", str(s).strip())
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a / b) if b else None
    try:
        return float(s)
    except ValueError:
        return None


def _log_ffprobe_video_stats(video_path: str, model_name: str) -> None:
    """送 vLLM 前印出影片檔本身資訊（非 vLLM 內部實際取樣幀數）。"""
    if not getattr(config, "VLLM_VIDEO_FFPROBE_LOG", True):
        return
    vp = Path(video_path).expanduser()
    if not vp.is_absolute():
        vp = (Path.cwd() / vp).resolve()
    if not vp.is_file():
        print(f"--- [vLLM-VideoDirect][ffprobe] 略過（非檔案）: {video_path} ---", flush=True)
        return
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-of",
        "json",
        str(vp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except FileNotFoundError:
        print(
            "--- [vLLM-VideoDirect][ffprobe] 未安裝 ffprobe，無法印出影片幀資訊 ---",
            flush=True,
        )
        return
    except subprocess.TimeoutExpired:
        print(f"--- [vLLM-VideoDirect][ffprobe] 逾時: {vp} ---", flush=True)
        return
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        err = (proc.stderr or proc.stdout or "").strip()[:300]
        print(f"--- [vLLM-VideoDirect][ffprobe] 失敗 rc={proc.returncode} {err} ---", flush=True)
        return
    try:
        meta: Dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"--- [vLLM-VideoDirect][ffprobe] JSON 解析失敗: {vp} ---", flush=True)
        return
    streams = meta.get("streams") or []
    st0 = streams[0] if streams else {}
    fmt = meta.get("format") or {}
    dur_s = st0.get("duration") or fmt.get("duration")
    try:
        duration_sec = float(dur_s) if dur_s not in (None, "N/A", "") else None
    except (TypeError, ValueError):
        duration_sec = None
    fps = _parse_fraction_fps(st0.get("r_frame_rate")) or _parse_fraction_fps(st0.get("avg_frame_rate"))
    nb = st0.get("nb_frames")
    nb_int: Optional[int] = None
    if nb not in (None, "N/A", ""):
        try:
            nb_int = int(float(nb))
        except (TypeError, ValueError):
            nb_int = None
    est_frames: Optional[int] = None
    if duration_sec is not None and fps is not None and fps > 0:
        est_frames = max(1, int(round(duration_sec * fps)))
    size_b = fmt.get("size")
    print(
        "--- [vLLM-VideoDirect][ffprobe] "
        f"file={vp.name} path={vp} | "
        f"codec={st0.get('codec_name')} {st0.get('width')}x{st0.get('height')} | "
        f"duration_sec={duration_sec} | fps_r={st0.get('r_frame_rate')} fps_avg={st0.get('avg_frame_rate')} "
        f"-> fps_parsed={fps} | "
        f"nb_frames_container={nb_int if nb_int is not None else nb} | "
        f"est_frames_duration_x_fps={est_frames} | "
        f"size_bytes={size_b} | "
        f"model={model_name} ---",
        flush=True,
    )
    print(
        "--- [vLLM-VideoDirect][hint] 上列為「影片檔」層級；若請求已附 video_url.num_frames 則由該值決定均勻取樣幀數，"
        "否則由 vLLM 預設（常見約 32）。可設 VLLM_LOGGING_LEVEL=DEBUG 看引擎 log。 ---",
        flush=True,
    )


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
            ptype = part.get("type")
            if ptype == "image_url":
                image_obj = part.get("image_url")
                if not isinstance(image_obj, dict):
                    continue
                url = image_obj.get("url")
                if isinstance(url, str) and url.startswith("data:image"):
                    image_obj["url"] = "<data:image;base64,...(omitted)>"
            elif ptype == "video_url":
                vo = part.get("video_url")
                if isinstance(vo, dict) and isinstance(vo.get("url"), str) and len(vo["url"]) > 120:
                    vo = dict(vo)
                    vo["url"] = vo["url"][:80] + "...(omitted)"
                    part["video_url"] = vo
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
    nreq = len(batch_requests)
    mw = max(1, min(max_workers, nreq or 1))
    print(f"--- [vLLM Batch] 開始並行處理 {nreq} 個任務 (Threads: {mw}) ---", flush=True)

    def _single_worker(req_item: Dict) -> str:
        try:
            req_enable_thinking = req_item.get("enable_thinking", enable_thinking)
            video_path = req_item.get("video_path")
            
            # 有 video_path 就走 video_direct，不做本地截圖
            if video_path:
                return _vllm_chat_video_direct(
                    model_name=model_name,
                    messages=req_item.get("messages", []),
                    video_path=video_path,
                    stream=False,
                    enable_thinking=req_enable_thinking,
                    video_num_frames=req_item.get("video_num_frames"),
                )
            # 沒有 video_path 才走原本截圖路徑
            return _vllm_chat(
                model_name=model_name,
                messages=req_item.get("messages", []),
                sampling_fps=req_item.get("sampling_fps"),
                frames_per_segment=req_item.get("frames_per_segment", 5),
                enable_thinking=req_enable_thinking,
                stream=False,
            )
        except Exception as e:
            print(f"--- [vLLM Batch Worker] 請求出錯: {e} ---")
            return ""

    results = ["" for _ in range(len(batch_requests))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=mw) as executor:
        future_to_index = {
            executor.submit(_single_worker, item): i
            for i, item in enumerate(batch_requests)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
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
    video_num_frames: Optional[int] = None,
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
        # 註：部分 vLLM 版本對 OpenAI 相容欄位 video_url.num_frames 會忽略（prompt_tokens 與省略時相同）。
        # 若要以伺服器端統一控制影片取樣，請在 vLLM 啟動參數加 --mm-processor-kwargs（見 docker-compose）。
        # Qwen3-VL 在 vLLM 0.17.x 通常用 fps；Qwen2.5-VL 等可能用 num_frames，請依映像／原始碼調整。
        # 另可改用截圖路徑（segment_pipeline_multipart + sampling_fps）在後端控制取樣。
        video_obj: Dict[str, Any] = {"url": video_uri}
        if video_num_frames is not None and int(video_num_frames) > 0:
            video_obj["num_frames"] = int(video_num_frames)
        msg["content"] = [
            {"type": "text", "text": str(text)},
            {"type": "video_url", "video_url": video_obj},
        ]
        break

    _max_out = int(getattr(config, "VLLM_VIDEO_DIRECT_MAX_COMPLETION_TOKENS", 4096) or 4096)
    payload = {
        "model": model_name,
        "messages": payload_messages,
        "stream": stream,
        "max_tokens": max(256, _max_out),
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

    _nf_log = f", num_frames={int(video_num_frames)}" if video_num_frames else ""
    print(
        f"--- [vLLM-VideoDirect] POST {url} (model={model_name}, video_path={video_path}{_nf_log}) ---",
        flush=True,
    )
    _log_ffprobe_video_stats(video_path, model_name)
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
        usage = data.get("usage")
        if isinstance(usage, dict):
            print(
                f"[usage] prompt_tokens={usage.get('prompt_tokens')} "
                f"completion_tokens={usage.get('completion_tokens')} "
                f"total_tokens={usage.get('total_tokens')} "
                f"video={Path(video_path).name}",
                flush=True,
            )
        else:
            print(
                f"[usage] (無 usage 欄位，無法用 token 反推視覺幀數) keys={list(data.keys())}",
                flush=True,
            )
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
            video_num_frames=req_item.get("video_num_frames"),
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