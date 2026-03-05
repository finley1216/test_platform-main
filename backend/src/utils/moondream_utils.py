# -*- coding: utf-8 -*-
"""
Moondream 模型載入與影片片段推論工具。
支援 moondream-2b-2025-04-14（單一權重檔）與 moondream3-preview（分片權重）。
供 AnalysisService 在 model_type=moondream 時使用。

模型目錄優先順序：
1. 環境變數 MOONDREAM_MODELS_DIR（例如 Docker 內設為 /app/models）
2. 否則依 __file__ 推算：專案根/models（若 backend 在 /backend 則改用 PROJECT_ROOT）
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# 專案根目錄：backend/src/utils -> backend -> (專案根或 /)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_PARENT = _BACKEND_DIR.parent


def _get_moondream_models_base() -> Path:
    """模型根目錄：支援環境變數，避免 Docker 下 _PARENT 變成 / 導致 /models。"""
    env_base = os.getenv("MOONDREAM_MODELS_DIR", "").strip()
    if env_base:
        return Path(env_base).resolve()
    # 依 __file__ 推算：專案根/models
    candidate = _PARENT / "models"
    if _PARENT != Path("/") and candidate.exists():
        return candidate.resolve()
    if _PARENT == Path("/"):
        # Docker 常見：backend 在 /backend，專案根用 PROJECT_ROOT 或 cwd
        try:
            from src.config import config
            proot = getattr(config, "PROJECT_ROOT", None)
            if proot is not None:
                p = (Path(proot) if isinstance(proot, str) else proot) / "models"
                if p.exists():
                    return p.resolve()
        except Exception:
            pass
        cwd_models = Path(os.getcwd()).resolve() / "models"
        if cwd_models.exists():
            return cwd_models
    return candidate.resolve()


MOONDREAM_MODELS_BASE = _get_moondream_models_base()
PROMPTS_DIR = _BACKEND_DIR.parent / "backend" / "prompts"
if _PARENT == Path("/"):
    try:
        from src.config import config
        proot = getattr(config, "PROJECT_ROOT", None)
        if proot is not None:
            proot_path = Path(proot) if isinstance(proot, str) else proot
            prompts_candidate = proot_path / "backend" / "prompts"
            if prompts_candidate.exists():
                PROMPTS_DIR = prompts_candidate.resolve()
    except Exception:
        pass

# 允許的模型版本（對應 models/ 底下的目錄名）
MOONDREAM_VERSIONS = ("moondream-2b-2025-04-14", "moondream3-preview")

# 模組級快取：同一 process 內同版本只載入一次
_model_cache: Dict[Tuple[str, str], Any] = {}


def get_moondream_model_dir(version: str) -> Path:
    """依版本名稱回傳模型目錄絕對路徑。"""
    if version not in MOONDREAM_VERSIONS:
        raise ValueError(f"不支援的 Moondream 版本: {version}，可選: {MOONDREAM_VERSIONS}")
    path = MOONDREAM_MODELS_BASE / version
    if not path.is_dir():
        hint = " 若在 Docker 執行，請設定環境變數 MOONDREAM_MODELS_DIR 指向模型目錄（例如 /app/models）。" if str(path).startswith("/models") or MOONDREAM_MODELS_BASE == Path("/models") else ""
        raise FileNotFoundError(f"Moondream 模型目錄不存在: {path}，請先下載模型。{hint}")
    return path


def _extract_first_json(text: str) -> Optional[Dict]:
    """從模型輸出文字中抽出第一個 JSON 物件。"""
    if not (text or (text and text.strip())):
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _sync_trust_remote_code_to_cache(model_dir: Path) -> None:
    """
    Hugging Face 只會把部分檔案複製到 cache，導致 from .lora import 等失敗。
    在載入前把模型目錄內所有 .py 同步到 cache 的 transformers_modules/{module_name}。
    """
    import shutil
    # 與 HF 相同：目錄名中 - 替換為 _hyphen_
    module_name = model_dir.name.replace("-", "_hyphen_")
    cache_base = Path(os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))
    cache_module_dir = cache_base / "modules" / "transformers_modules" / module_name
    cache_module_dir.mkdir(parents=True, exist_ok=True)
    for f in model_dir.iterdir():
        if f.suffix in (".py", ".json") or f.name == "config.py":
            dest = cache_module_dir / f.name
            try:
                shutil.copy2(f, dest)
            except Exception:
                pass


def _ensure_starmie_tokenizer_local(model_dir: Path) -> None:
    """
    Moondream3 內部會呼叫 Tokenizer.from_pretrained("moondream/starmie-v1")，離線會失敗。
    若模型目錄內有 starmie-v1-tokenizer/ 或 tokenizer/（含 tokenizer.json），
    則 monkey-patch 讓 from_pretrained("moondream/starmie-v1") 改從本地載入。
    """
    for subdir in ("starmie-v1-tokenizer", "tokenizer"):
        local_tokenizer_dir = model_dir / subdir
        tokenizer_json = local_tokenizer_dir / "tokenizer.json"
        if tokenizer_json.exists():
            import tokenizers
            _orig = tokenizers.Tokenizer.from_pretrained
            _local_json_str = str(tokenizer_json.resolve())

            def _patched(path, **kwargs):
                if path == "moondream/starmie-v1":
                    # 直接用 from_file 從本地載入，不經過 Hub（避免 repo_id 格式檢查）
                    return tokenizers.Tokenizer.from_file(_local_json_str)
                return _orig(path, **kwargs)

            tokenizers.Tokenizer.from_pretrained = _patched
            return


def load_moondream_model(version: str, device: str = "cuda"):
    """
    載入 Moondream 模型（支援 2b 單檔與 3 分片權重）。
    使用快取：同 (version, device) 只載入一次。
    """
    cache_key = (version, device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    import torch
    from transformers import AutoModelForCausalLM

    model_dir = get_moondream_model_dir(version)
    model_dir_str = str(model_dir.resolve())

    # Moondream3 需 starmie-v1 tokenizer；離線時改從模型目錄內 tokenizer 子目錄載入
    if version == "moondream3-preview":
        _ensure_starmie_tokenizer_local(model_dir)

    # 先同步所有 .py 到 HF cache，避免 trust_remote_code 只複製主檔導致 lora.py 等找不到
    _sync_trust_remote_code_to_cache(model_dir)

    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = {"": device} if device == "cuda" else None

    # 強制離線：避免 from_pretrained 嘗試連 Hub（tokenizer 已用上面 patch 改為本地）
    prev_offline = os.environ.get("TRANSFORMERS_OFFLINE"), os.environ.get("HF_HUB_OFFLINE")
    try:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        model = AutoModelForCausalLM.from_pretrained(
            model_dir_str,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        )
    finally:
        if prev_offline[0] is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_offline[0]
        if prev_offline[1] is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline[1]
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    _model_cache[cache_key] = model
    return model



def _resolve_segment_path(segment_path: str) -> str:
    """將片段路徑解析為絕對路徑；若相對路徑找不到檔案，嘗試以專案根目錄為基底。"""
    p = Path(segment_path)
    if p.is_absolute() and p.exists():
        return str(p)
    if not p.is_absolute():
        resolved = p.resolve()
        if resolved.exists():
            return str(resolved)
        # 可能 cwd 是 backend/，片段在 test_platform-main/segment/ 下
        for base in (MOONDREAM_MODELS_BASE.parent, _BACKEND_DIR):
            candidate = base / segment_path
            if candidate.exists():
                return str(candidate.resolve())
    return str(p.resolve())


def infer_segment_moondream(
    model_version: str,
    segment_path: str,
    event_detection_prompt: str,
    summary_prompt: str,
    frames_per_segment: int = 8,
    sampling_fps: Optional[float] = None,
    device: str = "cuda",
) -> Tuple[Dict[str, Any], str]:
    """
    使用 Moondream 分析單一影片片段：事件偵測 + 摘要。
    回傳 (frame_obj, summary_txt)，格式與 Qwen 分支一致，供 _normalize_vlm_output 使用。
    """
    import logging
    from src.utils.video_utils import _sample_frames_evenly_to_pil

    logger = logging.getLogger(__name__)
    default_events = {
        "water_flood": False,
        "fire": False,
        "abnormal_attire_face_cover_at_entry": False,
        "person_fallen_unmoving": False,
        "double_parking_lane_block": False,
        "smoking_outside_zone": False,
        "crowd_loitering": False,
        "security_door_tamper": False,
        "reason": "",
    }

    segment_path = _resolve_segment_path(segment_path)
    if not Path(segment_path).exists():
        err = f"片段檔案不存在: {segment_path}"
        logger.warning("--- [Moondream] %s ---", err)
        return {"error": err}, ""

    try:
        frames_pil = _sample_frames_evenly_to_pil(
            segment_path,
            max_frames=frames_per_segment,
            sampling_fps=sampling_fps,
        )
        logger.info("--- [Moondream] 已採樣 %d 張影格，開始推論 ---", len(frames_pil))
    except Exception as e:
        err = f"影格擷取失敗: {e}"
        logger.warning("--- [Moondream] %s ---", err)
        return {"error": err}, ""

    model = load_moondream_model(model_version, device=device)

    # 事件偵測：優先使用傳入的 prompt（與 frame_prompt.md 一致），否則讀取預設檔，最後才用精簡版
    event_instruction = ""
    if event_detection_prompt and event_detection_prompt.strip():
        event_instruction = event_detection_prompt.strip()[:2800] + "\n\n只輸出純 JSON，不要 Markdown 或解釋。"
    else:
        frame_prompt_path = PROMPTS_DIR / "frame_prompt.md"
        if frame_prompt_path.exists():
            event_instruction = frame_prompt_path.read_text(encoding="utf-8").strip()[:2800] + "\n\n只輸出純 JSON，不要 Markdown 或解釋。"
        else:
            # 精簡版仍須包含全部 8 個事件鍵，與 frame_prompt.md 格式一致
            event_instruction = (
                "你是一個嚴格的災害/人員異常偵測器。請僅依據畫面真實內容，嚴格遵守以下 JSON 格式輸出，不要猜測。\n"
                "{\"events\":{"
                "\"water_flood\":false,\"fire\":false,\"abnormal_attire_face_cover_at_entry\":false,"
                "\"person_fallen_unmoving\":false,\"double_parking_lane_block\":false,\"smoking_outside_zone\":false,"
                "\"crowd_loitering\":false,\"security_door_tamper\":false,\"reason\":\"\""
                "}}\n"
                "只輸出一個 JSON，不要其他文字。"
            )

    all_events: List[Dict] = []
    for i, img in enumerate(frames_pil):
        try:
            out = model.query(image=img, question=event_instruction)
            raw = out.get("answer", "") if isinstance(out, dict) else str(out)
            parsed = _extract_first_json(raw)
            if parsed and "events" in parsed:
                all_events.append(parsed["events"])
            else:
                if raw and len(raw.strip()) > 0:
                    logger.debug("--- [Moondream] 影格 %d 輸出無法解析為 events JSON，raw 前 200 字: %s ---", i, raw.strip()[:200])
                all_events.append(default_events.copy())
        except Exception as e:
            logger.warning("--- [Moondream] 影格 %d query 異常: %s ---", i, e)
            all_events.append(default_events.copy())

    merged = default_events.copy()
    reasons = []
    for ev in all_events:
        if not isinstance(ev, dict):
            continue
        for k in default_events:
            if k == "reason":
                if ev.get("reason"):
                    reasons.append(ev["reason"])
            else:
                if ev.get(k) is True:
                    merged[k] = True
    merged["reason"] = "；".join(reasons) if reasons else ""

    frame_obj = {"events": merged, "persons": []}

    # 摘要：使用傳入的 summary_prompt.md 內容（與後端預設一致），否則讀取預設檔
    summary_question = ""
    if summary_prompt and summary_prompt.strip():
        summary_question = summary_prompt.strip()[:800]
    if not summary_question:
        summary_prompt_path = PROMPTS_DIR / "summary_prompt.md"
        if summary_prompt_path.exists():
            summary_question = summary_prompt_path.read_text(encoding="utf-8").strip()[:800]
    if not summary_question:
        summary_question = "請根據影格內容，以客觀角度詳細描述畫面中出現的所有人、車、物體與動態，以及任何可能對事後查找有幫助的細節線索。約 150-250 字繁體中文。"

    summary_txt = ""
    try:
        out = model.query(image=frames_pil[0], question=summary_question)
        summary_txt = (out.get("answer", "") or "").strip() if isinstance(out, dict) else str(out).strip()
    except Exception as e:
        logger.warning("--- [Moondream] 摘要 query 異常: %s ---", e)
        summary_txt = ""

    return frame_obj, summary_txt
