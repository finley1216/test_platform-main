# -*- coding: utf-8 -*-
"""
以 vLLM（Qwen-VL 多模態）輔助判斷：查詢圖與多張候選是否為同一實例（如 Re-ID 候選篩選）。
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw, ImageFont

from src.config import config
from src.utils.image_utils import _pil_to_b64, _resize_short_side
from src.utils.vllm_utils import _vllm_chat

router = APIRouter(tags=["VLM / ReID 輔助"])

from src.main import get_api_key  # noqa: E402  — 與其他 api 模組相同，在 router 之後匯入

_MAX_CANDIDATES = 30
_MAX_THUMB_SHORT = 768
_MIN_THUMB_SHORT = 128


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """從模型輸出中盡量取出單一 JSON 物件。"""
    if not text or not text.strip():
        return None
    s = text.strip()
    # 去除 ```json ... ``` 包圍
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            return None
    return None


def _normalize_mismatch_indices(parsed: Dict[str, Any]) -> Tuple[List[int], List[str]]:
    """回傳 1-based 候選索引列表（第一張候選為 1）與警告訊息。"""
    warns: List[str] = []
    raw = parsed.get("mismatch_candidate_indices")
    if raw is None:
        raw = parsed.get("mismatch_indices") or parsed.get("not_same_object")
    if raw is None:
        return [], warns
    out: List[int] = []
    if isinstance(raw, list):
        for x in raw:
            try:
                n = int(x)
                if n >= 1:
                    out.append(n)
            except (TypeError, ValueError):
                warns.append(f"略過非整數 mismatch 項: {x!r}")
    elif isinstance(raw, dict):
        for k in raw.keys():
            try:
                n = int(k)
                if n >= 1:
                    out.append(n)
            except (TypeError, ValueError):
                pass
    return sorted(set(out)), warns


def _build_comparison_montage(
    query_img: Image.Image,
    candidate_imgs: List[Image.Image],
    mismatch_1based: Set[int],
    cell_short: int = 160,
    max_cols: int = 6,
) -> Tuple[Image.Image, str]:
    """
    將查詢圖與候選以多列網格排列；候選編號 1..N，不符者紅框、其餘綠框。
    回傳 (RGB PIL, JPEG base64 不含 data: 前綴)。
    """
    q = _resize_short_side(query_img.convert("RGB"), cell_short)
    cells: List[Image.Image] = [q]
    labels = ["Q"]
    borders: List[str] = ["#555555"]  # 查詢：中性框

    for i, im in enumerate(candidate_imgs, start=1):
        cells.append(_resize_short_side(im.convert("RGB"), cell_short))
        labels.append(str(i))
        borders.append("#cc2222" if i in mismatch_1based else "#228822")

    pad = 6
    label_h = 22
    n = len(cells)
    cols = min(max_cols, max(1, n))
    rows = (n + cols - 1) // cols

    col_widths: List[int] = [0] * cols
    row_heights: List[int] = [0] * rows
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        bw, bh = cell.size
        col_widths[c] = max(col_widths[c], bw)
        row_heights[r] = max(row_heights[r], bh)

    w_total = pad + sum(col_widths) + pad * cols
    h_total = pad + sum((rh + label_h + pad) for rh in row_heights)

    canvas = Image.new("RGB", (w_total, h_total), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    y0 = pad
    for r in range(rows):
        rh = row_heights[r]
        x0 = pad
        for c in range(cols):
            idx = r * cols + c
            if idx >= n:
                break
            cell, lab, color = cells[idx], labels[idx], borders[idx]
            bw, bh = cell.size
            cell_w = col_widths[c]
            x = x0 + (cell_w - bw) // 2
            y_img = y0 + (rh - bh) // 2
            draw.rectangle([x - 2, y_img - 2, x + bw + 1, y_img + bh + 1], outline=color, width=3)
            canvas.paste(cell, (x, y_img))
            if hasattr(draw, "textlength"):
                tw = float(draw.textlength(lab, font=font))
            else:
                tw = float(draw.textbbox((0, 0), lab, font=font)[2] - draw.textbbox((0, 0), lab, font=font)[0])
            draw.text((x0 + (cell_w - tw) / 2, y0 + rh + 4), lab, fill=(240, 240, 240), font=font)
            x0 += col_widths[c] + pad
        y0 += rh + label_h + pad

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=88, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return canvas, b64


_DEFAULT_SYSTEM = (
    "You are a vision expert for instance-level re-identification. "
    "Compare the reference image against each candidate crop. "
    "Answer with compact JSON only."
)


def _default_user_prompt(n_candidates: int) -> str:
    return f"""圖片順序（共 {1 + n_candidates} 張）：
- 第 1 張：查詢目標（REFERENCE），請以它為「同一物件實例」的基準。
- 第 2 張起：候選圖，依序編號為候選 1、候選 2、…、候選 {n_candidates}。

任務：判斷每張候選與查詢是否為「同一個實例」（同一台車／同一人／同一物件外觀與關鍵特徵可對應），
而非僅僅「同類別」。光線、角度、遮擋可能造成差異，但若高度可能是同一實例請不要列入不符。

請只輸出合法 JSON（不要 markdown、不要註解），格式如下：
{{
  "summary_zh": "一句繁體中文總結",
  "mismatch_candidate_indices": [ ],
  "mismatch_reasons_zh": {{ }},
  "uncertain_candidate_indices": [ ]
}}

說明：
- mismatch_candidate_indices：與查詢「明顯不是」同一實例的候選編號（1 到 {n_candidates} 的整數）陣列；若皆相符則為 []。
- mismatch_reasons_zh：可選，key 為字串形式的候選編號（如 "3"），value 為簡短繁體中文理由。
- uncertain_candidate_indices：無法判斷、需人工覆核的候選編號（可為 []）。
"""


@router.post("/v1/vlm/same_instance_judge")
def same_instance_judge(
    query: UploadFile = File(..., description="查詢目標圖（單張）"),
    candidates: List[UploadFile] = File(..., description="候選圖，可多檔同名欄位 candidates"),
    model: str = Form("Qwen/Qwen3-VL-8B-Instruct-AWQ"),
    extra_instruction: str = Form(""),
    return_comparison_image: bool = Form(True),
    thumb_short: int = Form(448),
    jpeg_quality: Optional[int] = Form(None),
    enable_thinking: bool = Form(False),
    api_key: str = Depends(get_api_key),
):
    """
    上傳 1 張查詢圖 + 多張候選圖，由 vLLM 判斷哪些候選與查詢**不是**同一實例；
    可選回傳一張橫向拼貼比對圖（JPEG base64）。

    客戶端範例（curl）::

        curl -sS -H "X-API-Key: $KEY" \\
          -F "query=@query.jpg" \\
          -F "candidates=@c1.jpg" -F "candidates=@c2.jpg" \\
          -F "model=Qwen/Qwen3-VL-8B-Instruct-AWQ" \\
          http://localhost:3000/api/v1/vlm/same_instance_judge

    請先啟動對應 vLLM（例如 `docker compose up vllm-qwen3-awq`），並讓後端 `QWEN3_AWQ_VLLM_BASE` 指向該服務。
    """
    t0 = time.time()
    if not candidates:
        raise HTTPException(status_code=422, detail="至少需要 1 張候選圖 candidates")
    if len(candidates) > _MAX_CANDIDATES:
        raise HTTPException(
            status_code=400,
            detail=f"候選張數過多（>{_MAX_CANDIDATES}），請分批呼叫",
        )

    ts = int(thumb_short)
    if ts < _MIN_THUMB_SHORT or ts > _MAX_THUMB_SHORT:
        raise HTTPException(
            status_code=400,
            detail=f"thumb_short 須介於 {_MIN_THUMB_SHORT}～{_MAX_THUMB_SHORT}",
        )
    qjpeg = int(jpeg_quality) if jpeg_quality is not None else int(getattr(config, "VLLM_JPEG_QUALITY", 70))

    def _read_upload_to_pil(up: UploadFile) -> Image.Image:
        raw = up.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"空檔案: {up.filename}")
        return Image.open(io.BytesIO(raw))

    try:
        q_pil = _read_upload_to_pil(query)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取查詢圖: {e}") from e

    cand_pils: List[Image.Image] = []
    cand_names: List[str] = []
    for c in candidates:
        try:
            cand_pils.append(_read_upload_to_pil(c))
            cand_names.append(c.filename or "")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"無法讀取候選圖: {e}") from e

    n = len(cand_pils)
    # 送 vLLM 前縮圖以控制 payload
    q_small = _resize_short_side(q_pil.convert("RGB"), ts)
    images_b64: List[str] = [_pil_to_b64(q_small, quality=qjpeg)]
    for im in cand_pils:
        images_b64.append(_pil_to_b64(_resize_short_side(im.convert("RGB"), ts), quality=qjpeg))

    user_text = _default_user_prompt(n)
    if extra_instruction and extra_instruction.strip():
        user_text += "\n\n補充說明（使用者提供）：\n" + extra_instruction.strip()

    messages = [
        {"role": "system", "content": _DEFAULT_SYSTEM},
        {"role": "user", "content": user_text},
    ]

    try:
        raw_text = _vllm_chat(
            model_name=model,
            messages=messages,
            images_b64=images_b64,
            enable_thinking=enable_thinking,
            max_tokens=4096,
            temperature=0.05,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vLLM 呼叫失敗: {type(e).__name__}: {e}") from e

    parsed = _extract_json_object(raw_text or "")
    mismatch_idx, parse_warns = ([], [])
    if isinstance(parsed, dict):
        mismatch_idx, parse_warns = _normalize_mismatch_indices(parsed)
    mismatch_set = set(mismatch_idx)

    montage_b64: Optional[str] = None
    if return_comparison_image:
        try:
            _, montage_b64 = _build_comparison_montage(q_pil, cand_pils, mismatch_set)
        except Exception:
            montage_b64 = None

    elapsed = round(time.time() - t0, 3)
    return {
        "ok": True,
        "model": model,
        "candidate_count": n,
        "candidate_filenames": cand_names,
        "mismatch_candidate_indices_1based": mismatch_idx,
        "parsed_json": parsed,
        "parse_warnings": parse_warns,
        "vlm_raw_text": raw_text,
        "comparison_image_jpeg_base64": montage_b64,
        "elapsed_sec": elapsed,
    }
