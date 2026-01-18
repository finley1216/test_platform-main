# -*- coding: utf-8 -*-
import io
import base64
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

def _resize_short_side(img: Image.Image, short: int) -> Image.Image:
    """將圖片短邊縮放到指定尺寸"""
    if not short or short <= 0: return img
    w, h = img.size
    s = min(w, h)
    if s == short: return img
    scale = short / float(s)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.BILINEAR)

def _pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    """PIL 轉 JPEG base64"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality or 85), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _resize_frame_for_vlm(frame: np.ndarray, target_short: int) -> np.ndarray:
    """調整幀大小用於 VLM"""
    h, w = frame.shape[:2]
    short = min(h, w)
    if short <= target_short: return frame
    
    scale = target_short / short
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
