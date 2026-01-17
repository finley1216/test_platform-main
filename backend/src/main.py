# -*- coding: utf-8 -*-
import os, io, re, json, base64, tempfile, subprocess, time, secrets, hashlib, copy, requests, cv2

import numpy as np
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("--- [WARNING] google.generativeai 未安裝，Gemini 功能將無法使用 ---")

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
from datetime import datetime, date, timedelta
from fastapi import FastAPI, Request, UploadFile,status , File, Form, Depends, Security, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
if HAS_GEMINI:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
else:
    # 定義假類別以避免導入錯誤
    class HarmCategory:
        HARM_CATEGORY_HARASSMENT = None
        HARM_CATEGORY_HATE_SPEECH = None
        HARM_CATEGORY_SEXUALLY_EXPLICIT = None
        HARM_CATEGORY_DANGEROUS_CONTENT = None
    class HarmBlockThreshold:
        BLOCK_NONE = None
# [DEPRECATED] RAGStore 已完全移除，現在完全使用 PostgreSQL + pgvector
# 不再需要 faiss，所有 RAG 功能都使用 PostgreSQL
HAS_RAG_STORE = False
RAGStore = None
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, select
from src.config import config

# Import SentenceTransformer for embedding generation
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False
    print("--- [WARNING] sentence-transformers not installed, embedding search will not work ---")
try:
    from src.database import get_db
    from src.models import Summary, ObjectCrop
    HAS_DB = True
except ImportError:
    HAS_DB = False
    print("--- [WARNING] 資料庫模組未找到，PostgreSQL 功能將無法使用 ---")

# ================== 環境變數 ==================

try:
    from prompts import EVENT_DETECTION_PROMPT, SUMMARY_PROMPT
except Exception:
    # 預設值已更新為符合您的新需求
    EVENT_DETECTION_PROMPT = "請根據提供的影格輸出事件 JSON。"
    SUMMARY_PROMPT = "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。"

# Use configuration from config.py
ADMIN_TOKEN = config.ADMIN_TOKEN
SESSION_TTL_SEC = config.SESSION_TTL_SEC
OLLAMA_BASE = config.OLLAMA_BASE
OWL_API_BASE = config.OWL_API_BASE
OWL_VIDEO_URL = config.OWL_VIDEO_URL

# RAG 索引開關（預設啟用）
AUTO_RAG_INDEX = config.AUTO_RAG_INDEX

# ReID 配置
ALLOW_REID_FALLBACK_TO_CLIP = config.ALLOW_REID_FALLBACK_TO_CLIP

# Gemini API Key (already configured in config.py)
GEMINI_API_KEY = config.GEMINI_API_KEY

# RAG 相關配置（已遷移到 PostgreSQL，這些變數保留用於向後兼容）
RAG_DIR = config.RAG_DIR  # [DEPRECATED] 不再使用，保留用於向後兼容
RAG_INDEX_PATH = config.RAG_INDEX_PATH  # [DEPRECATED] 不再使用，保留用於向後兼容
OLLAMA_EMBED_MODEL = config.OLLAMA_EMBED_MODEL

# Initialize SentenceTransformer for embedding generation
# Model: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
_embedding_model = None
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_model():
    """Get or initialize the SentenceTransformer model (CPU mode only)"""
    global _embedding_model
    if _embedding_model is None and HAS_SENTENCE_TRANSFORMER:
        try:
            # 強制使用 CPU 模式，避免 GPU 資源競爭
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 隱藏 GPU，強制使用 CPU
            # 嘗試使用本地緩存路徑
            local_model_path = "/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "modules.json")):
                # 使用本地路徑載入
                _embedding_model = SentenceTransformer(local_model_path, device='cpu')
            else:
                # 使用模型名稱載入（會嘗試從緩存讀取）
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
            print(f"✓ SentenceTransformer model loaded: {EMBEDDING_MODEL_NAME} (CPU Mode)")
        except Exception as e:
            print(f"⚠️  Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"無法載入 SentenceTransformer 模型: {e}")
    return _embedding_model

# CLIP 模型用於圖像 embedding（以圖搜圖）
_clip_model = None
_clip_processor = None
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 輸出 512 維向量

def get_clip_model():
    """獲取或初始化 CLIP 模型（用於圖像 embedding）"""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            import os
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"--- [CLIP] 載入模型: {CLIP_MODEL_NAME} (device: {device}) ---")
            
            # 直接使用模型名稱載入，transformers 會自動從緩存讀取（如果可用）
            # 如果緩存不完整，會嘗試從網路下載缺失的部分
            try:
                # 先嘗試只使用本地文件（如果緩存完整）
                print("--- [CLIP] 嘗試從本地緩存載入模型和處理器... ---")
                _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(device).eval()
                print("--- [CLIP] 模型載入完成，載入處理器... ---")
                _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
                print(f"✓ CLIP 模型和處理器從本地緩存載入完成")
            except Exception as local_e:
                # 如果本地緩存不完整，嘗試從網路下載（但可能因為網路問題失敗）
                print(f"--- [CLIP] 本地緩存不完整或載入失敗: {type(local_e).__name__}: {str(local_e)[:200]} ---")
                print("--- [CLIP] 嘗試從網路下載（如果網路可用，但可能很慢）... ---")
                try:
                    # 設置較短的超時，避免長時間等待
                    import requests
                    original_timeout = getattr(requests.adapters, 'DEFAULT_TIMEOUT', None)
                    if hasattr(requests.adapters, 'DEFAULT_TIMEOUT'):
                        requests.adapters.DEFAULT_TIMEOUT = 10  # 10 秒超時
                    
                    print("--- [CLIP] 從網路載入模型... ---")
                    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=False).to(device).eval()
                    print("--- [CLIP] 模型載入完成，從網路載入處理器... ---")
                    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=False)
                    print(f"✓ CLIP 模型和處理器從網路載入完成")
                    
                    # 恢復原始超時設置
                    if original_timeout is not None and hasattr(requests.adapters, 'DEFAULT_TIMEOUT'):
                        requests.adapters.DEFAULT_TIMEOUT = original_timeout
                except Exception as network_e:
                    print(f"⚠️  CLIP 模型網路載入也失敗: {type(network_e).__name__}: {str(network_e)[:200]}")
                    # 不拋出異常，讓應用繼續運行，但模型為 None
                    _clip_model = None
                    _clip_processor = None
                    print("⚠️  CLIP 功能將無法使用（無法從緩存或網路載入模型），但應用可以繼續運行")
                    print("⚠️  建議：檢查網路連線或確保模型緩存完整")
        except ImportError:
            print("--- [WARNING] transformers 未安裝，CLIP 功能將無法使用 ---")
            _clip_model = None
            _clip_processor = None
        except Exception as e:
            print(f"⚠️  Failed to load CLIP model: {e}")
            import traceback
            traceback.print_exc()
            _clip_model = None
            _clip_processor = None
            print("⚠️  CLIP 功能將無法使用，但應用可以繼續運行")
    return _clip_model, _clip_processor

def generate_image_embedding(image_path: str) -> Optional[List[float]]:
    """
    為圖像生成 CLIP embedding（用於以圖搜圖）
    
    Args:
        image_path: 圖像文件路徑
        
    Returns:
        embedding 向量（512 維）或 None
    """
    try:
        print(f"--- [generate_image_embedding] 開始處理圖片: {image_path} ---")
        import os
        file_exists = os.path.exists(image_path)
        print(f"--- [generate_image_embedding] 檢查文件是否存在: {file_exists} ---")
        
        if not file_exists:
            print(f"--- [generate_image_embedding] ✗ 文件不存在: {image_path} ---")
            return None
        
        clip_model, clip_processor = get_clip_model()
        if clip_model is None:
            print("--- [generate_image_embedding] ✗ CLIP 模型為 None（可能未載入或載入失敗）---")
            return None
        if clip_processor is None:
            print("--- [generate_image_embedding] ✗ CLIP 處理器為 None（可能未載入或載入失敗）---")
            return None
        print(f"--- [generate_image_embedding] CLIP 模型已加載（模型: {type(clip_model).__name__}, 處理器: {type(clip_processor).__name__}）---")
        import torch
        
        # 讀取圖像
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"--- [WARNING] 無法讀取圖像: {image_path}（可能是格式不支持或文件損壞）---")
            return None
        
        print(f"--- [generate_image_embedding] 圖片尺寸: {img_bgr.shape} ---")
        
        # 轉換為 RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 使用 CLIP 處理
        print(f"--- [generate_image_embedding] 使用 CLIP 處理器處理圖片... ---")
        inputs = clip_processor(images=[img_rgb], return_tensors="pt")
        device = next(clip_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"--- [generate_image_embedding] 輸入已移至設備: {device} ---")
        
        # 生成 embedding
        print(f"--- [generate_image_embedding] 生成 embedding... ---")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)[0]
            embedding = image_features.detach().cpu().numpy().astype(np.float32)
            # L2 正規化
            embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        print(f"--- [generate_image_embedding] ✓ embedding 生成成功，維度: {len(embedding)} ---")
        return embedding.tolist()
    except Exception as e:
        print(f"--- [ERROR] 生成圖像 embedding 失敗 ({image_path}): {e} ---")
        import traceback
        traceback.print_exc()
        return None

def generate_text_embedding(text: str) -> Optional[List[float]]:
    """
    為文字生成 CLIP embedding（用於文字搜圖）
    
    Args:
        text: 文字描述（例如 "藍色衣服的人"）
        
    Returns:
        embedding 向量（512 維）或 None
    """
    try:
        clip_model, clip_processor = get_clip_model()
        import torch
        
        # 使用 CLIP 處理文字
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        device = next(clip_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成 embedding
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)[0]
            embedding = text_features.detach().cpu().numpy().astype(np.float32)
            # L2 正規化
            embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        return embedding.tolist()
    except Exception as e:
        print(f"--- [WARNING] 生成文字 embedding 失敗 ({text}): {e} ---")
        return None

# Video library directory (歷史影片分類存放位置)
VIDEO_LIB_DIR = config.VIDEO_LIB_DIR

SERVER_API_KEY = config.SERVER_API_KEY
API_KEY_NAME = config.API_KEY_NAME

api_key_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# ================== FastAPI ==================

def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    return app

app = _make_app()

# ================== 啟動時預載入模型 ==================
@app.on_event("startup")
async def startup_event():
    """應用啟動時預載入模型，避免首次請求時延遲（在背景執行，不阻塞啟動）"""
    import asyncio
    import threading
    
    def preload_models_sync():
        """在背景線程中預載入模型（同步執行，避免阻塞）"""
        print("=" * 80)
        print("--- [啟動] 開始預載入模型（背景執行）... ---")
        
        # 預載入 CLIP 模型（以圖搜圖功能）
        try:
            print("--- [啟動] 預載入 CLIP 模型... ---")
            # 設置超時，避免無限等待
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("CLIP 模型載入超時")
            
            # 只在非 Windows 系統上使用 signal
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 60 秒超時
            except (AttributeError, ValueError):
                pass  # Windows 不支持 SIGALRM
            
            try:
                clip_model, clip_processor = get_clip_model()
                if clip_model is not None and clip_processor is not None:
                    print("✓ CLIP 模型預載入完成")
                else:
                    print("⚠️  CLIP 模型預載入失敗（模型為 None）")
            finally:
                try:
                    signal.alarm(0)  # 取消超時
                except (AttributeError, ValueError):
                    pass
        except TimeoutError:
            print("⚠️  CLIP 模型預載入超時（60秒），將在首次使用時載入")
        except Exception as e:
            print(f"⚠️  CLIP 模型預載入失敗: {e}")
            import traceback
            traceback.print_exc()
            # 不中斷啟動，讓應用繼續運行
        
        # 預載入 SentenceTransformer 模型（RAG 搜索功能）
        try:
            print("--- [啟動] 預載入 SentenceTransformer 模型... ---")
            embedding_model = get_embedding_model()
            if embedding_model is not None:
                print("✓ SentenceTransformer 模型預載入完成")
            else:
                print("⚠️  SentenceTransformer 模型預載入失敗（模型為 None）")
        except Exception as e:
            print(f"⚠️  SentenceTransformer 模型預載入失敗: {e}")
            import traceback
            traceback.print_exc()
            # 不中斷啟動，讓應用繼續運行
        
        print("--- [啟動] 模型預載入完成 ---")
        print("=" * 80)
    
    # 在背景線程中執行，不阻塞應用啟動
    thread = threading.Thread(target=preload_models_sync, daemon=True)
    thread.start()

app.mount("/segment", StaticFiles(directory="segment"), name="segment")

# ================== 小工具 ==================

#從網路下載影片
def _download_to_temp(url: str) -> str:
  """
  從網路下載影片並存到暫存資料夾 (/tmp 或類似位置)。
  使用了 stream=True 和分塊寫入 (1024*1024 bytes)，這是為了防止下載超大影片時把記憶體塞爆。
  """
  r = requests.get(url, stream=True, timeout=600)
  r.raise_for_status()
  suffix = Path(url).suffix or ".mp4"
  fd, path = tempfile.mkstemp(prefix="up_", suffix=suffix)
  with os.fdopen(fd, "wb") as f:
      for chunk in r.iter_content(1024*1024):
          if chunk: f.write(chunk)
  return path

# 轉成人類可讀的時間 ex: 把 3665.5 秒轉成 01:01:05
def _fmt_hms(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 精確查詢影片的總秒數。
def _probe_duration_seconds(path: str) -> float:
    """
    精確查詢影片的總秒數。
    它呼叫了 ffprobe (FFmpeg 的分析工具) 來讀取 metadata。
    """
    r = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path],
        capture_output=True, text=True, check=True
    )
    return float(r.stdout.strip())

# 計算每一段切片應該從幾秒開始。
def _compute_starts(duration: float, segment: float, overlap: float) -> List[float]:
    step = segment - overlap
    if step <= 0: raise ValueError("overlap 必須小於 segment_duration")
    starts, t = [], 0.0
    while t < duration:
        starts.append(round(t, 3)); t += step
    return starts

# 計算好切點 -> 迴圈執行 FFmpeg -> 產出一堆小影片檔
def _split_one_video(input_path: str, out_dir: str, segment: float, overlap: float, prefix: str="segment", 
                     resolution: Optional[int] = None, strict_mode: bool = False) -> List[str]:
    """
    切割影片，支援兩種模式：
    - strict_mode=False (預設): 使用 -c copy，速度快但不保證精確時間（優化：100x 速度提升）
    - strict_mode=True: 重新編碼，嚴格遵循 segment duration 和 overlap，並可設定解析度
    
    參數:
    - resolution: 長邊解析度（px），例如 720 表示長邊為 720px
    - strict_mode: 是否使用嚴格模式（重新編碼），默認為 False 使用 stream copy
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    duration = _probe_duration_seconds(input_path)
    starts = _compute_starts(duration, segment, overlap)
    ext = Path(input_path).suffix or ".mp4"
    outs = []
    
    for i, st in enumerate(starts):
        # 計算每段時長
        if strict_mode:
            # 嚴格模式：每段都是嚴格的 segment 秒（除了最後一段可能較短）
            dur = segment  # 嚴格使用 segment 秒
            # 如果超過總長度，則調整為剩餘長度
            if st + dur > duration:
                dur = max(0.0, duration - st)
        else:
            # 優化模式：允許最後一段較短
            dur = max(0.0, min(segment, duration - st))
        
        if dur <= 0.05:
            continue
        
        out_file = str(Path(out_dir) / f"{prefix}_{i:04d}{ext}")
        
        if strict_mode:
            # 嚴格模式：重新編碼以確保精確時間和解析度
            ffmpeg_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{st:.3f}",  # 精確到毫秒
                "-i", input_path,
                "-t", f"{dur:.3f}",  # 精確到毫秒
                "-c:v", "libx264",  # 使用 H.264 編碼
                "-preset", "medium",  # 編碼速度與品質平衡
                "-crf", "23",  # 品質設定（23 是較好的品質）
                "-c:a", "aac",  # 音訊編碼
                "-strict", "experimental",
            ]
            
            # 如果設定了解析度，強制縮放
            if resolution and resolution > 0:
                # 先獲取原始解析度
                probe_cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
                    input_path
                ]
                try:
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    orig_size = result.stdout.strip().split('x')
                    orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
                    
                    # 計算新尺寸（保持長寬比，長邊為 resolution）
                    if orig_w <= orig_h:
                        # 原始是直向（高度是長邊）
                        new_h = resolution
                        new_w = int(round(orig_w * (resolution / orig_h)))
                    else:
                        # 原始是橫向（寬度是長邊）
                        new_w = resolution
                        new_h = int(round(orig_h * (resolution / orig_w)))
                    
                    # 確保是偶數（H.264 要求）
                    new_w = new_w if new_w % 2 == 0 else new_w + 1
                    new_h = new_h if new_h % 2 == 0 else new_h + 1
                    
                    ffmpeg_cmd.extend(["-vf", f"scale={new_w}:{new_h}"])
                except Exception as e:
                    print(f"--- [WARNING] 無法獲取原始解析度，跳過解析度設定: {e} ---")
            
            ffmpeg_cmd.extend(["-y", out_file])
        else:
            # 優化模式：使用 -c copy（快速但不精確，100x 速度提升）
            # 注意：-ss 放在 -i 之前可以更快（輸入定位），但可能不夠精確
            # 為了速度，我們使用輸入定位
            ffmpeg_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{st:.3f}",  # 輸入定位（更快）
                "-i", input_path,
                "-t", f"{dur:.3f}",
                "-c", "copy",  # Stream copy，不重新編碼
                "-avoid_negative_ts", "make_zero",  # 避免負時間戳
                "-y", out_file
            ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        outs.append(out_file)
    
    return outs

# 從影片中根據 FPS 設定抓取截圖
def _sample_frames_evenly_to_pil(video_path: str, max_frames: int=8, sampling_fps: Optional[float] = None) -> List[Image.Image]:
    """
    從影片中抓取截圖，支援兩種模式：
    - 如果提供 sampling_fps: 嚴格按照 FPS 設定取樣（例如 0.5 fps = 每2秒取1 frame）
    - 如果未提供 sampling_fps: 使用均勻分佈（舊模式，向後兼容）
    
    參數:
    - video_path: 影片路徑
    - max_frames: 最大取樣幀數（當 sampling_fps 未設定時使用）
    - sampling_fps: 取樣 FPS（例如 0.5 表示每2秒取1 frame）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0: raise RuntimeError("invalid or zero-frame video")
        
        duration_sec = total_frames / fps
        
        frames = []
        
        if sampling_fps and sampling_fps > 0:
            # 嚴格模式：根據 FPS 設定取樣
            # 例如 0.5 fps = 每2秒取1 frame
            interval_sec = 1.0 / sampling_fps  # 取樣間隔（秒）
            
            # 計算取樣時間點
            sample_times = []
            t = 0.0
            while t < duration_sec:
                sample_times.append(t)
                t += interval_sec
            
            # 根據時間點取樣
            for sample_time in sample_times:
                frame_number = int(round(sample_time * fps))
                if frame_number >= total_frames:
                    frame_number = total_frames - 1
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, bgr = cap.read()
                if not ok: continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        else:
            # 舊模式：均勻分佈（向後兼容）
            n = min(max_frames, total_frames)
            idxs = np.linspace(0, total_frames-1, num=n, dtype=np.int64)
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, bgr = cap.read()
            if not ok: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        
        if not frames: raise RuntimeError("no frames sampled")
        return frames
    finally:
        cap.release()

# 這個函式將圖片短邊縮放到指定尺寸
def _resize_short_side(img: Image.Image, short: int) -> Image.Image:
    if not short or short <= 0:
        return img
    w, h = img.size
    s = min(w, h)
    if s == short:
        return img
    scale = short / float(s)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.BILINEAR)

# 將 Python 的 Pillow 圖片物件轉成 JPEG 格式的 Base64 字串，這是透過 API 傳送圖片的標準作法。
def _pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    """PIL 轉 JPEG base64（Ollama 圖像多採用 b64）。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality or 85), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# 檢查 Header 有沒有 X-API-Key，並檢查這個 Key 是不是跟我們伺服器設定的一樣
# 允許兩個 key：MY_API_KEY（一般使用者）和 ADMIN_TOKEN（管理者）
async def get_api_key(api_key_header: str = Security(api_key_scheme)):

    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="請提供 API Key (Could not validate credentials)")

    # 比對 Key：允許 MY_API_KEY（一般使用者）或 ADMIN_TOKEN（管理者）
    if api_key_header != SERVER_API_KEY and api_key_header != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的 API Key (Invalid Key)")

    # 驗證通過，回傳 Key 或使用者資訊供後續使用
    return api_key_header

def _now_ts() -> int:
    return int(time.time())

# ================== Ollama Chat ==================

# 重要的片段 : 負責發送請求給 Ollama
def _ollama_chat(
    model_name: str,
    messages: list[dict],
    images_b64: list[str] | None = None,
    stream: bool = False,
    timeout: int = 600,
    base: str | None = None,) -> str:

    """
    Ollama 的 API 規定，如果你要傳圖片，必須把圖片的 Base64 字串放在 user 說的話裡面 (images 欄位)。
    圖片通常是跟著「最新」的那一句話傳的。所以程式從最後面往回找第一個 role 為 user 的訊息。
    ** 參數有 stream，但程式碼行為是等待整個回應完成才回傳 (Blocking)，適合需要完整 JSON 結果的場景，不適合即時打字機效果。 **
    """
    base = base or OLLAMA_BASE
    url = f"{base.rstrip('/')}/api/chat"

    # 如果直接修改傳進來的 messages（例如把圖片塞進去），呼叫這個函式的原始變數也會被改變。所以用 copy
    msgs = copy.deepcopy(messages)

    # 若有影像，附到最後一個 user message
    if images_b64:
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["images"] = images_b64
                break
        else:
            # 如果沒有 user，就補一個
            msgs.append({"role": "user", "content": "", "images": images_b64})

    # Payload：建構標準的 Ollama API 請求格式。
    payload = {
        "model": model_name,
        "messages": msgs,
        "stream": bool(stream),
    }

    # 預設給了 600 秒（10 分鐘）。這是因為跑大型模型（例如 70B）或是在沒有 GPU 的機器上跑 AI，生成速度可能非常慢。
    # 如果使用預設的 HTTP timeout，程式很容易中斷報錯。
    r = requests.post(url, json=payload, timeout=timeout)

    # 報錯提示
    if r.status_code != 200:
        raise RuntimeError(f"Ollama chat failed [{r.status_code}]: {r.text[:500]}")

    # 抽純文字 content（支援多種常見鍵）
    j = r.json()

    # 應對「API 格式不統一」的問題所創建的 if 條件式
    if isinstance(j, dict):

        # 情況 A: 標準 Ollama 格式
        if "message" in j and isinstance(j["message"], dict):
            return j["message"].get("content", "") or ""

        # 情況 B: OpenAI 相容格式 (Ollama 也有支援)
        if "choices" in j and j["choices"]:
            ch0 = j["choices"][0]
            if isinstance(ch0, dict) and "message" in ch0:
                return (ch0["message"] or {}).get("content", "") or ""

        # 情況 C: 舊版或簡化版格式
        if "content" in j:
            return j.get("content", "") or ""

    # 回覆格式異常就回空字串，讓上游自行做 fallback
    return ""

# ================== JSON 處理相關 ==================

# 防止 json 解析失敗報錯的問題
def _safe_parse_json(text: str):
    """直接 json.loads，失敗則回 None。"""
    try:
        return json.loads(text)
    except Exception:
        return None

# 暴力提取，直接在文字堆裡抓出第一個看起來像 JSON 的物件。不管 AI 前面講了多少廢話，只抓重點。
def _extract_first_json(text: str):
    """從文字裡抓第一個 {...} 或 [...] 嘗試 parse 成 JSON。抓不到回 None。"""
    if not text:
        return None
    m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# 去 Markdown：AI 很喜歡用 json ... 把內容包起來。這個函式負責把這些外殼剝掉，只留內容。
def _clean_summary_text(s: str) -> str:
    """清掉 Markdown 圍欄、前後引號等，保留純文字摘要。"""
    s = (s or "").strip()
    s = re.sub(r"^```.*?\n|\n```$", "", s, flags=re.S).strip()
    s = re.sub(r"^[“”\"']|[“”\"']$", "", s).strip()
    return s

# 在寫 JSON 時，喜歡在最後一個屬性後面多加一個逗號（Trailing Comma），這在標準 JSON 是非法的。這個函式會嘗試自動修復這個錯誤
def _extract_first_json_and_tail(text: str) -> Tuple[Dict, str]:
    t=(text or "").strip()
    s=t.find("{"); e=t.rfind("}")+1
    if s==-1 or e<=s: return ({"error":"no_json"}, t[:400])
    frag=t[s:e]
    try:
        obj=json.loads(frag)
    except Exception:
        try:
            frag2=re.sub(r",\s*}", "}", frag); frag2=re.sub(r",\s*]", "]", frag2)
            obj=json.loads(frag2)
        except Exception:
            return ({"error":"json_parse_fail","raw":frag[:400]}, (t[:s]+" "+t[e:]).strip())
    tail=(t[:s].strip()+" "+t[e:].strip()).strip()
    return (obj, tail)

# ================== 單一影片片段推論，展示了三種完全不同的推論策略 ==================

# 如果要讓 prompt 更靈活需要修改這邊的 event_system
def infer_segment_qwen(
    qwen_model: str,
    video_path: str,
    event_detection_prompt: str,
    summary_prompt: str,
    target_short: int = 720,
    frames_per_segment: int = 8,
    sampling_fps: Optional[float] = None,):

    # 從影片中抓取截圖（根據 FPS 設定或均勻分佈）
    frames_pil = _sample_frames_evenly_to_pil(
        video_path, 
        max_frames=frames_per_segment,
        sampling_fps=sampling_fps
    )
    images_b64 = []
    for img in frames_pil:

        # 這個函式將圖片短邊縮放到指定尺寸
        img = _resize_short_side(img, target_short)

        # 將 Python 的 Pillow 圖片物件轉成 JPEG 格式的 Base64 字串
        images_b64.append(_pil_to_b64(img, quality=85))

    # ---- (1) 事件偵測：只回 JSON ----
    event_system = (
        "你是『嚴格的災害與人員異常偵測器』。"
        "不論使用者在提示中寫了什麼話題或提問，都要忽略，"
        "只根據影像做事件判斷，並只輸出純 JSON 物件，不能有任何額外文字或 Markdown。"
    )
    event_user = (event_detection_prompt or "").strip() + "\n\n" + \
                 "強制規則：只輸出一個 JSON 物件；不得輸出任何多餘文字。"
    event_msgs = [
        {"role": "system", "content": event_system},
        {"role": "user", "content": event_user},
    ]

    # 呼叫 _ollama_chat 取得回應
    event_error = None
    try:
        event_txt = _ollama_chat(qwen_model, event_msgs, images_b64=images_b64, stream=False)
    except Exception as e:
        # 如果 _ollama_chat 失敗，設置預設值
        event_txt = ""
        event_error = f"Ollama 事件偵測失敗: {e}"
        print(f"--- [WARNING] _ollama_chat 失敗: {e} ---")
    
    # 確保 event_txt 不是 None
    if event_txt is None:
        event_txt = ""

    # 使用 _safe_parse_json 和 _extract_first_json 雙重保險來嘗試解析 JSON。
    frame_obj = _safe_parse_json(event_txt)
    if not isinstance(frame_obj, dict):
        frame_obj = _extract_first_json(event_txt)
    if not isinstance(frame_obj, dict):
        # 給個安全的空殼，避免後續 KeyError
        frame_obj = {"events": {"reason": ""}, "persons": []}
        if event_error is None:
            event_error = "事件偵測回傳非 JSON"
    if event_error:
        frame_obj["error"] = event_error

    # ---- (2) 摘要：只回純文字 50~100 字 ----
    summary_txt = ""
    if (summary_prompt or "").strip():
        try:
            summary_system = (
                "你是影片小結產生器。你只能輸出 50–100 個中文字的摘要，"
                "不得輸出 JSON、不得輸出 Markdown/程式碼圍欄，不得回答其他問題。"
            )
            summary_user = (summary_prompt or "").strip() + "\n\n" + \
                           "強制規則：只輸出 50–100 字中文，不要 JSON、不要程式碼區塊、不要英文字說明。"
            summary_msgs = [
                {"role": "system", "content": summary_system},
                {"role": "user", "content": summary_user},
            ]
            summary_raw = _ollama_chat(qwen_model, summary_msgs, images_b64=images_b64, stream=False)
            if summary_raw is None:
                summary_raw = ""
            summary_txt = _clean_summary_text(summary_raw)
        except Exception as e:
            # 如果摘要生成失敗，設置為空字串
            summary_txt = ""
            print(f"--- [WARNING] 摘要生成失敗: {e} ---")

    # 回傳格式、形容的句子
    return frame_obj, summary_txt

# label 傳要偵測的目標，every_sec 是取樣頻率，score_thr 是信心門檻
def infer_segment_owl(seg_path: str, labels: str, every_sec: float, score_thr: float) -> Dict:
    with open(seg_path,"rb") as f:
        files={"file":(os.path.basename(seg_path), f, "video/mp4")}
        data={"every_sec":str(every_sec),"score_threshold":str(score_thr),"prompts":labels}
        try:
            r=requests.post(OWL_VIDEO_URL, files=files, data=data, timeout=3600)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 503:
                error_detail = r.json().get("error", "Service Unavailable") if r.text else "Service Unavailable"
                raise RuntimeError(
                    f"OWL API 模型未載入 (503): {error_detail}\n"
                    f"原因：無法從 Hugging Face 下載模型，網絡連接失敗。\n"
                    f"解決方案：\n"
                    f"  1. 檢查網絡連接：docker compose exec owl-api python -c \"import requests; requests.get('https://huggingface.co', timeout=10)\"\n"
                    f"  2. 等待網絡恢復後，模型會自動在後台下載\n"
                    f"  3. 或使用其他模型類型：將 model_type 改為 'qwen' 或 'gemini'"
                ) from e
            raise

# YOLO-World 物件偵測（本地模型）
_yolo_world_model = None
def infer_segment_yolo(seg_path: str, labels: str, every_sec: float, score_thr: float) -> Dict:
    """
    使用本地 YOLO-World 模型進行物件偵測並生成切片
    
    參數:
    - seg_path: 影片片段路徑
    - labels: 要偵測的物件類別（逗號分隔，例如 "person,pedestrian,car"）
    - every_sec: 取樣頻率（每幾秒處理一幀）
    - score_thr: 信心門檻（0.0-1.0）
    
    返回:
    - Dict: 包含偵測結果和物件切片的字典
    """
    global _yolo_world_model
    
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        raise RuntimeError("ultralytics 未安裝，請先安裝: pip install ultralytics")
    
    # 初始化模型（單例模式，避免重複載入）
    if _yolo_world_model is None:
        import os
        local_model_path = '/app/models/yolov8s-world.pt'
        if os.path.exists(local_model_path):
            print(f"--- [YOLO] 載入本地模型: {local_model_path} ---")
            _yolo_world_model = YOLOWorld(local_model_path)
        else:
            print(f"--- [YOLO] 本地模型不存在 ({local_model_path})，嘗試使用預設模型名稱 ---")
            try:
                _yolo_world_model = YOLOWorld('yolov8s-world.pt')
            except Exception as e:
                error_msg = (
                    f"無法載入 YOLO-World 模型：{str(e)}\n"
                    f"請確保模型文件存在於 {local_model_path} 或網路連接正常"
                )
                raise RuntimeError(error_msg) from e
        print("--- [YOLO] 模型載入完成 ---")
    
    # 解析標籤
    labels_list = [l.strip() for l in labels.split(",") if l.strip()]
    if not labels_list:
        labels_list = ["person", "pedestrian", "car", "motorcycle", "bus", "truck"]
    
    # 設定要偵測的類別
    try:
        _yolo_world_model.set_classes(labels_list)
        print(f"--- [YOLO] 設定偵測類別: {', '.join(labels_list)} ---")
    except Exception as e:
        error_msg = str(e)
        if "name resolution" in error_msg or "Temporary failure" in error_msg or "urlopen" in error_msg.lower():
            raise RuntimeError(
                f"無法設定偵測類別：網路連接失敗，CLIP 模型無法下載\n"
                f"錯誤詳情：{error_msg}\n"
                f"解決方案：請確保 CLIP 模型已預先下載到 ~/.cache/clip/ 目錄"
            ) from e
        raise
    
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
    
    # 準備輸出目錄
    seg_dir = Path(seg_path).parent
    output_dir = seg_dir / "yolo_output"
    output_dir.mkdir(exist_ok=True)
    
    # 物件切片目錄
    crops_dir = output_dir / "object_crops"
    crops_dir.mkdir(exist_ok=True)
    
    # 處理結果
    detections = []
    frame_count = 0
    processed_count = 0
    object_counter = {}
    crop_paths = []  # 記錄所有生成的切片路徑
    
    # 批量處理：收集所有 crop 圖像用於批量 ReID embedding（全面使用 ReID）
    reid_crops_batch = []  # 用於批量 ReID 處理的 crop 圖像
    reid_crops_metadata = []  # 對應的元數據（用於後續映射）
    
    print(f"--- [YOLO] 開始處理影片: {seg_path} ---")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 總幀數: {total_frames}")
    print(f"  - 取樣間隔: {frame_interval} 幀 (每 {every_sec} 秒)")
    print(f"  - 輸出目錄: {output_dir}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根據取樣間隔決定是否處理此幀
        if frame_count % frame_interval != 0:
            frame_count += 1
            continue
        
        timestamp = frame_count / fps
        
        # 轉換為 RGB（YOLO 需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 執行偵測
        results = _yolo_world_model.predict(pil_image, verbose=False, conf=score_thr)
        
        # 處理偵測結果
        frame_detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for idx, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float).tolist()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = labels_list[class_id] if class_id < len(labels_list) else f"class_{class_id}"
                    
                    # 確保座標在範圍內
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(width, int(x2)), min(height, int(y2))
                    
                    frame_detections.append({
                        "box": [x1, y1, x2, y2],
                        "score": confidence,
                        "label": class_name,
                        "label_idx": class_id
                    })
                    
                    # 裁剪物件並保存
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            # 生成文件名：類別_時間戳_序號.jpg
                            if class_name not in object_counter:
                                object_counter[class_name] = 0
                            object_counter[class_name] += 1
                            
                            timestamp_str = f"{timestamp:.3f}".replace(".", "_")
                            crop_filename = f"{class_name}_{timestamp_str}_{object_counter[class_name]:03d}.jpg"
                            crop_path = crops_dir / crop_filename
                            
                            # 保存物件切片圖片
                            cv2.imwrite(str(crop_path), crop)
                            
                            # 收集所有 crop 圖像到批量處理列表（全面使用 ReID）
                            reid_crops_batch.append(crop.copy())  # 複製 crop 圖像
                            reid_crops_metadata.append({
                                "index": len(crop_paths),  # 在 crop_paths 中的索引
                                "filename": crop_filename,
                                "label": class_name
                            })
                            
                            # 先添加 crop_paths 條目（ReID embedding 稍後批量填充）
                            crop_paths.append({
                                "path": str(crop_path),
                                "label": class_name,
                                "score": confidence,
                                "timestamp": timestamp,
                                "frame": frame_count,
                                "box": [x1, y1, x2, y2],
                                "clip_embedding": None,  # 不再使用 CLIP（保留字段以兼容舊代碼）
                                "reid_embedding": None  # ReID embedding（2048 維）- 稍後批量生成
                            })
        
        if frame_detections:
            detections.append({
                "timestamp": timestamp,
                "frame": frame_count,
                "detections": frame_detections
            })
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  - 進度: {frame_count}/{total_frames} 幀 ({frame_count/total_frames*100:.1f}%)")
        
        frame_count += 1
    
    cap.release()
    
    # 批量生成 ReID embedding（所有物件）
    if reid_crops_batch:
        print(f"--- [YOLO] 批量生成 ReID embedding（共 {len(reid_crops_batch)} 個物件 crops）---")
        try:
            reid_embeddings = generate_reid_embeddings_batch(reid_crops_batch)
            # 將批量生成的 ReID embedding 映射回對應的 crop_paths 條目
            success_count = 0
            for metadata, embedding in zip(reid_crops_metadata, reid_embeddings):
                if embedding is not None:
                    crop_paths[metadata["index"]]["reid_embedding"] = embedding
                    success_count += 1
                else:
                    print(f"  ⚠️  ReID embedding 生成失敗: {metadata['filename']}")
            print(f"--- [YOLO] ✓ 批量生成完成: {success_count}/{len(reid_crops_batch)} 個 ReID embedding 成功 ---")
        except Exception as e:
            print(f"--- [YOLO] ✗ 批量生成 ReID embedding 失敗: {e} ---")
            import traceback
            traceback.print_exc()
    
    print(f"--- [YOLO] 處理完成: 共處理 {processed_count} 幀，偵測到 {len(detections)} 個有物件的時間點，生成 {len(crop_paths)} 個物件切片 ---")
    
    # 返回格式
    return {
        "video_url": seg_path,
        "fps_input": fps,
        "every_sec": every_sec,
        "size": [width, height],
        "detections": detections,
        "total_frames_processed": processed_count,
        "total_detections": sum(len(d["detections"]) for d in detections),
        "crop_paths": crop_paths,  # 新增：所有物件切片路徑
        "object_count": object_counter
    }

# gemini 的輸入和 qwen 不一樣，qwen 用到 base64 字串
def infer_segment_gemini(model_name: str, seg_path: str, event_detection_prompt: str, summary_prompt: str, target_short: int=720, frames_per_segment: int=8, sampling_fps: Optional[float] = None) -> Tuple[Dict[str, Any], str]:

    # 0. 檢查 Key
    if not GEMINI_API_KEY:
        print("--- [DEBUG] 錯誤: 缺少 API Key")
        return ({"error": "missing_gemini_key"}, "請先設定 GEMINI_API_KEY 環境變數")

    try:
        # 1. Gemini 支援直接輸入多張圖片 (PIL Objects)，不像 Ollama 先轉成 Base64 字串，google-generativeai 套件會自動處理。
        print(f"--- [DEBUG] 正在處理影片: {seg_path}")
        frames = _sample_frames_evenly_to_pil(seg_path, max_frames=frames_per_segment, sampling_fps=sampling_fps)
        print(f"--- [DEBUG] 成功抽取 {len(frames)} 張影格")

        # 2. 準備 Prompt，這裡將圖片和文字混合在一起。先放圖片，後放文字指令
        prompt_content = []
        prompt_content.extend(frames)

        text_instruction = f"""
        你是一個專業的影像分析 AI。請分析附帶的連續影格。

        任務 1 (Event Detection): {event_detection_prompt}
        請確保輸出的 JSON 格式正確，欄位包含 events (Boolean) 與 persons (List)。

        任務 2 (Summary): {summary_prompt}

        請直接輸出 JSON 物件，不要使用 Markdown code block 圍繞。
        在 JSON 結束後，請換行並輸出中文摘要文字。
        """
        prompt_content.append(text_instruction)

        # 3. 設定安全過濾器，如果沒有把過濾器關掉 (BLOCK_NONE)，當畫面出現火災或受傷的人時，Gemini 預設的安全機制會觸發，導致你的監控系統失效。
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}

        # 4. 呼叫 Gemini
        print(f"--- [DEBUG] 正在呼叫模型: {model_name} (等待回應...)")
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt_content,
            generation_config=genai.types.GenerationConfig(temperature=0.2),
            safety_settings=safety_settings)

        # 5. 檢查 Prompt 是否因為太色情或暴力直接被擋下。
        print(f"--- [DEBUG]API 回應狀態 (Feedback): {response.prompt_feedback}")

        try:
            raw_text = response.text
            print("--- [DEBUG] 成功取得文字回應，前 500 字元如下:")
            print(raw_text[:500])
            print("---------------------------------------------")
        except Exception as e:

            # 如果沒有 text，代表被擋或是空回應
            print(f"--- [DEBUG] ❌ 無法取得 .text 屬性，原因: {e}")
            if response.candidates:
                print(f"--- [DEBUG] 結束原因 (Finish Reason): {response.candidates[0].finish_reason}")
                print(f"--- [DEBUG] 安全評級 (Safety Ratings): {response.candidates[0].safety_ratings}")
            return ({"error": "no_text_returned", "detail": str(e)}, "無法取得摘要")

        # 6. Markdown 清洗，Gemini 很喜歡用 ```json 包住回傳值，所以這裡會做清洗
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        return _extract_first_json_and_tail(clean_text)

    except Exception as e:
        print(f"--- [DEBUG] 發生未預期錯誤: {str(e)}")
        return ({"error": f"gemini_api_error: {str(e)}"}, "")

# ================== 確認狀態的路由 ==================

# Ping 的功能，確認 API 還有在運行
# 健康檢查和認證 API 已移至 src.api.health

# ================== 所有業務邏輯 ==================

# SegmentAnalysisRequest 定義保留在此處，供 video_analysis 模組使用
# 制定資料格式的正確標準，供 /v1/analyze_segment_result、/v1/segment_pipeline_multipart 使用
class SegmentAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # 檔案資訊
    segment_path: str
    segment_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    # 模型設定
    model_type: str # 'qwen', 'gemini', 'owl'
    qwen_model: str = "qwen2.5-vl:7b"
    frames_per_segment: int = 8
    target_short: int = 720
    sampling_fps: Optional[float] = None  # 取樣 FPS（如果提供則嚴格遵循）

    # Prompt
    event_detection_prompt: str
    summary_prompt: str

    # OWL 參數
    owl_labels: Optional[str] = None
    owl_every_sec: float = 2.0
    owl_score_thr: float = 0.15

    # YOLO 參數
    yolo_labels: Optional[str] = None
    yolo_every_sec: float = 2.0
    yolo_score_thr: float = 0.25

# 影片分析相關 API 已移至 src.api.video_analysis

# 它不親自做分析，而是負責調度資源與流程控制。影片，切割，片段影片填入標準格式，片段 API 處理，打包成大的 JSON
# 影片切割 API 已移至 src.api.video_analysis
# 保留函數定義以向後兼容（如果需要）
def _segment_video_legacy(
    request: Request,
    api_key: str = Depends(get_api_key),
    file: UploadFile = File(None),
    video_url: str = Form(None),
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    sampling_fps: float = Form(0.5),
    resolution: int = Form(720),
    output_dir: Optional[str] = Form(None),  # 可選：自訂輸出目錄
):
    """
    專門用於切割影片的 API，嚴格遵循所有參數：
    
    1. Segment Duration (s): 嚴格每 N 秒切割一段（除了最後一段可能較短）
    2. Overlap (s): 嚴格重疊設定的秒數
    3. Sampling Rate (FPS): 嚴格遵循，例如 0.5 fps = 每2秒取1 frame
    4. Resolution (px): 強制設定影片長邊解析度
    
    參數:
    - file: 上傳的影片檔案
    - video_url: 影片 URL（與 file 二選一）
    - segment_duration: 每段秒數（嚴格遵循）
    - overlap: 重疊秒數（嚴格遵循）
    - sampling_fps: 取樣 FPS（嚴格遵循，例如 0.5 = 每2秒取1 frame）
    - resolution: 長邊解析度（px，嚴格遵循）
    
    返回:
    - segments: 切割後的影片片段列表
    - segment_info: 每個片段的詳細資訊（路徑、時間範圍、取樣幀數等）
    """
    # 1. 下載或獲取影片
    if file is not None:
        target_filename = file.filename or "video.mp4"
        fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(file.filename or "video.mp4").suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(file.file.read())
        local_path, cleanup = tmp, True
    elif video_url:
        target_filename = Path(video_url).name or "video_url.mp4"
        local_path, cleanup = _download_to_temp(video_url), True
    else:
        raise HTTPException(status_code=422, detail="需要 file 或 video_url")
    
    # 2. 驗證參數
    if segment_duration <= 0:
        raise HTTPException(status_code=422, detail="segment_duration 必須大於 0")
    if overlap < 0:
        raise HTTPException(status_code=422, detail="overlap 不能為負數")
    if overlap >= segment_duration:
        raise HTTPException(status_code=422, detail="overlap 必須小於 segment_duration")
    if sampling_fps <= 0:
        raise HTTPException(status_code=422, detail="sampling_fps 必須大於 0")
    if resolution <= 0:
        raise HTTPException(status_code=422, detail="resolution 必須大於 0")
    
    try:
        # 3. 切割影片（使用嚴格模式）
        stem = Path(target_filename).stem
        if output_dir:
            # 使用自訂輸出目錄
            seg_dir = Path(output_dir)
        else:
            # 預設輸出目錄
            seg_dir = Path("segment") / f"{stem}_strict"
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"--- [INFO] 開始切割影片（嚴格模式）---")
        print(f"  - Segment Duration: {segment_duration} 秒")
        print(f"  - Overlap: {overlap} 秒")
        print(f"  - Sampling FPS: {sampling_fps} (每 {1.0/sampling_fps:.1f} 秒取1 frame)")
        print(f"  - Resolution: {resolution} px (長邊)")
        
        seg_files = _split_one_video(
            local_path, 
            seg_dir, 
            segment_duration, 
            overlap, 
            prefix="segment",
            resolution=resolution,
            strict_mode=True  # 使用嚴格模式
        )
        
        print(f"--- [INFO] 切割完成，共 {len(seg_files)} 個片段 ---")
        
        # 4. 為每個片段生成詳細資訊
        segment_info = []
        total_duration = _probe_duration_seconds(local_path)
        
        for i, seg_file in enumerate(seg_files):
            # 計算時間範圍
            start = i * (segment_duration - overlap)
            end = min(start + segment_duration, total_duration)
            
            # 驗證實際片段時長
            actual_duration = _probe_duration_seconds(seg_file)
            
            # 計算應該取樣的幀數（根據 FPS 設定）
            expected_frames = max(1, int(round(sampling_fps * actual_duration)))
            
            segment_info.append({
                "index": i,
                "path": str(seg_file),
                "relative_path": f"/segment/{Path(seg_file).parent.name}/{Path(seg_file).name}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "expected_duration": round(segment_duration, 3),
                "actual_duration": round(actual_duration, 3),
                "expected_frames": expected_frames,
                "sampling_fps": sampling_fps,
                "resolution": resolution,
            })
        
        # 5. 返回結果
        return {
            "status": "success",
            "message": f"成功切割 {len(seg_files)} 個片段",
            "segments": [str(f) for f in seg_files],
            "segment_info": segment_info,
            "total_segments": len(seg_files),
            "total_duration": round(total_duration, 3),
            "parameters": {
                "segment_duration": segment_duration,
                "overlap": overlap,
                "sampling_fps": sampling_fps,
                "resolution": resolution,
            }
        }
    
    except Exception as e:
        if cleanup and os.path.exists(local_path):
            os.remove(local_path)
        raise HTTPException(status_code=500, detail=f"切割失敗：{str(e)}")
    finally:
        if cleanup and os.path.exists(local_path):
            os.remove(local_path)

# segment_pipeline_multipart API 已移至 src.api.video_analysis
# 保留函數定義以向後兼容（如果需要）
def _segment_pipeline_multipart_legacy(
    request: Request,
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db) if HAS_DB else None,
    model_type: str = Form(...),
    file: UploadFile = File(None),
    video_url: str = Form(None),
    video_id: str = Form(None),  # 新增：重新分析已存在的影片
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    qwen_model: str = Form("qwen3-vl:8b"),
    frames_per_segment: int = Form(8),
    target_short: int = Form(720),
    sampling_fps: Optional[float] = Form(None),  # 新增：取樣 FPS（如果提供則嚴格遵循）
    strict_segmentation: bool = Form(False),  # 新增：是否使用嚴格切割模式
    owl_labels: str = Form("person,pedestrian,motorcycle,car,bus,scooter,truck"),
    owl_every_sec: float = Form(2.0),
    owl_score_thr: float = Form(0.15),
    event_detection_prompt: str = Form(EVENT_DETECTION_PROMPT),
    summary_prompt: str = Form(SUMMARY_PROMPT),
    save_json: bool = Form(True),
    save_basename: str = Form(None),
):

        target_filename = "unknown_video"

    # 1. 下載與儲存 (維持原樣)
        # 如果提供了 video_id，表示要重新分析已存在的影片，跳過下載和切割
        if video_id and video_id.strip():
            # 使用已存在的影片，不需要下載或切割
            local_path = None
            cleanup = False
        elif file is not None:
            # [修正 1] 抓取原始檔名 (例如 "my_video.mp4")
            target_filename = file.filename or "video.mp4"
            fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(file.filename or "video.mp4").suffix)
            with os.fdopen(fd, "wb") as f: f.write(file.file.read())
            local_path, cleanup = tmp, True
        elif video_url:
            # [修正 2] 如果是 URL，從網址抓檔名
            target_filename = Path(video_url).name or "video_url.mp4"
            local_path, cleanup = _download_to_temp(video_url), True
        else:
            raise HTTPException(status_code=422, detail="需要 file、video_url 或 video_id")

        # 2. 切割影片 (如果沒有使用已存在的影片)
    # [修正 3] 使用 "原始檔名" 來當作 ID，而不是用 local_path 的亂碼檔名
        if video_id and video_id.strip():
            video_id_clean = video_id.strip()
            
            # 檢查是否為 video_lib 格式 (category/video_name)
            if "/" in video_id_clean:
                # 從 video 資料夾讀取原始影片
                category, video_name = video_id_clean.split("/", 1)
                
                # 檢查是否已經在 segment 中處理過（使用 {category}_{video_name} 作為 ID）
                stem = f"{category}_{video_name}"  # 使用分類和影片名作為 ID
                seg_dir = Path("segment") / stem
                
                if seg_dir.exists() and list(seg_dir.glob("segment_*.mp4")):
                    # 已經處理過，直接使用現有的片段（不從 video 資料夾複製）
                    seg_files = sorted(seg_dir.glob("segment_*.mp4"))
                    try:
                        json_files = list(seg_dir.glob("*.json"))
                        if json_files:
                            with open(max(json_files, key=lambda p: p.stat().st_mtime), "r", encoding="utf-8") as f:
                                old_data = json.load(f)
                                total_duration = sum(r.get("duration_sec", segment_duration) for r in old_data.get("results", []))
                        else:
                            total_duration = len(seg_files) * segment_duration
                    except:
                        total_duration = len(seg_files) * segment_duration
                else:
                    # 尚未處理過，需要從 video 資料夾讀取原始影片並切割
                    video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
                    
                    # 嘗試其他擴展名
                    if not video_path.exists():
                        for ext in ['.avi', '.mov', '.mkv', '.flv']:
                            video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                            if video_path.exists():
                                break
                    
                    if not video_path.exists():
                        raise HTTPException(status_code=404, detail=f"Video {video_id_clean} not found in video library")
                    
                    # 尚未處理過，需要切割影片
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        # 複製影片到 segment 資料夾進行處理
                        import shutil
                        temp_video = seg_dir / video_path.name
                        shutil.copy2(video_path, temp_video)
                        seg_files = _split_one_video(
                            temp_video, 
                            seg_dir, 
                            segment_duration, 
                            overlap, 
                            prefix="segment",
                            resolution=target_short if strict_segmentation else None,
                            strict_mode=strict_segmentation
                        )
                        total_duration = _probe_duration_seconds(temp_video)
                        # 處理完後可以選擇刪除臨時副本（保留原始文件在 video 資料夾）
                        # os.remove(temp_video)  # 可選：刪除臨時副本
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"切割失敗：{e}")
            else:
                # 傳統的 segment 中的影片
                stem = video_id_clean
                seg_dir = Path("segment") / stem
                if not seg_dir.exists():
                    raise HTTPException(status_code=404, detail=f"Video {video_id_clean} not found")
                
                # 查找已存在的片段影片
                seg_files_existing = sorted(seg_dir.glob("segment_*.mp4"))
                if not seg_files_existing:
                    raise HTTPException(status_code=404, detail=f"No segment files found for video {video_id_clean}")
                
                seg_files = seg_files_existing
                # 估算總時長（從片段數量推斷，或從 JSON 讀取）
                try:
                    json_files = list(seg_dir.glob("*.json"))
                    if json_files:
                        with open(max(json_files, key=lambda p: p.stat().st_mtime), "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                            total_duration = sum(r.get("duration_sec", segment_duration) for r in old_data.get("results", []))
                    else:
                        total_duration = len(seg_files) * segment_duration
                except:
                    total_duration = len(seg_files) * segment_duration
        else:
            stem = Path(target_filename).stem
            # 建立固定的資料夾 segment/video_1/
            seg_dir = Path("segment") / stem
            seg_dir.mkdir(parents=True, exist_ok=True)
            try:
                seg_files = _split_one_video(
                    local_path, 
                    seg_dir, 
                    segment_duration, 
                    overlap, 
                    prefix="segment",
                    resolution=target_short if strict_segmentation else None,
                    strict_mode=strict_segmentation
                )
                total_duration = _probe_duration_seconds(local_path)
            except Exception as e:
                if cleanup and os.path.exists(local_path): os.remove(local_path)
                raise HTTPException(status_code=500, detail=f"切割失敗：{e}")

        # 3. 迴圈：Call API 取得結果
        results = []
        t0 = time.time()

        print(f"--- 開始處理 {len(seg_files)} 個片段，呼叫分析 API ---")

        for p in seg_files:
            # 3.1 計算時間區段資訊
            m = re.search(r"(\d+)", Path(p).name)
            idx = int(m.group(1)) if m else 0
            start = idx * (segment_duration - overlap)
            end = min(start + segment_duration, total_duration)

            # 3.2 準備參數 (Request Body)
            req_data = SegmentAnalysisRequest(
                segment_path=str(p), # 傳遞絕對路徑
                segment_index=idx,
                start_time=start,
                end_time=end,
                model_type=model_type,
                qwen_model=qwen_model,
                frames_per_segment=frames_per_segment,
                target_short=target_short,
            sampling_fps=sampling_fps,  # 傳遞取樣 FPS
                event_detection_prompt=event_detection_prompt,
                summary_prompt=summary_prompt,
                owl_labels=owl_labels,
                owl_every_sec=owl_every_sec,
                owl_score_thr=owl_score_thr
            )

            # 3.3 【關鍵步驟】Call API
            # 這裡直接呼叫函式，這等同於透過內部網路呼叫該 API，但更快
            from src.api.video_analysis import analyze_segment_result
            res = analyze_segment_result(req_data)
            results.append(res)

        # 4. 統計與存檔 (維持原樣)
        total_time = time.time() - t0
        ok_count = sum(1 for r in results if r.get("success"))

        resp = {
            "model_type": model_type,
            "total_segments": len(results),
            "success_segments": ok_count,
            "total_time_sec": round(total_time, 2),
            "results": results,
        }

        try:
            if save_json:
                filename = save_basename or f"{stem}.json"

                save_path = seg_dir / filename
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(resp, f, ensure_ascii=False, indent=2)
                resp["save_path"] = str(save_path)

                if AUTO_RAG_INDEX:
                    resp["rag_auto_indexed"] = _auto_index_to_rag(resp)
        except Exception: pass

        # 5. 保存分析結果到 PostgreSQL（與 RAG 同步）
        if HAS_DB and db:
            try:
                _save_results_to_postgres(db, results, stem)
            except Exception as e:
                print(f"--- [WARNING] 保存到 PostgreSQL 失敗: {e} ---")
                # 不中斷流程，只記錄警告

        if cleanup and os.path.exists(local_path):
            try: os.remove(local_path)
            except: pass

        return JSONResponse(resp, media_type="application/json; charset=utf-8")

# ================== 前端網頁取得 prompt 的來源 ==================

# 新增一個 GET 路由
# Prompt API 已移至 src.api.prompts
# 保留函數定義以向後兼容（如果需要）
def _get_default_prompts_legacy():
    """回傳後端設定的預設 Prompts（動態讀取文件，無需重啟服務）"""
    # 動態讀取 prompt 文件，而不是使用啟動時緩存的變數
    prompts_dir = Path(__file__).parent.parent / "prompts"
    
    def _read_prompt_file(filename: str) -> str:
        """讀取 prompt 文件"""
        file_path = prompts_dir / filename
        try:
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                # 清理：去掉 BOM 與首尾空白
                if content and content[0] == "\ufeff":
                    content = content[1:]
                return content.strip()
            else:
                # 如果文件不存在，回退到緩存的變數
                return EVENT_DETECTION_PROMPT if filename == "frame_prompt.md" else SUMMARY_PROMPT
        except Exception as e:
            # 讀取失敗時回退到緩存的變數
            print(f"[警告] 無法讀取 {filename}：{e}，使用緩存值")
            return EVENT_DETECTION_PROMPT if filename == "frame_prompt.md" else SUMMARY_PROMPT
    
    return {
        "event_prompt": _read_prompt_file("frame_prompt.md"),
        "summary_prompt": _read_prompt_file("summary_prompt.md")
    }

# ================== RAG 路由 ==================

# 將程式碼裡的英文代號（如 water_flood）轉成人類可讀的中文（如 淹水積水）。
def _event_cn_name(k: str) -> str:
    mapping = {
        "water_flood": "淹水積水",
        "fire": "火災濃煙",
        "abnormal_attire_face_cover_at_entry": "門禁遮臉入場",
        "person_fallen_unmoving": "人員倒地不起",
        "double_parking_lane_block": "車道併排阻塞",
        "smoking_outside_zone": "離開吸菸區吸菸",
        "crowd_loitering": "聚眾逗留",
        "security_door_tamper": "安全門破壞/撬動",
    }
    return mapping.get(k, k)

# 將 save_path（segment/upload_群聚/upload_群聚.json）拆分成 video_display、folder_rel
# video_display (segment/upload_群聚)
# folder_rel (upload_群聚)
def _derive_video_and_folder(src_resp: Any) -> Tuple[str, Optional[str]]:

    # 預設
    video_display = str(src_resp.get("video") or src_resp.get("input") or "unknown_video")
    folder_rel: Optional[str] = None

    save_path = src_resp.get("save_path")
    if isinstance(save_path, str) and save_path.strip():

        # 標準化路徑分隔符
        p = Path(save_path).as_posix()

        # [邏輯確認] 假設路徑結構是 segment/影片名/檔名.json
        parts = p.strip("/").split("/")

        if "segment" in parts:
            idx = parts.index("segment")
            # 如果結構是 .../segment/video_name/xxx.json
            if len(parts) > idx + 1:
                folder_name = parts[idx + 1]

                # [關鍵] 這是我們在 RAG 裡面的 Unique Key (Video ID)
                # 只要是同一個 folder_name，就視為同一部影片
                video_display = f"/segment/{folder_name}"
                folder_rel = folder_name

    return video_display, folder_rel

# 把分析完的複雜 JSON 結構（給程式看的），轉換成 「適合 RAG 搜尋的文件格式」
# 輸入參數 src_resp 預期是字典或列表，輸出是一個由字典組成的列表（標準 RAG 文件格式）。
def _results_to_docs(src_resp: Any) -> List[Dict[str, Any]]:

    # 建立一個空列表 docs，用來裝轉換好的文件，最後會回傳它。
    docs: List[Dict[str, Any]] = []

    # 情況 A：完整的 API 回應包 (剛跑完 API 的完整回應)
    if isinstance(src_resp, dict) and "results" in src_resp:

        # src_resp 包含 results 代表他剛跑完
        video_display, folder_rel = _derive_video_and_folder(src_resp)
        items = src_resp.get("results") or []

    # 情況 B：如果輸入直接是一個列表（List），代表它可能只是 results 部分，沒有外層包裝。 (從硬碟讀取的舊存檔)
    elif isinstance(src_resp, list):
        first = src_resp[0] if src_resp else {}
        tmp_src = {"video": (first.get("video") if isinstance(first, dict) else None)}
        video_display, folder_rel = _derive_video_and_folder(tmp_src)
        items = src_resp
    else:
        return docs

    # 遍歷每一個分析片段（Segment)，抓每個 /v1/analyze_segment_result 跑出來的標準小片段格式
    for it in items:
        seg = it.get("segment")
        time_range = it.get("time_range")
        parsed = (it.get("parsed") or {})
        frame = (parsed.get("frame_analysis") or {})
        events = (frame.get("events") or {})
        summary = (parsed.get("summary_independent") or "").strip()

        # 它檢查 events 字典，只保留值為 True 的鍵
        events_true = [k for k, v in events.items() if isinstance(v, bool) and v]
        reason = str(events.get("reason", "") or "")

        # 事件代碼轉成中文。fire -> 火災
        evt_text = "；".join([_event_cn_name(k) for k in events_true]) if events_true else "未見事件"

        # 這段是為了 RAG 搜尋優化，將結構化資料變成一段「自然語言描述」。
        content = (
            f"影片：{video_display}\n"
            f"片段：{seg}（{time_range}）\n"
            f"事件：{evt_text}\n"
            f"說明：{reason}\n"
            f"摘要：{summary}")

        # 為了避免重複索引，將影片名、資料夾、片段名、時間組合成字串，進行 SHA1 雜湊計算，取前 16 碼作為 ID。
        doc_id_base = f"{video_display}|{folder_rel or ''}|{seg}|{time_range}"
        doc_id = hashlib.sha1(doc_id_base.encode("utf-8")).hexdigest()[:16]

        # 這些欄位不會被變成向量（Embed），如果使用者搜尋「火災」，我們可以先用 metadata 過濾 events_true 包含 fire 的文件，再進行語意搜尋。
        meta = {
            "video": video_display,       # 顯示用（/segment/<folder>）
            "folder": folder_rel,         # 給 /rag/search → video_url 用（相對 MEDIA_ROOT）
            "segment": seg,
            "time_range": time_range,
            "duration_sec": it.get("duration_sec"),
            "events_true": events_true,
            "reason": reason,
            "summary": summary,
        }
        docs.append({"id": doc_id, "content": content, "metadata": meta})

    return docs

# [DEPRECATED] _remove_old_rag_records 已不再需要
# PostgreSQL 使用更新或新增邏輯（在 _save_results_to_postgres 中實現），不需要手動刪除舊記錄
def _remove_old_rag_records(target_video_id: str):
    """
    [DEPRECATED] 此函數已不再使用
    PostgreSQL 使用更新或新增邏輯，不需要手動刪除舊記錄
    """
    return 0

# 當影片分析完成後，順便自動把結果存進向量資料庫。
# 注意：數據已經通過 _save_results_to_postgres 自動保存到 PostgreSQL（包含 embedding）
# 此函數現在只返回成功狀態，實際索引已在 PostgreSQL 中完成
def _auto_index_to_rag(resp: Dict[str, Any]) -> Dict[str, Any]:

    # 先看全域變數 AUTO_RAG_INDEX 是否為 True。如果關閉就不做。
    if not AUTO_RAG_INDEX:
        return {"enabled": False, "message": "自動 RAG 索引已停用"}

    # 檢查資料庫是否可用
    if not HAS_DB:
            return {
                "success": False,
            "error": "Database not available",
            "message": "RAG 索引失敗：資料庫不可用"
        }

    try:
        # 數據已經通過 _save_results_to_postgres 自動保存到 PostgreSQL（包含 embedding）
        # 這裡只返回成功狀態
        results = resp.get("results", [])
        success_count = len([r for r in results if r.get("success", False)])
        
        # 從 PostgreSQL 查詢總數（可選，如果需要準確數字）
        total = 0
        try:
            from src.database import SessionLocal
            from src.models import Summary, ObjectCrop
            from sqlalchemy import func
            db = SessionLocal()
            try:
                total = db.query(func.count(Summary.id)).filter(
                    Summary.message.isnot(None),
                    Summary.message != "",
                    Summary.embedding.isnot(None)
                ).scalar() or 0
            finally:
                db.close()
        except Exception:
            pass  # 查詢失敗不影響返回

        return {
            "success": True,
            "removed_old": 0,  # PostgreSQL 使用更新或新增邏輯，不需要刪除
            "added_new": success_count,
            "total": total,
            "message": f"✓ RAG 更新完成（已保存 {success_count} 筆到 PostgreSQL）"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"RAG 自動索引失敗：{e}"
        }

# ================== RAG 相關 API ==================

# [DEPRECATED] RAG index API 已移至 src.api.rag，並改為使用 PostgreSQL
# 此函數已不再使用，保留僅用於向後兼容
async def _rag_index_legacy(request: Request):
    """[DEPRECATED] 此函數已不再使用，請使用 /rag/index API（已遷移到 PostgreSQL）"""
    raise HTTPException(status_code=410, detail="此 API 已棄用，請使用新的 /rag/index API（PostgreSQL）")

# ================== 影片管理 API ==================

# 影片事件標籤存儲（簡單的 JSON 文件）
VIDEO_EVENTS_FILE = Path("segment") / "_video_events.json"

def _load_video_events() -> Dict[str, Dict[str, Any]]:
    """載入影片事件標籤"""
    if VIDEO_EVENTS_FILE.exists():
        try:
            with open(VIDEO_EVENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_video_events(events: Dict[str, Dict[str, Any]]):
    """保存影片事件標籤"""
    VIDEO_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VIDEO_EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

def _get_video_lib_categories() -> Dict[str, List[str]]:
    """獲取 video 資料夾中的分類和影片列表"""
    categories = {}
    if VIDEO_LIB_DIR.exists() and VIDEO_LIB_DIR.is_dir():
        for category_dir in VIDEO_LIB_DIR.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                video_files = []
                for video_file in category_dir.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                        video_files.append(video_file.name)
                if video_files:
                    categories[category_name] = sorted(video_files)
    return categories

# 影片管理 API 已移至 src.api.video_management
# 保留函數定義以向後兼容（如果需要）
def _list_videos_legacy():
    """獲取已上傳的影片列表（統一管理 segment 和 video 兩個位置）"""
    seg_dir = Path("segment")
    videos = []
    events = _load_video_events()
    video_lib_categories = _get_video_lib_categories()
    
    # 用於追蹤已處理的 video_lib 影片，避免重複顯示
    # key: video_lib 格式的 video_id (例如 "火災生成/Video_火災2")
    # value: segment 中的實際 ID (例如 "火災生成_Video_火災2")
    processed_video_lib = {}  # 改為字典，記錄對應關係
    
    # 1. 從 segment 資料夾讀取已處理的影片
    if seg_dir.exists():
        for video_dir in seg_dir.iterdir():
            if video_dir.is_dir() and not video_dir.name.startswith("_"):
                segment_id = video_dir.name
                # 查找 JSON 文件
                json_files = list(video_dir.glob("*.json"))
                if json_files:
                    # 獲取最新的 JSON 文件
                    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                    try:
                        with open(latest_json, "r", encoding="utf-8") as f:
                            video_data = json.load(f)
                        
                        # 檢查是否為 video_lib 影片的處理結果（格式：{category}_{video_name}）
                        original_video_id = None
                        category = None
                        display_name = segment_id
                        
                        if "_" in segment_id:
                            # 嘗試還原為 video_lib 格式
                            parts = segment_id.split("_", 1)
                            if len(parts) == 2:
                                potential_category = parts[0]
                                potential_video_name = parts[1]
                                # 檢查是否存在對應的 video_lib 影片
                                if potential_category in video_lib_categories:
                                    for vf in video_lib_categories[potential_category]:
                                        if Path(vf).stem == potential_video_name:
                                            original_video_id = f"{potential_category}/{potential_video_name}"
                                            category = potential_category
                                            display_name = vf  # 使用原始檔案名作為顯示名稱
                                            # 記錄對應關係：video_lib 的 video_id -> segment 的 ID
                                            processed_video_lib[original_video_id] = segment_id
                                            break
                        
                        video_info = {
                            "video_id": original_video_id if original_video_id else segment_id,
                            "display_name": display_name,
                            "source": "segment",  # 標記來源
                            "json_path": str(latest_json.relative_to(Path("."))),
                            "total_segments": video_data.get("total_segments", 0),
                            "success_segments": video_data.get("success_segments", 0),
                            "model_type": video_data.get("model_type", "unknown"),
                            "last_modified": latest_json.stat().st_mtime,
                            "event_label": events.get(original_video_id or segment_id, {}).get("event_label") or (category if category else None),
                            "event_description": events.get(original_video_id or segment_id, {}).get("event_description", ""),
                            "category": category,  # 如果有對應的分類
                            "segment_id": segment_id,  # 保留 segment 中的實際 ID，用於重新分析
                        }
                        videos.append(video_info)
                    except Exception as e:
                        print(f"Warning: Failed to load video info for {segment_id}: {e}")
    
    # 2. 從 video 資料夾讀取歷史影片（按分類），但跳過已經在 segment 中處理過的
    for category_name, video_files in video_lib_categories.items():
        for video_file in video_files:
            video_id = f"{category_name}/{Path(video_file).stem}"
            
            # 如果這個影片已經在 segment 中處理過，跳過（避免重複顯示）
            # processed_video_lib 的 key 是 video_lib 格式的 video_id
            if video_id in processed_video_lib:
                continue
            
            video_path = VIDEO_LIB_DIR / category_name / video_file
            
            # 檢查是否有對應的事件標籤
            event_info = events.get(video_id, {})
            if not event_info.get("event_label"):
                # 如果沒有標籤，使用分類名稱作為預設標籤
                event_info = {"event_label": category_name, "event_description": ""}
            
            video_info = {
                "video_id": video_id,
                "display_name": video_file,
                "source": "video_lib",  # 標記來源
                "json_path": None,  # video_lib 中的影片可能沒有分析結果
                "total_segments": 0,
                "success_segments": 0,
                "model_type": "unknown",
                "last_modified": video_path.stat().st_mtime if video_path.exists() else 0,
                "event_label": event_info.get("event_label", category_name),
                "event_description": event_info.get("event_description", ""),
                "category": category_name,  # 分類名稱
                "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)) if video_path.exists() else None,
            }
            videos.append(video_info)
    
    # 按最後修改時間排序（最新的在前）
    videos.sort(key=lambda x: x["last_modified"], reverse=True)
    
    return {
        "videos": videos,
        "total": len(videos),
        "categories": list(video_lib_categories.keys())  # 返回所有分類
    }

# 影片詳情 API 已移至 src.api.video_management
# 保留函數定義以向後兼容（如果需要）
def _get_video_info_legacy(video_id: str):
    """獲取特定影片的詳細信息（支持 segment 和 video_lib 兩個來源）"""
    # 檢查是否為 video_lib 格式 (category/video_name)
    if "/" in video_id:
        category, video_name = video_id.split("/", 1)
        video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
        # 嘗試其他擴展名
        if not video_path.exists():
            for ext in ['.avi', '.mov', '.mkv', '.flv']:
                video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                if video_path.exists():
                    break
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found in video library")
        
        # 檢查是否已經在 segment 中處理過
        stem = f"{category}_{video_name}"
        seg_dir = Path("segment") / stem
        
        events = _load_video_events()
        event_info = events.get(video_id, {})
        
        # 如果有分析結果，返回分析數據
        json_files = list(seg_dir.glob("*.json")) if seg_dir.exists() else []
        if json_files:
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                return {
                    "video_id": video_id,
                    "display_name": video_path.name,
                    "source": "video_lib",
                    "json_path": str(latest_json.relative_to(Path("."))),
                    "analysis_data": video_data,
                    "event_label": event_info.get("event_label", category),
                    "event_description": event_info.get("event_description", ""),
                    "event_set_by": event_info.get("set_by", ""),
                    "event_set_at": event_info.get("set_at", ""),
                    "category": category,
                    "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load video data: {e}")
        else:
            # 沒有分析結果，只返回基本信息
            return {
                "video_id": video_id,
                "display_name": video_path.name,
                "source": "video_lib",
                "json_path": None,
                "analysis_data": None,
                "event_label": event_info.get("event_label", category),
                "event_description": event_info.get("event_description", ""),
                "event_set_by": event_info.get("set_by", ""),
                "event_set_at": event_info.get("set_at", ""),
                "category": category,
                "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)),
            }
    else:
        # segment 中的影片
        seg_dir = Path("segment") / video_id
        if not seg_dir.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        # 查找 JSON 文件
        json_files = list(seg_dir.glob("*.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail=f"No analysis result found for {video_id}")
        
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_json, "r", encoding="utf-8") as f:
                video_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load video data: {e}")
        
        events = _load_video_events()
        event_info = events.get(video_id, {})
        
        return {
            "video_id": video_id,
            "display_name": video_id,
            "source": "segment",
            "json_path": str(latest_json.relative_to(Path("."))),
            "analysis_data": video_data,
            "event_label": event_info.get("event_label"),
            "event_description": event_info.get("event_description", ""),
            "event_set_by": event_info.get("set_by", ""),
            "event_set_at": event_info.get("set_at", ""),
            "category": None,
    }

# [DEPRECATED] RAG stats API 已移至 src.api.rag，並改為使用 PostgreSQL
# 此函數已不再使用，保留僅用於向後兼容
def _rag_stats_legacy():
    """[DEPRECATED] 此函數已不再使用，請使用 /rag/stats API（已遷移到 PostgreSQL）"""
    return {
        "count": 0,
        "path": "PostgreSQL (此 API 已棄用)"
    }

# ================== 影片管理 API ==================

# 影片事件標籤存儲（簡單的 JSON 文件）
VIDEO_EVENTS_FILE = Path("segment") / "_video_events.json"

def _load_video_events() -> Dict[str, Dict[str, Any]]:
    """載入影片事件標籤"""
    if VIDEO_EVENTS_FILE.exists():
        try:
            with open(VIDEO_EVENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# 影片事件 API 已移至 src.api.video_management
# 保留函數定義以向後兼容（如果需要）
async def _set_video_event_legacy(video_id: str, request: Request):
    """設置影片的事件標籤（管理者功能）"""
    # 驗證影片是否存在（支持 segment 和 video_lib 格式）
    video_exists = False
    
    # 檢查是否為 video_lib 格式 (category/video_name)
    if "/" in video_id:
        category, video_name = video_id.split("/", 1)
        video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
        # 嘗試其他擴展名
        if not video_path.exists():
            for ext in ['.avi', '.mov', '.mkv', '.flv']:
                video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                if video_path.exists():
                    break
        video_exists = video_path.exists()
    else:
        # segment 中的影片
        seg_dir = Path("segment") / video_id
        video_exists = seg_dir.exists()
    
    if not video_exists:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    payload = await request.json()
    event_label = payload.get("event_label", "").strip()
    event_description = payload.get("event_description", "").strip()
    
    if not event_label:
        raise HTTPException(status_code=422, detail="event_label is required")
    
    events = _load_video_events()
    events[video_id] = {
        "event_label": event_label,
        "event_description": event_description,
        "set_by": "admin",  # 可以從 API key 或 session 獲取
        "set_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    _save_video_events(events)
    
    return {
        "success": True,
        "video_id": video_id,
        "event_label": event_label,
        "event_description": event_description,
        "message": f"影片 {video_id} 已標記為「{event_label}」"
    }

# 移除影片事件 API 已移至 src.api.video_management
# 保留函數定義以向後兼容（如果需要）
def _remove_video_event_legacy(video_id: str):
    """移除影片的事件標籤"""
    events = _load_video_events()
    if video_id in events:
        del events[video_id]
        _save_video_events(events)
        return {"success": True, "message": f"已移除影片 {video_id} 的事件標籤"}
    return {"success": False, "message": f"影片 {video_id} 沒有事件標籤"}

# 影片分類 API 已移至 src.api.video_management
# 保留函數定義以向後兼容（如果需要）
def _get_video_categories_legacy():
    """獲取 video 資料夾中的所有分類"""
    categories = _get_video_lib_categories()
    return {
        "categories": list(categories.keys()),
        "category_details": {cat: len(videos) for cat, videos in categories.items()}
    }

# 移動影片 API 已移至 src.api.video_management
# 保留函數定義以向後兼容（如果需要）
async def _move_video_to_category_legacy(video_id: str, request: Request):
    """將影片移動到 video 資料夾的指定分類（管理者功能）"""
    # 檢查是否為管理者
    api_key = request.headers.get("X-API-Key", "")
    if api_key != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="此功能僅限管理者使用")
    
    payload = await request.json()
    category = payload.get("category", "").strip()
    event_description = payload.get("event_description", "").strip()
    
    if not category:
        raise HTTPException(status_code=422, detail="category is required")
    
    # 檢查影片是否存在（只支持 segment 中的影片進行移動）
    # 注意：video_lib 中的影片已經在分類資料夾中，不需要移動
    seg_dir = Path("segment") / video_id
    if not seg_dir.exists():
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in segment. Only videos in segment can be moved to categories.")
    
    # 創建目標分類資料夾
    target_category_dir = VIDEO_LIB_DIR / category
    target_category_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找原始影片文件（可能在 segment 目錄中，或需要從片段重建）
    original_video = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        potential_file = seg_dir / f"{video_id}{ext}"
        if potential_file.exists():
            original_video = potential_file
            break
    
    # 如果沒有找到原始影片，嘗試從第一個片段推斷
    if not original_video:
        seg_files = sorted(seg_dir.glob("segment_*.mp4"))
        if seg_files:
            # 使用第一個片段作為參考（實際應該合併所有片段，這裡簡化處理）
            original_video = seg_files[0]
    
    if not original_video:
        raise HTTPException(status_code=404, detail=f"找不到影片文件：{video_id}")
    
    # 複製影片到目標分類資料夾
    target_video_path = target_category_dir / original_video.name
    import shutil
    shutil.copy2(original_video, target_video_path)
    
    # 更新事件標籤
    events = _load_video_events()
    new_video_id = f"{category}/{Path(original_video).stem}"
    events[new_video_id] = {
        "event_label": category,
        "event_description": event_description,
        "set_by": "admin",
        "set_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "moved_from": video_id
    }
    _save_video_events(events)
    
    return {
        "success": True,
        "message": f"影片已移動到分類「{category}」",
        "new_video_id": new_video_id,
        "target_path": str(target_video_path)
    }

# 幫你找影片片段，但不負責解釋內容
# RAG search API 已移至 src.api.rag
# 保留函數定義以向後兼容（如果需要）
async def _rag_search_legacy(request: Request, db: Session = Depends(get_db) if HAS_DB else None):
    """
    使用 PostgreSQL + pgvector 進行混合搜索
    - Filter 1 (Hard Filter): 時間範圍、事件類型、關鍵字過濾
    - Filter 2 (Vector Search): 使用 embedding 的 cosine_distance 進行語義搜索
    """
    try:
        payload = await request.json()

        # query: 你要找什麼？
        query = (payload.get("query") or "").strip()
        top_k = int(payload.get("top_k") or 5)

        # [NEW] 新增分數門檻參數 (預設 0.0 代表不過濾，前端可傳 0.6 代表 60%)
        score_threshold = float(payload.get("score_threshold") or 0.0)

        if not query:
            raise HTTPException(status_code=422, detail="missing query")

        if not HAS_DB or not db:
            raise HTTPException(status_code=503, detail="Database not available")

        # 步驟 1: 解析查詢條件
        query_filters = {}
        date_info = None
        
        try:
            query_filters = _parse_query_filters(query)
            
            # 提取日期解析資訊，用於返回給前端
            if query_filters.get("date_filter") or query_filters.get("time_start"):
                date_filter = query_filters.get("date_filter")
                if date_filter:
                    if hasattr(date_filter, 'isoformat'):
                        picked_date_str = date_filter.isoformat()
                    else:
                        picked_date_str = str(date_filter)
                else:
                    picked_date_str = None
                
                date_info = {
                    "mode": query_filters.get("date_mode", "NONE"),
                    "picked_date": picked_date_str,
                    "time_start": query_filters.get("time_start"),
                    "time_end": query_filters.get("time_end"),
                }
                print(f"\n{'='*60}")
                print(f"[日期解析] 查詢: {query}")
                print(f"[日期解析] 模式: {date_info['mode']}")
                print(f"[日期解析] 解析到的日期: {date_info['picked_date']}")
                if date_info['time_start']:
                    print(f"[日期解析] 時間範圍: {date_info['time_start'][:19]} ~ {date_info['time_end'][:19]}")
                print(f"{'='*60}\n")
        except Exception as e:
            print(f"--- [WARNING] 查詢解析失敗: {e} ---")
            import traceback
            traceback.print_exc()
            query_filters = {}

        # 步驟 2: 生成查詢向量（去除日期後的 clean query）
        # 從查詢中移除日期相關文字，保留語義查詢部分
        clean_query = query
        # 簡單的日期文字移除（可以根據需要改進）
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}',
            r'今天|昨天|明天|本週|上週|下週',
            r'\d{4}年\d{1,2}月\d{1,2}日',
        ]
        for pattern in date_patterns:
            clean_query = re.sub(pattern, '', clean_query)
        clean_query = clean_query.strip()
        
        # 如果 clean_query 為空，使用原始查詢
        if not clean_query:
            clean_query = query

        # 生成 embedding
        embedding_model = get_embedding_model()
        if not embedding_model:
            raise HTTPException(status_code=503, detail="Embedding model not available")
        
        query_embedding = embedding_model.encode(clean_query, normalize_embeddings=True)
        print(f"--- [DEBUG] 查詢向量生成完成 (維度: {len(query_embedding)}) ---")

        # 步驟 3: 構建 PostgreSQL + pgvector 混合查詢
        # Filter 1: Hard filters (時間、事件、關鍵字)
        stmt = select(Summary).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)  # 只查詢有 embedding 的記錄
        )

        # 時間範圍過濾（最高優先級）
        if query_filters.get("time_start") and query_filters.get("time_end"):
            try:
                time_start_str = query_filters["time_start"]
                time_end_str = query_filters["time_end"]
                
                if "T" in time_start_str:
                    t0 = datetime.fromisoformat(time_start_str.replace("Z", "+00:00"))
                else:
                    t0 = datetime.fromisoformat(time_start_str)
                
                if "T" in time_end_str:
                    t1 = datetime.fromisoformat(time_end_str.replace("Z", "+00:00"))
                else:
                    t1 = datetime.fromisoformat(time_end_str)
                
                if t0.tzinfo:
                    t0 = t0.astimezone().replace(tzinfo=None)
                if t1.tzinfo:
                    t1 = t1.astimezone().replace(tzinfo=None)
                
                stmt = stmt.filter(
                    Summary.start_timestamp >= t0,
                    Summary.start_timestamp < t1
                )
                print(f"--- [DEBUG] 應用時間範圍過濾: {t0} ~ {t1} ---")
            except Exception as e:
                print(f"--- [WARNING] 時間範圍解析失敗: {e} ---")
        elif query_filters.get("date_filter"):
            target_date = query_filters["date_filter"]
            t0 = datetime.combine(target_date, datetime.min.time())
            next_day = target_date + timedelta(days=1)
            t1 = datetime.combine(next_day, datetime.min.time())
            stmt = stmt.filter(
                Summary.start_timestamp >= t0,
                Summary.start_timestamp < t1
            )
            print(f"--- [DEBUG] 應用日期過濾: {target_date} ({t0} ~ {t1}) ---")

        # 事件類型過濾
        if query_filters.get("event_types"):
            event_conditions = []
            for event_type in query_filters["event_types"]:
                if event_type == "fire":
                    event_conditions.append(Summary.fire == True)
                elif event_type == "water_flood":
                    event_conditions.append(Summary.water_flood == True)
                elif event_type == "abnormal_attire_face_cover_at_entry":
                    event_conditions.append(Summary.abnormal_attire_face_cover_at_entry == True)
                elif event_type == "person_fallen_unmoving":
                    event_conditions.append(Summary.person_fallen_unmoving == True)
                elif event_type == "double_parking_lane_block":
                    event_conditions.append(Summary.double_parking_lane_block == True)
                elif event_type == "smoking_outside_zone":
                    event_conditions.append(Summary.smoking_outside_zone == True)
                elif event_type == "crowd_loitering":
                    event_conditions.append(Summary.crowd_loitering == True)
                elif event_type == "security_door_tamper":
                    event_conditions.append(Summary.security_door_tamper == True)
            
            if event_conditions:
                stmt = stmt.filter(or_(*event_conditions))
                print(f"--- [DEBUG] 應用事件過濾: {query_filters['event_types']} ---")

        # 關鍵字過濾
        if query_filters.get("message_keywords"):
            message_conditions = []
            for keyword in query_filters["message_keywords"]:
                message_conditions.append(Summary.message.ilike(f"%{keyword}%"))
            if message_conditions:
                stmt = stmt.filter(or_(*message_conditions))
                print(f"--- [DEBUG] 應用關鍵字過濾: {query_filters['message_keywords']} ---")

        # Filter 2: Vector search - 使用 cosine_distance 排序
        try:
            from pgvector.sqlalchemy import Vector
            # 計算 cosine_distance 並作為分數欄位
            # cosine_distance 範圍是 [0, 2]，我們需要轉換為相似度分數 [0, 1]
            # cosine_similarity = 1 - cosine_distance
            # 但為了讓分數在 [0, 1] 範圍內，我們使用: score = 1 - (distance / 2)
            distance_expr = Summary.embedding.cosine_distance(query_embedding)
            
            # 選擇需要的欄位，並計算距離
            stmt = stmt.add_columns(
                distance_expr.label('cosine_distance')
            ).order_by(
                distance_expr
            ).limit(top_k * 3)  # 多取一些，後續可以根據分數過濾
            
            print(f"--- [DEBUG] 執行 PostgreSQL + pgvector 混合查詢 ---")
            results = db.execute(stmt).all()
            print(f"--- [DEBUG] 查詢返回 {len(results)} 筆結果 ---")
        except ImportError:
            raise HTTPException(status_code=503, detail="pgvector not available")
        except Exception as e:
            print(f"--- [ERROR] pgvector 查詢失敗: {e} ---")
            import traceback
            traceback.print_exc()
            results = []

        # 步驟 4: 計算相似度分數並格式化結果
        norm_hits = []
        for row in results:
            result = row[0]  # Summary 對象
            cosine_distance = row[1]  # cosine_distance 值
            
            # 將 cosine_distance 轉換為相似度分數
            # cosine_distance 範圍: [0, 2]
            # cosine_similarity 範圍: [-1, 1]
            # 我們使用: score = 1 - (distance / 2) 來將距離轉換為 [0, 1] 範圍的分數
            score = max(0.0, min(1.0, 1.0 - (cosine_distance / 2.0)))

            # 過濾低於門檻的結果
            if score < score_threshold:
                continue

            # 過濾低於門檻的結果
            if score < score_threshold:
                continue

            # 構建事件列表
            events_true = []
            if result.fire:
                events_true.append("fire")
            if result.water_flood:
                events_true.append("water_flood")
            if result.abnormal_attire_face_cover_at_entry:
                events_true.append("abnormal_attire_face_cover_at_entry")
            if result.person_fallen_unmoving:
                events_true.append("person_fallen_unmoving")
            if result.double_parking_lane_block:
                events_true.append("double_parking_lane_block")
            if result.smoking_outside_zone:
                events_true.append("smoking_outside_zone")
            if result.crowd_loitering:
                events_true.append("crowd_loitering")
            if result.security_door_tamper:
                events_true.append("security_door_tamper")

            norm_hits.append({
                "score": round(score, 4),
                "video": result.video or "",
                "segment": result.segment or "",
                "time_range": result.time_range or "",
                "events_true": events_true,
                "summary": result.message or "",
                "reason": result.event_reason or "",
                "doc_id": None,
            })

        # 取 top_k
        norm_hits = norm_hits[:top_k]
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")

        # 構建響應
        response = {"backend": EMBEDDING_MODEL_NAME, "hits": norm_hits}
        if date_info:
            response["date_parsed"] = date_info
        # [NEW] 添加關鍵字資訊
        if query_filters.get("message_keywords"):
            response["keywords_found"] = query_filters["message_keywords"]
        if query_filters.get("event_types"):
            response["event_types_found"] = query_filters["event_types"]
        # [NEW] 添加 embedding 查詢資訊（用於向量搜索的 clean query）
        response["embedding_query"] = clean_query
        print(f"--- [DEBUG] Embedding 查詢文本: '{clean_query}' ---")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"--- [ERROR] 搜索失敗: {e} ---")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        # [簡化] 如果有日期過濾，PostgreSQL 先過濾日期，然後 RAG 在這些結果中進行向量搜索
        if has_date_filter:
            if filtered_set and len(filtered_set) > 0:
                # PostgreSQL 已過濾日期，RAG 在這些結果中進行向量搜索
                print(f"--- [DEBUG] PostgreSQL 日期過濾找到 {len(filtered_set)} 筆記錄，開始 RAG 向量搜索 ---")
                
                # RAG 正常搜索（不限制數量，因為後面會過濾）
                search_top_k = top_k * 50  # 增加搜尋數量以確保能找到所有匹配的記錄
                raw_hits = store.search(query, top_k=search_top_k, filters=filters)
                
                # 只保留在 PostgreSQL 過濾結果中的記錄
                filtered_hits = []
                # [新增] 構建事件相關關鍵字列表（用於判斷是否相關）
                event_keyword_map = {
                    "fire": ["火災", "火", "fire", "smoke", "濃煙", "煙"],
                    "water_flood": ["水災", "水", "淹水", "積水", "flood", "water"],
                    "person_fallen_unmoving": ["倒地", "倒地不起", "fallen", "unmoving"],
                    "crowd_loitering": ["群聚", "聚眾", "逗留", "crowd", "loitering"],
                    "abnormal_attire_face_cover_at_entry": ["遮臉", "異常著裝", "face", "cover"],
                    "double_parking_lane_block": ["併排", "停車", "阻塞", "parking", "block"],
                    "smoking_outside_zone": ["吸菸", "抽菸", "smoking"],
                    "security_door_tamper": ["闖入", "突破", "安全門", "tamper", "door"],
                }
                # 收集所有需要匹配的關鍵字
                required_keywords = set()
                if query_filters.get("message_keywords"):
                    required_keywords.update(query_filters["message_keywords"])
                if query_filters.get("event_types"):
                    for event_type in query_filters["event_types"]:
                        if event_type in event_keyword_map:
                            required_keywords.update(event_keyword_map[event_type])
                
                if required_keywords:
                    print(f"--- [DEBUG] 需要匹配的關鍵字: {required_keywords} ---")
                
                for h in raw_hits:
                    m = h.get("metadata", {})
                    seg = m.get("segment")
                    tr = m.get("time_range")
                    if seg and tr:
                        # 寬鬆匹配（處理格式差異）
                        seg_base = seg.rsplit('.', 1)[0] if '.' in seg else seg
                        tr_normalized = tr.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
                        # [修改] 檢查是否在 PostgreSQL 過濾結果中（使用 (video, segment, time_range) 三元組）
                        # 從 RAG metadata 中提取 video 名稱
                        video_from_rag = m.get("video", "").replace("/segment/", "").strip("/") if m.get("video") else None
                        folder_from_rag = m.get("folder", "")
                        video_name = folder_from_rag if folder_from_rag else video_from_rag
                        
                        # 匹配邏輯：檢查 (video, segment, time_range) 是否在過濾集合中
                        matched = False
                        for db_video, db_seg, db_tr in filtered_set:
                            # 檢查 segment 和 time_range 是否匹配（寬鬆匹配）
                            seg_match = (seg == db_seg) or (seg_base == db_seg) or (seg == (db_seg.rsplit('.', 1)[0] if '.' in db_seg else db_seg))
                            tr_match = (tr == db_tr) or (tr_normalized == db_tr) or (tr == db_tr.replace(" - ", "-")) or (tr_normalized == db_tr.replace(" - ", "-"))
                            
                            if seg_match and tr_match:
                                # 如果 video 匹配，或者其中一個為 None（舊記錄），則匹配
                                if (video_name and db_video and video_name == db_video) or \
                                   (not video_name and not db_video) or \
                                   (video_name and not db_video) or \
                                   (not video_name and db_video):
                                    matched = True
                                    break
                        
                        if matched:
                            # [修改] 只有當查詢中沒有事件關鍵字時，才給予高分數（1.0）
                            # 如果有事件關鍵字，則檢查是否有事件標記或摘要中包含關鍵字
                            original_score = h.get("score", 0.0)
                            
                            if not required_keywords:
                                # 沒有事件關鍵字：給予高分數（因為已經通過日期過濾）
                                h["score"] = 1.0
                            else:
                                # 有事件關鍵字：檢查是否相關
                                events_true = m.get("events_true", [])
                                summary = str(m.get("summary", "")).lower()
                                summary_original = str(m.get("summary", ""))
                                
                                has_event_match = False
                                has_keyword_match = False
                                
                                # 檢查是否有匹配的事件標記
                                if query_filters.get("event_types"):
                                    for event_type in query_filters["event_types"]:
                                        if event_type in events_true:
                                            has_event_match = True
                                            break
                                
                                # 檢查摘要中是否包含關鍵字
                                if not has_event_match:
                                    for keyword in required_keywords:
                                        keyword_lower = keyword.lower()
                                        if keyword in summary_original or keyword_lower in summary:
                                                    has_keyword_match = True
                                                    break
                                
                                # 只有當有事件標記或摘要中包含關鍵字時，才給予高分數
                                if has_event_match or has_keyword_match:
                                    h["score"] = 1.0
                                    print(f"--- [DEBUG] 給予高分數: segment={seg}, 事件匹配={has_event_match}, 關鍵字匹配={has_keyword_match}, events_true={events_true} ---")
                                else:
                                    # 保持原始 RAG 分數（不修改）
                                    print(f"--- [DEBUG] 保持原始分數 {original_score:.4f}: segment={seg}, 無事件標記且摘要中無關鍵字, events_true={events_true}, summary前100字={summary_original[:100]}... ---")
                            
                            filtered_hits.append(h)
                
                raw_hits = filtered_hits
                print(f"--- [DEBUG] RAG 向量搜索後，匹配 PostgreSQL 日期過濾的結果: {len(raw_hits)} 筆 ---")
            else:
                # PostgreSQL 日期過濾沒有找到匹配結果：返回空結果
                print(f"--- [DEBUG] PostgreSQL 日期過濾沒有找到匹配結果，返回空結果 ---")
                raw_hits = []
        else:
            # 沒有日期過濾：正常 RAG 搜尋（完全由 RAG 處理）
            # [修改] 當沒有日期過濾時，不限制搜索數量，讓 RAG 進行全面搜索
            # 事件過濾會在搜索後進行，而不是在搜索時使用 events_true_any 過濾器
            # 這樣可以確保即使沒有事件標記，但摘要中包含關鍵字的結果也能被找到
            search_top_k = top_k * 50 if query_filters.get("event_types") or query_filters.get("message_keywords") else top_k
            # [修改] 不使用 events_true_any 過濾器，讓 RAG 進行全面搜索，然後在結果中進行事件過濾
            raw_hits = store.search(query, top_k=search_top_k, filters={})

        # [新增] 如果查詢中有事件關鍵字，進行 RAG 層面的事件過濾（無論是否有日期過濾）
        if query_filters.get("event_types") or query_filters.get("message_keywords"):
            strict_filtered_hits = []
            required_event_types = query_filters.get("event_types", [])
            required_message_keywords = query_filters.get("message_keywords", [])
            # [新增] 保存 has_date_filter 變數，供事件過濾邏輯使用
            # 直接使用 has_date_filter（已在函數開始時定義）
            try:
                has_date_filter_for_event = has_date_filter
            except NameError:
                has_date_filter_for_event = False
            
            # 構建事件關鍵字映射（用於檢查摘要）
            event_keyword_map = {
                "fire": ["火災", "火", "fire", "smoke", "濃煙", "煙"],
                "water_flood": ["水災", "水", "淹水", "積水", "flood", "water"],
                "person_fallen_unmoving": ["倒地", "倒地不起", "fallen", "unmoving"],
                "crowd_loitering": ["群聚", "聚眾", "逗留", "crowd", "loitering"],
                "abnormal_attire_face_cover_at_entry": ["遮臉", "異常著裝", "face", "cover"],
                "double_parking_lane_block": ["併排", "停車", "阻塞", "parking", "block"],
                "smoking_outside_zone": ["吸菸", "抽菸", "smoking"],
                "security_door_tamper": ["闖入", "突破", "安全門", "tamper", "door"],
            }
            
            # 收集所有需要匹配的關鍵字
            all_keywords = set(required_message_keywords)
            for event_type in required_event_types:
                if event_type in event_keyword_map:
                    all_keywords.update(event_keyword_map[event_type])
            
            print(f"--- [DEBUG] RAG 事件過濾: 需要的事件類型={required_event_types}, 關鍵字={all_keywords} ---")
            
            for h in raw_hits:
                m = h.get("metadata", {})
                events_true = m.get("events_true", [])
                summary = str(m.get("summary", "")).lower()
                summary_original = str(m.get("summary", ""))
                
                # 檢查是否有匹配的事件標記
                has_event_match = False
                if required_event_types:
                    for event_type in required_event_types:
                        if event_type in events_true:
                            has_event_match = True
                            break
                
                # 檢查摘要中是否包含關鍵字（更嚴格的匹配）
                has_keyword_match = False
                if all_keywords:
                    for keyword in all_keywords:
                        keyword_lower = keyword.lower()
                        # [修改] 更嚴格的關鍵字匹配：檢查關鍵字是否在摘要中出現
                        # 對於中文關鍵字（如「火災」），直接檢查是否包含
                        # 對於英文關鍵字（如「fire」），使用不區分大小寫的匹配
                        if keyword in summary_original or keyword_lower in summary:
                            # [新增] 額外檢查：確保不是誤匹配
                            # 例如：避免「無火災」被匹配為「火災」
                            # 檢查關鍵字前後是否有否定詞
                            keyword_pos = summary_original.lower().find(keyword_lower)
                            if keyword_pos >= 0:
                                # 檢查關鍵字前後的上下文
                                context_before = summary_original[max(0, keyword_pos-5):keyword_pos].lower()
                                context_after = summary_original[keyword_pos+len(keyword):min(len(summary_original), keyword_pos+len(keyword)+5)].lower()
                                
                                # 如果關鍵字前有否定詞，則不匹配
                                negation_words = ['無', '沒有', '不', '非', '未', 'no', 'not', 'none', 'without']
                                has_negation = any(neg in context_before for neg in negation_words)
                                
                                if not has_negation:
                                    has_keyword_match = True
                                    print(f"--- [DEBUG] 關鍵字匹配: segment={m.get('segment')}, keyword={keyword}, summary片段={summary_original[:100]}... ---")
                                    break
                
                # 如果有事件類型或關鍵字要求，則必須至少匹配其中一個
                # [修改] 如果沒有找到匹配的結果，但 RAG 搜索返回了結果，可能是因為：
                # 1. 摘要中沒有明確寫出關鍵字，但語義相關
                # 2. 向量搜索找到了相關結果，但摘要描述方式不同
                # 在這種情況下，如果 RAG 分數較高（>0.5），且查詢中只有事件關鍵字（沒有日期），
                # 可以考慮放寬過濾條件，允許高分數的結果通過
                if has_event_match or has_keyword_match:
                    strict_filtered_hits.append(h)
                    print(f"--- [DEBUG] 保留符合事件要求的結果: segment={m.get('segment')}, 事件匹配={has_event_match}, 關鍵字匹配={has_keyword_match}, events_true={events_true} ---")
                else:
                    # [新增] 如果沒有日期過濾，且 RAG 分數較高，可以考慮保留（但優先級較低）
                    # 這適用於向量搜索找到了語義相關但摘要中沒有明確關鍵字的情況
                    score = h.get("score", 0.0)
                    # [修改] 檢查是否有日期過濾
                    if not has_date_filter_for_event and score > 0.5:
                        # 檢查摘要中是否有相關的描述（即使沒有明確的關鍵字）
                        # 例如：對於「倒地」，檢查是否有「躺」、「臥」、「不動」等相關詞
                        related_keywords_map = {
                            "person_fallen_unmoving": ["躺", "臥", "不動", "靜止", "趴", "倒", "fall", "lie", "still"],
                            "fire": ["燃燒", "燒", "煙", "火光", "burn", "flame"],
                            "water_flood": ["濕", "水", "積", "淹", "wet", "water"],
                        }
                        
                        has_related_keyword = False
                        for event_type in required_event_types:
                            if event_type in related_keywords_map:
                                for related_kw in related_keywords_map[event_type]:
                                    if related_kw in summary_original.lower() or related_kw in summary:
                                        has_related_keyword = True
                                        print(f"--- [DEBUG] 找到相關關鍵字: segment={m.get('segment')}, related_keyword={related_kw}, score={score:.2f} ---")
                                        break
                                if has_related_keyword:
                                    break
                        
                        if has_related_keyword:
                            strict_filtered_hits.append(h)
                            print(f"--- [DEBUG] 保留高分數且包含相關關鍵字的結果: segment={m.get('segment')}, score={score:.2f} ---")
                        else:
                            print(f"--- [DEBUG] 過濾掉不符合事件要求的結果: segment={m.get('segment')}, events_true={events_true}, score={score:.2f}, summary前100字={summary_original[:100]}... ---")
                    else:
                        print(f"--- [DEBUG] 過濾掉不符合事件要求的結果: segment={m.get('segment')}, events_true={events_true}, summary前100字={summary_original[:100]}... ---")
            
            raw_hits = strict_filtered_hits
            print(f"--- [DEBUG] RAG 事件過濾後剩餘 {len(raw_hits)} 筆結果 ---")

        # 3. [MODIFIED] 過濾與包裝結果，並根據事件和關鍵字進行優先排序
        norm_hits = []
        print(f"--- [DEBUG] 開始處理 {len(raw_hits)} 筆 RAG 結果，分數門檻: {score_threshold} ---")
        for h in raw_hits:
            score = float(h.get("score", 0.0))

            # [NEW] 過濾低於門檻的結果
            if score < score_threshold:
                continue

            m = h.get("metadata", {})
            
            # [新增] 計算優先級：有事件標記或摘要中包含關鍵字的結果優先
            priority = 0
            events_true = m.get("events_true", [])
            summary = str(m.get("summary", "")).lower()
            message_keywords = query_filters.get("message_keywords", [])
            
            # 檢查是否有匹配的事件標記
            if query_filters.get("event_types"):
                for event_type in query_filters["event_types"]:
                    if event_type in events_true:
                        priority = 2  # 有事件標記，最高優先級
                        break
            
            # 檢查摘要中是否包含關鍵字
            if priority < 2 and message_keywords:
                for keyword in message_keywords:
                    if keyword in summary or keyword.lower() in summary:
                        priority = 1  # 摘要中包含關鍵字，次高優先級
                        break
            
            norm_hits.append({
                "score": round(score, 4), # 回傳小數點後四位
                "priority": priority,  # [新增] 優先級，用於排序
                "video": m.get("video"),
                "segment": m.get("segment"),
                "time_range": m.get("time_range"),
                "events_true": m.get("events_true", []),
                "summary": m.get("summary", ""),
                "reason": m.get("reason", ""),
                "doc_id": h.get("id"),
            })
        
        # [新增] 根據優先級和分數排序：優先級高的在前，同優先級按分數降序
        norm_hits.sort(key=lambda x: (-x.get("priority", 0), -x.get("score", 0)))
        
        # 移除 priority 欄位（不需要返回給前端）
        for hit in norm_hits:
            hit.pop("priority", None)
        
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")

        # [NEW] 在響應中包含日期解析資訊
        response = {"backend": store.embed_model, "hits": norm_hits}
        if date_info:
            response["date_parsed"] = date_info
        return response

    except Exception as e:
        print(f"--- [RAG Search Error] ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"RAG Search Failed: {str(e)}", "detail": str(e)}
        )

# 它不僅幫你找資料，還會 「閱讀資料並回答問題」。
# RAG answer API 已移至 src.api.rag
# 保留函數定義以向後兼容（如果需要）
async def _rag_answer_legacy(request: Request, db: Session = Depends(get_db) if HAS_DB else None):
    """
    使用 PostgreSQL + pgvector 進行混合搜索，然後使用 LLM 生成回答
    搜索邏輯與 /rag/search 完全相同
    """
    try:
        payload = await request.json()
        question = (payload.get("query") or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail="missing query")

        top_k = int(payload.get("top_k") or 5)

        # [NEW] 分數門檻 (建議 RAG 回答時可以設高一點，例如 0.5)
        score_threshold = float(payload.get("score_threshold") or 0.0)

        # 指定用哪個 LLM 來回答
        llm_model = (payload.get("model") or "qwen2.5vl:latest").strip()

        if not HAS_DB or not db:
            raise HTTPException(status_code=503, detail="Database not available")

        # 步驟 1: 解析查詢條件（與 /rag/search 統一）
        query_filters = {}
        date_info = None
        
        try:
            query_filters = _parse_query_filters(question)
            
            # 提取日期解析資訊，用於返回給前端
            if query_filters.get("date_filter") or query_filters.get("time_start"):
                if not date_info:  # 如果 LLM 工具調用已經設置了 date_info，就不需要重複設置
                    date_filter = query_filters.get("date_filter")
                    if date_filter:
                        if hasattr(date_filter, 'isoformat'):
                            picked_date_str = date_filter.isoformat()
                        else:
                            picked_date_str = str(date_filter)
                    else:
                        picked_date_str = None
                    
                    date_info = {
                        "mode": query_filters.get("date_mode", "NONE"),
                        "picked_date": picked_date_str,
                        "time_start": query_filters.get("time_start"),
                        "time_end": query_filters.get("time_end"),
                    }
                
                if date_info:
                    print(f"\n{'='*60}")
                    print(f"[日期解析] 查詢: {question}")
                    print(f"[日期解析] 模式: {date_info['mode']}")
                    print(f"[日期解析] 解析到的日期: {date_info['picked_date']}")
                    if date_info.get('time_start'):
                        print(f"[日期解析] 時間範圍: {date_info['time_start'][:19]} ~ {date_info.get('time_end', '')[:19]}")
                    print(f"{'='*60}\n")
        except Exception as e:
            print(f"--- [WARNING] 查詢解析失敗: {e} ---")
            import traceback
            traceback.print_exc()
            query_filters = {}

        # 步驟 2: 生成查詢向量（與 /rag/search 相同）
        clean_query = question
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}',
            r'今天|昨天|明天|本週|上週|下週',
            r'\d{4}年\d{1,2}月\d{1,2}日',
        ]
        for pattern in date_patterns:
            clean_query = re.sub(pattern, '', clean_query)
        clean_query = clean_query.strip()
        
        if not clean_query:
            clean_query = question

        embedding_model = get_embedding_model()
        if not embedding_model:
            raise HTTPException(status_code=503, detail="Embedding model not available")
        
        query_embedding = embedding_model.encode(clean_query, normalize_embeddings=True)
        print(f"--- [DEBUG] 查詢向量生成完成 (維度: {len(query_embedding)}) ---")

        # 步驟 3: 構建 PostgreSQL + pgvector 混合查詢（與 /rag/search 相同）
        stmt = select(Summary).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.isnot(None)
        )

        # 時間範圍過濾
        if query_filters.get("time_start") and query_filters.get("time_end"):
            try:
                time_start_str = query_filters["time_start"]
                time_end_str = query_filters["time_end"]
                
                if "T" in time_start_str:
                    t0 = datetime.fromisoformat(time_start_str.replace("Z", "+00:00"))
                else:
                    t0 = datetime.fromisoformat(time_start_str)
                
                if "T" in time_end_str:
                    t1 = datetime.fromisoformat(time_end_str.replace("Z", "+00:00"))
                else:
                    t1 = datetime.fromisoformat(time_end_str)
                
                if t0.tzinfo:
                    t0 = t0.astimezone().replace(tzinfo=None)
                if t1.tzinfo:
                    t1 = t1.astimezone().replace(tzinfo=None)
                
                stmt = stmt.filter(
                    Summary.start_timestamp >= t0,
                    Summary.start_timestamp < t1
                )
                print(f"--- [DEBUG] 應用時間範圍過濾: {t0} ~ {t1} ---")
            except Exception as e:
                print(f"--- [WARNING] 時間範圍解析失敗: {e} ---")
        elif query_filters.get("date_filter"):
            target_date = query_filters["date_filter"]
            t0 = datetime.combine(target_date, datetime.min.time())
            next_day = target_date + timedelta(days=1)
            t1 = datetime.combine(next_day, datetime.min.time())
            stmt = stmt.filter(
                Summary.start_timestamp >= t0,
                Summary.start_timestamp < t1
            )
            print(f"--- [DEBUG] 應用日期過濾: {target_date} ({t0} ~ {t1}) ---")

        # 事件類型過濾
        if query_filters.get("event_types"):
            event_conditions = []
            for event_type in query_filters["event_types"]:
                if event_type == "fire":
                    event_conditions.append(Summary.fire == True)
                elif event_type == "water_flood":
                    event_conditions.append(Summary.water_flood == True)
                elif event_type == "abnormal_attire_face_cover_at_entry":
                    event_conditions.append(Summary.abnormal_attire_face_cover_at_entry == True)
                elif event_type == "person_fallen_unmoving":
                    event_conditions.append(Summary.person_fallen_unmoving == True)
                elif event_type == "double_parking_lane_block":
                    event_conditions.append(Summary.double_parking_lane_block == True)
                elif event_type == "smoking_outside_zone":
                    event_conditions.append(Summary.smoking_outside_zone == True)
                elif event_type == "crowd_loitering":
                    event_conditions.append(Summary.crowd_loitering == True)
                elif event_type == "security_door_tamper":
                    event_conditions.append(Summary.security_door_tamper == True)
            
            if event_conditions:
                stmt = stmt.filter(or_(*event_conditions))
                print(f"--- [DEBUG] 應用事件過濾: {query_filters['event_types']} ---")

        # 關鍵字過濾
        if query_filters.get("message_keywords"):
            message_conditions = []
            for keyword in query_filters["message_keywords"]:
                message_conditions.append(Summary.message.ilike(f"%{keyword}%"))
            if message_conditions:
                stmt = stmt.filter(or_(*message_conditions))
                print(f"--- [DEBUG] 應用關鍵字過濾: {query_filters['message_keywords']} ---")

        # Vector search
        try:
            from pgvector.sqlalchemy import Vector
            distance_expr = Summary.embedding.cosine_distance(query_embedding)
            
            stmt = stmt.add_columns(
                distance_expr.label('cosine_distance')
            ).order_by(
                distance_expr
            ).limit(top_k * 3)
            
            print(f"--- [DEBUG] 執行 PostgreSQL + pgvector 混合查詢 ---")
            results = db.execute(stmt).all()
            print(f"--- [DEBUG] 查詢返回 {len(results)} 筆結果 ---")
        except ImportError:
            raise HTTPException(status_code=503, detail="pgvector not available")
        except Exception as e:
            print(f"--- [ERROR] pgvector 查詢失敗: {e} ---")
            import traceback
            traceback.print_exc()
            results = []

        # 步驟 4: 計算相似度分數並格式化結果（與 /rag/search 相同）
        norm_hits = []
        for row in results:
            result = row[0]
            cosine_distance = row[1]
            
            score = max(0.0, min(1.0, 1.0 - (cosine_distance / 2.0)))

            if score < score_threshold:
                continue
            
            events_true = []
            if result.fire:
                events_true.append("fire")
            if result.water_flood:
                events_true.append("water_flood")
            if result.abnormal_attire_face_cover_at_entry:
                events_true.append("abnormal_attire_face_cover_at_entry")
            if result.person_fallen_unmoving:
                events_true.append("person_fallen_unmoving")
            if result.double_parking_lane_block:
                events_true.append("double_parking_lane_block")
            if result.smoking_outside_zone:
                events_true.append("smoking_outside_zone")
            if result.crowd_loitering:
                events_true.append("crowd_loitering")
            if result.security_door_tamper:
                events_true.append("security_door_tamper")

            norm_hits.append({
                "score": round(score, 4),
                "video": result.video or "",
                "segment": result.segment or "",
                "time_range": result.time_range or "",
                "events_true": events_true,
                "summary": result.message or "",
                "reason": result.event_reason or "",
                "doc_id": None,
            })

        norm_hits = norm_hits[:top_k]
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")
        
        # 2. 組裝 Context (A) - 使用 norm_hits 作為上下文
        if not norm_hits:
            # 如果因為門檻過濾後導致沒有資料，也視為找不到
            # [NEW] 如果 Ollama 失敗，直接返回空結果而不是報錯
            try:
                msgs = [
                    {"role": "system", "content": "你只能根據系統提供的資料回答。現在沒有資料可用，請直接說你找不到答案。"},
                    {"role": "user", "content": question},
                ]
                rj = _ollama_chat(llm_model, msgs, timeout=1800)
                msg = ""
                if isinstance(rj, dict):
                    msg = (rj.get("message") or {}).get("content", "").strip()
                elif isinstance(rj, str):
                    msg = rj
            except Exception as ollama_error:
                # Ollama 失敗時，返回空結果而不是報錯
                print(f"--- [WARNING] Ollama 失敗，返回空結果: {ollama_error} ---")
                msg = "目前索引到的片段裡找不到答案（或是相似度過低）。LLM 服務暫時無法使用。"

            response = {
                "backend": {"embed_model": EMBEDDING_MODEL_NAME, "llm": llm_model},
                "hits": [],
                "answer": msg or "目前索引到的片段裡找不到答案（或是相似度過低）。",
            }
            if date_info:
                response["date_parsed"] = date_info
            if query_filters.get("message_keywords"):
                response["keywords_found"] = query_filters["message_keywords"]
            if query_filters.get("event_types"):
                response["event_types_found"] = query_filters["event_types"]
            # [NEW] 添加 embedding 查詢資訊
            response["embedding_query"] = clean_query
            return response

        # 組裝 Context (A) - 使用已經格式化好的 norm_hits
        context_blocks = []
        for i, hit in enumerate(norm_hits, start=1):
            summary = hit.get("summary", "")
            video = hit.get("video")
            time_range = hit.get("time_range")

            # 在 context 中加入分數資訊讓 LLM 參考也不錯，但這邊先保持簡潔
            context_blocks.append(
                f"[{i}] 影片: {video}  時間: {time_range}\n摘要: {summary}"
            )

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "你是工廠監控影片說明助理，必須嚴格根據提供的片段摘要回答問題。"
            "如果資料中沒有答案，就回答「我在目前索引到的片段裡找不到相關資訊」。"
        )
        user_prompt = (
            f"使用下面這些片段摘要回答問題：\n\n{context_text}\n\n"
            f"問題：{question}\n\n"
            "請用繁體中文回答，並在回答中附上你參考的片段編號（例如 [1]、[2]）。"
        )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 3. 呼叫 LLM (G) - 如果失敗，返回搜尋結果
        try:
            ans_content = _ollama_chat(llm_model, msgs, timeout=1800)
            # 防呆處理
            if isinstance(ans_content, dict):
                answer = (ans_content.get("message") or {}).get("content", "").strip()
            else:
                answer = str(ans_content).strip()
        except Exception as ollama_error:
            # [NEW] Ollama 失敗時，返回搜尋結果而不是報錯
            print(f"--- [WARNING] Ollama 失敗，返回搜尋結果: {ollama_error} ---")
            answer = f"抱歉，LLM 服務暫時無法使用（錯誤：{str(ollama_error)[:100]}）。以下是根據您的查詢找到的相關片段：\n\n"
            # 將搜尋結果轉換為文字描述
            for i, hit in enumerate(norm_hits, 1):
                answer += f"[{i}] 影片: {hit.get('video', 'N/A')}  時間: {hit.get('time_range', 'N/A')}\n"
                answer += f"    摘要: {hit.get('summary', 'N/A')[:100]}...\n\n"

        # [NEW] 在響應中包含日期解析資訊和關鍵字資訊
        response = {
            "backend": {"embed_model": EMBEDDING_MODEL_NAME, "llm": llm_model},
            "hits": norm_hits,
            "answer": answer,
        }
        if date_info:
            response["date_parsed"] = date_info
        # [NEW] 添加關鍵字資訊
        if query_filters.get("message_keywords"):
            response["keywords_found"] = query_filters["message_keywords"]
        if query_filters.get("event_types"):
            response["event_types_found"] = query_filters["event_types"]
        # [NEW] 添加 embedding 查詢資訊
        response["embedding_query"] = clean_query
        return response

    except Exception as e:
        print(f"--- [RAG Answer Error] ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"RAG Answer Failed: {str(e)}", "detail": str(e)}
        )

# ================== PostgreSQL 保存與過濾 ==================

def _parse_time_range(time_range_str: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    解析時間範圍字串，例如 "00:00:00 - 00:00:08"
    返回 (start_time, end_time) 作為 datetime 對象
    """
    try:
        if not time_range_str or not isinstance(time_range_str, str):
            return None, None
            
        if " - " in time_range_str:
            start_str, end_str = time_range_str.split(" - ", 1)
            start_str = start_str.strip()
            end_str = end_str.strip()
            
            # 解析 HH:MM:SS 格式
            start_time_obj = datetime.strptime(start_str, "%H:%M:%S").time()
            end_time_obj = datetime.strptime(end_str, "%H:%M:%S").time()
            
            # 創建 datetime 對象（使用今天的日期作為基準）
            today = datetime.now().date()
            
            start_time = datetime.combine(today, start_time_obj)
            end_time = datetime.combine(today, end_time_obj)
            
            return start_time, end_time
    except ValueError as e:
        print(f"Warning: Could not parse time range '{time_range_str}': {e}")
    except Exception as e:
        print(f"Warning: Unexpected error parsing time range '{time_range_str}': {e}")
    
    return None, None


def _create_alert_if_needed(db: Session, summary_id: int, events: Dict[str, Any], video_stem: str, segment: str, location: str = None):
    """
    如果偵測到事件，自動創建 Alert 記錄
    
    Args:
        db: 資料庫 session
        summary_id: Summary 記錄的 ID
        events: 事件資料 (來自 parsed.frame_analysis.events)
        video_stem: 影片名稱
        segment: 片段名稱
        location: 發生位置（可選）
    """
    from src.models import Alert, DetectionItem
    
    # 檢查是否有任何事件被偵測到
    detected_events = []
    for event_name, detected in events.items():
        if event_name != "reason" and detected:
            detected_events.append(event_name)
    
    # 如果沒有偵測到任何事件，不創建 Alert
    if not detected_events:
        return
    
    # 查詢 DetectionItem 以獲取事件的中英文名稱
    detection_items = db.query(DetectionItem).filter(DetectionItem.name.in_(detected_events)).all()
    item_dict = {item.name: item for item in detection_items}
    
    # 為每個偵測到的事件創建 Alert
    for event_name in detected_events:
        item = item_dict.get(event_name)
        if not item:
            print(f"  ⚠️  找不到偵測項目: {event_name}")
            continue
        
        # 建立警報標題
        title = f"偵測到{item.name_zh}"
        if location:
            title += f"於{location}"
        
        # 建立警報訊息
        event_reason = events.get("reason", "")
        message = f"偵測到{item.name_zh}"
        if event_reason:
            message += f"，{event_reason}"
        else:
            message += "，請立即進行疏散與應處理。"
        
        # 創建 Alert 記錄
        try:
            alert = Alert(
                summary_id=summary_id,
                detection_item_id=item.id,
                alert_type=event_name,
                title=title,
                message=message,
                location=location,
                video=video_stem,
                segment=segment,
                timestamp=datetime.now(),
                severity='high' if event_name in ['fire', 'water_flood'] else 'medium',
                is_read=False,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            db.add(alert)
            
            # 更新 DetectionItem 的 alert_count
            item.alert_count += 1
            
            print(f"  ✓ 建立警報：{title}")
        except Exception as e:
            print(f"  ⚠️  建立警報失敗 ({event_name}): {e}")


def _save_results_to_postgres(db: Session, results: List[Dict[str, Any]], video_stem: str):
    """
    將分析結果保存到 PostgreSQL 資料庫
    與 migrate_segments_to_db.py 的邏輯一致：影片相同則更新，新的則新增
    
    Args:
        db: 資料庫 session
        results: 分析結果列表（來自 segment_pipeline_multipart 的 results）
        video_stem: 影片名稱（用於識別，例如 "fire_1" 或 "火災生成_Video_火災"）
    """
    if not HAS_DB:
        print("--- [PostgreSQL] HAS_DB = False，跳過保存 ---")
        return
    
    if not db:
        print("--- [PostgreSQL] db session 為 None，跳過保存 ---")
        return
    
    print(f"--- [PostgreSQL] 開始保存 {len(results)} 筆結果到資料庫 (video: {video_stem}) ---")
    
    # [修改] 採用「更新或新增」的邏輯，與 migrate_segments_to_db.py 保持一致
    # 不再先刪除舊記錄，而是檢查是否存在，存在則更新，不存在則新增
    saved_count = 0
    updated_count = 0
    inserted_count = 0
    skipped_count = 0
    
    for idx, result in enumerate(results):
        # 只處理成功的結果
        if not result.get("success", False):
            print(f"--- [PostgreSQL] 片段 {idx}：success=False，跳過 ---")
            skipped_count += 1
            continue
        
        # 獲取摘要文字
        parsed = result.get("parsed", {})
        summary_text = parsed.get("summary_independent", "")
        
        # [修改] 如果有 YOLO 結果，即使沒有摘要也要保存
        raw_detection = result.get("raw_detection")
        has_yolo = raw_detection and isinstance(raw_detection, dict) and raw_detection.get("yolo")
        
        # 如果沒有摘要且沒有 YOLO 結果，跳過
        if (not summary_text or not summary_text.strip()) and not has_yolo:
            print(f"--- [PostgreSQL] 片段 {idx}：無摘要且無 YOLO 結果，跳過 ---")
            skipped_count += 1
            continue
        
        # 如果沒有摘要但有 YOLO 結果，使用預設摘要
        if not summary_text or not summary_text.strip():
            summary_text = "YOLO 偵測結果（無 VLM 摘要）"
            print(f"--- [PostgreSQL] 片段 {idx}：無摘要但有 YOLO 結果，使用預設摘要 ---")
        
        # 解析時間範圍
        time_range = result.get("time_range", "")
        start_time, end_time = _parse_time_range(time_range)
        
        # 獲取其他欄位
        segment = result.get("segment", "")
        duration_sec = result.get("duration_sec")
        time_sec = result.get("time_sec")
        
        # 獲取事件檢測資料
        frame_analysis = parsed.get("frame_analysis", {})
        events = frame_analysis.get("events", {})
        
        # 檢查記錄是否已存在（根據 video、segment 和 time_range）
        # 與 migrate_segments_to_db.py 的邏輯一致：影片相同則更新，新的則新增
        # [修改] 加入 video 欄位判斷，避免不同影片的相同 segment 和 time_range 被誤判為同一筆記錄
        existing = db.query(Summary).filter(
            Summary.video == video_stem,
            Summary.segment == segment,
            Summary.time_range == time_range
        ).first()
        
        if existing:
            # 更新現有記錄（影片相同則更新）
            existing.start_timestamp = start_time if start_time else datetime.now()
            existing.end_timestamp = end_time
            existing.video = video_stem  # [新增] 確保 video 欄位也被更新
            existing.message = summary_text.strip()
            existing.duration_sec = float(duration_sec) if duration_sec is not None else None
            existing.time_sec = float(time_sec) if time_sec is not None else None
            # 更新事件檢測欄位（舊欄位，保留以向後兼容）
            existing.water_flood = bool(events.get("water_flood", False))
            existing.fire = bool(events.get("fire", False))
            existing.abnormal_attire_face_cover_at_entry = bool(events.get("abnormal_attire_face_cover_at_entry", False))
            existing.person_fallen_unmoving = bool(events.get("person_fallen_unmoving", False))
            existing.double_parking_lane_block = bool(events.get("double_parking_lane_block", False))
            existing.smoking_outside_zone = bool(events.get("smoking_outside_zone", False))
            existing.crowd_loitering = bool(events.get("crowd_loitering", False))
            existing.security_door_tamper = bool(events.get("security_door_tamper", False))
            existing.event_reason = events.get("reason", "") if events.get("reason") else None
            
            # [已移除] events_en, events_zh, events_json 欄位（改用個別布林欄位）
            # 個別的事件布林欄位已經在上面更新（行 3286-3294）
            
            # [NEW] 更新 YOLO 結果（如果有的話）
            raw_detection = result.get("raw_detection")
            if raw_detection and isinstance(raw_detection, dict):
                yolo_result = raw_detection.get("yolo")
                if yolo_result:
                    existing.yolo_detections = json.dumps(yolo_result.get("detections", []), ensure_ascii=False)
                    existing.yolo_object_count = json.dumps(yolo_result.get("object_count", {}), ensure_ascii=False)
                    existing.yolo_total_detections = yolo_result.get("total_detections", 0)
                    existing.yolo_total_frames_processed = yolo_result.get("total_frames_processed", 0)
                    # 設置物件切片目錄路徑（優化版本可能不包含 path，改為 None）
                    if yolo_result.get("crop_paths"):
                        first_crop = yolo_result["crop_paths"][0]
                        if isinstance(first_crop, dict) and first_crop.get("path"):
                            crops_dir = str(Path(first_crop["path"]).parent)
                            existing.yolo_crops_dir = crops_dir
                        else:
                            # 優化版本：不保存文件，設置為 None
                            existing.yolo_crops_dir = None
                    
                    # [NEW] 更新 ObjectCrop 記錄：先刪除舊的，再添加新的
                    try:
                        # 刪除該 summary 的所有舊 ObjectCrop 記錄
                        db.query(ObjectCrop).filter(ObjectCrop.summary_id == existing.id).delete()
                        
                        # 添加新的 ObjectCrop 記錄
                        if yolo_result.get("crop_paths"):
                            crop_paths = yolo_result.get("crop_paths", [])
                            for crop_data in crop_paths:
                                if isinstance(crop_data, dict):
                                    try:
                                        object_crop = ObjectCrop(
                                            summary_id=existing.id,
                                            crop_path=crop_data.get("path"),
                                            label=crop_data.get("label"),
                                            score=crop_data.get("score"),
                                            timestamp=crop_data.get("timestamp"),
                                            frame=crop_data.get("frame"),
                                            box=json.dumps(crop_data.get("box", []), ensure_ascii=False) if crop_data.get("box") else None,
                                            clip_embedding=crop_data.get("clip_embedding"),  # CLIP embedding（512 維）
                                            reid_embedding=crop_data.get("reid_embedding"),  # ReID embedding（2048 維）
                                            created_at=datetime.now()
                                        )
                                        db.add(object_crop)
                                    except Exception as e:
                                        print(f"  ⚠️  更新 ObjectCrop 失敗: {e}")
                                        continue
                    except Exception as e:
                        print(f"  ⚠️  更新 ObjectCrop 記錄時出錯: {e}")
            
            # [NEW] 如果 message 有變更，重新生成 embedding
            if existing.message != summary_text.strip() and summary_text.strip():
                try:
                    embedding_model = get_embedding_model()
                    if embedding_model:
                        existing.embedding = embedding_model.encode(
                            summary_text.strip(),
                            normalize_embeddings=True
                        ).tolist()
                except Exception as e:
                    print(f"  ⚠️  更新 embedding 失敗: {e}")
            # 更新 updated_at 時間戳
            existing.updated_at = datetime.now()
            saved_count += 1
            updated_count += 1
            
            # [NEW] 如果偵測到事件，創建 Alert 記錄
            _create_alert_if_needed(db, existing.id, events, video_stem, segment, location=None)
        else:
            # 新增記錄（新的則新增）
            # [NEW] 自動生成 embedding
            embedding = None
            if summary_text.strip():
                try:
                    embedding_model = get_embedding_model()
                    if embedding_model:
                        embedding = embedding_model.encode(
                            summary_text.strip(),
                            normalize_embeddings=True
                        ).tolist()
                except Exception as e:
                    print(f"  ⚠️  生成 embedding 失敗: {e}")
            
            summary = Summary(
                start_timestamp=start_time if start_time else datetime.now(),
                end_timestamp=end_time,
                location=None,  # 之後可以從其他地方填入
                camera=None,    # 之後可以從其他地方填入
                video=video_stem,  # [新增] 保存影片名稱，用於區分不同影片的相同 segment
                message=summary_text.strip(),
                segment=segment if segment else None,
                time_range=time_range if time_range else None,
                duration_sec=float(duration_sec) if duration_sec is not None else None,
                time_sec=float(time_sec) if time_sec is not None else None,
                # 事件檢測欄位（舊欄位，保留以向後兼容）
                water_flood=bool(events.get("water_flood", False)),
                fire=bool(events.get("fire", False)),
                abnormal_attire_face_cover_at_entry=bool(events.get("abnormal_attire_face_cover_at_entry", False)),
                person_fallen_unmoving=bool(events.get("person_fallen_unmoving", False)),
                double_parking_lane_block=bool(events.get("double_parking_lane_block", False)),
                smoking_outside_zone=bool(events.get("smoking_outside_zone", False)),
                crowd_loitering=bool(events.get("crowd_loitering", False)),
                security_door_tamper=bool(events.get("security_door_tamper", False)),
                event_reason=events.get("reason", "") if events.get("reason") else None,
                embedding=embedding,  # [NEW] 自動生成的 embedding
                # [NEW] YOLO 結果（整合到 summaries 表）
                yolo_detections=None,
                yolo_object_count=None,
                yolo_crops_dir=None,
                yolo_total_detections=None,
                yolo_total_frames_processed=None,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # [NEW] 如果有 YOLO 結果，添加到記錄中
            raw_detection = result.get("raw_detection")
            if raw_detection and isinstance(raw_detection, dict):
                yolo_result = raw_detection.get("yolo")
                if yolo_result:
                    summary.yolo_detections = json.dumps(yolo_result.get("detections", []), ensure_ascii=False)
                    summary.yolo_object_count = json.dumps(yolo_result.get("object_count", {}), ensure_ascii=False)
                    summary.yolo_total_detections = yolo_result.get("total_detections", 0)
                    summary.yolo_total_frames_processed = yolo_result.get("total_frames_processed", 0)
                    # 設置物件切片目錄路徑（優化版本可能不包含 path，改為 None）
                    if yolo_result.get("crop_paths"):
                        first_crop = yolo_result["crop_paths"][0]
                        if isinstance(first_crop, dict) and first_crop.get("path"):
                            crops_dir = str(Path(first_crop["path"]).parent)
                            summary.yolo_crops_dir = crops_dir
                        else:
                            # 優化版本：不保存文件，設置為 None
                            summary.yolo_crops_dir = None
            
            # [已移除] events_en, events_zh, events_json 欄位（改用個別布林欄位）
            # 個別的事件布林欄位已經在 Summary 建構時設定（行 3412-3420）
            
            try:
                db.add(summary)
                db.flush()  # 獲取 summary.id
                saved_count += 1
                inserted_count += 1
                
                # [NEW] 如果偵測到事件，創建 Alert 記錄
                _create_alert_if_needed(db, summary.id, events, video_stem, segment, location=None)
                
                # [NEW] 保存 ObjectCrop 記錄（如果有的話）
                if raw_detection and isinstance(raw_detection, dict):
                    yolo_result = raw_detection.get("yolo")
                    if yolo_result and yolo_result.get("crop_paths"):
                        crop_paths = yolo_result.get("crop_paths", [])
                        for crop_data in crop_paths:
                            if isinstance(crop_data, dict):
                                try:
                                    object_crop = ObjectCrop(
                                        summary_id=summary.id,
                                        crop_path=crop_data.get("path"),
                                        label=crop_data.get("label"),
                                        score=crop_data.get("score"),
                                        timestamp=crop_data.get("timestamp"),
                                        frame=crop_data.get("frame"),
                                        box=json.dumps(crop_data.get("box", []), ensure_ascii=False) if crop_data.get("box") else None,
                                        clip_embedding=crop_data.get("clip_embedding"),  # CLIP embedding（512 維）
                                        reid_embedding=crop_data.get("reid_embedding"),  # ReID embedding（2048 維）
                                        created_at=datetime.now()
                                    )
                                    db.add(object_crop)
                                except Exception as e:
                                    print(f"  ⚠️  保存 ObjectCrop 失敗: {e}")
                                    continue
            except Exception as e:
                print(f"Warning: Failed to add summary to session: {e}")
                continue
    
    # 批量提交
    print(f"--- [PostgreSQL] 準備提交：已處理 {saved_count} 筆（新增: {inserted_count}, 更新: {updated_count}, 跳過: {skipped_count}）---")
    
    if saved_count > 0:
        try:
            db.commit()
            print(f"--- [PostgreSQL] ✓ 成功保存/更新 {saved_count} 筆分析結果到資料庫 (video: {video_stem}, 新增: {inserted_count}, 更新: {updated_count}) ---")
        except Exception as e:
            db.rollback()
            print(f"--- [PostgreSQL ERROR] ✗ 提交失敗: {e} ---")
            import traceback
            traceback.print_exc()
    else:
        # 即使沒有新記錄，也要提交（可能只有更新操作）
        try:
            db.commit()
            print(f"--- [PostgreSQL] 無新記錄需要保存（跳過: {skipped_count}）---")
        except Exception as e:
            db.rollback()
            print(f"--- [PostgreSQL ERROR] ✗ 提交失敗: {e} ---")
            import traceback
            traceback.print_exc()

# ReID 模型（用於物件 re-identification）
_reid_model = None
_reid_device = None

def get_reid_model():
    """獲取或初始化 ReID 模型（用於物件 re-identification，優先使用 torchreid/FastReID）"""
    global _reid_model, _reid_device
    
    if _reid_model is not None:
        return _reid_model, _reid_device
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _reid_device = device
    
    errors = []  # 記錄所有嘗試的錯誤
    
    # 優先使用 torchreid（FastReID 風格）
    try:
        import torchreid
        print(f"--- [ReID] 嘗試載入 torchreid 模型 (device: {device}) ---")
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=751,  # Market-1501 dataset classes
            loss='softmax',
            pretrained=True
        )
        model = model.to(device).eval()
        model.classifier = None  # 移除分類層，只保留特徵提取器
        
        # 驗證模型輸出維度
        test_input = torch.randn(1, 3, 256, 128).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            output_dim = test_output.shape[1] if len(test_output.shape) > 1 else test_output.shape[0]
        
        if output_dim != 2048:
            error_msg = f"torchreid 模型輸出維度錯誤: 預期 2048，實際 {output_dim}"
            print(f"--- [ReID] ✗ {error_msg} ---")
            errors.append(f"torchreid: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"✓ torchreid ResNet50 模型載入完成（輸出維度: {output_dim}）")
        _reid_model = model
        return model, device
    except ImportError as e:
        error_msg = f"torchreid 未安裝: {e}"
        print(f"--- [ReID] {error_msg}，嘗試備用方案... ---")
        errors.append(error_msg)
    except Exception as e:
        error_msg = f"torchreid 載入失敗: {e}"
        print(f"--- [ReID] ✗ {error_msg}，嘗試備用方案... ---")
        errors.append(error_msg)
        import traceback
        traceback.print_exc()
    
    # 備用：使用 timm ResNet50（設置離線模式避免長時間下載）
    try:
        import timm
        import os
        print(f"--- [ReID] 嘗試載入 timm ResNet50 模型 (device: {device}) ---")
        
        # 設置 HuggingFace 為離線模式，避免網絡下載超時
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            # 嘗試從本地緩存載入預訓練模型
            model = timm.create_model('resnet50', pretrained=True, num_classes=0)
        except Exception as download_err:
            # 如果本地沒有緩存，使用不預訓練的模型（避免網絡下載）
            print(f"--- [ReID] 本地無緩存，使用未預訓練的 ResNet50: {download_err} ---")
            model = timm.create_model('resnet50', pretrained=False, num_classes=0)
        finally:
            # 恢復網絡模式
            os.environ.pop('HF_HUB_OFFLINE', None)
        
        model = model.to(device).eval()
        
        # 驗證模型輸出維度
        test_input = torch.randn(1, 3, 256, 128).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            output_dim = test_output.shape[1] if len(test_output.shape) > 1 else test_output.shape[0]
        
        if output_dim != 2048:
            error_msg = f"timm 模型輸出維度錯誤: 預期 2048，實際 {output_dim}"
            print(f"--- [ReID] ✗ {error_msg} ---")
            errors.append(f"timm: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"✓ timm ResNet50 模型載入完成（輸出維度: {output_dim}）")
        _reid_model = model
        return model, device
    except ImportError as e:
        error_msg = f"timm 未安裝: {e}"
        print(f"--- [ReID] {error_msg}，嘗試最後備用方案... ---")
        errors.append(error_msg)
    except Exception as e:
        error_msg = f"timm 載入失敗: {e}"
        print(f"--- [ReID] ✗ {error_msg}，嘗試最後備用方案... ---")
        errors.append(error_msg)
        import traceback
        traceback.print_exc()
    
    # 最後備用：使用 torchvision ResNet50（避免網絡下載）
    try:
        import torchvision.models as models
        print(f"--- [ReID] 嘗試載入 torchvision ResNet50 模型 (device: {device}) ---")
        
        try:
            # 嘗試載入預訓練模型（從本地緩存）
            model = models.resnet50(pretrained=True)
        except Exception as download_err:
            # 如果下載失敗，使用未預訓練的模型
            print(f"--- [ReID] 無法載入預訓練權重，使用未預訓練的 ResNet50: {download_err} ---")
            model = models.resnet50(pretrained=False)
        
        model.fc = torch.nn.Identity()  # 移除分類層
        model = model.to(device).eval()
        
        # 驗證模型輸出維度
        test_input = torch.randn(1, 3, 256, 128).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            output_dim = test_output.shape[1] if len(test_output.shape) > 1 else test_output.shape[0]
        
        if output_dim != 2048:
            error_msg = f"torchvision 模型輸出維度錯誤: 預期 2048，實際 {output_dim}"
            print(f"--- [ReID] ✗ {error_msg} ---")
            errors.append(f"torchvision: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"✓ torchvision ResNet50 模型載入完成（輸出維度: {output_dim}）")
        _reid_model = model
        return model, device
    except ImportError as e:
        error_msg = f"torchvision 未安裝: {e}"
        print(f"--- [ReID] ✗ {error_msg} ---")
        errors.append(error_msg)
    except Exception as e:
        error_msg = f"torchvision 載入失敗: {e}"
        print(f"--- [ReID] ✗ {error_msg} ---")
        errors.append(error_msg)
        import traceback
        traceback.print_exc()
    
    # 所有方案都失敗
    error_summary = "所有 ReID 模型載入方案都失敗：\n" + "\n".join(f"  - {err}" for err in errors)
    print(f"--- [ReID] ✗ {error_summary} ---")
    print("--- [ReID] 請檢查：1) 是否安裝 torchreid: pip install torchreid 2) 或確保 timm/torchvision 已正確安裝 3) 檢查後端日誌以獲取詳細錯誤信息 ---")
    return None, None

def generate_reid_embeddings_batch(crop_images: List[np.ndarray], reid_model=None, reid_device=None) -> List[Optional[List[float]]]:
    """
    批量生成 ReID embedding（優化版本，用於內存中的裁剪圖像）
    
    Args:
        crop_images: 裁剪的圖像列表（numpy arrays，BGR 格式）
        reid_model: ReID 模型（如果為 None 則自動獲取）
        reid_device: 設備（如果為 None 則自動獲取）
        
    Returns:
        embedding 向量列表（每個 2048 維）或 None
    """
    if not crop_images:
        return []
    
    try:
        if reid_model is None or reid_device is None:
            reid_model, reid_device = get_reid_model()
        
        if reid_model is None:
            # 備用：使用 CLIP（但 CLIP 不支持批量，需要逐個處理）
            print("--- [WARNING] ReID 模型不可用，跳過批量 embedding 生成 ---")
            return [None] * len(crop_images)
        
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        # 批量預處理
        transform = transforms.Compose([
            transforms.Resize((256, 128)),  # ReID 標準尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 轉換所有圖像為 tensor
        tensors = []
        valid_indices = []
        for i, crop in enumerate(crop_images):
            try:
                # 轉換 BGR 到 RGB
                if len(crop.shape) == 3 and crop.shape[2] == 3:
                    rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = crop
                
                # 轉換為 PIL Image
                pil_image = Image.fromarray(rgb_image)
                
                # 預處理
                tensor = transform(pil_image)
                tensors.append(tensor)
                valid_indices.append(i)
            except Exception as e:
                print(f"--- [WARNING] 預處理圖像失敗 (index {i}): {e} ---")
                valid_indices.append(None)
        
        if not tensors:
            return [None] * len(crop_images)
        
        # 批量推理
        batch_tensor = torch.stack(tensors).to(reid_device)
        
        with torch.no_grad():
            features = reid_model(batch_tensor)
            # L2 正規化
            features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-12)
            embeddings = features.cpu().numpy()
        
        # 映射回原始索引
        result = [None] * len(crop_images)
        for idx, valid_idx in enumerate(valid_indices):
            if valid_idx is not None:
                result[valid_idx] = embeddings[idx].tolist()
        
        return result
    except Exception as e:
        print(f"--- [WARNING] 批量生成 ReID embedding 失敗: {e} ---")
        import traceback
        traceback.print_exc()
        return [None] * len(crop_images)

def generate_reid_embedding(image_path: str) -> tuple[Optional[List[float]], str]:
    """
    為圖像生成 ReID embedding（用於物件 re-identification）
    
    Args:
        image_path: 圖像文件路徑
        
    Returns:
        (embedding 向量, embedding_type): 
        - embedding: 2048 維（ReID）或 None
        - embedding_type: "reid" 或 None
        
    Raises:
        RuntimeError: 如果 ReID 模型未載入
    """
    try:
        reid_model, reid_device = get_reid_model()
        if reid_model is None:
            error_msg = "ReID 模型未載入。請檢查：1) 是否安裝 torchreid: pip install torchreid 2) 後端日誌中的詳細錯誤信息"
            print(f"--- [ERROR] {error_msg} ---")
            raise RuntimeError(error_msg)
        
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        # 讀取圖像
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # ReID 標準預處理（256x128）
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(pil_image).unsqueeze(0).to(reid_device)
        
        # 生成 embedding
        with torch.no_grad():
            features = reid_model(input_tensor)
            embedding = features.cpu().numpy().flatten()
            # L2 正規化
            embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        # 驗證維度（應該是 2048）
        if len(embedding) != 2048:
            error_msg = f"ReID embedding 維度錯誤: 預期 2048 維，實際 {len(embedding)} 維"
            print(f"--- [ERROR] {error_msg} ---")
            raise ValueError(error_msg)
        
        return embedding.tolist(), "reid"
    except (RuntimeError, ValueError) as e:
        # 否則重新拋出異常
        raise
    except Exception as e:
        error_msg = f"生成 ReID embedding 失敗 ({image_path}): {e}"
        print(f"--- [ERROR] {error_msg} ---")
        import traceback
        traceback.print_exc()
        raise RuntimeError(error_msg) from e

# [DEPRECATED] _save_object_crops_to_postgres 已不再使用
# YOLO 結果現在直接整合到 summaries 表中，通過 _save_results_to_postgres 保存
# 保留函數定義以向後兼容
def _save_object_crops_to_postgres(db: Session, yolo_result: Dict, video_stem: str, segment_name: str, segment_index: int, time_range: str):
    """
    [DEPRECATED] 此函數已不再使用
    YOLO 結果現在直接整合到 summaries 表中，通過 _save_results_to_postgres 保存
    """
    return


# ================== Prompt 解析與 PostgreSQL 過濾 ==================

def _parse_query_filters(question: str) -> Dict[str, Any]:
    """
    從用戶問題中解析出過濾條件：
    - 日期/時間（支援多種格式：今天/昨天/本週/MMDD/YYYYMMDD/自然語言日期）
    - 地點（如 "路口" -> location 欄位）
    - 事件類型（如 "火災"、"水災"、"闖入"）
    
    返回一個字典，包含：
    - date_filter: Optional[date] - 日期過濾（向後兼容）
    - time_start: Optional[datetime] - 開始時間（ISO 字串）
    - time_end: Optional[datetime] - 結束時間（ISO 字串）
    - date_mode: str - 日期解析模式
    - location_keywords: List[str] - 地點關鍵字
    - event_types: List[str] - 事件類型（對應到資料庫欄位）
    """
    # [NEW] 使用 MCP 客戶端進行日期解析
    from mcp_client import parse_date_via_mcp
    
    filters = {
        "date_filter": None,
        "time_start": None,
        "time_end": None,
        "date_mode": "NONE",
        "location_keywords": [],
        "event_types": [],
    }
    
    # [NEW] 通過 MCP 調用日期解析工具
    try:
        date_result = parse_date_via_mcp(question)
    except Exception as e:
        print(f"--- [WARNING] MCP 日期解析失敗，回退到直接調用: {e} ---")
        # 回退到直接調用
        try:
            from mcp.tools.parse_time import parse_query_time_window
        except ImportError:
            from src.mcp.tools.parse_time import parse_query_time_window
        date_result = parse_query_time_window(question)
    
    # 處理 date_filter（可能是字串或 date 對象）
    if date_result.get("date_filter"):
        date_filter_value = date_result["date_filter"]
        # 如果是字串，轉換為 date 對象
        if isinstance(date_filter_value, str):
            from datetime import date as date_type
            filters["date_filter"] = date_type.fromisoformat(date_filter_value)
        else:
            filters["date_filter"] = date_filter_value
        
        filters["time_start"] = date_result.get("time_start")
        filters["time_end"] = date_result.get("time_end")
        filters["date_mode"] = date_result.get("mode", "NONE")
        print(f"--- [DEBUG] 日期解析成功 (模式: {date_result.get('mode', 'NONE')}): {filters['date_filter']} ({date_result.get('time_start', '')} ~ {date_result.get('time_end', '')}) ---")
    
    # 事件類型映射（中文 -> 資料庫欄位）
    event_mapping = {
        "火災": "fire",
        "火": "fire",
        "水災": "water_flood",
        "水": "water_flood",
        "淹水": "water_flood",
        "積水": "water_flood",
        "闖入": "security_door_tamper",
        "突破": "security_door_tamper",
        "安全門": "security_door_tamper",
        "遮臉": "abnormal_attire_face_cover_at_entry",
        "異常著裝": "abnormal_attire_face_cover_at_entry",
        "倒地": "person_fallen_unmoving",
        "倒地不起": "person_fallen_unmoving",
        "併排": "double_parking_lane_block",
        "停車": "double_parking_lane_block",
        "阻塞": "double_parking_lane_block",
        "吸菸": "smoking_outside_zone",
        "抽菸": "smoking_outside_zone",
        "群聚": "crowd_loitering",
        "聚眾": "crowd_loitering",
        "逗留": "crowd_loitering",
    }
    
    # [NEW] message 關鍵字過濾：如果查詢中包含事件相關關鍵字，也在 message 中搜尋
    # 這可以幫助找到 message 中提到相關事件的記錄（例如：火災、倒地、群聚等）
    event_keywords_in_message = ["火災", "倒地", "群聚", "聚眾", "水災", "淹水", "闖入", "遮臉", "吸菸", "停車", "阻塞"]
    message_keywords_found = []
    for keyword in event_keywords_in_message:
        if keyword in question:
            message_keywords_found.append(keyword)
    
    # [NEW] 添加描述性關鍵字（顏色、衣服、車輛等）
    descriptive_keywords = [
        # 顏色 + 衣服
        "黃色衣服", "黑色衣服", "白色衣服", "紅色衣服", "藍色衣服", "綠色衣服",
        "深色衣服", "淺色衣服", "灰色衣服",
        "黃色制服", "黑色制服", "白色制服", "紅色制服", "藍色制服", "綠色制服",
        "深色制服", "淺色制服", "灰色制服",
        # 顏色 + 車輛
        "藍色貨車", "白色貨車", "紅色貨車", "黑色貨車", "綠色貨車", "黃色貨車",
        "藍色卡車", "白色卡車", "紅色卡車", "黑色卡車",
        "藍色汽車", "白色汽車", "紅色汽車", "黑色汽車",
        "藍色機車", "白色機車", "紅色機車", "黑色機車",
        "藍色車", "白色車", "紅色車", "黑色車", "黃色車",
        # 單獨的顏色（用於匹配摘要中的顏色描述）
        "黃色", "黑色", "白色", "紅色", "藍色", "綠色", "灰色", "深色", "淺色",
        # 衣服相關
        "衣服", "上衣", "褲子", "帽子",
        # 車輛相關
        "貨車", "卡車", "汽車", "機車", "車"
    ]
    
    # 先檢查完整的多字關鍵字（如「黃色衣服」、「藍色貨車」）
    for keyword in descriptive_keywords:
        if keyword in question and keyword not in message_keywords_found:
            message_keywords_found.append(keyword)
    
    if message_keywords_found:
        # 將找到的關鍵字添加到 filters 中，用於後續的 message 過濾
        filters["message_keywords"] = message_keywords_found
        print(f"--- [DEBUG] 找到 message 關鍵字: {message_keywords_found} ---")
    
    # 解析事件類型
    question_lower = question.lower()
    for keyword, db_field in event_mapping.items():
        if keyword in question_lower:
            if db_field not in filters["event_types"]:
                filters["event_types"].append(db_field)
    
    # 解析地點關鍵字（簡單關鍵字匹配）
    location_keywords = ["路口", "入口", "出口", "停車場", "大門", "側門", "後門"]
    for keyword in location_keywords:
        if keyword in question:
            filters["location_keywords"].append(keyword)
    
    return filters


def _expand_query_keywords(query: str) -> List[str]:
    """
    對 query 做關鍵字擴展（例如：積水 / 水災 / 淹水）
    返回擴展後的查詢列表
    """
    expanded_queries = [query]  # 原始查詢
    
    # 關鍵字擴展映射
    keyword_expansions = {
        "積水": ["積水", "水災", "淹水", "水"],
        "水災": ["水災", "積水", "淹水", "水"],
        "淹水": ["淹水", "積水", "水災", "水"],
        "火災": ["火災", "火", "煙", "濃煙"],
        "火": ["火", "火災", "煙", "濃煙"],
        "倒地": ["倒地", "倒地不起", "躺", "臥"],
        "群聚": ["群聚", "聚眾", "逗留"],
        "闖入": ["闖入", "突破", "安全門"],
        "遮臉": ["遮臉", "異常著裝"],
        "吸菸": ["吸菸", "抽菸"],
        "停車": ["停車", "併排", "阻塞"],
    }
    
    # 檢查查詢中是否包含可擴展的關鍵字
    for keyword, expansions in keyword_expansions.items():
        if keyword in query:
            for expanded_keyword in expansions:
                if expanded_keyword not in query:
                    # 替換關鍵字生成新的查詢
                    expanded_query = query.replace(keyword, expanded_keyword)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
    
    return expanded_queries


def _filter_summaries_by_query(
    db: Session,
    filters: Dict[str, Any],
    limit: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    根據過濾條件從 PostgreSQL 查詢 summaries，返回符合條件的記錄
    
    返回: Dict[segment_uid] -> sql_payload
    segment_uid 格式: f"{video}|{segment}|{time_range}"
    sql_payload 包含: start_timestamp, video, segment, time_range, event_reason, message
    """
    if not HAS_DB:
        return {}
    
    # 查詢需要的欄位
    query = db.query(
        Summary.start_timestamp,
        Summary.video,
        Summary.segment,
        Summary.time_range,
        Summary.event_reason,
        Summary.message
    ).filter(
        Summary.message.isnot(None),
        Summary.message != ""
    )
    
    # [關鍵] 日期/時間範圍過濾 - 使用 start_timestamp 做 range filter
    # 優先使用 time_start/time_end（支援週範圍等），否則使用 date_filter（向後兼容）
    if filters.get("time_start") and filters.get("time_end"):
        # 使用新的時間範圍過濾（支援相對日期、週範圍等）
        try:
            # 解析 ISO 格式的時間字串
            time_start_str = filters["time_start"]
            time_end_str = filters["time_end"]
            
            # 處理 ISO 格式（可能包含時區資訊）
            if "T" in time_start_str:
                t0 = datetime.fromisoformat(time_start_str.replace("Z", "+00:00"))
            else:
                # 如果只是日期，加上時間部分
                t0 = datetime.fromisoformat(time_start_str)
            
            if "T" in time_end_str:
                t1 = datetime.fromisoformat(time_end_str.replace("Z", "+00:00"))
            else:
                t1 = datetime.fromisoformat(time_end_str)
            
            # 轉換為 naive datetime（資料庫存的是 naive datetime）
            if t0.tzinfo:
                t0 = t0.astimezone().replace(tzinfo=None)
            if t1.tzinfo:
                t1 = t1.astimezone().replace(tzinfo=None)
            
            query = query.filter(
                Summary.start_timestamp >= t0,
                Summary.start_timestamp < t1
            )
            print(f"--- [DEBUG] 應用時間範圍過濾 (模式: {filters.get('date_mode', 'UNKNOWN')}): {t0} ~ {t1} ---")
        except Exception as e:
            print(f"--- [WARNING] 時間範圍解析失敗: {e}，回退到 date_filter ---")
            # 回退到 date_filter
            if filters.get("date_filter"):
                target_date = filters["date_filter"]
                t0 = datetime.combine(target_date, datetime.min.time())
                next_day = target_date + timedelta(days=1)
                t1 = datetime.combine(next_day, datetime.min.time())
                query = query.filter(
                    Summary.start_timestamp >= t0,
                    Summary.start_timestamp < t1
                )
                print(f"--- [DEBUG] 應用日期過濾 (回退): {target_date} ({t0} ~ {t1}) ---")
    elif filters.get("date_filter"):
        # 向後兼容：使用單一日期過濾
        target_date = filters["date_filter"]
        # 計算時間範圍：從當天 00:00:00 到次日 00:00:00
        t0 = datetime.combine(target_date, datetime.min.time())
        next_day = target_date + timedelta(days=1)
        t1 = datetime.combine(next_day, datetime.min.time())
        query = query.filter(
            Summary.start_timestamp >= t0,
            Summary.start_timestamp < t1
        )
        print(f"--- [DEBUG] 應用日期過濾 (使用 start_timestamp range): {target_date} ({t0} ~ {t1}) ---")
    
    # [關鍵] 事件類型過濾 - 必須嚴格匹配事件 boolean 欄位
    if filters.get("event_types"):
        event_conditions = []
        for event_type in filters["event_types"]:
            if event_type == "fire":
                event_conditions.append(Summary.fire == True)
            elif event_type == "water_flood":
                event_conditions.append(Summary.water_flood == True)
            elif event_type == "abnormal_attire_face_cover_at_entry":
                event_conditions.append(Summary.abnormal_attire_face_cover_at_entry == True)
            elif event_type == "person_fallen_unmoving":
                event_conditions.append(Summary.person_fallen_unmoving == True)
            elif event_type == "double_parking_lane_block":
                event_conditions.append(Summary.double_parking_lane_block == True)
            elif event_type == "smoking_outside_zone":
                event_conditions.append(Summary.smoking_outside_zone == True)
            elif event_type == "crowd_loitering":
                event_conditions.append(Summary.crowd_loitering == True)
            elif event_type == "security_door_tamper":
                event_conditions.append(Summary.security_door_tamper == True)
        
        if event_conditions:
            query = query.filter(or_(*event_conditions))
            print(f"--- [DEBUG] 應用事件過濾: {filters['event_types']} ---")
    
    # [NEW] message 關鍵字過濾 - 在 SQL 查詢中過濾
    if filters.get("message_keywords"):
        message_conditions = []
        for keyword in filters["message_keywords"]:
            message_conditions.append(Summary.message.ilike(f"%{keyword}%"))
        if message_conditions:
            query = query.filter(or_(*message_conditions))
            print(f"--- [DEBUG] 應用 message 關鍵字過濾: {filters['message_keywords']} ---")
    
    # 執行查詢
    results = query.limit(limit).all()
    print(f"--- [DEBUG] PostgreSQL 查詢返回 {len(results)} 筆原始結果 ---")
    
    # 構建 Dict[segment_uid] -> sql_payload
    sql_results = {}
    for row in results:
        start_ts, video, segment, time_range, event_reason, message = row
        
        # 生成 segment_uid
        video_str = str(video) if video else ""
        segment_str = str(segment) if segment else ""
        time_range_str = str(time_range) if time_range else ""
        segment_uid = f"{video_str}|{segment_str}|{time_range_str}"
        
        # 構建 sql_payload
        sql_results[segment_uid] = {
            "start_timestamp": start_ts,
            "video": video_str,
            "segment": segment_str,
            "time_range": time_range_str,
            "event_reason": str(event_reason) if event_reason else "",
            "message": str(message) if message else ""
        }
    
    print(f"--- [DEBUG] 過濾後的有效結果: {len(sql_results)} 筆 ---")
    return sql_results


def _calculate_sql_score(
    sql_data: Dict[str, Any],
    query_filters: Dict[str, Any],
    message_keywords: List[str]
) -> float:
    """
    計算 SQL 結果的分數
    
    規則：
    - 符合時間：+0.4
    - 符合事件 boolean：+0.5
    - summary / message 命中關鍵字：+0.1
    """
    score = 0.0
    
    # 符合時間：+0.4（如果有日期過濾，已經在 SQL 查詢中過濾了，所以這裡直接給分）
    if query_filters.get("date_filter"):
        score += 0.4
    
    # 符合事件 boolean：+0.5
    # 注意：如果 query_filters 中有 event_types，且 sql_data 存在，
    # 表示這個記錄已經通過了 SQL 的事件 boolean 過濾，所以直接給分
    if query_filters.get("event_types"):
        score += 0.5
    
    # summary / message 命中關鍵字：根據匹配程度給予不同分數
    if message_keywords:
        message = sql_data.get("message", "") or ""
        event_reason = sql_data.get("event_reason", "") or ""
        combined_text = f"{message} {event_reason}".lower()
        
        matched_keywords = []
        for keyword in message_keywords:
            keyword_lower = keyword.lower()
            # 檢查關鍵字是否在文本中
            if keyword_lower in combined_text:
                matched_keywords.append(keyword)
                # 精確匹配（完整關鍵字）：+0.4
                score += 0.4
            # 檢查關鍵字的部分是否在文本中（例如「黃色衣服」中的「黃色」）
            elif len(keyword) >= 2:
                # 對於多字關鍵字，檢查每個字是否在文本中
                keyword_parts = [part for part in keyword_lower.split() if len(part) >= 2]
                matched_parts = [part for part in keyword_parts if part in combined_text]
                if matched_parts:
                    matched_keywords.append(keyword)
                    # 部分匹配：+0.1（每個匹配的部分）
                    score += 0.1 * len(matched_parts)
        
        # 如果匹配到多個關鍵字，給予額外分數
        if len(matched_keywords) > 1:
            score += 0.15 * (len(matched_keywords) - 1)  # 每多一個關鍵字 +0.15
        
        # 如果完全沒有匹配到任何關鍵字，但有關鍵字要求，大幅減分
        if len(matched_keywords) == 0 and len(message_keywords) > 0:
            score -= 0.2
    
    return score


def _merge_and_rank_results(
    vector_hits: List[Dict[str, Any]],
    sql_results: Dict[str, Dict[str, Any]],
    query_filters: Dict[str, Any],
    message_keywords: List[str]
) -> List[Dict[str, Any]]:
    """
    融合排序：合併 Vector 和 SQL 的結果
    
    保留三類結果：
    1. 只在 SQL 命中
    2. 只在 Vector 命中（但如果有日期過濾，則不保留）
    3. 兩邊都命中（加分）
    
    最終排序規則：
    final_score = 0.65 * vector_score + 0.35 * sql_score + 0.15 bonus if in_both
    """
    merged = {}
    has_date_filter = bool(query_filters.get("date_filter"))
    
    # 處理 Vector 結果
    for vector_hit in vector_hits:
        metadata = vector_hit.get("metadata", {})
        video = metadata.get("video", "")
        segment = metadata.get("segment", "")
        time_range = metadata.get("time_range", "")
        
        # 生成 segment_uid（與 SQL 一致）
        video_str = str(video) if video else ""
        segment_str = str(segment) if segment else ""
        time_range_str = str(time_range) if time_range else ""
        segment_uid = f"{video_str}|{segment_str}|{time_range_str}"
        
        vector_score = float(vector_hit.get("score", 0.0))
        
        # [NEW] 檢查 Vector 結果中是否包含關鍵字，如果有則提高分數
        if message_keywords:
            summary = str(metadata.get("summary", "")).lower()
            message = str(metadata.get("message", "")).lower()
            combined_text = f"{summary} {message}".lower()
            
            keyword_match_bonus = 0.0
            matched_count = 0
            
            for keyword in message_keywords:
                keyword_lower = keyword.lower()
                # 精確匹配（完整關鍵字）
                if keyword_lower in combined_text:
                    keyword_match_bonus += 0.3  # 每個精確匹配 +0.3
                    matched_count += 1
                # 部分匹配（例如「黃色衣服」中的「黃色」）
                elif len(keyword) >= 2:
                    keyword_parts = [part for part in keyword_lower.split() if len(part) >= 2]
                    matched_parts = [part for part in keyword_parts if part in combined_text]
                    if matched_parts:
                        keyword_match_bonus += 0.15 * len(matched_parts)  # 每個部分匹配 +0.15
                        matched_count += 1
            
            # 如果匹配到多個關鍵字，給予額外分數
            if matched_count > 1:
                keyword_match_bonus += 0.2 * (matched_count - 1)
            
            # 將關鍵字匹配分數加到 vector_score
            vector_score = min(1.0, vector_score + keyword_match_bonus)  # 限制在 1.0 以內
        
        # 檢查是否也在 SQL 結果中
        sql_data = sql_results.get(segment_uid)
        if sql_data:
            # 兩邊都命中：計算 SQL 分數並加分
            sql_score = _calculate_sql_score(sql_data, query_filters, message_keywords)
            final_score = 0.65 * vector_score + 0.35 * sql_score + 0.15  # bonus for in_both
            in_both = True
        else:
            # 只在 Vector 命中
            # [FIX] 如果有日期過濾，則不保留只在 Vector 命中的結果（因為它們不符合日期條件）
            if has_date_filter:
                continue  # 跳過不符合日期條件的 Vector 結果
            
            sql_score = 0.0
            final_score = 0.65 * vector_score + 0.35 * sql_score
            in_both = False
        
        merged[segment_uid] = {
            "segment_uid": segment_uid,
            "vector_score": vector_score,
            "sql_score": sql_score,
            "final_score": final_score,
            "in_both": in_both,
            "vector_hit": vector_hit,
            "sql_data": sql_data
        }
    
    # 處理只在 SQL 命中的結果
    for segment_uid, sql_data in sql_results.items():
        if segment_uid not in merged:
            # 只在 SQL 命中
            sql_score = _calculate_sql_score(sql_data, query_filters, message_keywords)
            final_score = 0.65 * 0.0 + 0.35 * sql_score  # vector_score = 0
            merged[segment_uid] = {
                "segment_uid": segment_uid,
                "vector_score": 0.0,
                "sql_score": sql_score,
                "final_score": final_score,
                "in_both": False,
                "vector_hit": None,
                "sql_data": sql_data
            }
    
    # 按 final_score 排序
    sorted_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)
    
    return sorted_results

# ================== 註冊 API 路由 ==================
# 必須在所有函數定義之後註冊，避免循環導入
from src.api import health, prompts, video_analysis, rag, video_management, detection_items

app.include_router(health.router)
app.include_router(prompts.router)
app.include_router(video_analysis.router)
app.include_router(rag.router)
app.include_router(video_management.router)
app.include_router(detection_items.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8080, reload=True)