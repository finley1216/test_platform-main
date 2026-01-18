# -*- coding: utf-8 -*-
import os
import torch
from typing import Optional, List, Tuple
from src.config import config

# --- SentenceTransformer ---
_embedding_model = None
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_model():
    """Get or initialize the SentenceTransformer model (CPU mode only)"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # 強制使用 CPU 模式，避免 GPU 資源競爭
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 隱藏 GPU，強制使用 CPU
            # 嘗試使用本地緩存路徑
            local_model_path = "/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "modules.json")):
                _embedding_model = SentenceTransformer(local_model_path, device='cpu')
            else:
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
            print(f"✓ SentenceTransformer model loaded: {EMBEDDING_MODEL_NAME} (CPU Mode)")
        except Exception as e:
            print(f"⚠️  Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"無法載入 SentenceTransformer 模型: {e}")
    return _embedding_model

# --- CLIP ---
_clip_model = None
_clip_processor = None
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def get_clip_model():
    """獲取或初始化 CLIP 模型"""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"--- [CLIP] 載入模型: {CLIP_MODEL_NAME} (device: {device}) ---")
            
            try:
                _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(device).eval()
                _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
                print(f"✓ CLIP 模型和處理器從本地緩存載入完成")
            except Exception:
                print("--- [CLIP] 嘗試從網路下載... ---")
                _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=False).to(device).eval()
                _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=False)
                print(f"✓ CLIP 模型和處理器從網路載入完成")
        except Exception as e:
            print(f"⚠️  Failed to load CLIP model: {e}")
            _clip_model, _clip_processor = None, None
    return _clip_model, _clip_processor

# --- ReID ---
# 統一從 src.main 獲取以保持一致性
def get_reid_model():
    """獲取或載入 ReID 模型（與 main.py 保持一致）"""
    from src.main import get_reid_model as _get_reid
    return _get_reid()

# --- YOLO-World ---
_yolo_world_model = None

def get_yolo_model():
    """獲取或初始化 YOLO-World 模型"""
    global _yolo_world_model
    if _yolo_world_model is None:
        try:
            from ultralytics import YOLOWorld
            local_model_path = '/app/models/yolov8s-world.pt'
            if os.path.exists(local_model_path):
                print(f"--- [YOLO] 載入本地模型: {local_model_path} ---")
                _yolo_world_model = YOLOWorld(local_model_path)
            else:
                print("--- [YOLO] 本地模型不存在，下載預訓練模型... ---")
                _yolo_world_model = YOLOWorld('yolov8s-world.pt')
            print("✓ YOLO-World 模型載入完成")
        except Exception as e:
            print(f"⚠️  Failed to load YOLO model: {e}")
            _yolo_world_model = None
    return _yolo_world_model
