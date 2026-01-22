# backend/src/core/model_loader.py
# 全域模型管理標準化：唯一真理來源
# 絕對禁止從 src.main 或 src.services import 任何東西（防止循環引用）
import os
import torch
import numpy as np
from ultralytics import YOLOWorld

# YOLO-World 模型
_yolo_world_model = None

# ReID 模型
_reid_model = None
_reid_device = None

# CLIP 模型
_clip_model = None
_clip_processor = None
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 輸出 512 維向量

def get_yolo_model():
    """
    獲取或初始化 YOLO-World 模型（帶健康檢查，防止殭屍模型）
    """
    global _yolo_world_model
    
    # 健康檢查：必須有 .model 且 .model 不能是 None
    if _yolo_world_model is not None:
        # 檢查是否有 model 屬性
        if hasattr(_yolo_world_model, 'model'):
            # 如果 model.model 為 None，代表是殭屍物件
            if _yolo_world_model.model is None:
                print("--- [ModelLoader] 檢測到殭屍 YOLO 模型（model.model 為 None），強制重載 ---")
                _yolo_world_model = None  # 強制重載
            else:
                try:
                    # 測試訪問 names 屬性，若噴錯代表是殭屍模型
                    _ = _yolo_world_model.model.names
                except (AttributeError, TypeError) as e:
                    print(f"--- [ModelLoader] 檢測到殭屍 YOLO 模型，強制重載: {e} ---")
                    _yolo_world_model = None  # 強制重載
        else:
            # 如果沒有 model 屬性，也視為殭屍模型
            print("--- [ModelLoader] 檢測到殭屍 YOLO 模型（缺少 model 屬性），強制重載 ---")
            _yolo_world_model = None  # 強制重載
    
    if _yolo_world_model is None:
        print("--- [ModelLoader] 載入 YOLO-World 模型... ---")
        local_path = '/app/models/yolov8s-world.pt'
        _yolo_world_model = YOLOWorld(local_path if os.path.exists(local_path) else 'yolov8s-world.pt')
        try:
            # 觸發初始化，確保模型完全載入
            _yolo_world_model.set_classes(["person"])
            print("--- [ModelLoader] ✓ YOLO-World 模型載入完成 ---")
        except Exception as e:
            print(f"--- [ModelLoader] ⚠️ YOLO-World 初始化警告: {e} ---")
            # 即使初始化失敗，也返回模型（可能後續會重試）
    
    return _yolo_world_model

def get_reid_model():
    """
    獲取或初始化 ReID 模型（強制 FP32，防止循環引用）
    優先嘗試 torchreid，失敗則 fallback 到 torchvision.models.resnet50
    """
    global _reid_model, _reid_device
    
    if _reid_model is None:
        print("--- [ModelLoader] 載入 ReID 模型... ---")
        _reid_device = "cuda" if torch.cuda.is_available() else "cpu"
        errors = []  # 記錄所有嘗試的錯誤
        
        # 方案 1: 優先使用 torchreid（FastReID 風格）
        try:
            import torchreid
            print(f"--- [ModelLoader] 嘗試載入 torchreid 模型 (device: {_reid_device}) ---")
            model = torchreid.models.build_model(
                name='resnet50',
                num_classes=751,  # Market-1501 dataset classes
                loss='softmax',
                pretrained=True
            )
            model = model.to(_reid_device).eval()
            # 移除分類層，只保留特徵提取器
            model.classifier = None
            
            # 驗證模型輸出維度（應該是 2048）
            test_input = torch.randn(1, 3, 256, 128).to(_reid_device)
            with torch.no_grad():
                test_output = model(test_input)
                output_dim = test_output.shape[1] if len(test_output.shape) > 1 else test_output.shape[0]
            
            if output_dim != 2048:
                error_msg = f"torchreid 模型輸出維度錯誤: 預期 2048，實際 {output_dim}"
                print(f"--- [ModelLoader] ✗ {error_msg} ---")
                errors.append(f"torchreid: {error_msg}")
                raise ValueError(error_msg)
            
            # 強制型別安全：確保模型權重為 FP32
            model = model.float()
            
            print(f"--- [ModelLoader] ✓ torchreid ResNet50 模型載入完成 (device: {_reid_device}, dtype: FP32, 輸出維度: {output_dim}) ---")
            _reid_model = model
            return _reid_model, _reid_device
            
        except ImportError as e:
            error_msg = f"torchreid 未安裝: {e}"
            print(f"--- [ModelLoader] {error_msg}，嘗試備用方案... ---")
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"torchreid 載入失敗: {e}"
            print(f"--- [ModelLoader] {error_msg}，嘗試備用方案... ---")
            errors.append(error_msg)
        
        # 方案 2: Fallback 到 torchvision.models.resnet50
        try:
            print(f"--- [ModelLoader] 嘗試載入 torchvision ResNet50 模型 (device: {_reid_device}) ---")
            import torchvision.models as models
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Identity()  # 移除分類層，只保留特徵提取器
            model = model.to(_reid_device).eval()
            
            # 強制型別安全：確保模型權重為 FP32
            model = model.float()
            
            print(f"--- [ModelLoader] ✓ torchvision ResNet50 模型載入完成 (device: {_reid_device}, dtype: FP32) ---")
            _reid_model = model
            return _reid_model, _reid_device
            
        except Exception as e:
            error_msg = f"torchvision ResNet50 載入失敗: {e}"
            print(f"--- [ModelLoader] ✗ {error_msg} ---")
            errors.append(error_msg)
            print(f"--- [ModelLoader] 所有 ReID 模型載入方案均失敗: {errors} ---")
            raise RuntimeError(f"無法載入 ReID 模型。嘗試的方案: {errors}") from e
    
    return _reid_model, _reid_device

def get_clip_model():
    """
    獲取或初始化 CLIP 模型（用於圖像 embedding）
    """
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"--- [ModelLoader] 載入 CLIP 模型: {CLIP_MODEL_NAME} (device: {device}) ---")
            
            # 直接使用模型名稱載入，transformers 會自動從緩存讀取（如果可用）
            # 如果緩存不完整，會嘗試從網路下載缺失的部分
            try:
                # 先嘗試只使用本地文件（如果緩存完整）
                print("--- [ModelLoader] 嘗試從本地緩存載入 CLIP 模型和處理器... ---")
                _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(device).eval()
                if device == "cuda":
                    _clip_model = _clip_model.half()  # 啟用半精度
                print("--- [ModelLoader] CLIP 模型載入完成，載入處理器... ---")
                _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
                print(f"--- [ModelLoader] ✓ CLIP 模型和處理器從本地緩存載入完成 (FP16: {device == 'cuda'}) ---")
            except Exception as local_e:
                # 如果本地緩存不完整，嘗試從網路下載（但可能因為網路問題失敗）
                print(f"--- [ModelLoader] 本地緩存不完整或載入失敗: {type(local_e).__name__}: {str(local_e)[:200]} ---")
                print("--- [ModelLoader] 嘗試從網路下載（如果網路可用，但可能很慢）... ---")
                try:
                    # 設置較短的超時，避免長時間等待
                    import requests
                    original_timeout = getattr(requests.adapters, 'DEFAULT_TIMEOUT', None)
                    if hasattr(requests.adapters, 'DEFAULT_TIMEOUT'):
                        requests.adapters.DEFAULT_TIMEOUT = 10  # 10 秒超時
                    
                    print("--- [ModelLoader] 從網路載入 CLIP 模型... ---")
                    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=False).to(device).eval()
                    print("--- [ModelLoader] CLIP 模型載入完成，從網路載入處理器... ---")
                    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=False)
                    print(f"--- [ModelLoader] ✓ CLIP 模型和處理器從網路載入完成 ---")
                    
                    # 恢復原始超時設置
                    if original_timeout is not None and hasattr(requests.adapters, 'DEFAULT_TIMEOUT'):
                        requests.adapters.DEFAULT_TIMEOUT = original_timeout
                except Exception as network_e:
                    print(f"--- [ModelLoader] ⚠️ CLIP 模型網路載入也失敗: {type(network_e).__name__}: {str(network_e)[:200]}")
                    # 不拋出異常，讓應用繼續運行，但模型為 None
                    _clip_model = None
                    _clip_processor = None
                    print("--- [ModelLoader] ⚠️ CLIP 功能將無法使用（無法從緩存或網路載入模型），但應用可以繼續運行 ---")
                    print("--- [ModelLoader] ⚠️ 建議：檢查網路連線或確保模型緩存完整 ---")
        except ImportError:
            print("--- [ModelLoader] ⚠️ transformers 未安裝，CLIP 功能將無法使用 ---")
            _clip_model = None
            _clip_processor = None
        except Exception as e:
            print(f"--- [ModelLoader] ⚠️ Failed to load CLIP model: {e} ---")
            import traceback
            traceback.print_exc()
            _clip_model = None
            _clip_processor = None
            print("--- [ModelLoader] ⚠️ CLIP 功能將無法使用，但應用可以繼續運行 ---")
    
    return _clip_model, _clip_processor
