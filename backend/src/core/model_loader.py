# backend/src/core/model_loader.py
# 全域模型管理標準化：唯一真理來源（Singleton，每 process 僅載入一次）
# 絕對禁止從 src.main 或 src.services import 任何東西（防止循環引用）
# 注意：僅 Hugging Face 系模型（如 Qwen）使用 device_map；YOLO/ReID 為一般 PyTorch/Ultralytics，不使用 device_map，載入後 .to(device) 為正確用法。
import os
import threading
import time
from contextlib import contextmanager
import torch
import numpy as np
from ultralytics import YOLOWorld

# 載入鎖：確保高併發下只有一個 thread 執行載入，避免重複載入導致記憶體崩潰
_yolo_load_lock = threading.Lock()
_reid_load_lock = threading.Lock()

# 推理鎖：YOLO/ReID 模型非 thread-safe，多 thread 並行 predict 會導致設備不一致、模型狀態損壞、CUDA illegal memory access。
# run_full_pipeline 的 ThreadPoolExecutor 會有多 thread 同時呼叫 infer_segment_yolo，必須串行化 GPU 推理。
_gpu_inference_lock = threading.Lock()


@contextmanager
def gpu_inference_lock_traced():
    """
    取得 YOLO/ReID 共用的 _gpu_inference_lock，並印出等待時間。
    多請求同時進入時等待會變長，可用來對照 500/逾時是否為 GPU 鎖排隊或 CPU 解碼過載。
    """
    t0 = time.time()
    th = threading.current_thread().name
    ident = threading.get_ident()
    _gpu_inference_lock.acquire()
    try:
        wait_sec = time.time() - t0
        print(
            f"--- [GPU_LOCK] YOLO/ReID 已取得鎖 | 等待 {wait_sec:.3f}s | thread={th} ident={ident} ---",
            flush=True,
        )
        yield
    finally:
        _gpu_inference_lock.release()


# YOLO-World 模型（單例）
_yolo_world_model = None

# ReID 模型（單例）
_reid_model = None
_reid_device = None

# CLIP 模型
_clip_model = None
_clip_processor = None
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 輸出 512 維向量

def get_yolo_model():
    """
    獲取 YOLO-World 模型單例。僅在首次呼叫時載入一次，後續請求共用同一實例。
    嚴禁在請求處理內重複 load；載入由鎖保護，高併發下不會重複載入。
    """
    global _yolo_world_model
    # 快速路徑：已有有效實例則直接回傳（無需取鎖）
    if _yolo_world_model is not None:
        if hasattr(_yolo_world_model, "model") and _yolo_world_model.model is not None:
            try:
                _ = _yolo_world_model.model.names
                return _yolo_world_model
            except (AttributeError, TypeError):
                pass
        # 殭屍模型，清空後需重載（進入下方取鎖區）
        _yolo_world_model = None

    with _yolo_load_lock:
        # 雙重檢查：可能其他 thread 已載入
        if _yolo_world_model is not None:
            try:
                _ = _yolo_world_model.model.names
                return _yolo_world_model
            except (AttributeError, TypeError):
                _yolo_world_model = None
        print("--- [ModelLoader] 載入 YOLO-World 模型（單例，僅此一次）... ---")
        # YOLO-World 為 Ultralytics 模型，不使用 device_map；載入後由 Ultralytics 內部管理設備，無 meta tensor 衝突。
        yolo_path = os.getenv("YOLO_WEIGHTS_PATH", "").strip()
        if yolo_path and os.path.exists(yolo_path):
            model_path = yolo_path
            print(f"--- [ModelLoader] 使用 YOLO_WEIGHTS_PATH: {model_path} ---")
        elif os.path.exists("/app/models/yolov8s-world.pt"):
            model_path = "/app/models/yolov8s-world.pt"
        else:
            model_path = "yolov8s-world.pt"
            print("--- [ModelLoader] 未找到本地權重，將嘗試下載（離線請設定 YOLO_WEIGHTS_PATH）---")
        _yolo_world_model = YOLOWorld(model_path)
        try:
            _yolo_world_model.set_classes(["person"])
            print("--- [ModelLoader] ✓ YOLO-World 模型載入完成 ---")
        except Exception as e:
            print(f"--- [ModelLoader] ⚠️ YOLO-World 初始化警告: {e} ---")
    return _yolo_world_model

def get_reid_model():
    """
    獲取 ReID 模型單例。僅在首次呼叫時載入一次，後續請求共用同一實例。
    """
    global _reid_model, _reid_device
    if _reid_model is not None:
        return _reid_model, _reid_device

    with _reid_load_lock:
        if _reid_model is not None:
            return _reid_model, _reid_device
        print("--- [ModelLoader] 載入 ReID 模型（單例，僅此一次）... ---")
        _reid_device = "cuda" if torch.cuda.is_available() else "cpu"
        errors = []  # 記錄所有嘗試的錯誤
        
        # 方案 1: 優先使用 torchreid（FastReID 風格）。不在本模組使用 device_map，故載入後 .to(device) 正確。
        # 在 CPU 上建立模型再 .to(device)，避免與同 process 內其他 meta tensor 狀態衝突導致「Cannot copy out of meta tensor」。
        try:
            import torchreid
            print(f"--- [ModelLoader] 嘗試載入 torchreid 模型 (device: {_reid_device}) ---")
            with torch.device("cpu"):
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
        
        # 方案 2: Fallback 到 torchvision ResNet50。
        # 強制在 CPU 上建立模型（with torch.device("cpu")），再 .to(cuda)，避免同 process 內 meta 污染導致「Cannot copy out of meta tensor」。
        # 推論在 GPU；僅建立階段短暫用 CPU。
        try:
            print(f"--- [ModelLoader] 嘗試載入 torchvision ResNet50 模型 (device: {_reid_device}) ---")
            import torchvision.models as models
            with torch.device("cpu"):
                if hasattr(models, "ResNet50_Weights"):
                    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                else:
                    model = models.resnet50(pretrained=True)
                model.fc = torch.nn.Identity()
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
            
            # 僅從本地緩存載入，不連線 Huggingface，避免離線/容器內 DNS 失敗導致重試與卡住
            try:
                print("--- [ModelLoader] 從本地緩存載入 CLIP 模型和處理器 (local_files_only)... ---")
                _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(device).eval()
                if device == "cuda":
                    _clip_model = _clip_model.half()  # 啟用半精度
                _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
                print(f"--- [ModelLoader] ✓ CLIP 從本地緩存載入完成 (FP16: {device == 'cuda'}) ---")
            except Exception as local_e:
                _clip_model = None
                _clip_processor = None
                print(f"--- [ModelLoader] CLIP 本地緩存不可用，略過（不連線 Huggingface）: {type(local_e).__name__} ---")
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
