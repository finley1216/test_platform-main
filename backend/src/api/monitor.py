# -*- coding: utf-8 -*-
from fastapi import APIRouter
import psutil
import os
import torch
from typing import Dict, Any, Optional

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

from src.api.video_analysis import get_active_requests

router = APIRouter(tags=["系統監控"])

def get_gpu_status() -> Optional[Dict[str, Any]]:
    """獲取 GPU 使用狀況"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = []
    if HAS_NVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                gpu_info.append({
                    "id": i,
                    "name": name if isinstance(name, str) else name.decode('utf-8'),
                    "mem_total_mb": round(mem_info.total / (1024 * 1024), 2),
                    "mem_used_mb": round(mem_info.used / (1024 * 1024), 2),
                    "mem_free_mb": round(mem_info.free / (1024 * 1024), 2),
                    "gpu_util_percent": util.gpu,
                    "mem_util_percent": util.memory
                })
            pynvml.nvmlShutdown()
        except Exception as e:
            return {"error": f"NVML Error: {str(e)}"}
    else:
        # 備用方案：使用 torch 內建功能（無法獲取 GPU 使用率，只能獲取記憶體）
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_allocated = torch.cuda.memory_allocated(i)
            mem_reserved = torch.cuda.memory_reserved(i)
            mem_total = props.total_memory
            
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "mem_total_mb": round(mem_total / (1024 * 1024), 2),
                "mem_used_mb": round(mem_allocated / (1024 * 1024), 2),
                "mem_reserved_mb": round(mem_reserved / (1024 * 1024), 2),
                "gpu_util_percent": None,  # torch 無法獲取 GPU 使用率，需要 nvidia-smi
                "mem_util_percent": round((mem_allocated / mem_total) * 100, 2) if mem_total > 0 else 0
            })
            
    return {"devices": gpu_info}

@router.get("/v1/system/status")
def get_system_status():
    """獲取系統詳細健康狀況"""
    # CPU/Memory 用量 (Point 1, 10)
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # 模型預載狀態 (Point 1) - 使用 model_loader 獲取模型狀態
    try:
        from src.core.model_loader import get_yolo_model, get_reid_model
        _yolo_world_model = get_yolo_model()
        _reid_model, _ = get_reid_model()
    except (ImportError, AttributeError):
        _yolo_world_model = None
        _reid_model = None
    
    model_status = {
        "yolo_world": _yolo_world_model is not None,
        "reid_model": _reid_model is not None
    }
    
    # 殭屍請求偵測 (Point 7)
    active_requests = get_active_requests()
    
    # GPU 用量 (Point 1)
    gpu_status = get_gpu_status()
    
    return {
        "timestamp": os.getpid(), # 借用 PID 標識進程
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count()
        },
        "memory": {
            "total_mb": round(mem.total / (1024 * 1024), 2),
            "used_mb": round(mem.used / (1024 * 1024), 2),
            "available_mb": round(mem.available / (1024 * 1024), 2),
            "percent": mem.percent
        },
        "gpu": gpu_status,
        "models": model_status,
        "active_requests": active_requests,
        "disk": {
            "percent": disk.percent,
            "free_gb": round(disk.free / (1024**3), 2)
        }
    }
