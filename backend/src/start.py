#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start script for backend API using uvicorn
"""
import multiprocessing
import uvicorn
from src.config import config

if __name__ == "__main__":
    # PyTorch/CUDA 與 fork 不相容：fork 後的子進程無法正確初始化 CUDA，會導致 worker 崩潰成殭屍。
    # 必須使用 spawn 讓每個 worker 以全新進程啟動，才能正常載入 GPU 模型。
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 已設定過則略過

    # VRAM 提醒：Qwen2.5-VL-7B (FP16) ~15GB，YOLO+ReID ~2GB/process。WORKERS=4 約需 68GB，請先以 WORKERS=1 測試。
    if config.WORKERS > 1:
        print(f"[start] 注意: WORKERS={config.WORKERS}，每個 worker 會各載一份 Qwen+YOLO+ReID，VRAM 約需 {(15 + 2) * config.WORKERS}GB。建議先設 WORKERS=1 測試。")
    # 啟動時印出實際 worker 數，方便對照 nvidia-smi（Uvicorn 會產生 1 個 master + N 個 worker process，僅 worker 載模型；若看到 2 個 python 都佔大量 VRAM = 實際為 2 workers）
    import os as _os
    print(f"[start] Uvicorn workers={config.WORKERS} (env WORKERS={_os.environ.get('WORKERS', '<unset>')})", flush=True)
    print(f"[start] 本 process PID={_os.getpid()}（此為 master；worker 會是別的 PID。若 nvidia-smi 有 2 個 python 且都佔 ~15GB+，代表實際 workers≥2）", flush=True)

    # Run the application using configuration from config.py
    uvicorn.run(
        "src.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        workers=config.WORKERS if not config.RELOAD else 1,  # reload mode doesn't support multiple workers
        log_level=config.LOG_LEVEL,
        access_log=True,
    )

