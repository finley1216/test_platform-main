#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for backend API
Centralizes all environment variable handling
"""
import os
from pathlib import Path
from typing import Optional

class Config:
    """Application configuration from environment variables"""

    def __init__(self):
        """Initialize configuration from environment variables"""
        # ================== Authentication & Security ==================
        self.ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "admin-dev")
        self.SESSION_TTL_SEC: int = int(os.getenv("SESSION_TTL_SEC", "604800"))  # 7 days
        self.SERVER_API_KEY: str = os.getenv("MY_API_KEY", "123")
        self.API_KEY_NAME: str = "X-API-Key"

        # ================== API Configuration ==================
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8080"))
        self.RELOAD: bool = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")
        self.WORKERS: int = int(os.getenv("WORKERS", "1"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
        self.PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", os.getcwd())

        # ================== Ollama Configuration ==================
        self.OLLAMA_BASE: str = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
        self.OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        self.OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5vl:latest")
        self.OLLAMA_REQUEST_TIMEOUT: int = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))  # VLM 運算較久時避免連線中斷（秒）
        # Qwen HF 本機載入參數（對應 LlamaCpp 的 n_ctx / n_batch / n_gpu_layers）
        self.QWEN_MAX_NEW_TOKENS: int = int(os.getenv("QWEN_MAX_NEW_TOKENS", "512"))   # 等同 n_ctx 的生成長度上限
        self.QWEN_HF_MAX_INFERENCE_BATCH_SIZE: int = int(os.getenv("QWEN_HF_MAX_INFERENCE_BATCH_SIZE", "4"))  # 等同 n_batch：VLM 單次 forward 段數上限
        self.QWEN_N_GPU_LAYERS: str = os.getenv("QWEN_N_GPU_LAYERS", "-1")  # -1=全 GPU (auto)，0=僅 CPU
        # RAG 回答優化配置
        self.RAG_ANSWER_TOP_K: int = int(os.getenv("RAG_ANSWER_TOP_K", "5"))  # 生成回答時使用的片段數量（預設5個以確保完整性）
        self.RAG_ANSWER_SUMMARY_MAX_LEN: int = int(os.getenv("RAG_ANSWER_SUMMARY_MAX_LEN", "500"))  # 每個摘要的最大長度（增加到500以保留更多資訊）

        # ================== Gemini API Configuration ==================
        self.GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

        # ================== vLLM Configuration ==================
        self.VLLM_BASE: str = os.getenv("VLLM_BASE", "http://127.0.0.1:8440")
        # Qwen3 專用 vLLM endpoint（若未設定則回退 VLLM_BASE）
        self.QWEN3_VLLM_BASE: str = os.getenv("QWEN3_VLLM_BASE", "http://host.docker.internal:8441")
        # Qwen3-AWQ 專用 vLLM endpoint（若未設定則回退 QWEN3_VLLM_BASE 再回退 VLLM_BASE）
        self.QWEN3_AWQ_VLLM_BASE: str = os.getenv("QWEN3_AWQ_VLLM_BASE", "").strip()
        self.VLLM_API_KEY: Optional[str] = os.getenv("VLLM_API_KEY") or None
        self.VLLM_REQUEST_TIMEOUT: int = int(os.getenv("VLLM_REQUEST_TIMEOUT", "600"))
        # 送 vLLM 的 JPEG 壓縮品質（降低 payload、縮短 api_total_time）
        self.VLLM_JPEG_QUALITY: int = int(os.getenv("VLLM_JPEG_QUALITY", "70"))
        # 非同步批次時同時在飛的 HTTP 數上限（應 ≤ vLLM --max-num-seqs）
        self.VLLM_BATCH_MAX_IN_FLIGHT: int = int(os.getenv("VLLM_BATCH_MAX_IN_FLIGHT", "16"))
        # 前端切換 VLM 時可選：掛載 docker.sock + 設定 compose 檔路徑後，由後端執行 stop/start（具 root 等價風險，僅建議內網）
        self.VLM_ORCHESTRATION_ENABLED: bool = os.getenv("VLM_ORCHESTRATION_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        self.VLM_COMPOSE_FILE: str = os.getenv("VLM_COMPOSE_FILE", "").strip()
        self.VLM_COMPOSE_PROJECT_DIR: str = os.getenv("VLM_COMPOSE_PROJECT_DIR", "").strip()
        # backend 在容器內執行 compose 時，若需讓 daemon 以主機路徑解析 bind mount，可指定 host 專用 project dir
        self.VLM_COMPOSE_PROJECT_DIR_HOST: str = os.getenv("VLM_COMPOSE_PROJECT_DIR_HOST", "").strip()
        # 須與主機 docker compose 專案名一致；否則 --project-directory 掛在 /vlm-compose-host 時預設名會變成 vlm-compose-host，stop/up 找不到容器
        self.VLM_COMPOSE_PROJECT_NAME: str = os.getenv("VLM_COMPOSE_PROJECT_NAME", "").strip()
        # 選 vLLM profile 時是否一併 docker stop ollama（單卡時釋放 VRAM；若需同時用 RAG「回答」依賴 Ollama LLM 請設 false）
        self.VLM_ORCHESTRATION_STOP_OLLAMA_ON_VLLM: bool = os.getenv(
            "VLM_ORCHESTRATION_STOP_OLLAMA_ON_VLLM", "false"
        ).lower() in ("true", "1", "yes")

        # ================== Database Configuration ==================
        # PostgreSQL connection URL
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db_name = os.getenv("POSTGRES_DB", "postgres")
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL",
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        # ================== RAG Configuration ==================
        self.AUTO_RAG_INDEX: bool = os.getenv("AUTO_RAG_INDEX", "true").lower() in ("true", "1", "yes")
        self.RAG_DIR: Path = Path(os.getenv("RAG_DIR", "./rag_store"))
        self.RAG_STORE_DIR: str = os.getenv("RAG_STORE_DIR", "./rag_store")
        self.RAG_INDEX_PATH: Path = self.RAG_DIR / "index.json"
        
        # ================== ReID Configuration ==================
        # 是否允許在 ReID 模型不可用時回退到 CLIP（預設為 false，強制使用 ReID）
        self.ALLOW_REID_FALLBACK_TO_CLIP: bool = os.getenv("ALLOW_REID_FALLBACK_TO_CLIP", "false").lower() in ("true", "1", "yes")

        # ================== Startup / Preload ==================
        # 是否在啟動時預載入 CLIP（以圖搜圖）。關閉可避免離線環境連線 Huggingface 導致 DNS 重試與卡住
        self.PRELOAD_CLIP: bool = os.getenv("PRELOAD_CLIP", "false").lower() in ("true", "1", "yes")
        # 是否在啟動時預載入 YOLO+ReID（segment_pipeline 用）。關閉可省 GPU，改為首次請求時載入
        self.PRELOAD_YOLO_REID: bool = os.getenv("PRELOAD_YOLO_REID", "true").lower() in ("true", "1", "yes")
        # 是否在啟動時預載入 SentenceTransformer（RAG 用）。關閉可省 GPU
        self.PRELOAD_SENTENCE_TRANSFORMER: bool = os.getenv("PRELOAD_SENTENCE_TRANSFORMER", "true").lower() in ("true", "1", "yes")
        # segment_pipeline 完成後是否刪除 yolo_output/object_crops（預設 false：保留圖檔供以圖搜圖/預覽）
        self.CLEANUP_YOLO_CROPS: bool = os.getenv("CLEANUP_YOLO_CROPS", "false").lower() in ("true", "1", "yes")

        # ================== File Paths ==================
        self.SEGMENT_DIR: Path = Path(os.getenv("SEGMENT_DIR", "./segment"))
        self.PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", os.getcwd()))
        # Video library directory (歷史影片分類存放位置)
        video_lib_path = os.getenv("VIDEO_LIB_DIR")
        if video_lib_path:
            self.VIDEO_LIB_DIR: Path = Path(video_lib_path)
        else:
            # 預設為容器內的 /app/video（對應主機的 ../video）
            self.VIDEO_LIB_DIR: Path = Path("/app/video")

        # Ensure directories exist
        self.RAG_DIR.mkdir(parents=True, exist_ok=True)
        self.SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Configure Gemini if API key is provided
        if self.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.GEMINI_API_KEY)
            except ImportError:
                pass

    def validate(self) -> bool:
        """Validate configuration values"""
        errors = []

        if not self.SERVER_API_KEY or self.SERVER_API_KEY == "123":
            errors.append("Warning: Using default API key. Set MY_API_KEY for production.")

        if self.AUTO_RAG_INDEX and not self.OLLAMA_BASE:
            errors.append("Warning: AUTO_RAG_INDEX is enabled but OLLAMA_BASE is not set.")

        if errors:
            for error in errors:
                print(f"[Config] {error}")

        return len(errors) == 0

    def __repr__(self) -> str:
        """String representation of configuration (hides sensitive data)"""
        return (
            f"Config("
            f"host={self.HOST}, "
            f"port={self.PORT}, "
            f"reload={self.RELOAD}, "
            f"workers={self.WORKERS}, "
            f"ollama_base={self.OLLAMA_BASE}, "
            f"rag_dir={self.RAG_DIR}, "
            f"auto_rag_index={self.AUTO_RAG_INDEX}"
            f")"
        )


# Create global config instance
config = Config()

# Validate on import
config.validate()

