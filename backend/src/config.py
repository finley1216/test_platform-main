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

        # ================== OWL API Configuration ==================
        self.OWL_API_BASE: str = os.getenv("OWL_API_BASE", "http://127.0.0.1:18001")
        self.OWL_VIDEO_URL: str = f"{self.OWL_API_BASE}/video_detect"

        # ================== Gemini API Configuration ==================
        self.GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

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
            f"owl_api_base={self.OWL_API_BASE}, "
            f"rag_dir={self.RAG_DIR}, "
            f"auto_rag_index={self.AUTO_RAG_INDEX}"
            f")"
        )


# Create global config instance
config = Config()

# Validate on import
config.validate()

