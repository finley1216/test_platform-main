#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start script for backend API using uvicorn
"""
import uvicorn
from src.config import config

if __name__ == "__main__":
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

