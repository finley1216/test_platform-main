#!/usr/bin/env python3
"""
Centralized configuration loader utility for RTSP Recorder.
Handles both YAML and JSON configuration files with environment variable substitution.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import logging
from dotenv import load_dotenv

# Load environment variables from .env file, overriding existing ones
load_dotenv(override=True)
logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file with environment variable substitution.

    Args:
        config_path: Path to the configuration file (YAML or JSON)

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the configuration file is not found
        yaml.YAMLError: If there's an error parsing YAML
        json.JSONDecodeError: If there's an error parsing JSON
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        # Read the file content
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace environment variables
        content = os.path.expandvars(content)

        # Parse based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(content)
        elif config_path.suffix.lower() == '.json':
            config = json.loads(content)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        logger.info(f"Configuration loaded successfully from: {config_path}")
        return config

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration from {config_path}: {e}")
        raise


def _infer_rtsp_root_from_file() -> str:
    """依本檔位置推斷 rtsp-recorder 根目錄（config_loader 在 src/utils/ → 往上兩層到 rtsp-recorder）。"""
    _here = Path(__file__).resolve().parent  # .../rtsp-recorder/src/utils
    return str(_here.parent.parent)  # .../rtsp-recorder


def get_project_root() -> str:
    """
    Get the PROJECT_ROOT. 僅當該路徑下 config 檔存在時才用環境變數，
    否則依本檔位置推斷（避免 .env 或 Docker 設 PROJECT_ROOT=/app 時找不到 config）。
    若使用推斷路徑，會寫回 os.environ 讓 YAML 內 ${PROJECT_ROOT} 展開正確。
    """
    env_root = os.getenv("PROJECT_ROOT", "").strip()
    if env_root:
        if (Path(env_root) / "config" / "recorder-config.yaml").exists():
            return env_root
    inferred = _infer_rtsp_root_from_file()
    os.environ["PROJECT_ROOT"] = inferred
    return inferred


def get_config_path(config_name: str = "recorder-config.yaml") -> Path:
    """
    Get the full path to a configuration file in the config directory.

    Args:
        config_name: Name of the configuration file (defaults to recorder-config.yaml)

    Returns:
        Full path to the configuration file
    """
    project_root = get_project_root()
    return Path(project_root) / "config" / config_name


def get_recorder_config_path() -> Path:
    """
    Get the recorder configuration file path (recorder-config.yaml).

    Returns:
        Full path to the recorder config.yaml file
    """
    return get_config_path("recorder-config.yaml")


def get_stream_config_path() -> Path:
    """
    Get the stream configuration file path (stream-config.yaml).

    Returns:
        Full path to the stream config.yaml file
    """
    return get_config_path("stream-config.yaml")

