import time
from utils.logger_config import setup_logger
from pathlib import Path

logger = setup_logger(__name__)

def housekeeping(output_dir, housekeeping_days):
    """Clean up old video files based on housekeeping configuration."""

    now = time.time()
    cutoff_time = now - (housekeeping_days * 24 * 60 * 60)

    logger.info(f"Starting housekeeping - removing files older than {housekeeping_days} days from {output_dir}")

    deleted_count = 0
    for file in Path(output_dir).glob("*.avi"):
        if file.stat().st_mtime < cutoff_time:
            logger.info(f"Deleting old file: {file}")
            file.unlink()
            deleted_count += 1

    logger.info(f"Housekeeping completed - deleted {deleted_count} files")