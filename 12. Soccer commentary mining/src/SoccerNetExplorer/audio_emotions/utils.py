import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def configure_logger(task_name: str, log_pth: Path):
    """Configure logger."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_pth / f"{task_name}" / f"log_{timestamp}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level="INFO",
    )
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level="INFO",
    )
