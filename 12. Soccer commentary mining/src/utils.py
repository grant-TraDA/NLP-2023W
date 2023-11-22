from datetime import datetime
from loguru import logger
import sys


def configure_logger(task_name: str):
    """Configure logger."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./../.logs/{task_name}/log_{timestamp}.log"
    logger.add(log_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
               level="INFO")
    logger.add(sys.stdout, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
               level="INFO")
