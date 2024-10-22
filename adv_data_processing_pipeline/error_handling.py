import logging
import traceback

logger = logging.getLogger(__name__)

def handle_error(error: Exception) -> None:
    """Handle and log errors."""
    error_type = type(error).__name__
    error_message = str(error)
    stack_trace = traceback.format_exc()
    
    logger.error(f"Error Type: {error_type}")
    logger.error(f"Error Message: {error_message}")
    logger.error(f"Stack Trace:\n{stack_trace}")

