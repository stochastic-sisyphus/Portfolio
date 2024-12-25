import logging
import traceback
import sys

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = 'pipeline.log', log_level: int = logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def handle_error(error: Exception, step: str = 'Unknown') -> None:
    """Handle and log errors with more context."""
    error_type = type(error).__name__
    error_message = str(error)
    stack_trace = traceback.format_exc()
    
    logger.error(f"Error in step: {step}")
    logger.error(f"Error Type: {error_type}")
    logger.error(f"Error Message: {error_message}")
    logger.error(f"Stack Trace:\n{stack_trace}")

    # You can add custom error handling logic here, such as sending notifications or writing to a database

def log_step(step: str, message: str):
    """Log the start and end of each pipeline step."""
    logger.info(f"Starting step: {step}")
    logger.info(message)
    logger.info(f"Completed step: {step}")
