"""Utility functions for the MARA application."""

import logging
from typing import Any, Callable, Tuple
from functools import wraps
from config import config

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MARAError(Exception):
    """Base exception class for MARA application."""
    pass

class ValidationError(MARAError):
    """Raised when input validation fails."""
    pass

class ProcessingError(MARAError):
    """Raised when processing operations fail."""
    pass

def validate_topic(topic: str) -> Tuple[bool, str]:
    """Validate the input topic string.
    
    Args:
        topic: The input topic string to validate.
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
        
    Raises:
        ValidationError: If the topic is invalid.
    """
    logger.debug(f"Validating topic: {topic}")
    
    if not topic or not topic.strip():
        msg = "Topic cannot be empty."
        logger.warning(msg)
        return False, msg
    
    if len(topic) < config.MIN_TOPIC_LENGTH:
        msg = f"Topic must be at least {config.MIN_TOPIC_LENGTH} characters long."
        logger.warning(msg)
        return False, msg
    
    if len(topic) > config.MAX_TOPIC_LENGTH:
        msg = f"Topic must be no more than {config.MAX_TOPIC_LENGTH} characters long."
        logger.warning(msg)
        return False, msg
    
    logger.debug("Topic validation successful")
    return True, ""

def sanitize_topic(topic: str) -> str:
    """Sanitize the topic string for safe use in prompts.
    
    Args:
        topic: The raw topic string.
        
    Returns:
        Sanitized topic string with only alphanumeric, space, and basic punctuation.
        
    Raises:
        ValidationError: If the topic cannot be sanitized.
    """
    logger.debug(f"Sanitizing topic: {topic}")
    
    if not isinstance(topic, str):
        raise ValidationError(f"Expected string, got {type(topic)}")
    
    # Remove any leading/trailing whitespace
    topic = topic.strip()
    
    # Keep only alphanumeric, spaces, and basic punctuation
    sanitized = ''.join(c for c in topic if c.isalnum() or c.isspace() or c in '.,!?-_()[]{}')
    
    # Normalize whitespace
    sanitized = ' '.join(sanitized.split())
    
    if not sanitized:
        raise ValidationError("Topic became empty after sanitization")
    
    logger.debug(f"Sanitized topic: {sanitized}")
    return sanitized

def handle_error(func: Callable) -> Callable:
    """Decorator to handle errors consistently.
    
    Args:
        func: The function to wrap.
        
    Returns:
        Wrapped function with error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except MARAError as e:
            # Log and re-raise application-specific errors
            logger.error(f"Application error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            # Log and wrap unexpected errors
            logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
            raise ProcessingError(f"Operation failed: {str(e)}") from e
    return wrapper 