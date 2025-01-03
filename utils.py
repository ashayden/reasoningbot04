"""Utility functions for the MARA application."""

import logging
from typing import Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

def validate_topic(topic: str) -> tuple[bool, str]:
    """Validate the input topic string.
    
    Args:
        topic: The input topic string to validate.
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not topic or not topic.strip():
        return False, "Topic cannot be empty."
    
    if len(topic) < MIN_TOPIC_LENGTH:
        return False, f"Topic must be at least {MIN_TOPIC_LENGTH} characters long."
    
    if len(topic) > MAX_TOPIC_LENGTH:
        return False, f"Topic must be no more than {MAX_TOPIC_LENGTH} characters long."
    
    return True, ""

def handle_error(func: Callable) -> Callable:
    """Decorator to handle errors consistently."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper 