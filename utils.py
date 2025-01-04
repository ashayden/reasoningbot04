"""Utility functions for the MARA application."""

import logging
from typing import Any, Callable, Tuple
from functools import wraps
from config import MIN_TOPIC_LENGTH, MAX_TOPIC_LENGTH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MARAError(Exception):
    """Base exception class for MARA application."""
    pass

class QuotaExceededError(MARAError):
    """Raised when API quota is exhausted."""
    pass

def rate_limit_decorator(func: Callable) -> Callable:
    """Simple decorator to handle API errors."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):
                raise QuotaExceededError("API quota exceeded. Please wait 5 minutes before trying again.")
            raise
    return wrapper

def validate_topic(topic: str) -> Tuple[bool, str]:
    """Validate the input topic string."""
    if not topic:
        return False, "Topic cannot be empty."
    
    sanitized = topic.strip()
    if not sanitized:
        return False, "Topic cannot be empty or contain only whitespace."
    
    if len(sanitized) < MIN_TOPIC_LENGTH:
        return False, f"Topic must be at least {MIN_TOPIC_LENGTH} characters long."
    
    if len(sanitized) > MAX_TOPIC_LENGTH:
        return False, f"Topic must be no more than {MAX_TOPIC_LENGTH} characters long."
    
    return True, ""

def sanitize_topic(topic: str) -> str:
    """Clean and sanitize the input topic."""
    if not topic:
        return ""
    
    # Remove leading/trailing whitespace
    sanitized = topic.strip()
    
    # Log the sanitized topic for debugging
    logger.info(f"Sanitized topic: {sanitized}")
    
    return sanitized 