"""Utility functions for the MARA application."""

import logging
import time
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

class ValidationError(MARAError):
    """Raised when input validation fails."""
    pass

def rate_limit_decorator(func: Callable) -> Callable:
    """Simple decorator to handle rate limiting."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            # Add small delay between calls
            time.sleep(0.5)
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):
                raise QuotaExceededError("API quota exceeded. Please try again in 5 minutes.")
            raise
    return wrapper

def validate_topic(topic: str) -> Tuple[bool, str]:
    """Validate the input topic string."""
    if not topic or len(topic.strip()) == 0:
        return False, "Topic cannot be empty."
    
    if len(topic) < MIN_TOPIC_LENGTH:
        return False, f"Topic must be at least {MIN_TOPIC_LENGTH} characters long."
    
    if len(topic) > MAX_TOPIC_LENGTH:
        return False, f"Topic must be no more than {MAX_TOPIC_LENGTH} characters long."
    
    return True, ""

def sanitize_topic(topic: str) -> str:
    """Clean and sanitize the input topic."""
    return topic.strip() 