"""Utility functions for the MARA application."""

import logging
import time
import random
from typing import Any, Callable, Tuple
from functools import wraps
from config import MIN_TOPIC_LENGTH, MAX_TOPIC_LENGTH

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

def rate_limit_decorator(func: Callable) -> Callable:
    """Decorator to implement rate limiting with exponential backoff.
    
    Args:
        func: The function to rate limit
        
    Returns:
        The wrapped function with rate limiting
    """
    last_call_time = 0
    base_delay = 1  # Base delay in seconds
    max_retries = 3
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        nonlocal last_call_time
        
        # Ensure minimum time between calls
        current_time = time.time()
        time_since_last_call = current_time - last_call_time
        if time_since_last_call < base_delay:
            sleep_time = base_delay - time_since_last_call
            logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Implement exponential backoff for retries
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                last_call_time = time.time()
                return result
            except Exception as e:
                if "429" in str(e):  # API quota error
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        delay = (2 ** attempt * base_delay) + (random.random() * 0.1)
                        logger.warning(f"API quota exceeded. Attempt {attempt + 1}/{max_retries}. "
                                     f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"API quota exceeded after {max_retries} attempts")
                        raise Exception("API quota exceeded. Please try again later.") from e
                else:
                    logger.error(f"Error in rate-limited function {func.__name__}: {str(e)}")
                    raise
        
        return None  # Should never reach here due to raise in loop
    
    return wrapper

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
    
    if not topic or len(topic.strip()) == 0:
        msg = "Topic cannot be empty."
        logger.warning(msg)
        return False, msg
    
    if len(topic) < MIN_TOPIC_LENGTH:
        msg = f"Topic must be at least {MIN_TOPIC_LENGTH} characters long."
        logger.warning(msg)
        return False, msg
    
    if len(topic) > MAX_TOPIC_LENGTH:
        msg = f"Topic must be no more than {MAX_TOPIC_LENGTH} characters long."
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