"""Utility functions for the MARA application."""

import logging
import time
import random
from typing import Any, Callable, Tuple, Dict
from functools import wraps
from datetime import datetime, timedelta
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

class QuotaExceededError(MARAError):
    """Raised when API quota is exhausted."""
    def __init__(self, retry_after: int = 300, message: str = None):
        self.retry_after = retry_after
        if message is None:
            message = f"API quota exceeded. Please try again in {retry_after} seconds."
        super().__init__(message)
        
    def get_retry_after(self) -> int:
        """Get the recommended retry delay in seconds."""
        return self.retry_after

class ValidationError(MARAError):
    """Raised when input validation fails."""
    pass

class ProcessingError(MARAError):
    """Raised when processing operations fail."""
    pass

class APIRateLimiter:
    """Manages API rate limiting and quota tracking."""
    
    def __init__(self):
        self.last_call_time = 0
        self.quota_reset_time = None
        self.calls_remaining = 60  # Default per-minute quota
        self.daily_calls = 0
        self.daily_limit = 1000  # Conservative daily limit
        self.base_delay = 2.0  # Increased base delay
        self.quota_exceeded = False
        self.error_counts: Dict[str, int] = {}
        self.last_reset = time.time()
    
    def check_quota(self) -> None:
        """Check if we're within quota limits."""
        current_time = time.time()
        
        # Reset per-minute quota if a minute has passed
        if current_time - self.last_reset >= 60:
            self.calls_remaining = 60
            self.last_reset = current_time
            self.error_counts.clear()
            logger.info("Per-minute quota reset")
        
        # Reset daily quota at midnight UTC
        current_day = datetime.utcnow().date()
        last_call_day = datetime.fromtimestamp(self.last_call_time).date()
        if current_day != last_call_day:
            self.daily_calls = 0
            logger.info("Daily quota reset")
        
        # Check if we're in quota exceeded state
        if self.quota_exceeded:
            wait_time = int(self.quota_reset_time - current_time) if self.quota_reset_time else 300
            raise QuotaExceededError(retry_after=wait_time)
        
        # Check quota limits
        if self.calls_remaining <= 0:
            self.quota_exceeded = True
            self.quota_reset_time = self.last_reset + 60
            raise QuotaExceededError(retry_after=int(self.quota_reset_time - current_time))
        
        if self.daily_calls >= self.daily_limit:
            self.quota_exceeded = True
            tomorrow = (datetime.utcnow() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self.quota_reset_time = tomorrow.timestamp()
            raise QuotaExceededError(retry_after=int(self.quota_reset_time - current_time))
        
        # Enforce minimum delay between calls with jitter
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.base_delay:
            sleep_time = self.base_delay - time_since_last_call + (random.random() * 0.5)
            logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    def handle_error(self, error: Exception) -> None:
        """Handle API errors and update quota tracking."""
        error_str = str(error)
        self.error_counts[error_str] = self.error_counts.get(error_str, 0) + 1
        
        if "429" in error_str:
            # Implement progressive cooldown
            cooldown_time = min(300 * self.error_counts[error_str], 3600)  # Max 1 hour
            self.quota_exceeded = True
            self.quota_reset_time = time.time() + cooldown_time
            logger.warning(f"Quota error detected. Implementing {cooldown_time} second cooldown.")
            raise QuotaExceededError(retry_after=cooldown_time)
    
    def update_last_call(self) -> None:
        """Update the last successful call time and quotas."""
        self.last_call_time = time.time()
        self.calls_remaining -= 1
        self.daily_calls += 1

# Global rate limiter instance
rate_limiter = APIRateLimiter()

def rate_limit_decorator(func: Callable) -> Callable:
    """Decorator to implement rate limiting with quota management."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Check quota before making call
                rate_limiter.check_quota()
                
                # Make the API call
                result = func(*args, **kwargs)
                
                # Update successful call time
                rate_limiter.update_last_call()
                return result
                
            except Exception as e:
                # Handle the error and update quota tracking
                rate_limiter.handle_error(e)
                
                if attempt < max_retries - 1:
                    # Calculate backoff delay
                    delay = (2 ** attempt * rate_limiter.base_delay) + (random.random() * 0.1)
                    logger.warning(f"API error on attempt {attempt + 1}/{max_retries}. "
                                 f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    # If we've exhausted retries, raise the error
                    if isinstance(e, QuotaExceededError):
                        raise
                    logger.error(f"Error in rate-limited function {func.__name__}: {str(e)}")
                    raise
        
        return None  # Should never reach here
    
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