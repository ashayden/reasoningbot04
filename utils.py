"""Utility functions for the MARA application."""

import logging
import time
from functools import wraps
from typing import Dict, Tuple, Optional, Union, List

import streamlit as st
from config import MIN_TOPIC_LENGTH, MAX_TOPIC_LENGTH, MAX_REQUESTS_PER_MINUTE

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter implementation."""
    def __init__(self, max_requests: int, time_window: int = 60):
        if not isinstance(max_requests, int) or max_requests <= 0:
            raise ValueError("max_requests must be a positive integer")
        if not isinstance(time_window, int) or time_window <= 0:
            raise ValueError("time_window must be a positive integer")
            
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []

    def can_proceed(self) -> bool:
        """Check if a new request can proceed."""
        current_time = time.time()
        # Remove old requests in one pass
        cutoff_time = current_time - self.time_window
        self.requests = [req for req in self.requests if req > cutoff_time]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        return False

# Initialize rate limiter as a singleton
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

def rate_limit_decorator(func):
    """Decorator to apply rate limiting to functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not rate_limiter.can_proceed():
            raise Exception("Rate limit exceeded. Please wait before making more requests.")
        return func(*args, **kwargs)
    return wrapper

def validate_topic(topic: str) -> Tuple[bool, Optional[str]]:
    """Validate the input topic string.
    
    Args:
        topic: The input topic string to validate.
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(topic, str):
        return False, "Topic must be a string."
        
    topic = topic.strip()
    if not topic:
        return False, "Topic cannot be empty."
    
    topic_length = len(topic)
    if topic_length < MIN_TOPIC_LENGTH:
        return False, f"Topic must be at least {MIN_TOPIC_LENGTH} characters long."
    
    if topic_length > MAX_TOPIC_LENGTH:
        return False, f"Topic must be no more than {MAX_TOPIC_LENGTH} characters long."
    
    return True, None

def parse_title_content(text: str) -> Dict[str, str]:
    """Parse title, subtitle, and content from generated text.
    
    Args:
        text: The raw text to parse.
        
    Returns:
        Dictionary containing 'title', 'subtitle', and 'content'.
    """
    if not isinstance(text, str):
        logger.error("Input to parse_title_content must be a string")
        return {'title': '', 'subtitle': '', 'content': ''}

    text = text.strip()
    if not text:
        return {'title': '', 'subtitle': '', 'content': ''}

    try:
        # Split only once to get lines more efficiently
        lines = text.split('\n')
        result = {'title': '', 'subtitle': '', 'content': text}  # Default to full text as content
        
        # Use enumerate sparingly and break early when possible
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('Title:'):
                result['title'] = line[6:].strip()  # More efficient than replace
            elif line.startswith('Subtitle:'):
                result['subtitle'] = line[9:].strip()  # More efficient than replace
                # We found both title and subtitle, use rest as content
                if result['title']:  # Only if we also found a title
                    result['content'] = '\n'.join(lines[i+1:]).strip()
                break
                
        return result
        
    except Exception as e:
        logger.error(f"Error parsing content: {str(e)}")
        return {'title': '', 'subtitle': '', 'content': text}

def sanitize_topic(topic: str) -> str:
    """Sanitize the topic string for safe use in prompts.
    
    Args:
        topic: The raw topic string.
        
    Returns:
        Sanitized topic string.
    """
    if not isinstance(topic, str):
        logger.error("Input to sanitize_topic must be a string")
        return ""
        
    # Use set for more efficient character filtering
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_()')
    return ''.join(c for c in topic if c in allowed_chars) 