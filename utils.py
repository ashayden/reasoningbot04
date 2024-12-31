"""Utility functions for the MARA application."""

import time
from functools import wraps
from typing import Dict, Tuple, Optional

import streamlit as st
from config import MIN_TOPIC_LENGTH, MAX_TOPIC_LENGTH, MAX_REQUESTS_PER_MINUTE

class RateLimiter:
    """Simple rate limiter implementation."""
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def can_proceed(self) -> bool:
        """Check if a new request can proceed."""
        current_time = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        return False

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
    if not topic or not topic.strip():
        return False, "Topic cannot be empty."
    
    if len(topic) < MIN_TOPIC_LENGTH:
        return False, f"Topic must be at least {MIN_TOPIC_LENGTH} characters long."
    
    if len(topic) > MAX_TOPIC_LENGTH:
        return False, f"Topic must be no more than {MAX_TOPIC_LENGTH} characters long."
    
    return True, None

def parse_title_content(text: str) -> Dict[str, str]:
    """Parse title, subtitle, and content from generated text.
    
    Args:
        text: The raw text to parse.
        
    Returns:
        Dictionary containing 'title', 'subtitle', and 'content'.
    """
    if not text:
        return {
            'title': '',
            'subtitle': '',
            'content': ''
        }

    try:
        lines = text.split('\n')
        result = {
            'title': '',
            'subtitle': '',
            'content': ''
        }
        
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('Title:'):
                result['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Subtitle:'):
                result['subtitle'] = line.replace('Subtitle:', '').strip()
                content_start = i + 1
                break
        
        # If we found a title/subtitle, get the remaining content
        if content_start > 0:
            result['content'] = '\n'.join(lines[content_start:]).strip()
        else:
            # If no title/subtitle found, treat entire text as content
            result['content'] = text.strip()
        
        # Ensure we never return None values
        return {k: v if v is not None else '' for k, v in result.items()}
        
    except Exception as e:
        logger.error(f"Error parsing content: {str(e)}")
        return {
            'title': '',
            'subtitle': '',
            'content': text if text else ''
        }

def sanitize_topic(topic: str) -> str:
    """Sanitize the topic string for safe use in prompts.
    
    Args:
        topic: The raw topic string.
        
    Returns:
        Sanitized topic string.
    """
    # Remove any potentially harmful characters
    return ''.join(c for c in topic if c.isalnum() or c.isspace() or c in '.,!?-_()') 