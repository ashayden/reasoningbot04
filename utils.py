"""Utility functions for the MARA application."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, List
from google.api_core import retry
from google.generativeai.types import GenerateContentResponse

logger = logging.getLogger(__name__)

T = TypeVar('T')

class APIError(Exception):
    """Custom exception for API-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

def safe_api_call(retries: int = 3, backoff: float = 2.0) -> Callable:
    """Decorator for safe API calls with retry logic."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None
            for attempt in range(retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    last_error = e
                    if attempt < retries - 1:
                        wait_time = backoff ** attempt
                        logger.warning(f"API call failed (attempt {attempt + 1}/{retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API call failed after {retries} attempts: {str(e)}")
                        raise APIError(f"API call failed: {str(e)}", response=last_error)
        return wrapper
    return decorator

def parse_gemini_response(response: GenerateContentResponse) -> Dict[str, Any]:
    """Safely parse Gemini API response with enhanced error handling."""
    if not response:
        raise APIError("Empty response from Gemini API")
        
    try:
        # Extract text content
        text = response.text.strip() if hasattr(response, 'text') else None
        
        # Handle different response formats
        if not text:
            raise APIError("No text content in response")
            
        # Clean up response text
        text = text.replace('\\"', '"')  # Fix escaped quotes
        text = text.replace('\\n', '\n')  # Fix escaped newlines
        
        # Try multiple parsing approaches
        try:
            # First try ast.literal_eval for safety
            import ast
            result = ast.literal_eval(text)
        except:
            try:
                # Try json.loads as fallback
                import json
                result = json.loads(text)
            except:
                # Return raw text if parsing fails
                result = {"content": text}
                
        return result
        
    except Exception as e:
        raise APIError(f"Failed to parse Gemini response: {str(e)}", response=response)

def rate_limit_decorator(calls: int = 60, period: float = 60.0) -> Callable:
    """Rate limiting decorator with token bucket algorithm."""
    bucket = TokenBucket(calls, period)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            bucket.consume(1)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class TokenBucket:
    """Token bucket for rate limiting."""
    def __init__(self, tokens: int, period: float):
        self.tokens = tokens
        self.period = period
        self.last_update = time.time()
        self.current_tokens = tokens
        
    def consume(self, tokens: int = 1) -> None:
        now = time.time()
        elapsed = now - self.last_update
        
        # Replenish tokens based on elapsed time
        self.current_tokens = min(
            self.tokens,
            self.current_tokens + (elapsed * self.tokens / self.period)
        )
        
        # Update timestamp
        self.last_update = now
        
        # Check if we have enough tokens
        if self.current_tokens < tokens:
            sleep_time = (tokens - self.current_tokens) * self.period / self.tokens
            time.sleep(sleep_time)
            self.current_tokens = tokens
            
        self.current_tokens -= tokens

def validate_response_format(response: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate response format against required keys."""
    return all(key in response for key in required_keys)

def clean_markdown_content(content: str) -> str:
    """Clean and standardize markdown content."""
    # Remove multiple consecutive newlines
    content = '\n'.join(line for line in content.splitlines() if line.strip())
    
    # Ensure proper heading hierarchy
    lines = content.split('\n')
    current_level = 0
    cleaned_lines = []
    
    for line in lines:
        if line.startswith('#'):
            level = len(line.split()[0])
            if level > current_level + 1:
                level = current_level + 1
            current_level = level
            cleaned_lines.append('#' * level + line[level:])
        else:
            cleaned_lines.append(line)
            if not line.strip():
                current_level = 0
                
    return '\n'.join(cleaned_lines) 