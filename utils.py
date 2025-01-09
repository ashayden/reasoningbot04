"""Utility functions for the MARA application."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, List
from google.api_core import retry, exceptions
from google.generativeai.types import GenerateContentResponse
import google.generativeai.types as gemini_types

logger = logging.getLogger(__name__)

T = TypeVar('T')

class GeminiAPIError(Exception):
    """Custom exception for Gemini API-related errors that follows Google's guidelines."""
    def __init__(self, message: str, error_type: Optional[str] = None):
        self.error_type = error_type
        super().__init__(message)

def safe_api_call(retries: int = 3, backoff: float = 2.0) -> Callable:
    """Decorator for safe Gemini API calls with compliant retry logic."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except exceptions.GoogleAPIError as e:
                    # Handle Google API specific errors
                    if attempt < retries - 1:
                        wait_time = backoff ** attempt
                        logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Gemini API call failed after {retries} attempts")
                        raise GeminiAPIError(str(e), error_type="API_ERROR")
                except Exception as e:
                    # Handle other errors
                    logger.error(f"Unexpected error in Gemini API call: {str(e)}")
                    raise GeminiAPIError(str(e), error_type="UNEXPECTED_ERROR")
        return wrapper
    return decorator

def parse_gemini_response(response: GenerateContentResponse) -> Dict[str, Any]:
    """Safely parse Gemini API response following Google's guidelines."""
    if not response:
        raise GeminiAPIError("Empty response from Gemini API", error_type="EMPTY_RESPONSE")
        
    try:
        # Check response finish reason
        if hasattr(response, 'prompt_feedback'):
            feedback = response.prompt_feedback
            if feedback and feedback.block_reason:
                raise GeminiAPIError(
                    f"Content blocked: {feedback.block_reason}",
                    error_type="CONTENT_BLOCKED"
                )
        
        # Extract text content
        if not hasattr(response, 'text'):
            raise GeminiAPIError("Invalid response format", error_type="INVALID_FORMAT")
            
        text = response.text.strip()
        if not text:
            raise GeminiAPIError("Empty text in response", error_type="EMPTY_CONTENT")
            
        # Clean up response text
        text = text.replace('\\"', '"')
        text = text.replace('\\n', '\n')
        
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
        
    except GeminiAPIError:
        raise
    except Exception as e:
        raise GeminiAPIError(f"Failed to parse Gemini response: {str(e)}", error_type="PARSE_ERROR")

def rate_limit_decorator(calls: int = 60, period: float = 60.0) -> Callable:
    """Rate limiting decorator compliant with Gemini API quotas."""
    bucket = TokenBucket(calls, period)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                bucket.consume(1)
                return func(*args, **kwargs)
            except Exception as e:
                raise GeminiAPIError(f"Rate limit exceeded: {str(e)}", error_type="RATE_LIMIT")
        return wrapper
    return decorator

class TokenBucket:
    """Token bucket for API rate limiting."""
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