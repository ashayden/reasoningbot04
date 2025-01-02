"""Utility functions for the MARA application."""

import logging
import re
import time
from functools import wraps
from typing import Dict, Optional, Any, Callable

import streamlit as st

logger = logging.getLogger(__name__)

def rate_limit_decorator(func: Callable) -> Callable:
    """Decorator to rate limit API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get or initialize rate limit state
        if 'last_call_time' not in st.session_state:
            st.session_state.last_call_time = 0
        
        # Calculate time since last call
        current_time = time.time()
        time_since_last_call = current_time - st.session_state.last_call_time
        
        # If less than 1 second has passed, wait
        if time_since_last_call < 1:
            time.sleep(1 - time_since_last_call)
        
        # Update last call time and execute function
        st.session_state.last_call_time = time.time()
        return func(*args, **kwargs)
    
    return wrapper

def parse_title_content(text: str) -> Optional[Dict[str, str]]:
    """Parse title, subtitle, and content from generated text.
    
    Args:
        text: The text to parse
        
    Returns:
        Dictionary containing 'title', 'subtitle', and 'content' keys,
        or None if parsing fails
    """
    if not text:
        return None
    
    try:
        # Initialize result dictionary
        result = {
            'title': '',
            'subtitle': '',
            'content': ''
        }
        
        # Split text into lines for processing
        lines = text.strip().split('\n')
        current_line = 0
        total_lines = len(lines)
        
        # Process title
        while current_line < total_lines:
            line = lines[current_line].strip()
            
            # Skip empty lines
            if not line:
                current_line += 1
                continue
            
            # Check for title markers
            title_match = re.match(r'^(?:Title:|#)\s*(.+)$', line, re.IGNORECASE)
            if title_match:
                result['title'] = title_match.group(1).strip()
                current_line += 1
                break
            
            # If no explicit title marker but line looks like a title
            if re.match(r'^[A-Z].*[^.!?]$', line) and len(line.split()) <= 10:
                result['title'] = line
                current_line += 1
                break
            
            current_line += 1
        
        # Process subtitle
        while current_line < total_lines:
            line = lines[current_line].strip()
            
            # Skip empty lines
            if not line:
                current_line += 1
                continue
            
            # Check for subtitle markers
            subtitle_match = re.match(r'^(?:Subtitle:|##)\s*(.+)$', line, re.IGNORECASE)
            if subtitle_match:
                result['subtitle'] = subtitle_match.group(1).strip()
                current_line += 1
                break
            
            # If no explicit subtitle marker but line looks like a subtitle
            if (line.startswith('*') and line.endswith('*')) or \
               (line.startswith('_') and line.endswith('_')):
                result['subtitle'] = line.strip('*_ ')
                current_line += 1
                break
            
            current_line += 1
        
        # Combine remaining lines as content
        content_lines = []
        while current_line < total_lines:
            line = lines[current_line].strip()
            
            # Skip empty lines at the start of content
            if not line and not content_lines:
                current_line += 1
                continue
            
            content_lines.append(line)
            current_line += 1
        
        # Clean up and format content
        if content_lines:
            content = '\n'.join(content_lines)
            
            # Clean up markdown formatting
            content = re.sub(r'\n{3,}', '\n\n', content)  # Remove extra newlines
            content = re.sub(r'(?m)^[-*]\s', '* ', content)  # Standardize bullet points
            content = re.sub(r'(?m)^(\d+\.)\s', r'\1 ', content)  # Clean up numbered lists
            
            result['content'] = content.strip()
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing text: {str(e)}")
        return None 