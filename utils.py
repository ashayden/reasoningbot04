"""Utility functions for the MARA application."""

import logging
import time
from functools import wraps
from typing import Dict, Tuple, Optional
import io
import markdown
from datetime import datetime
import mdpdf
import tempfile
import os

import streamlit as st
from config import MIN_TOPIC_LENGTH, MAX_TOPIC_LENGTH, MAX_REQUESTS_PER_MINUTE

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter implementation."""
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def can_proceed(self) -> bool:
        """Check if a new request can proceed."""
        current_time = time.time()
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
        Tuple of (is_valid, error_message). If valid, error_message is None.
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
        If parsing fails, returns text as content with empty title/subtitle.
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
        
        # Look for title and subtitle with more flexible matching
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Match title with or without "Title:" prefix
            if line.lower().startswith('title:'):
                result['title'] = line.replace('Title:', '').replace('title:', '').strip()
            elif line.startswith('#'):  # Markdown title format
                result['title'] = line.replace('#', '').strip()
            elif not result['title'] and len(line) < 100:  # First short line could be title
                result['title'] = line
                
            # Match subtitle with flexible prefix
            if line.lower().startswith('subtitle:'):
                result['subtitle'] = line.replace('Subtitle:', '').replace('subtitle:', '').strip()
                content_start = i + 1
                break
            elif line.startswith('*') and line.endswith('*'):  # Markdown italic format
                result['subtitle'] = line.strip('*').strip()
                content_start = i + 1
                break
        
        # If no explicit content start was found, start after the first two non-empty lines
        if content_start == 0:
            non_empty_lines = 0
            for i, line in enumerate(lines):
                if line.strip():
                    non_empty_lines += 1
                    if non_empty_lines == 2:
                        content_start = i + 1
                        break
            
            # If still no content start, use the whole text
            if content_start == 0:
                content_start = 2 if len(lines) > 2 else 0
        
        # Join remaining lines as content, ensuring no duplicate title/subtitle
        content_lines = lines[content_start:]
        if content_lines:
            content = '\n'.join(content_lines).strip()
            # Remove title/subtitle if they appear at the start of content
            if result['title'] and content.startswith(result['title']):
                content = content[len(result['title']):].strip()
            if result['subtitle'] and content.startswith(result['subtitle']):
                content = content[len(result['subtitle']):].strip()
            result['content'] = content
        
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
        Sanitized topic string with only alphanumeric, space, and basic punctuation.
    """
    return ''.join(c for c in topic if c.isalnum() or c.isspace() or c in '.,!?-_()') 

class CitationFormatter:
    """Utility class for handling citations and references."""
    
    @staticmethod
    def format_citation(citation: str) -> str:
        """Format a citation line according to APA style."""
        citation = citation.strip()
        if not citation.endswith('.'):
            citation += '.'
        return citation
    
    @staticmethod
    def format_reference_section(lines: list) -> list:
        """Format a list of references according to APA style."""
        formatted_refs = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not line.startswith('*'):
                line = f"* {line}"
            if not line.endswith('.'):
                line += '.'
            formatted_refs.append(line)
        return formatted_refs

class MarkdownFormatter:
    """Utility class for markdown formatting."""
    
    @staticmethod
    def format_section_header(title: str, level: int = 1) -> str:
        """Format a section header with proper markdown."""
        return f"{'#' * level} {title.strip()}"
    
    @staticmethod
    def format_bullet_point(text: str, indent_level: int = 0) -> str:
        """Format a bullet point with proper indentation."""
        indent = '  ' * indent_level
        return f"{indent}* {text.strip()}"
    
    @staticmethod
    def clean_spacing(text: str) -> str:
        """Clean up markdown spacing."""
        text = text.replace('\n\n\n', '\n\n')  # Remove extra blank lines
        text = text.replace('\n*', '\n\n*')    # Ensure space before lists
        text = text.replace('\n#', '\n\n#')    # Ensure space before headers
        return text.strip() 

def generate_pdf(markdown_content: str) -> bytes:
    """Convert markdown content to PDF using mdpdf.
    
    Args:
        markdown_content: The markdown content to convert
        
    Returns:
        PDF file as bytes
    """
    try:
        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as md_file:
            md_file.write(markdown_content)
            md_filepath = md_file.name
        
        # Create a temporary PDF file path
        pdf_filepath = md_filepath.replace('.md', '.pdf')
        
        # Convert markdown to PDF
        mdpdf.convert(md_filepath, pdf_filepath, theme='github')
        
        # Read the PDF file
        with open(pdf_filepath, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Clean up temporary files
        os.unlink(md_filepath)
        os.unlink(pdf_filepath)
        
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        # Return a simple PDF with error message
        return f"Error generating PDF: {str(e)}".encode('utf-8')

def generate_markdown(content: str) -> bytes:
    """Prepare markdown content for download.
    
    Args:
        content: The markdown content
        
    Returns:
        Markdown file as bytes
    """
    return content.encode('utf-8')

def get_safe_filename(topic: str) -> str:
    """Generate a safe filename from the topic.
    
    Args:
        topic: The research topic
        
    Returns:
        A sanitized filename with timestamp
    """
    # Remove invalid filename characters
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_topic = safe_topic.replace(' ', '_')[:50]  # Limit length
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"MARA_Report_{safe_topic}_{timestamp}" 