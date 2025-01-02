"""Configuration settings for the MARA application."""

import os
from typing import Dict, Any

# API Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-1.0-pro"
GEMINI_VISION_MODEL = "gemini-1.0-pro-vision"

# Analysis Configuration
MAX_ITERATIONS = 2
ANALYSIS_BASE_TEMP = 0.7
ANALYSIS_TEMP_INCREMENT = 0.1
ANALYSIS_MAX_TEMP = 0.9

# Generation Configurations
PROMPT_DESIGN_CONFIG: Dict[str, Any] = {
    "candidate_count": 1,
    "stop_sequences": [],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}

FRAMEWORK_CONFIG: Dict[str, Any] = {
    "candidate_count": 1,
    "stop_sequences": [],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

ANALYSIS_CONFIG: Dict[str, Any] = {
    "candidate_count": 1,
    "stop_sequences": [],
    "temperature": 0.7,  # Base temperature, will be adjusted dynamically
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 4096,
}

SYNTHESIS_CONFIG: Dict[str, Any] = {
    "candidate_count": 1,
    "stop_sequences": [],
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# Input validation
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 200

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60 