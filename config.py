"""Configuration settings for the MARA application."""

import os
from typing import Dict, Any

# Model configuration
GEMINI_MODEL = "gemini-exp-1206"
PREANALYSIS_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}

# Generation configurations
PROMPT_DESIGN_CONFIG = {
    "temperature": 0.1,
    "candidate_count": 1,
    "max_output_tokens": 1024
}

FRAMEWORK_CONFIG = {
    "temperature": 0.1,
    "candidate_count": 1,
    "max_output_tokens": 4096
}

# Research Analysis settings
ANALYSIS_BASE_TEMP = 0.7
ANALYSIS_TEMP_INCREMENT = 0.1
ANALYSIS_MAX_TEMP = 0.9

ANALYSIS_CONFIG = {
    "temperature": ANALYSIS_BASE_TEMP,
    "candidate_count": 1,
    "max_output_tokens": 8192
}

SYNTHESIS_CONFIG = {
    "temperature": 0.5,
    "candidate_count": 1,
    "max_output_tokens": 8192
}

# Input validation
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 200

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60 