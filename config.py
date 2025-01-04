"""Configuration settings for the MARA application."""

# Model Configuration
GEMINI_MODEL = "learnlm-1.5-pro-experimental"

# Topic Validation
MIN_TOPIC_LENGTH = 10
MAX_TOPIC_LENGTH = 500

# Agent Configurations
PREANALYSIS_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 1024,
}

ANALYSIS_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 40,
    'max_output_tokens': 2048,
}

SYNTHESIS_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 40,
    'max_output_tokens': 4096,
}

# Analysis Temperature Settings
ANALYSIS_BASE_TEMP = 0.7
ANALYSIS_TEMP_INCREMENT = 0.1
ANALYSIS_MAX_TEMP = 0.9

# Cache Settings
CACHE_TTL = 3600  # 1 hour in seconds 