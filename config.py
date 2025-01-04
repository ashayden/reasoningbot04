"""Configuration settings for the MARA application."""

# Model configuration
GEMINI_MODEL = "gemini-pro"

# Generation configurations
PREANALYSIS_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}

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

ANALYSIS_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1,
    "max_output_tokens": 8192
}

SYNTHESIS_CONFIG = {
    "temperature": 0.5,
    "candidate_count": 1,
    "max_output_tokens": 8192
}

# Analysis temperature settings
ANALYSIS_BASE_TEMP = 0.7
ANALYSIS_TEMP_INCREMENT = 0.1
ANALYSIS_MAX_TEMP = 0.9

# Input validation
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 200 