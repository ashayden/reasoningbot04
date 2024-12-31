"""Configuration settings for the MARA application."""

# Model configurations
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-1219"

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

# Research Analysis temperature range
ANALYSIS_BASE_TEMP = 0.7
ANALYSIS_TEMP_INCREMENT = 0.1
ANALYSIS_MAX_TEMP = 0.9

ANALYSIS_CONFIG = {
    "temperature": ANALYSIS_BASE_TEMP,  # Will be dynamically adjusted
    "candidate_count": 1,
    "max_output_tokens": 8192
}

# Synthesis with increased creativity
SYNTHESIS_CONFIG = {
    "temperature": 0.5,  # Increased from 0.3 for more creative synthesis
    "candidate_count": 1,
    "max_output_tokens": 8192
}

# Analysis depth settings
DEPTH_ITERATIONS = {
    "Quick": 1,
    "Balanced": 2,
    "Deep": 3,
    "Comprehensive": 4
}

# Input validation
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 200

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60 