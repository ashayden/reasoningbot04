"""Configuration settings for the MARA application."""

# Model configurations
GEMINI_MODEL = "gemini-1.5-pro-latest"

# Generation configurations
PROMPT_DESIGN_CONFIG = {
    "temperature": 0.1,
    "candidate_count": 1,
    "max_output_tokens": 1024
}

FRAMEWORK_CONFIG = {
    "temperature": 0.1,
    "candidate_count": 1,
    "max_output_tokens": 1024
}

ANALYSIS_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1,
    "max_output_tokens": 2048
}

SYNTHESIS_CONFIG = {
    "temperature": 0.3,
    "candidate_count": 1,
    "max_output_tokens": 4096
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
REQUEST_TIMEOUT = 30  # seconds 