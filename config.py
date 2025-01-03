"""Configuration settings for the MARA application."""

# Model configurations
GEMINI_MODEL = "gemini-1.5-pro-latest"

# Generation configurations
PROMPT_DESIGN_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "stop_sequences": ["I will", "Let me", "First", "Now I will"]
}

FRAMEWORK_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1,
    "max_output_tokens": 4096,
    "stop_sequences": ["I will", "Let me", "First", "Now I will"]
}

# Research Analysis settings
ANALYSIS_BASE_TEMP = 0.7
ANALYSIS_TEMP_INCREMENT = 0.1
ANALYSIS_MAX_TEMP = 0.9

ANALYSIS_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1,
    "max_output_tokens": 8192,
    "stop_sequences": ["I will", "Let me", "First", "Now I will"]
}

SYNTHESIS_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1,
    "max_output_tokens": 8192,
    "stop_sequences": ["I will", "Let me", "First", "Now I will"]
}

# Input validation
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 200

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60 