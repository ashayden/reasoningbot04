"""Configuration settings for the MARA application."""

# Model Configuration
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-1219"

# Generation Configurations
PROMPT_DESIGN_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 2048,
    'candidate_count': 1
}

FRAMEWORK_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 4096,
    'candidate_count': 1
}

ANALYSIS_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 8192,
    'candidate_count': 1
}

SYNTHESIS_CONFIG = {
    'temperature': 0.5,
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 8192,
    'candidate_count': 1
}

# Analysis Depth Settings
DEPTH_ITERATIONS = {
    'Quick': 1,
    'Balanced': 2,
    'Deep': 3,
    'Comprehensive': 4
}

# Input validation
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 200

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60 