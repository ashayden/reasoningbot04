"""Configuration settings for the MARA application."""

# Model configurations
GEMINI_MODEL = "gemini-exp-1206"

# Generation configurations
PROMPT_DESIGN_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1
}

FRAMEWORK_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1
}

ANALYSIS_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1
}

SYNTHESIS_CONFIG = {
    "temperature": 0.7,
    "candidate_count": 1
}

# Input validation
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 200 