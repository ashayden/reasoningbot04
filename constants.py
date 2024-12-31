"""Constants and shared configuration for the MARA application."""

# Status Messages
STATUS_MESSAGES = {
    'PROMPT_DESIGN': {
        'start': 'Optimizing prompt for analysis...',
        'complete': 'Prompt optimization complete.'
    },
    'FRAMEWORK': {
        'start': 'Creating analysis framework...',
        'complete': 'Analysis framework complete.'
    },
    'ANALYSIS': {
        'start': 'Conducting research analysis...',
        'complete': 'Research analysis complete.'
    },
    'SYNTHESIS': {
        'start': 'Synthesizing findings...',
        'complete': 'Final synthesis complete.'
    }
}

# UI Elements
CUSTOM_CSS = """
<style>
.block-container { max-width: 800px; padding: 2rem 1rem; }
.stButton > button { width: 100%; }
div[data-testid="stImage"] { text-align: center; }
div[data-testid="stImage"] > img { max-width: 800px; width: 100%; }
.status-message {
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.status-running {
    background-color: #f0f2f6;
    border-left: 5px solid #1f77b4;
}
.status-complete {
    background-color: #e8f4ea;
    border-left: 5px solid #28a745;
}
.status-error {
    background-color: #fdf1f1;
    border-left: 5px solid #dc3545;
}
.analysis-title {
    font-size: 1.5em;
    font-weight: bold;
    margin: 15px 0;
}
.analysis-subtitle {
    font-size: 1.2em;
    font-style: italic;
    color: #666;
    margin: 10px 0;
}
.section-header {
    font-size: 1.2em;
    font-weight: bold;
    margin: 15px 0 10px 0;
}
</style>
"""

# Form Elements
TOPIC_INPUT = {
    'LABEL': "What would you like to explore?",
    'PLACEHOLDER': "e.g., 'Artificial Intelligence' or 'Climate Change'"
}

DEPTH_SELECTOR = {
    'LABEL': "Analysis Depth",
    'DEFAULT': "Balanced"
}

# Success Messages
SUCCESS_MESSAGE = "Analysis complete! Review the results above."

# Error Messages
ERROR_MESSAGES = {
    'API_INIT': "Failed to initialize Gemini API: {error}",
    'RATE_LIMIT': "Rate limit exceeded. Please wait before making more requests.",
    'EMPTY_RESPONSE': "Empty response from model",
    'CONTENT_ERROR': "Error generating content: {error}",
    'ANALYSIS_ERROR': "Analysis error: {error}"
}

# Session State Keys
SESSION_STATE_KEYS = {
    'CURRENT_ANALYSIS': 'current_analysis',
    'ANALYSIS_CONTAINER': 'analysis_container'
}

# Default Session State
DEFAULT_ANALYSIS_STATE = {
    'topic': None,
    'framework': None,
    'analysis': None,
    'summary': None
} 