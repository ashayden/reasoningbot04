"""Constants and shared configuration for the MARA application."""

# Status Messages
STATUS_MESSAGES = {
    'FRAMEWORK': {
        'START': "🎯 Creating analysis framework...",
        'COMPLETE': "🎯 Analysis Framework"
    },
    'ANALYSIS': {
        'START': "🔄 Performing research analysis #{iteration}...",
        'COMPLETE': "🔄 Research Analysis #{iteration}"
    },
    'SYNTHESIS': {
        'START': "📊 Generating final report...",
        'COMPLETE': "📊 Final Report"
    }
}

# UI Elements
CUSTOM_CSS = """
<style>
.block-container { max-width: 800px; padding: 2rem 1rem; }
.stButton > button { width: 100%; }
div[data-testid="stImage"] { text-align: center; }
div[data-testid="stImage"] > img { max-width: 800px; width: 100%; }
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