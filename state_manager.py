"""State management for the MARA application."""

import streamlit as st
from typing import Optional, Dict, Any

from constants import (
    SESSION_STATE_KEYS,
    DEFAULT_ANALYSIS_STATE,
    STATUS_MESSAGES,
    ERROR_MESSAGES,
    SUCCESS_MESSAGE
)

class StateManager:
    """Manages application state and UI updates."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize or reset session state."""
        if SESSION_STATE_KEYS['CURRENT_ANALYSIS'] not in st.session_state:
            st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']] = DEFAULT_ANALYSIS_STATE.copy()
    
    @staticmethod
    def clear_results():
        """Clear all analysis results."""
        st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']] = DEFAULT_ANALYSIS_STATE.copy()
        if SESSION_STATE_KEYS['ANALYSIS_CONTAINER'] in st.session_state:
            st.session_state[SESSION_STATE_KEYS['ANALYSIS_CONTAINER']].empty()
    
    @staticmethod
    def update_analysis_results(topic: str, framework: Optional[str] = None, 
                              analysis: Optional[list] = None, summary: Optional[str] = None):
        """Update analysis results in session state."""
        st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']].update({
            'topic': topic,
            'framework': framework if framework is not None else None,
            'analysis': analysis if analysis is not None else None,
            'summary': summary if summary is not None else None
        })
    
    @staticmethod
    def get_current_topic() -> Optional[str]:
        """Get the current topic being analyzed."""
        return st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']]['topic']
    
    @staticmethod
    def show_status(status_key: str, iteration: int = 0) -> Any:
        """Create and return a status object with the appropriate message."""
        message = STATUS_MESSAGES[status_key]['START']
        if '{iteration}' in message:
            message = message.format(iteration=iteration)
        return st.status(message)
    
    @staticmethod
    def update_status(status: Any, status_key: str, iteration: int = 0):
        """Update a status object with completion message."""
        message = STATUS_MESSAGES[status_key]['COMPLETE']
        if '{iteration}' in message:
            message = message.format(iteration=iteration)
        status.update(label=message)
    
    @staticmethod
    def show_error(error_key: str, error: Exception = None):
        """Display an error message."""
        message = ERROR_MESSAGES[error_key]
        if error and '{error}' in message:
            message = message.format(error=str(error))
        st.error(message)
    
    @staticmethod
    def show_success():
        """Display success message."""
        st.success(SUCCESS_MESSAGE)
    
    @staticmethod
    def create_analysis_container() -> Any:
        """Create and store a container for analysis output."""
        if SESSION_STATE_KEYS['ANALYSIS_CONTAINER'] not in st.session_state:
            st.session_state[SESSION_STATE_KEYS['ANALYSIS_CONTAINER']] = st.container()
        return st.session_state[SESSION_STATE_KEYS['ANALYSIS_CONTAINER']] 