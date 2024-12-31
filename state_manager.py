"""State management for the MARA application."""

import logging
import streamlit as st
from typing import Optional, Dict, Any, TypedDict
from dataclasses import dataclass

from constants import (
    SESSION_STATE_KEYS,
    DEFAULT_ANALYSIS_STATE,
    STATUS_MESSAGES,
    ERROR_MESSAGES,
    SUCCESS_MESSAGE
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AnalysisState:
    """Type-safe analysis state."""
    topic: Optional[str] = None
    framework: Optional[str] = None
    analysis: Optional[list] = None
    summary: Optional[str] = None

class StateManager:
    """Manages application state and UI updates."""
    
    @staticmethod
    def validate_state(state: Dict) -> bool:
        """Validate state structure."""
        required_keys = ['topic', 'framework', 'analysis', 'summary']
        return all(key in state for key in required_keys)
    
    @staticmethod
    def initialize_session_state():
        """Initialize or reset session state."""
        if SESSION_STATE_KEYS['CURRENT_ANALYSIS'] not in st.session_state:
            st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']] = AnalysisState()
    
    @staticmethod
    def clear_results():
        """Clear all analysis results."""
        st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']] = AnalysisState()
        if SESSION_STATE_KEYS['ANALYSIS_CONTAINER'] in st.session_state:
            st.session_state[SESSION_STATE_KEYS['ANALYSIS_CONTAINER']].empty()
    
    @staticmethod
    def update_analysis_results(topic: str, framework: Optional[str] = None, 
                              analysis: Optional[list] = None, summary: Optional[str] = None):
        """Update analysis results in session state."""
        current_state = st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']]
        new_state = AnalysisState(
            topic=topic,
            framework=framework if framework is not None else current_state.framework,
            analysis=analysis if analysis is not None else current_state.analysis,
            summary=summary if summary is not None else current_state.summary
        )
        st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']] = new_state
    
    @staticmethod
    def get_current_topic() -> Optional[str]:
        """Get the current topic being analyzed."""
        return st.session_state[SESSION_STATE_KEYS['CURRENT_ANALYSIS']].topic
    
    @staticmethod
    def show_status(status_key: str, iteration: int = 0) -> Any:
        """Create and return a status object with the appropriate message."""
        try:
            if status_key not in STATUS_MESSAGES:
                logger.error(f"Invalid status key: {status_key}")
                return st.status("Processing...")
                
            message = STATUS_MESSAGES[status_key]['START']
            if '{iteration}' in message:
                message = message.format(iteration=iteration)
            return st.status(message)
        except Exception as e:
            logger.error(f"Error showing status: {str(e)}")
            return st.status("Processing...")
    
    @staticmethod
    def update_status(status: Any, status_key: str, iteration: int = 0):
        """Update a status object with completion message."""
        try:
            if status_key not in STATUS_MESSAGES:
                logger.error(f"Invalid status key: {status_key}")
                status.update(label="Complete")
                return
                
            message = STATUS_MESSAGES[status_key]['COMPLETE']
            if '{iteration}' in message:
                message = message.format(iteration=iteration)
            status.update(label=message)
        except Exception as e:
            logger.error(f"Error updating status: {str(e)}")
            status.update(label="Complete")
    
    @staticmethod
    def show_error(error_key: str, error: Exception = None):
        """Display an error message."""
        try:
            if error_key not in ERROR_MESSAGES:
                st.error(str(error) if error else "An error occurred")
                return
                
            message = ERROR_MESSAGES[error_key]
            if error and '{error}' in message:
                message = message.format(error=str(error))
            st.error(message)
        except Exception as e:
            logger.error(f"Error showing error message: {str(e)}")
            st.error(str(error) if error else "An error occurred")
    
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