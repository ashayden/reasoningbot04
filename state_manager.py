"""State management for the MARA application."""

import logging
from typing import Optional, Dict, Any

import streamlit as st

from constants import STATUS_MESSAGES

logger = logging.getLogger(__name__)

class StateManager:
    """Manages application state and UI updates."""
    
    @staticmethod
    def init_session_state() -> None:
        """Initialize session state if not already done."""
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = {
                'topic': None,
                'framework': None,
                'analysis': None,
                'summary': None
            }
        if 'analysis_container' not in st.session_state:
            st.session_state.analysis_container = None
    
    @staticmethod
    def clear_results() -> None:
        """Clear all analysis results."""
        st.session_state.current_analysis = {
            'topic': None,
            'framework': None,
            'analysis': None,
            'summary': None
        }
        if 'analysis_container' in st.session_state:
            st.session_state.analysis_container = None
    
    @staticmethod
    def show_status(phase: str, status: str, iteration: Optional[int] = None) -> None:
        """Show a status message for the current operation."""
        try:
            if phase not in STATUS_MESSAGES:
                logger.error(f"Invalid status phase: {phase}")
                return
                
            if status not in ['start', 'complete']:
                logger.error(f"Invalid status key: {status}")
                return
                
            message = STATUS_MESSAGES[phase][status]
            
            # Add iteration number for analysis phase if provided
            if phase == 'ANALYSIS' and iteration is not None:
                message = message.replace('...', f' (Iteration {iteration})...')
            
            # Create a container for the status message if it doesn't exist
            if 'status_container' not in st.session_state:
                st.session_state.status_container = st.empty()
            
            # Show the status message with appropriate styling
            status_class = 'status-running' if status == 'start' else 'status-complete'
            st.session_state.status_container.markdown(
                f'<div class="status-message {status_class}">{message}</div>',
                unsafe_allow_html=True
            )
            
        except Exception as e:
            logger.error(f"Error showing status: {str(e)}")
    
    @staticmethod
    def update_analysis(key: str, value: Any) -> None:
        """Update a specific key in the current analysis."""
        try:
            if 'current_analysis' not in st.session_state:
                logger.error("Session state not initialized")
                return
                
            st.session_state.current_analysis[key] = value
        except Exception as e:
            logger.error(f"Error updating analysis: {str(e)}")
    
    @staticmethod
    def get_analysis(key: str) -> Optional[Any]:
        """Get a specific value from the current analysis."""
        try:
            if 'current_analysis' not in st.session_state:
                logger.error("Session state not initialized")
                return None
                
            return st.session_state.current_analysis.get(key)
        except Exception as e:
            logger.error(f"Error getting analysis: {str(e)}")
            return None
    
    @staticmethod
    def create_container() -> None:
        """Create a container for analysis output."""
        try:
            if 'analysis_container' not in st.session_state:
                st.session_state.analysis_container = st.container()
        except Exception as e:
            logger.error(f"Error creating container: {str(e)}")
    
    @staticmethod
    def get_container():
        """Get the analysis container."""
        try:
            return st.session_state.get('analysis_container')
        except Exception as e:
            logger.error(f"Error getting container: {str(e)}")
            return None 