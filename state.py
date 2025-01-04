"""State management for the MARA application."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import streamlit as st

@dataclass
class AppState:
    """Application state container."""
    topic: str = ''
    last_topic: str = ''
    stage: str = 'input'
    insights: Optional[Dict[str, str]] = None
    focus_areas: Optional[List[str]] = None
    selected_focus_areas: List[str] = None
    analysis_results: List[Dict[str, str]] = None
    synthesis: Optional[Dict[str, str]] = None
    iterations: int = 2
    focus_container_expanded: bool = True
    selected_areas: List[str] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.selected_focus_areas is None:
            self.selected_focus_areas = []
        if self.analysis_results is None:
            self.analysis_results = []
        if self.selected_areas is None:
            self.selected_areas = []

    @classmethod
    def load_state(cls) -> 'AppState':
        """Load state from session_state or create new."""
        if 'app_state' not in st.session_state:
            st.session_state.app_state = cls()
        return st.session_state.app_state

    def save_state(self):
        """Save current state to session_state."""
        st.session_state.app_state = self

    def soft_reset(self):
        """Reset state while preserving the last topic."""
        self.stage = 'input'
        self.insights = None
        self.focus_areas = None
        self.selected_focus_areas = []
        self.analysis_results = []
        self.synthesis = None
        self.iterations = 2
        self.focus_container_expanded = True
        self.selected_areas = []
        self.save_state()

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return asdict(self) 