"""State management for the MARA application."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import streamlit as st

@dataclass
class AppState:
    """Application state management with validation and persistence."""
    
    # User input state
    last_topic: str = field(default="")
    iterations: int = field(default=1)
    
    # Analysis state
    stage: str = field(default="input")
    insights: Optional[Dict[str, str]] = field(default=None)
    focus_areas: List[str] = field(default_factory=list)
    selected_focus_areas: List[str] = field(default_factory=list)
    focus_container_expanded: bool = field(default=True)
    
    # Research state
    current_iteration: int = field(default=0)
    analysis_results: List[Dict[str, str]] = field(default_factory=list)
    synthesis: Optional[Dict[str, str]] = field(default=None)
    
    def __post_init__(self):
        """Validate state after initialization."""
        self.validate_state()
    
    def validate_state(self) -> None:
        """Validate state integrity."""
        # Validate stage
        valid_stages = {'input', 'analysis', 'research', 'complete'}
        if self.stage not in valid_stages:
            self.stage = 'input'
        
        # Validate iterations
        if not isinstance(self.iterations, int) or self.iterations < 1:
            self.iterations = 1
        elif self.iterations > 5:
            self.iterations = 5
            
        # Validate current_iteration
        if self.current_iteration > self.iterations:
            self.current_iteration = self.iterations
            
        # Validate focus areas
        if len(self.selected_focus_areas) > 5:
            self.selected_focus_areas = self.selected_focus_areas[:5]
    
    def soft_reset(self) -> None:
        """Reset analysis state while preserving last topic."""
        self.stage = 'input'
        self.insights = None
        self.focus_areas = []
        self.selected_focus_areas = []
        self.focus_container_expanded = True
        self.current_iteration = 0
        self.analysis_results = []
        self.synthesis = None
        
        # Persist state
        self.persist_state()
    
    def hard_reset(self) -> None:
        """Complete state reset."""
        self.last_topic = ""
        self.iterations = 1
        self.soft_reset()
    
    def persist_state(self) -> None:
        """Persist state to session storage."""
        for field_name, field_value in self.__dict__.items():
            st.session_state[f'mara_{field_name}'] = field_value
    
    def load_persisted_state(self) -> None:
        """Load state from session storage."""
        for field_name in self.__dict__.keys():
            session_key = f'mara_{field_name}'
            if session_key in st.session_state:
                setattr(self, field_name, st.session_state[session_key])
        
        # Validate after loading
        self.validate_state()
    
    def update_stage(self, new_stage: str) -> None:
        """Update stage with validation."""
        valid_stages = {'input', 'analysis', 'research', 'complete'}
        if new_stage in valid_stages:
            self.stage = new_stage
            self.persist_state()
    
    def add_analysis_result(self, result: Dict[str, str]) -> None:
        """Add analysis result with validation."""
        if not isinstance(result, dict):
            return
            
        required_keys = {'title', 'subtitle', 'content'}
        if not all(key in result for key in required_keys):
            return
            
        self.analysis_results.append(result)
        self.current_iteration += 1
        self.persist_state()
    
    def set_synthesis(self, synthesis: Dict[str, str]) -> None:
        """Set synthesis result with validation."""
        if not isinstance(synthesis, dict):
            return
            
        required_keys = {'title', 'subtitle', 'content'}
        if not all(key in synthesis for key in required_keys):
            return
            
        self.synthesis = synthesis
        self.persist_state()
    
    @property
    def is_complete(self) -> bool:
        """Check if research is complete."""
        return (
            self.stage == 'complete' and
            self.synthesis is not None and
            len(self.analysis_results) == self.iterations
        ) 