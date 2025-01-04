"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai
import os
from typing import Dict
import time

from config import GEMINI_MODEL
from utils import validate_topic, sanitize_topic, QuotaExceededError
from agents import PreAnalysisAgent, PromptDesigner, FrameworkEngineer, ResearchAnalyst, SynthesisExpert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="M.A.R.A. - Multi-Agent Reasoning Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS and Logo
st.markdown("""
<style>
.block-container { 
    max-width: 800px; 
    padding: 2rem 1rem; 
}

.stButton > button { 
    width: 100%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

/* Focus area buttons */
[data-testid="baseButton-secondary"] {
    background-color: #f8f9fa !important;
    border: 1px solid #dee2e6 !important;
    color: #2c3338 !important;
    padding: 0.75rem !important;
    min-height: 3rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="baseButton-secondary"]:hover {
    background-color: #e9ecef !important;
    border-color: #ced4da !important;
}

[data-testid="baseButton-primary"] {
    background-color: rgba(0, 102, 204, 0.1) !important;
    border: 1px solid #0066cc !important;
    box-shadow: 0 0 0 1px #0066cc !important;
    color: #0066cc !important;
    font-weight: 500 !important;
    padding: 0.75rem !important;
    min-height: 3rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="baseButton-primary"]:hover {
    background-color: rgba(0, 102, 204, 0.2) !important;
}

[data-testid="baseButton-primary"]:disabled {
    background-color: #f8f9fa !important;
    border-color: #dee2e6 !important;
    box-shadow: none !important;
    color: #6c757d !important;
    cursor: not-allowed !important;
}

div[data-testid="stImage"] { 
    text-align: center; 
}

div[data-testid="stImage"] > img { 
    max-width: 800px; 
    width: 100%; 
}

textarea {
    font-size: 1.1em !important;
    line-height: 1.5 !important;
    padding: 0.5em !important;
    height: 150px !important;
    background-color: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    color: #2c3338 !important;
}

/* Number input styling */
div[data-testid="stNumberInput"] input {
    color: #2c3338 !important;
    background-color: #ffffff !important;
    border: 1px solid #dee2e6 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: #f8f9fa !important;
    border: 1px solid #dee2e6 !important;
    color: #2c3338 !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: #e9ecef !important;
}
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

def initialize_state():
    """Initialize the application state."""
    return {
        'topic': None,
        'iterations': 2,
        'show_insights': False,
        'show_focus': False,
        'show_framework': False,
        'show_analysis': False,
        'show_summary': False,
        'insights': None,
        'focus_areas': None,
        'framework': None,
        'analysis_results': [],
        'focus_selection_complete': False
    }

def reset_state(topic, iterations):
    """Reset the application state."""
    st.session_state.app_state = initialize_state()
    st.session_state.app_state.update({
        'topic': topic,
        'iterations': iterations,
        'show_insights': True
    })
    
    # Clear focus area states
    if 'focus_area_expanded' in st.session_state:
        del st.session_state.focus_area_expanded
    if 'current_focus_areas' in st.session_state:
        del st.session_state.current_focus_areas

def display_insights(insights: dict):
    """Display insights in proper containers."""
    with st.container():
        with st.expander("💡 Did You Know?", expanded=True):
            st.markdown(insights['did_you_know'])
        
        with st.expander("⚡ ELI5", expanded=True):
            st.markdown(insights['eli5'])

def display_focus_areas(focus_areas: list):
    """Display focus area selection with proper state handling."""
    # Track if the section should be expanded in session state
    if 'focus_area_expanded' not in st.session_state:
        st.session_state.focus_area_expanded = True
        
    # Track selected areas in session state to persist between reruns
    if 'current_focus_areas' not in st.session_state:
        st.session_state.current_focus_areas = []
    
    # Create container for focus area selection
    focus_container = st.container()
    
    with focus_container:
        with st.expander("🎯 Focus Areas", expanded=st.session_state.focus_area_expanded):
            st.markdown("Choose specific aspects you'd like the analysis to emphasize (optional):")
            
            # Use multiselect with max_selections parameter to allow multiple selections
            selected = st.multiselect(
                "Focus Areas",
                options=focus_areas,
                default=st.session_state.current_focus_areas,
                key="focus_multiselect",
                label_visibility="collapsed",
                max_selections=None  # Allow unlimited selections
            )
            
            # Update session state with current selections
            st.session_state.current_focus_areas = selected
            
            st.markdown("---")
            
            # Only show buttons if selection is not complete
            if not st.session_state.app_state.get('focus_selection_complete'):
                col1, col2 = st.columns(2)
                
                # Handle Skip button
                if col1.button(
                    "Skip",
                    key="skip_focus",
                    use_container_width=True,
                    on_click=lambda: setattr(st.session_state, 'focus_area_expanded', False)
                ):
                    st.session_state.app_state['focus_selection_complete'] = True
                    st.session_state.app_state['show_framework'] = True
                    st.session_state.focus_area_expanded = False
                    st.rerun()
                
                # Handle Continue button
                continue_disabled = len(selected) == 0
                if col2.button(
                    "Continue",
                    key="continue_focus",
                    disabled=continue_disabled,
                    type="primary",
                    use_container_width=True,
                    on_click=lambda: setattr(st.session_state, 'focus_area_expanded', False)
                ):
                    st.session_state.app_state['focus_selection_complete'] = True
                    st.session_state.app_state['show_framework'] = True
                    st.session_state.focus_area_expanded = False
                    st.rerun()
            
            return False, selected

def cleanup_partial_results(context: str):
    """Clean up any partial results when errors occur."""
    if context == 'analysis':
        if 'analysis_results' in st.session_state.app_state:
            # Remove the last incomplete analysis
            if st.session_state.app_state['analysis_results']:
                st.session_state.app_state['analysis_results'].pop()
    elif context == 'framework':
        st.session_state.app_state['framework'] = None
        st.session_state.app_state['show_analysis'] = False
    elif context == 'focus':
        st.session_state.app_state['focus_areas'] = None
        st.session_state.app_state['selected_areas'] = []
        st.session_state.app_state['show_framework'] = False

def process_stage(stage_name, container, stage_fn, next_stage=None, spinner_text=None, display_fn=None, **kwargs):
    """Process a single stage of the analysis."""
    if stage_name not in st.session_state.app_state:
        with container, st.spinner(spinner_text):
            result = stage_fn(**kwargs)
            if result is not None:
                st.session_state.app_state[stage_name] = result
                if next_stage:
                    # Only show framework after focus selection is complete
                    if next_stage == 'framework' and not st.session_state.app_state.get('focus_selection_complete'):
                        return
                    st.session_state.app_state[f'show_{next_stage}'] = True
                st.rerun()
    elif display_fn and st.session_state.app_state.get(stage_name) is not None:
        display_fn(st.session_state.app_state[stage_name])

def main():
    """Main application flow."""
    # Initialize app_state if not exists
    if 'app_state' not in st.session_state:
        st.session_state.app_state = initialize_state()
    
    # Check if quota timer has expired
    if hasattr(st.session_state, 'quota_reset_time'):
        if time.time() >= st.session_state.quota_reset_time:
            st.session_state.form_disabled = False
            del st.session_state.quota_reset_time
    
    # Input form
    with st.form("analysis_form"):
        topic = st.text_area(
            "What would you like to explore?",
            help="Enter your research topic or question.",
            placeholder="e.g., 'Examine the impact of artificial intelligence on healthcare...'"
        )
        
        iterations = st.number_input(
            "Number of Analysis Iterations",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Choose 1-5 iterations. More iterations = deeper insights = longer wait."
        )
        
        submit = st.form_submit_button(
            "🚀 Start Analysis",
            use_container_width=True,
            disabled=st.session_state.get('form_disabled', False)
        )
    
    # Process submission
    if submit:
        is_valid, error_msg, sanitized_topic = validate_and_sanitize_input(topic)
        if not is_valid:
            st.error(error_msg)
            return
        
        # Reset state and start analysis
        reset_state(sanitized_topic, iterations)
        st.rerun()

    # Only proceed with analysis if we have a topic
    if not st.session_state.app_state.get('topic'):
        return

    try:
        # Initialize containers for each stage
        containers = {
            'insights': st.empty(),
            'focus': st.empty(),
            'framework': st.empty(),
            'analysis': st.empty(),
            'summary': st.empty()
        }
        
        # Process each stage
        if st.session_state.app_state.get('show_insights'):
            process_stage('insights', containers['insights'],
                         lambda **kwargs: PreAnalysisAgent(model).generate_insights(st.session_state.app_state['topic']),
                         'focus', spinner_text="💡 Generating insights...",
                         display_fn=display_insights)
            
            # Generate prompt silently in the background if not already generated
            if not st.session_state.app_state.get('prompt'):
                optimized_prompt = PromptDesigner(model).design_prompt(st.session_state.app_state['topic'])
                if optimized_prompt:
                    st.session_state.app_state['prompt'] = optimized_prompt
        
        if st.session_state.app_state.get('show_focus'):
            process_stage('focus', containers['focus'],
                         lambda **kwargs: PromptDesigner(model).generate_focus_areas(st.session_state.app_state['topic']),
                         'framework', spinner_text="🎯 Generating focus areas...",
                         display_fn=display_focus_areas)
        
        # Only show framework after focus selection is complete
        if st.session_state.app_state.get('show_framework') and st.session_state.app_state.get('focus_selection_complete'):
            process_stage('framework', containers['framework'],
                         lambda **kwargs: FrameworkEngineer(model).create_framework(
                             st.session_state.app_state.get('prompt', ''),
                             st.session_state.app_state.get('enhanced_prompt')
                         ),
                         'analysis', spinner_text="🔨 Building analysis framework...",
                         display_fn=lambda x: st.expander("🎯 Research Framework", expanded=False).markdown(x))
        
        # Process analysis (special handling due to iterations)
        if st.session_state.app_state.get('show_analysis'):
            with containers['analysis']:
                if len(st.session_state.app_state.get('analysis_results', [])) < st.session_state.app_state.get('iterations', 2):
                    with st.spinner("🔄 Performing analysis..."):
                        result = ResearchAnalyst(model).analyze(
                            st.session_state.app_state['topic'],
                            st.session_state.app_state.get('framework', ''),
                            st.session_state.app_state.get('analysis_results', [])[-1] if st.session_state.app_state.get('analysis_results') else None
                        )
                        if result:
                            content = format_analysis_result(result)
                            if 'analysis_results' not in st.session_state.app_state:
                                st.session_state.app_state['analysis_results'] = []
                            st.session_state.app_state['analysis_results'].append(content)
                            if len(st.session_state.app_state['analysis_results']) == st.session_state.app_state['iterations']:
                                st.session_state.app_state['show_summary'] = True
                            st.rerun()
                
                for i, result in enumerate(st.session_state.app_state.get('analysis_results', [])):
                    with st.expander(f"🔄 Research Analysis #{i + 1}", expanded=False):
                        st.markdown(result)
        
        if st.session_state.app_state.get('show_summary'):
            process_stage('summary', containers['summary'],
                         lambda **kwargs: SynthesisExpert(model).synthesize(
                             st.session_state.app_state['topic'],
                             st.session_state.app_state.get('analysis_results', [])
                         ),
                         spinner_text="📊 Generating final report...",
                         display_fn=lambda x: st.expander("📊 Final Report", expanded=False).markdown(x))
                     
    except Exception as e:
        handle_error(e, "analysis")
        return

def format_analysis_result(result: Dict[str, str]) -> str:
    """Format analysis result with consistent structure."""
    content = ""
    if result['title']:
        content += f"# {result['title']}\n\n"
    if result['subtitle']:
        content += f"*{result['subtitle']}*\n\n"
    if result['content']:
        content += result['content']
    return content

if __name__ == "__main__":
    main() 