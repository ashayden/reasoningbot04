"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai
import os
import time
import random
from typing import Any, Optional, Tuple, List
from google.generativeai.types import GenerationConfig

from config import (
    config,
    GEMINI_MODEL,
    PROMPT_DESIGN_CONFIG
)
from utils import validate_topic, sanitize_topic
from agents import (
    PreAnalysisAgent,
    PromptDesigner,
    FrameworkEngineer,
    ResearchAnalyst,
    SynthesisExpert,
    ResearchResult
)

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
}

/* Focus area buttons */
[data-testid="baseButton-secondary"] {
    background-color: rgba(30, 30, 30, 0.6) !important;
    border: 1px solid #333 !important;
    color: white !important;
    padding: 0.75rem !important;
    min-height: 3rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="baseButton-secondary"]:hover {
    background-color: rgba(42, 42, 42, 0.8) !important;
    border-color: #444 !important;
}

[data-testid="baseButton-primary"] {
    background-color: rgba(0, 102, 204, 0.2) !important;
    border: 1px solid #0066cc !important;
    box-shadow: 0 0 0 1px #0066cc !important;
    color: #0066cc !important;
    font-weight: 500 !important;
    padding: 0.75rem !important;
    min-height: 3rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="baseButton-primary"]:hover {
    background-color: rgba(0, 102, 204, 0.3) !important;
}

[data-testid="baseButton-primary"]:disabled {
    background-color: #1E1E1E !important;
    border-color: #333 !important;
    box-shadow: none !important;
    color: #4a4a4a !important;
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
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
    color: #fff !important;
}

/* Number input styling */
div[data-testid="stNumberInput"] input {
    color: #fff !important;
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: #333 !important;
    border: none !important;
    color: #fff !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: #444 !important;
}

/* Multiselect styling */
div[data-testid="stMultiSelect"] {
    width: 100% !important;
    max-width: none !important;
}

div[data-testid="stMultiSelect"] > div {
    width: 100% !important;
    max-width: none !important;
}

/* Hide the inner text input */
div[data-testid="stMultiSelect"] input[type="text"] {
    display: none !important;
}

/* Style the select container */
div[data-testid="stMultiSelect"] [data-baseweb="select"] {
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
    border-radius: 4px !important;
}

/* Style the placeholder and selected items */
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
    background-color: #1E1E1E !important;
    border: none !important;
    color: #fff !important;
    min-height: 40px !important;
    padding: 8px 12px !important;
}

/* Style the dropdown arrow */
div[data-testid="stMultiSelect"] [role="button"] {
    color: #fff !important;
}

/* Style selected items */
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: rgba(0, 102, 204, 0.2) !important;
    border: 1px solid #0066cc !important;
    color: #fff !important;
}

/* Style the dropdown list */
div[data-testid="stMultiSelect"] [role="listbox"] {
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
    margin-top: 4px !important;
}

/* Style dropdown options */
div[data-testid="stMultiSelect"] [role="option"] {
    background-color: #1E1E1E !important;
    color: #fff !important;
    padding: 8px 12px !important;
}

/* Style hover state of options */
div[data-testid="stMultiSelect"] [role="option"]:hover {
    background-color: #333 !important;
}

/* Style selected options in dropdown */
div[data-testid="stMultiSelect"] [aria-selected="true"] {
    background-color: rgba(0, 102, 204, 0.2) !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

# Initialize state
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'topic': None,
        'iterations': None,
        'insights': None,
        'focus_areas': None,
        'selected_areas': [],
        'prompt': None,
        'framework': None,
        'analysis_results': [],
        'summary': None,
        'show_insights': False,
        'show_focus': False,
        'show_framework': False,
        'show_analysis': False,
        'show_summary': False,
        'focus_selection_complete': False
    }

# Initialize Gemini
@st.cache_resource
def initialize_gemini() -> Optional[Any]:
    """Initialize and test the Gemini model."""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["GOOGLE_API_KEY"]
        if not api_key:
            logger.error("No API key found in secrets")
            st.error("Missing API key. Please configure GOOGLE_API_KEY in Streamlit secrets.")
            return None
            
        logger.info("Testing Gemini model initialization...")
        
        # Configure the model
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            config.GEMINI_MODEL,
            safety_settings=config.SAFETY_SETTINGS
        )
        
        # Simple test prompt
        test_response = model.generate_content(
            "Respond with a single word: Hello",
            generation_config=config.PROMPT_DESIGN_CONFIG.model_dump()
        )
        
        if test_response and test_response.text:
            logger.info("Gemini model initialized successfully")
            return model
            
        logger.error("Empty response from model test")
        return None
                
    except Exception as e:
        logger.error(f"Failed to initialize or test Gemini model: {str(e)}")
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        return None

# Initialize model early
model = initialize_gemini()
if not model:
    st.stop()

def validate_and_sanitize_input(topic: str) -> tuple[bool, str, str]:
    """Validate and sanitize user input."""
    if not topic or len(topic.strip()) == 0:
        return False, "Please enter a topic to analyze.", ""
    
    if len(topic) > 1000:
        return False, "Topic is too long. Please keep it under 1000 characters.", ""
    
    sanitized = sanitize_topic(topic)
    if not sanitized:
        return False, "Invalid topic format. Please try again.", ""
    
    return True, "", sanitized

def handle_error(e: Exception, context: str):
    """Handle errors consistently."""
    error_msg = f"Error during {context}: {str(e)}"
    logger.error(error_msg)
    st.error(error_msg)
    
    # Reset relevant state
    if context in st.session_state.app_state:
        st.session_state.app_state[context] = None
    
    # Stop showing subsequent stages
    stages = ['show_insights', 'show_prompt', 'show_focus', 'show_framework', 'show_analysis', 'show_summary']
    current_stage_idx = stages.index(f'show_{context}')
    for stage in stages[current_stage_idx:]:
        st.session_state.app_state[stage] = False

def reset_state(topic, iterations):
    """Reset application state with memory cleanup."""
    # Clear previous results and states
    if 'app_state' in st.session_state:
        old_state = st.session_state.app_state
        if old_state.get('analysis_results'):
            old_state['analysis_results'].clear()
    
    # Reset all session state variables
    for key in list(st.session_state.keys()):
        if key != 'app_state':
            del st.session_state[key]
    
    # Initialize new state
    st.session_state.app_state = {
        'topic': topic,
        'iterations': iterations,
        'insights': None,
        'focus_areas': None,
        'selected_areas': [],
        'prompt': None,
        'framework': None,
        'analysis_results': [],
        'summary': None,
        'show_insights': True,
        'show_focus': False,
        'show_framework': False,
        'show_analysis': False,
        'show_summary': False,
        'focus_selection_complete': False
    }
    
    # Initialize focus area state with expanded UI
    st.session_state.focus_area_state = {
        'expanded': True,
        'selected': [],
        'complete': False,
        'just_completed': False
    }

def display_insights(insights: dict):
    """Display insights in proper containers."""
    with st.container():
        with st.expander("💡 Did You Know?", expanded=True):
            st.markdown(insights['did_you_know'])
        
        with st.expander("⚡ ELI5", expanded=True):
            st.markdown(insights['eli5'])

def display_focus_selection(focus_areas: list, selected_areas: list) -> tuple[bool, list]:
    """Display focus area selection with proper state handling."""
    # Initialize session state for focus areas
    if 'focus_area_state' not in st.session_state:
        st.session_state.focus_area_state = {
            'expanded': True,
            'selected': [],
            'complete': False,
            'just_completed': False
        }
    
    # Create container for focus area selection
    focus_container = st.container()
    
    with focus_container:
        # Show content with proper expansion state
        with st.expander("🎯 Focus Areas", expanded=st.session_state.focus_area_state['expanded']):
            # Only show selection UI if not completed
            if not st.session_state.focus_area_state['complete']:
                # Handle selection changes
                def on_selection_change():
                    st.session_state.focus_area_state['selected'] = st.session_state.focus_select
                
                # Create the multiselect with callback
                selected = st.multiselect(
                    "Select Focus Areas",  # Add proper label
                    options=focus_areas,
                    default=st.session_state.focus_area_state['selected'],
                    key="focus_select",
                    on_change=on_selection_change,
                    label_visibility="collapsed",  # Hide label but keep it for accessibility
                    placeholder="Select one or more focus areas..."
                )
                
                st.markdown("---")
                
                # Action buttons
                button_col1, button_col2 = st.columns(2)
                
                # Handle Skip button
                if button_col1.button("Skip", key="skip_focus", use_container_width=True):
                    st.session_state.focus_area_state.update({
                        'complete': True,
                        'expanded': False,
                        'just_completed': True,
                        'selected': []
                    })
                    st.session_state.app_state['focus_selection_complete'] = True
                    st.session_state.app_state['show_framework'] = True
                    st.rerun()
                    return True, []
                
                # Handle Continue button
                if button_col2.button(
                    "Continue",
                    key="continue_focus",
                    disabled=len(selected) == 0,
                    type="primary",
                    use_container_width=True
                ):
                    st.session_state.focus_area_state.update({
                        'complete': True,
                        'expanded': False,
                        'just_completed': True,
                        'selected': selected
                    })
                    st.session_state.app_state['focus_selection_complete'] = True
                    st.session_state.app_state['show_framework'] = True
                    st.rerun()
                    return True, selected
            else:
                # Show selected areas in collapsed state
                if st.session_state.focus_area_state['selected']:
                    st.markdown("**Selected Focus Areas:**")
                    for area in st.session_state.focus_area_state['selected']:
                        st.markdown(f"* {area}")
                else:
                    st.markdown("*No focus areas selected*")
        
        return False, st.session_state.focus_area_state['selected']

def display_analysis(analysis_results: List[ResearchResult]):
    """Display analysis results."""
    for i, result in enumerate(analysis_results):
        with st.expander(f"🔄 Research Analysis #{i + 1}", expanded=False):
            if result.subtitle:
                # Split title if it contains a colon
                if ':' in result.title:
                    main_title, subtitle_part = result.title.split(':', 1)
                    st.markdown(f"## {subtitle_part.strip()}\n\n*{result.subtitle}*\n\n{result.content}")
                else:
                    st.markdown(f"## {result.title}\n\n*{result.subtitle}*\n\n{result.content}")
            else:
                st.markdown(f"## {result.title}\n\n{result.content}")

def display_final_report(summary: str):
    """Display final report with proper title formatting."""
    with st.expander("📊 Final Report", expanded=False):
        # Split the summary into lines
        lines = summary.split('\n', 2)
        if len(lines) >= 2:
            # If first line contains a colon, use the part after colon as main title
            if ':' in lines[0]:
                main_title, subtitle_part = lines[0].split(':', 1)
                st.markdown(f"## {subtitle_part.strip()}\n\n*{lines[1]}*\n\n{lines[2] if len(lines) > 2 else ''}")
            else:
                st.markdown(f"## {lines[0]}\n\n*{lines[1]}*\n\n{lines[2] if len(lines) > 2 else ''}")
        else:
            st.markdown(summary)

def main():
    """Main application flow."""
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
            use_container_width=True
        )
    
    # Process submission
    if submit:
        # Validate input
        is_valid, error_msg, sanitized_topic = validate_and_sanitize_input(topic)
        if not is_valid:
            st.error(error_msg)
            return
        
        # Reset state if topic changed
        if st.session_state.app_state['topic'] != sanitized_topic:
            reset_state(sanitized_topic, iterations)
    
    try:
        # Create containers with proper styling
        insights_container = st.container()
        focus_container = st.container()
        framework_container = st.container()
        analysis_container = st.container()
        summary_container = st.container()
        
        # Process insights
        if st.session_state.app_state['show_insights']:
            with insights_container:
                try:
                    if not st.session_state.app_state['insights']:
                        with st.spinner("💡 Generating insights..."):
                            insights = PreAnalysisAgent(model).generate_insights(topic)
                            if insights:
                                st.session_state.app_state['insights'] = insights
                                st.session_state.app_state['show_focus'] = True
                                st.rerun()
                    
                    if st.session_state.app_state['insights']:
                        display_insights(st.session_state.app_state['insights'])
                except Exception as e:
                    handle_error(e, "insights")
                    return
        
        # Handle focus areas
        if st.session_state.app_state['show_focus']:
            with focus_container:
                if not st.session_state.app_state['focus_areas']:
                    with st.spinner("🎯 Generating focus areas..."):
                        try:
                            focus_areas = PromptDesigner(model).generate_focus_areas(topic)
                            if focus_areas:
                                st.session_state.app_state['focus_areas'] = focus_areas
                                st.rerun()
                        except Exception as e:
                            logger.error(f"Error generating focus areas: {str(e)}")
                            st.error(
                                "Failed to generate focus areas. This can happen if the topic is too broad or specific. "
                                "You can either try again or skip focus area selection."
                            )
                            # Add a retry button
                            if st.button("🔄 Retry Focus Areas", use_container_width=True):
                                st.session_state.app_state['focus_areas'] = None
                                st.rerun()
                            # Add a skip button
                            if st.button("⏭️ Skip Focus Areas", use_container_width=True):
                                st.session_state.app_state['show_framework'] = True
                                st.rerun()
                            return
                
                if st.session_state.app_state['focus_areas']:
                    proceed, selected = display_focus_selection(
                        st.session_state.app_state['focus_areas'],
                        st.session_state.app_state['selected_areas']
                    )
                    st.session_state.app_state['selected_areas'] = selected
                    
                    if proceed:
                        st.session_state.app_state['show_framework'] = True
                        st.rerun()
        
        # Process framework
        if st.session_state.app_state['show_framework']:
            with framework_container:
                if not st.session_state.app_state['framework']:
                    with st.spinner("🔨 Building research framework..."):
                        try:
                            # First, create optimized prompt for the Framework Engineer
                            prompt_for_framework = PromptDesigner(model).create_optimized_prompt(
                                topic,
                                st.session_state.app_state.get('selected_areas')
                            )
                            
                            if prompt_for_framework:
                                st.session_state.app_state['prompt'] = prompt_for_framework
                                
                                # Then have Framework Engineer create topic-specific framework
                                framework = FrameworkEngineer(model).create_framework(
                                    topic,
                                    prompt_for_framework,
                                    st.session_state.app_state.get('selected_areas')
                                )
                                
                                if framework:
                                    st.session_state.app_state['framework'] = framework
                                    st.session_state.app_state['show_analysis'] = True
                                    st.rerun()
                                    
                        except Exception as e:
                            logger.error(f"Error generating framework: {str(e)}")
                            st.error("Failed to generate research framework. Please try again.")
                            return
                
                if st.session_state.app_state['prompt']:
                    with st.expander("✍️ Optimized Prompt", expanded=False):
                        st.markdown("---")
                        st.markdown("**Instructions for Framework Development:**")
                        st.markdown(st.session_state.app_state['prompt'])
                        st.markdown("---")
                
                if st.session_state.app_state['framework']:
                    with st.expander("📄 Topic-Specific Research Framework", expanded=False):
                        try:
                            st.markdown("---")
                            st.markdown("**Research Framework to Guide Analysis:**")
                            st.markdown(st.session_state.app_state['framework'])
                            st.markdown("---")
                            st.success("Framework generated successfully! Proceeding with research analysis following this framework...")
                        except Exception as e:
                            logger.error(f"Error displaying framework: {str(e)}")
                            st.error("Error displaying framework. Please try again.")
        
        # Process analysis
        if st.session_state.app_state['show_analysis']:
            with analysis_container:
                if len(st.session_state.app_state['analysis_results']) < st.session_state.app_state['iterations']:
                    with st.spinner("🔄 Performing analysis..."):
                        # Get previous analysis content if it exists
                        previous_analysis = None
                        if st.session_state.app_state['analysis_results']:
                            last_result = st.session_state.app_state['analysis_results'][-1]
                            previous_analysis = f"{last_result.title}\n{last_result.subtitle or ''}\n{last_result.content}"
                        
                        result = ResearchAnalyst(model).analyze(
                            topic,
                            st.session_state.app_state['framework'],
                            st.session_state.app_state['prompt'],
                            len(st.session_state.app_state['analysis_results']),
                            st.session_state.app_state['iterations'],
                            previous_analysis
                        )
                        if result:
                            st.session_state.app_state['analysis_results'].append(result)
                            
                            if len(st.session_state.app_state['analysis_results']) == st.session_state.app_state['iterations']:
                                st.session_state.app_state['show_summary'] = True
                            st.rerun()
                
                # Display analysis results
                display_analysis(st.session_state.app_state['analysis_results'])
        
        # Process summary
        if st.session_state.app_state['show_summary']:
            with summary_container:
                if not st.session_state.app_state['summary']:
                    with st.spinner("📊 Generating final report..."):
                        summary = SynthesisExpert(model).synthesize(
                            topic,
                            st.session_state.app_state['analysis_results']
                        )
                        if summary:
                            st.session_state.app_state['summary'] = summary
                            st.rerun()
                
                if st.session_state.app_state['summary']:
                    display_final_report(st.session_state.app_state['summary'])
    except Exception as e:
        handle_error(e, "analysis")
        return

if __name__ == "__main__":
    main() 