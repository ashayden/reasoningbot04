"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai
import time
import re
from typing import Optional, Tuple, List, Dict, Any

from config import GEMINI_MODEL
from utils import validate_topic, sanitize_topic, QuotaExceededError, MARAError
from agents import PreAnalysisAgent, PromptDesigner, FrameworkEngineer, ResearchAnalyst, SynthesisExpert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

def initialize_gemini():
    """Initialize the Gemini model with caching."""
    try:
        # Check if API key exists in secrets
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("Google API key not found in Streamlit secrets.")
            logger.error("Google API key not found in secrets")
            return None
            
        api_key = st.secrets["GOOGLE_API_KEY"]
        if not api_key:
            st.error("Google API key is empty. Please check your Streamlit secrets.")
            logger.error("Google API key is empty")
            return None
            
        # Configure the API
        genai.configure(api_key=api_key)
        
        try:
            # Initialize the model
            model = genai.GenerativeModel(GEMINI_MODEL)
            logger.info("Successfully initialized Gemini model")
            return model
                
        except Exception as e:
            if "429" in str(e):
                st.error("API quota exceeded. Please wait a few minutes and try again.")
                logger.error("API quota exceeded during initialization")
            else:
                st.error(f"Failed to initialize Gemini model: {str(e)}")
                logger.error(f"Failed to initialize or test Gemini model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        logger.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

# Initialize model early
@st.cache_resource
def get_model():
    return initialize_gemini()

model = get_model()
if not model:
    st.error("Failed to initialize the AI model. Please check your API key in Streamlit secrets and try again.")
    st.stop()

def initialize_app_state():
    """Initialize the application state if it doesn't exist."""
    logger.info("Initializing new app_state")
    
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
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
            'focus_selection_complete': False,
            'selected_areas': [],
            'prompt': None
        }
        logger.info("Created new app_state")
    else:
        logger.info("Using existing app_state")
        
    logger.info(f"Current app_state: {st.session_state.app_state}")
    
    # Return early if no topic is set
    if not st.session_state.app_state.get('topic'):
        logger.info("No topic in app_state, returning")
        return

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
        'focus_selection_complete': False,
        'selected_areas': []
    }

def reset_state(topic, iterations):
    """Reset the application state."""
    logger.info(f"Resetting state with topic: '{topic}' and iterations: {iterations}")
    logger.info(f"Previous app_state: {st.session_state.app_state}")
    
    st.session_state.app_state = initialize_state()
    st.session_state.app_state.update({
        'topic': topic,
        'iterations': iterations,
        'show_insights': True
    })
    
    # Clear focus area states
    if 'focus_area_expanded' in st.session_state:
        logger.info("Clearing focus_area_expanded from session state")
        del st.session_state.focus_area_expanded
    if 'current_focus_areas' in st.session_state:
        logger.info("Clearing current_focus_areas from session state")
        del st.session_state.current_focus_areas
    
    logger.info(f"New app_state: {st.session_state.app_state}")

def display_insights(insights: dict):
    """Display insights in proper containers."""
    with st.container():
        with st.expander("💡 Did You Know?", expanded=True):
            st.markdown(insights['did_you_know'])
        
        with st.expander("⚡ ELI5", expanded=True):
            st.markdown(insights['eli5'])

def display_focus_areas(focus_areas):
    """Display focus areas for selection."""
    logger.info("Displaying focus areas for selection")
    
    if not focus_areas:
        logger.error("No focus areas provided to display")
        st.error("Failed to load focus areas. Please try again.")
        return
    
    with st.expander("🎯 Focus Areas", expanded=True):
        st.write("Choose 3-5 areas to focus your analysis on:")
        
        # Initialize selected_areas in session state if not present
        if 'selected_areas' not in st.session_state.app_state:
            st.session_state.app_state['selected_areas'] = []
        
        # Create columns for focus area selection
        cols = st.columns(2)
        for i, area in enumerate(focus_areas):
            col_idx = i % 2
            with cols[col_idx]:
                key = f"focus_area_{i}"
                is_selected = area in st.session_state.app_state['selected_areas']
                if st.checkbox(area, value=is_selected, key=key):
                    if area not in st.session_state.app_state['selected_areas']:
                        logger.info(f"Selected focus area: {area}")
                        st.session_state.app_state['selected_areas'].append(area)
                elif area in st.session_state.app_state['selected_areas']:
                    logger.info(f"Deselected focus area: {area}")
                    st.session_state.app_state['selected_areas'].remove(area)
        
        # Show warning if too few or too many areas selected
        num_selected = len(st.session_state.app_state['selected_areas'])
        if num_selected < 3:
            st.warning("Please select at least 3 focus areas")
        elif num_selected > 5:
            st.warning("Please select no more than 5 focus areas")
        else:
            st.success(f"You have selected {num_selected} focus areas")
            
            # Add a confirm button when the selection is valid
            if st.button("Confirm Selection", type="primary"):
                logger.info(f"Focus areas confirmed: {st.session_state.app_state['selected_areas']}")
                st.session_state.app_state['focus_selection_complete'] = True
                st.session_state.app_state['show_framework'] = True
                st.rerun()

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
    logger.info(f"Processing stage: {stage_name}")
    logger.info(f"Current app_state: {st.session_state.app_state}")
    
    try:
        # Check if we need to generate content
        if stage_name not in st.session_state.app_state or st.session_state.app_state[stage_name] is None:
            logger.info(f"Stage {stage_name} not in app_state or is None, generating content")
            
            with container, st.spinner(spinner_text):
                try:
                    # Call the stage function with error handling
                    logger.info(f"Calling stage function for {stage_name}")
                    result = stage_fn(**kwargs)
                    
                    # Log the result details
                    logger.info(f"Stage function result type: {type(result)}")
                    if result is not None:
                        logger.info(f"Stage function result: {result}")
                    else:
                        logger.error(f"Stage function returned None for {stage_name}")
                        raise ValueError(f"Failed to generate content for {stage_name}")
                        
                except Exception as e:
                    logger.error(f"Error in stage function: {str(e)}", exc_info=True)
                    st.error(f"Failed to generate content: {str(e)}")
                    # Reset the show flag for this stage
                    st.session_state.app_state[f'show_{stage_name}'] = False
                    return
                
                if result is not None:
                    logger.info(f"Stage {stage_name} generated content successfully")
                    # Store the result in app state
                    st.session_state.app_state[stage_name] = result
                    
                    # Handle next stage transition
                    if next_stage:
                        # Only show framework after focus selection is complete
                        if next_stage == 'framework' and not st.session_state.app_state.get('focus_selection_complete'):
                            logger.info("Skipping framework stage until focus selection is complete")
                            return
                        logger.info(f"Setting show_{next_stage} to True")
                        st.session_state.app_state[f'show_{next_stage}'] = True
                    
                    # Trigger rerun to update UI
                    st.rerun()
                else:
                    logger.error(f"Stage {stage_name} failed to generate content")
                    st.error(f"Failed to generate content for {stage_name}. Please try again.")
                    # Reset the show flag for this stage
                    st.session_state.app_state[f'show_{stage_name}'] = False
                    return
                    
        # Display existing content if available
        elif display_fn and st.session_state.app_state.get(stage_name) is not None:
            logger.info(f"Displaying existing content for stage {stage_name}")
            try:
                with container:
                    display_fn(st.session_state.app_state[stage_name])
                
                # Generate optimized prompt after insights are displayed
                if stage_name == 'insights' and not st.session_state.app_state.get('prompt'):
                    logger.info("Generating optimized prompt after insights display")
                    try:
                        optimized_prompt = PromptDesigner(model).design_prompt(st.session_state.app_state['topic'])
                        if optimized_prompt:
                            st.session_state.app_state['prompt'] = optimized_prompt
                            logger.info("Optimized prompt generated successfully")
                    except Exception as e:
                        logger.error(f"Error generating optimized prompt: {str(e)}", exc_info=True)
                        
            except Exception as e:
                logger.error(f"Error displaying content for stage {stage_name}: {str(e)}", exc_info=True)
                st.error(f"Failed to display content for {stage_name}. Please try again.")
                # Reset the show flag for this stage
                st.session_state.app_state[f'show_{stage_name}'] = False
                
    except Exception as e:
        logger.error(f"Unexpected error in stage {stage_name}: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred during {stage_name}. Please try again.")
        # Reset the show flag for this stage
        st.session_state.app_state[f'show_{stage_name}'] = False

def validate_and_sanitize_input(topic: str) -> tuple[bool, str, str]:
    """Validate and sanitize user input."""
    logger.info(f"Validating topic: '{topic}'")
    
    # First check if topic is None or empty
    if topic is None:
        logger.error("Topic is None")
        return False, "Please enter a topic to analyze.", ""
    
    # Validate the topic
    is_valid, error_msg = validate_topic(topic)
    if not is_valid:
        logger.error(f"Topic validation failed: {error_msg}")
        return False, error_msg, ""
    
    # Sanitize the topic
    sanitized = sanitize_topic(topic)
    if not sanitized:
        logger.error("Sanitized topic is empty")
        return False, "Please enter a valid topic to analyze.", ""
    
    logger.info(f"Topic validated and sanitized successfully: '{sanitized}'")
    return True, "", sanitized

def handle_error(e: Exception, context: str):
    """Handle errors consistently."""
    error_msg = f"Error during {context}: {str(e)}"
    logger.error(error_msg)
    
    # Handle quota exceeded errors with clear user feedback
    if isinstance(e, QuotaExceededError):
        st.error("⚠️ API quota limit reached. Please wait 5 minutes before trying again.")
        st.info("💡 This helps ensure fair usage of the API for all users.")
        # Disable the form temporarily
        st.session_state.form_disabled = True
        # Schedule re-enable after 5 minutes
        st.session_state.quota_reset_time = time.time() + 300  # 5 minutes
        return
    
    # Provide user-friendly error message for other errors
    user_msg = {
        'insights': "Failed to generate initial insights. Please try again.",
        'prompt': "Failed to optimize the prompt. Please try again.",
        'focus': "Failed to generate focus areas. Please try again.",
        'framework': "Failed to build analysis framework. Please try again.",
        'analysis': "Failed during analysis. Please try again.",
        'summary': "Failed to generate final report. Please try again."
    }.get(context, "An unexpected error occurred. Please try again.")
    
    st.error(user_msg)
    
    # Reset state for the current stage
    if context in st.session_state.app_state:
        st.session_state.app_state[context] = None

def handle_form_submission(topic: str, iterations: int):
    """Handle form submission and validate input."""
    logger.info(f"Form submitted with topic: '{topic}' and iterations: {iterations}")
    
    # Validate and sanitize input
    is_valid, error_msg, sanitized_topic = validate_and_sanitize_input(topic)
    if not is_valid:
        logger.error(f"Topic validation failed: {error_msg}")
        st.error(error_msg)
        return
    
    logger.info(f"Topic validated successfully. Sanitized topic: '{sanitized_topic}'")
    
    # Reset state and start analysis
    reset_state(sanitized_topic, iterations)
    logger.info("State reset, initiating rerun")
    st.rerun()

def main():
    """Main application function."""
    # Initialize app_state if not exists
    if 'app_state' not in st.session_state:
        logger.info("Initializing new app_state")
        st.session_state.app_state = initialize_state()
    else:
        logger.info("Using existing app_state")
        logger.info(f"Current app_state: {st.session_state.app_state}")
    
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
        handle_form_submission(topic, iterations)
    
    # Only proceed with analysis if we have a topic
    if not st.session_state.app_state.get('topic'):
        logger.info("No topic in app_state, returning")
        return
    
    try:
        logger.info("Starting analysis process")
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
            logger.info("Processing insights stage")
            try:
                process_stage('insights', containers['insights'],
                            lambda **kwargs: PreAnalysisAgent(model).generate_insights(st.session_state.app_state['topic']),
                            'focus', spinner_text="💡 Generating insights...",
                            display_fn=display_insights)
            except Exception as e:
                logger.error(f"Error in insights stage: {str(e)}", exc_info=True)
                st.error("Failed to generate insights. Please try again.")
                return
        
        if st.session_state.app_state.get('show_focus'):
            logger.info("Processing focus areas stage")
            process_stage('focus', containers['focus'],
                         lambda **kwargs: PreAnalysisAgent(model).generate_focus_areas(st.session_state.app_state['topic']),
                         'framework', spinner_text="🎯 Generating focus areas...",
                         display_fn=display_focus_areas)
        
        # Only show framework after focus selection is complete
        if st.session_state.app_state.get('show_framework') and st.session_state.app_state.get('focus_selection_complete'):
            logger.info("Processing framework stage")
            process_stage('framework', containers['framework'],
                         lambda **kwargs: FrameworkEngineer(model).create_framework(
                             st.session_state.app_state.get('prompt', ''),
                             st.session_state.app_state.get('enhanced_prompt')
                         ),
                         'analysis', spinner_text="🔨 Building analysis framework...",
                         display_fn=lambda x: st.expander("🎯 Research Framework", expanded=False).markdown(x))
        
        # Process analysis (special handling due to iterations)
        if st.session_state.app_state.get('show_analysis'):
            logger.info("Processing analysis stage")
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
            logger.info("Processing summary stage")
            process_stage('summary', containers['summary'],
                         lambda **kwargs: SynthesisExpert(model).synthesize(
                             st.session_state.app_state['topic'],
                             st.session_state.app_state.get('analysis_results', [])
                         ),
                         spinner_text="📊 Generating final report...",
                         display_fn=lambda x: st.expander("📊 Final Report", expanded=False).markdown(x))
                     
    except Exception as e:
        logger.error(f"Error in analysis process: {str(e)}", exc_info=True)
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