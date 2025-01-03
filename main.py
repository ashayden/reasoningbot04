"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai
import os
from typing import Dict

from config import GEMINI_MODEL
from utils import validate_topic, sanitize_topic
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
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

def initialize_state():
    """Initialize application state with default values."""
    return {
        'topic': None,
        'iterations': None,
        'insights': None,
        'prompt': None,
        'focus_areas': None,
        'selected_areas': [],
        'framework': None,
        'analysis_results': [],
        'summary': None,
        'show_insights': False,
        'show_prompt': False,
        'show_focus': False,
        'show_framework': False,
        'show_analysis': False,
        'show_summary': False,
        'error': None,
        'focus_selection_complete': False
    }

# Initialize state
if 'app_state' not in st.session_state:
    st.session_state.app_state = initialize_state()

# Initialize Gemini
@st.cache_resource
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
            
            # Test the model with a minimal prompt
            logger.info("Testing Gemini model initialization...")
            response = model.generate_content("Hello")
            
            # Extract content from response, ignoring thought process
            try:
                # Get the full response text
                full_text = response.candidates[0].content.parts[0].text
                
                # Split into lines and filter out thought process
                lines = full_text.split('\n')
                content_lines = []
                in_thought_block = False
                
                thought_starters = [
                    "first i will", "i will", "thoughts", "thinking process", "let me",
                    "i need to", "i'll", "step 1", "step 2", "the user wants",
                    "draft answer", "self-critique", "i am thinking", "i should",
                    "my approach", "i'm going to", "let's", "here's how", "my plan",
                    "i understand", "to answer this", "to respond", "my response",
                    "my analysis", "i can see", "i notice", "i observe",
                    "first, ", "second, ", "third, ", "finally, ", "next, ",
                    "to start", "to begin", "i must", "i have to"
                ]
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line starts a thought block
                    if any(line.lower().startswith(starter) for starter in thought_starters):
                        in_thought_block = True
                        continue
                    
                    # Check if this line is part of a thought block
                    if any(marker in line.lower() for marker in [
                        "i will", "i'll", "i should", "i need", "i must",
                        "i have to", "i am", "i'm", "let me", "let's"
                    ]):
                        in_thought_block = True
                        continue
                    
                    # If we're not in a thought block, add the line to content
                    if not in_thought_block:
                        content_lines.append(line)
                    
                    # Reset thought block if we hit a clear content marker
                    if any(marker in line for marker in ["🌟", "💡", "🔍", "📝", "🎯", "⚡"]):
                        in_thought_block = False
                
                # Join the content lines
                test_text = "\n".join(content_lines).strip()
                if not test_text:
                    st.error("Model initialization test failed - no valid content in response")
                    logger.error("Model initialization test failed - no valid content in response")
                    return None
                    
                logger.info("Successfully extracted content from test response")
                return model
                
            except (AttributeError, IndexError) as e:
                st.error(f"Model initialization test failed - invalid response structure: {str(e)}")
                logger.error(f"Model initialization test failed - invalid response structure: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"Failed to initialize or test Gemini model: {str(e)}")
            logger.error(f"Failed to initialize or test Gemini model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        logger.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

# Initialize model early
model = initialize_gemini()
if not model:
    st.error("Failed to initialize the AI model. Please check your API key in Streamlit secrets and try again.")
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
    """Handle errors consistently with proper cleanup and user feedback."""
    error_msg = f"Error during {context}: {str(e)}"
    logger.error(error_msg)
    
    # Provide user-friendly error message
    user_msg = {
        'insights': "Failed to generate initial insights. Please try again.",
        'prompt': "Failed to optimize the prompt. Please try again.",
        'focus': "Failed to generate focus areas. Please try again.",
        'framework': "Failed to build analysis framework. Please try again.",
        'analysis': "Failed during analysis. Please try again.",
        'summary': "Failed to generate final report. Please try again."
    }.get(context, "An unexpected error occurred. Please try again.")
    
    st.error(user_msg)
    
    # Reset relevant state and stop showing subsequent stages
    if context in st.session_state.app_state:
        st.session_state.app_state[context] = None
    
    stages = ['show_insights', 'show_prompt', 'show_focus', 'show_framework', 'show_analysis', 'show_summary']
    current_stage_idx = stages.index(f'show_{context}')
    for stage in stages[current_stage_idx:]:
        st.session_state.app_state[stage] = False
    
    # Clean up any partial results
    cleanup_partial_results(context)

def reset_state(topic: str, iterations: int):
    """Reset application state with memory cleanup."""
    # Clear previous results and perform cleanup
    if hasattr(st.session_state, 'app_state'):
        if st.session_state.app_state.get('analysis_results'):
            st.session_state.app_state['analysis_results'].clear()
    
    # Reset focus area states
    st.session_state.focus_area_expanded = True
    if hasattr(st.session_state, 'current_focus_areas'):
        st.session_state.current_focus_areas = []
    
    # Initialize new state with provided values
    new_state = initialize_state()
    new_state.update({
        'topic': topic,
        'iterations': iterations,
        'show_insights': True  # Start with insights
    })
    
    st.session_state.app_state = new_state

def display_insights(insights: dict):
    """Display insights in proper containers."""
    with st.container():
        with st.expander("💡 Did You Know?", expanded=True):
            st.markdown(insights['did_you_know'])
        
        with st.expander("⚡ ELI5", expanded=True):
            st.markdown(insights['eli5'])

def display_focus_selection(focus_areas: list, selected_areas: list) -> tuple[bool, list]:
    """Display focus area selection with proper state handling."""
    # Track if the section should be expanded in session state
    if 'focus_area_expanded' not in st.session_state:
        st.session_state.focus_area_expanded = True
        
    # Track selected areas in session state to persist between reruns
    if 'current_focus_areas' not in st.session_state:
        st.session_state.current_focus_areas = selected_areas
    
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
                    return True, selected
                
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
                    return True, selected
            
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

def process_stage(stage_name: str, container, process_fn, next_stage: str = None, **kwargs):
    """Process a single stage of the analysis pipeline."""
    if not st.session_state.app_state[f'show_{stage_name}']:
        return
        
    with container:
        try:
            state_key = stage_name if stage_name != 'focus' else 'focus_areas'
            
            # Check if we need to process this stage
            if not st.session_state.app_state[state_key]:
                with st.spinner(f"{kwargs.get('spinner_text', 'Processing...')}"):
                    logger.info(f"Starting {stage_name} stage processing...")
                    try:
                        # Process the stage
                        result = process_fn(**kwargs)
                        logger.info(f"Process function for {stage_name} completed. Result exists: {result is not None}")
                        
                        if result:
                            # Store the result and update state
                            st.session_state.app_state[state_key] = result
                            if next_stage:
                                st.session_state.app_state[f'show_{next_stage}'] = True
                                logger.info(f"Moving to next stage: {next_stage}")
                                st.rerun()  # Use experimental_rerun for more reliable state updates
                        else:
                            # Handle failed processing
                            logger.error(f"Process function for {stage_name} returned None")
                            handle_error(Exception(f"Failed to generate {stage_name}"), stage_name)
                            return
                            
                    except Exception as e:
                        # Handle processing errors
                        logger.error(f"Error in process function for {stage_name}: {str(e)}")
                        handle_error(e, stage_name)
                        return
            
            # Display the result if we have it
            if st.session_state.app_state[state_key]:
                display_fn = kwargs.get('display_fn')
                if display_fn:
                    try:
                        display_fn(st.session_state.app_state[state_key])
                    except Exception as e:
                        logger.error(f"Error displaying {stage_name} result: {str(e)}")
                        handle_error(e, stage_name)
                        return
                    
        except Exception as e:
            # Handle any other errors
            logger.error(f"Outer error in {stage_name} stage: {str(e)}")
            handle_error(e, stage_name)
            return

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
        is_valid, error_msg, sanitized_topic = validate_and_sanitize_input(topic)
        if not is_valid:
            st.error(error_msg)
            return
        
        # Reset state and start analysis
        st.session_state.app_state = initialize_state()  # Ensure complete state reset
        st.session_state.app_state.update({
            'topic': sanitized_topic,
            'iterations': iterations,
            'show_insights': True  # Start with insights
        })
        
        # Clear any previous focus area state
        if 'focus_area_expanded' in st.session_state:
            del st.session_state.focus_area_expanded
        if 'current_focus_areas' in st.session_state:
            del st.session_state.current_focus_areas
            
        st.rerun()

    # Only proceed with analysis if we have a topic
    if not st.session_state.app_state.get('topic'):
        return

    try:
        # Create containers
        containers = {
            'insights': st.container(),
            'prompt': st.container(),
            'focus': st.container(),
            'framework': st.container(),
            'analysis': st.container(),
            'summary': st.container()
        }
        
        # Process each stage
        if st.session_state.app_state['show_insights']:
            process_stage('insights', containers['insights'],
                         lambda **kwargs: PreAnalysisAgent(model).generate_insights(st.session_state.app_state['topic']),
                         'prompt', spinner_text="💡 Generating insights...",
                         display_fn=display_insights)
        
        if st.session_state.app_state['show_prompt']:
            process_stage('prompt', containers['prompt'],
                         lambda **kwargs: PromptDesigner(model).design_prompt(st.session_state.app_state['topic']),
                         'focus', spinner_text="✍️ Optimizing prompt...",
                         display_fn=lambda x: st.expander("✍️ Optimized Prompt", expanded=False).markdown(x))
        
        if st.session_state.app_state['show_focus']:
            with containers['focus']:
                if not st.session_state.app_state['focus_areas']:
                    with st.spinner("🎯 Generating focus areas..."):
                        focus_areas = PromptDesigner(model).generate_focus_areas(st.session_state.app_state['topic'])
                        if focus_areas:
                            st.session_state.app_state['focus_areas'] = focus_areas
                            st.rerun()  # Ensure UI updates with new focus areas
                
                if st.session_state.app_state['focus_areas']:
                    proceed, selected = display_focus_selection(
                        st.session_state.app_state['focus_areas'],
                        st.session_state.app_state['selected_areas']
                    )
                    st.session_state.app_state['selected_areas'] = selected
                    
                    if proceed:
                        with st.spinner("Enhancing prompt with focus areas..."):
                            enhanced_prompt = PromptDesigner(model).design_prompt(st.session_state.app_state['topic'], selected)
                            st.session_state.app_state['enhanced_prompt'] = enhanced_prompt
                            st.rerun()
        
        if st.session_state.app_state['show_framework']:
            process_stage('framework', containers['framework'],
                         lambda **kwargs: FrameworkEngineer(model).create_framework(
                             st.session_state.app_state['prompt'],
                             st.session_state.app_state.get('enhanced_prompt')
                         ),
                         'analysis', spinner_text="🔨 Building analysis framework...",
                         display_fn=lambda x: st.expander("🎯 Research Framework", expanded=False).markdown(x))
        
        # Process analysis (special handling due to iterations)
        if st.session_state.app_state['show_analysis']:
            with containers['analysis']:
                if len(st.session_state.app_state.get('analysis_results', [])) < st.session_state.app_state['iterations']:
                    with st.spinner("🔄 Performing analysis..."):
                        result = ResearchAnalyst(model).analyze(
                            st.session_state.app_state['topic'],
                            st.session_state.app_state['framework'],
                            st.session_state.app_state['analysis_results'][-1] if st.session_state.app_state.get('analysis_results') else None
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
        
        if st.session_state.app_state['show_summary']:
            process_stage('summary', containers['summary'],
                         lambda **kwargs: SynthesisExpert(model).synthesize(
                             st.session_state.app_state['topic'],
                             st.session_state.app_state['analysis_results']
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