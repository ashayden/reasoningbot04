"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai

from config import GEMINI_MODEL, DEPTH_ITERATIONS
from utils import validate_topic, sanitize_topic
from agents import PromptDesigner, FrameworkEngineer, ResearchAnalyst, SynthesisExpert
from state_manager import StateManager
from constants import (
    CUSTOM_CSS,
    SIDEBAR_CONTENT,
    TOPIC_INPUT,
    DEPTH_SELECTOR,
    ERROR_MESSAGES
)
from exceptions import MARAException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="M.A.R.A. - Multi-Agent Reasoning Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

# Initialize Gemini
@st.cache_resource
def initialize_gemini():
    """Initialize the Gemini model with caching."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logger.error(ERROR_MESSAGES['API_INIT'].format(error=str(e)))
        StateManager.show_error('API_INIT', e)
        return None

def analyze_topic(model, topic: str, iterations: int = 1):
    """Perform multi-agent analysis of a topic.
    
    Args:
        model: The initialized Gemini model
        topic: The topic to analyze
        iterations: Number of analysis iterations to perform
        
    Returns:
        Tuple of (framework, analysis_results, summary) or (None, None, None) on error
    """
    try:
        # Validate and sanitize input
        is_valid, error_msg = validate_topic(topic)
        if not is_valid:
            st.error(error_msg)
            return None, None, None
            
        topic = sanitize_topic(topic)
        
        # Initialize agents
        framework_engineer = FrameworkEngineer(model)
        research_analyst = ResearchAnalyst(model)
        synthesis_expert = SynthesisExpert(model)
        
        # Get analysis container
        container = StateManager.create_analysis_container()
        
        with container:
            # Agent 1: Framework Engineer
            with StateManager.show_status('FRAMEWORK') as status:
                framework = framework_engineer.create_framework(topic)
                if not framework:
                    return None, None, None
                st.markdown(framework)
                StateManager.update_status(status, 'FRAMEWORK')
            
            # Agent 2: Research Analyst
            analysis_results = []
            previous_analysis = None
            
            for iteration_num in range(iterations):
                with StateManager.show_status('ANALYSIS', iteration_num + 1) as status:
                    st.divider()
                    
                    result = research_analyst.analyze(topic, framework, previous_analysis)
                    if not result:
                        return None, None, None
                    
                    if result['title']:
                        st.markdown(f"# {result['title']}")
                    if result['subtitle']:
                        st.markdown(f"*{result['subtitle']}*")
                    if result['content']:
                        st.markdown(result['content'])
                    
                    analysis_results.append(result['content'])
                    previous_analysis = result['content']
                    st.divider()
                    StateManager.update_status(status, 'ANALYSIS', iteration_num + 1)
            
            # Agent 3: Synthesis Expert
            with StateManager.show_status('SYNTHESIS') as status:
                summary = synthesis_expert.synthesize(topic, analysis_results)
                if not summary:
                    return None, None, None
                st.markdown(summary)
                StateManager.update_status(status, 'SYNTHESIS')
                
            return framework, analysis_results, summary
            
    except MARAException as e:
        logger.error(f"MARA error: {str(e)}")
        st.error(str(e))
        return None, None, None
    except Exception as e:
        logger.error(ERROR_MESSAGES['ANALYSIS_ERROR'].format(error=str(e)))
        StateManager.show_error('ANALYSIS_ERROR', e)
        return None, None, None

# Initialize session state
StateManager.initialize_session_state()

# Main UI
with st.sidebar:
    st.markdown(SIDEBAR_CONTENT)

# Initialize model
model = initialize_gemini()
if not model:
    st.stop()

# Input form
with st.form("analysis_form"):
    topic = st.text_input(
        TOPIC_INPUT['LABEL'],
        placeholder=TOPIC_INPUT['PLACEHOLDER']
    )
    
    depth = st.select_slider(
        DEPTH_SELECTOR['LABEL'],
        options=list(DEPTH_ITERATIONS.keys()),
        value=DEPTH_SELECTOR['DEFAULT']
    )
    
    submit = st.form_submit_button("🚀 Start Analysis")

if submit and topic:
    # Clear previous results if topic changed
    current_topic = StateManager.get_current_topic()
    if current_topic != topic:
        StateManager.clear_results()
        StateManager.update_analysis_results(topic)
    
    iterations = DEPTH_ITERATIONS[depth]
    framework, analysis, summary = analyze_topic(model, topic, iterations)
    
    if framework and analysis and summary:
        StateManager.update_analysis_results(topic, framework, analysis, summary)
        StateManager.show_success() 