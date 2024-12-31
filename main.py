"""Main application file for MARA."""

import logging
from typing import Optional, Tuple, List, Dict, Any
import streamlit as st
import google.generativeai as genai

from config import GEMINI_MODEL, DEPTH_ITERATIONS
from utils import validate_topic, sanitize_topic
from agents import PromptDesigner, FrameworkEngineer, ResearchAnalyst, SynthesisExpert
from state_manager import StateManager
from constants import (
    CUSTOM_CSS,
    TOPIC_INPUT,
    DEPTH_SELECTOR,
    ERROR_MESSAGES
)
from exceptions import MARAException, ModelError, EmptyResponseError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AnalysisManager:
    """Manages the analysis workflow."""
    
    def __init__(self, model: Any):
        self.model = model
        self.prompt_designer = PromptDesigner(model)
        self.framework_engineer = FrameworkEngineer(model)
        self.research_analyst = ResearchAnalyst(model)
        self.synthesis_expert = SynthesisExpert(model)
    
    def run_prompt_design(self, topic: str, container: Any) -> Optional[str]:
        """Run prompt design phase."""
        with container:
            with StateManager.show_status('PROMPT_DESIGN') as status:
                try:
                    prompt = self.prompt_designer.design_prompt(topic)
                    if not prompt:
                        raise EmptyResponseError("Empty prompt received")
                    st.markdown(prompt)
                    StateManager.update_status(status, 'PROMPT_DESIGN')
                    return prompt
                except Exception as e:
                    logger.error(f"Prompt design error: {str(e)}")
                    raise ModelError(f"Prompt design failed: {str(e)}")
    
    def run_framework_analysis(self, prompt: str, container: Any) -> Optional[str]:
        """Run framework analysis phase."""
        with container:
            with StateManager.show_status('FRAMEWORK') as status:
                try:
                    framework = self.framework_engineer.create_framework(prompt)
                    if not framework:
                        raise EmptyResponseError("Empty framework received")
                    st.markdown(framework)
                    StateManager.update_status(status, 'FRAMEWORK')
                    return framework
                except Exception as e:
                    logger.error(f"Framework analysis error: {str(e)}")
                    raise ModelError(f"Framework analysis failed: {str(e)}")
    
    def run_research_analysis(
        self, topic: str, framework: str, iterations: int, container: Any
    ) -> List[str]:
        """Run research analysis phase."""
        analysis_results = []
        previous_analysis = None
        
        with container:
            for iteration_num in range(iterations):
                with StateManager.show_status('ANALYSIS', iteration_num + 1) as status:
                    try:
                        st.divider()
                        result = self.research_analyst.analyze(topic, framework, previous_analysis)
                        if not result:
                            raise EmptyResponseError("Empty analysis received")
                        
                        self._display_analysis_result(result)
                        analysis_results.append(result['content'])
                        previous_analysis = result['content']
                        
                        st.divider()
                        StateManager.update_status(status, 'ANALYSIS', iteration_num + 1)
                    except Exception as e:
                        logger.error(f"Research analysis error: {str(e)}")
                        raise ModelError(f"Research analysis failed: {str(e)}")
        
        return analysis_results
    
    def run_synthesis(self, topic: str, analyses: List[str], container: Any) -> Optional[str]:
        """Run synthesis phase."""
        with container:
            with StateManager.show_status('SYNTHESIS') as status:
                try:
                    summary = self.synthesis_expert.synthesize(topic, analyses)
                    if not summary:
                        raise EmptyResponseError("Empty synthesis received")
                    st.markdown(summary)
                    StateManager.update_status(status, 'SYNTHESIS')
                    return summary
                except Exception as e:
                    logger.error(f"Synthesis error: {str(e)}")
                    raise ModelError(f"Synthesis failed: {str(e)}")
    
    @staticmethod
    def _display_analysis_result(result: Dict[str, str]):
        """Display analysis result with proper formatting."""
        if result['title']:
            st.markdown(f"# {result['title']}")
        if result['subtitle']:
            st.markdown(f"*{result['subtitle']}*")
        if result['content']:
            st.markdown(result['content'])
    
    def analyze_topic(self, topic: str, iterations: int = 1) -> Tuple[Optional[str], Optional[List[str]], Optional[str]]:
        """Perform complete multi-agent analysis of a topic."""
        try:
            # Validate and sanitize input
            is_valid, error_msg = validate_topic(topic)
            if not is_valid:
                st.error(error_msg)
                return None, None, None
                
            topic = sanitize_topic(topic)
            container = StateManager.create_analysis_container()
            
            # Run analysis pipeline
            prompt = self.run_prompt_design(topic, container)
            framework = self.run_framework_analysis(prompt, container)
            analyses = self.run_research_analysis(topic, framework, iterations, container)
            summary = self.run_synthesis(topic, analyses, container)
            
            return framework, analyses, summary
                
        except MARAException as e:
            logger.error(f"MARA error: {str(e)}")
            st.error(str(e))
            return None, None, None
        except Exception as e:
            logger.error(ERROR_MESSAGES['ANALYSIS_ERROR'].format(error=str(e)))
            StateManager.show_error('ANALYSIS_ERROR', e)
            return None, None, None

# Page setup
st.set_page_config(
    page_title="M.A.R.A. - Multi-Agent Reasoning Assistant",
    page_icon="🤖",
    layout="wide"
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.image("assets/mara-logo.png", use_container_width=True)

# Initialize session state and model
StateManager.initialize_session_state()
model = initialize_gemini()
if not model:
    st.stop()

analysis_manager = AnalysisManager(model)

# Input form
with st.form(key="analysis_form"):
    topic = st.text_input(
        TOPIC_INPUT['LABEL'],
        placeholder=TOPIC_INPUT['PLACEHOLDER'],
        key="topic_input"
    )
    
    depth = st.select_slider(
        DEPTH_SELECTOR['LABEL'],
        options=list(DEPTH_ITERATIONS.keys()),
        value=DEPTH_SELECTOR['DEFAULT'],
        key="depth_selector"
    )
    
    submit = st.form_submit_button("🚀 Start Analysis")

# Create main content area AFTER the form
main_content = st.container()

if submit and topic:
    try:
        # Clear previous results if topic changed
        current_topic = StateManager.get_current_topic()
        if current_topic != topic:
            StateManager.clear_results()
            StateManager.update_analysis_results(topic)
        
        with main_content:
            iterations = DEPTH_ITERATIONS[depth]
            framework, analysis, summary = analysis_manager.analyze_topic(topic, iterations)
            
            if framework and analysis and summary:
                StateManager.update_analysis_results(topic, framework, analysis, summary)
                StateManager.show_success()
    except Exception as e:
        logger.error(f"Form submission error: {str(e)}")
        StateManager.show_error('ANALYSIS_ERROR', e) 