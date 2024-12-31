"""Main application file for MARA."""

import logging
import os
from typing import Optional, Dict, Any

import google.generativeai as genai
import streamlit as st

from config import (
    GEMINI_MODEL,
    DEPTH_ITERATIONS
)
from constants import (
    CUSTOM_CSS,
    TOPIC_INPUT,
    DEPTH_SELECTOR
)
from agents import (
    PromptDesigner,
    FrameworkEngineer,
    ResearchAnalyst,
    SynthesisExpert
)
from state_manager import StateManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_gemini() -> Optional[Any]:
    """Initialize the Gemini model."""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
            return None
            
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {str(e)}")
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

def run_analysis(model: Any, topic: str, depth: str) -> None:
    """Run the complete analysis process."""
    try:
        # Clear previous results
        StateManager.clear_results()
        
        # Initialize agents
        prompt_designer = PromptDesigner(model)
        framework_engineer = FrameworkEngineer(model)
        research_analyst = ResearchAnalyst(model)
        synthesis_expert = SynthesisExpert(model)
        
        # Design prompt
        StateManager.show_status('PROMPT_DESIGN', 'start')
        prompt = prompt_designer.design_prompt(topic)
        if not prompt:
            raise ValueError("Empty prompt received")
        StateManager.update_analysis('prompt', prompt)
        StateManager.show_status('PROMPT_DESIGN', 'complete')
        
        # Create framework
        StateManager.show_status('FRAMEWORK', 'start')
        framework = framework_engineer.create_framework(prompt)
        if not framework:
            raise ValueError("Empty framework received")
        StateManager.update_analysis('framework', framework)
        StateManager.show_status('FRAMEWORK', 'complete')
        
        # Perform iterative analysis
        analyses = []
        previous_analysis = None
        iterations = DEPTH_ITERATIONS[depth]
        
        for i in range(iterations):
            StateManager.show_status('ANALYSIS', 'start', i + 1)
            
            result = research_analyst.analyze(topic, framework, previous_analysis)
            if not result:
                raise ValueError(f"Empty analysis received for iteration {i + 1}")
                
            # Validate analysis result structure
            if not isinstance(result, dict) or not all(k in result for k in ['title', 'subtitle', 'content']):
                raise ValueError(f"Invalid analysis result structure in iteration {i + 1}")
                
            analyses.append(result)
            previous_analysis = result.get('content')
            StateManager.show_status('ANALYSIS', 'complete', i + 1)
        
        # Generate final synthesis
        StateManager.show_status('SYNTHESIS', 'start')
        analysis_contents = [a['content'] for a in analyses if a and isinstance(a, dict) and 'content' in a]
        synthesis = synthesis_expert.synthesize(topic, analysis_contents)
        if not synthesis:
            raise ValueError("Empty synthesis received")
        
        # Update analysis results
        StateManager.update_analysis('topic', topic)
        StateManager.update_analysis('analysis', analyses)
        StateManager.update_analysis('summary', synthesis)
        StateManager.show_status('SYNTHESIS', 'complete')
        
    except Exception as e:
        logger.error(f"MARA error: {str(e)}")
        st.error(f"MARA error: {str(e)}")
        # Clear status on error
        if 'status_container' in st.session_state:
            st.session_state.status_container.empty()

def display_results() -> None:
    """Display analysis results in the UI."""
    try:
        container = StateManager.get_container()
        if not container:
            return
            
        with container:
            # Display optimized prompt
            prompt = StateManager.get_analysis('prompt')
            if prompt:
                st.markdown("### Optimized Prompt")
                st.markdown(prompt)
                st.markdown("---")
            
            # Display framework
            framework = StateManager.get_analysis('framework')
            if framework:
                st.markdown("### Analysis Framework")
                st.markdown(framework)
                st.markdown("---")
            
            # Display analyses
            analyses = StateManager.get_analysis('analysis')
            if analyses and isinstance(analyses, list):
                st.markdown("### Research Analysis")
                for i, analysis in enumerate(analyses, 1):
                    if (analysis and isinstance(analysis, dict) and 
                        all(k in analysis for k in ['title', 'subtitle', 'content'])):
                        title = analysis.get('title', f'Analysis {i}')
                        subtitle = analysis.get('subtitle', '')
                        content = analysis.get('content', '')
                        
                        st.markdown(f"#### {title}")
                        if subtitle:
                            st.markdown(f"*{subtitle}*")
                        st.markdown(content)
                        st.markdown("---")
            
            # Display synthesis
            synthesis = StateManager.get_analysis('summary')
            if synthesis:
                st.markdown("### Final Report")
                st.markdown(synthesis)
                
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        st.error(f"Error displaying results: {str(e)}")

# Set page config
st.set_page_config(
    page_title="MARA - Multi-Agent Reasoning & Analysis",
    page_icon="🤖",
    layout="centered"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Display logo
st.image("assets/mara-logo.png", use_container_width=True)

# Initialize session state and model
StateManager.init_session_state()
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
        options=DEPTH_SELECTOR['OPTIONS'],
        value=DEPTH_SELECTOR['DEFAULT'],
        help=DEPTH_SELECTOR['HELP']
    )
    
    submitted = st.form_submit_button("Start Analysis")

# Create main content area after form
StateManager.create_container()
    
# Run analysis if form submitted
if submitted and topic:
    run_analysis(model, topic, depth)

# Display results
display_results() 