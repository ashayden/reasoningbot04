"""Main application module for MARA."""

import streamlit as st
import google.generativeai as genai
from typing import List, Optional

from agents import PreAnalysisAgent, ResearchAnalyst, SynthesisExpert
from components import (
    display_logo, input_form, display_insights,
    display_focus_areas
)
from config import (
    GEMINI_MODEL, MIN_TOPIC_LENGTH, MAX_TOPIC_LENGTH,
    ProgressiveConfig, API_RATE_LIMIT
)
from state import AppState
from utils import (
    safe_api_call, parse_gemini_response, rate_limit_decorator,
    clean_markdown_content, APIError
)

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="MARA Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def initialize_state() -> None:
    """Initialize or reset application state."""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()

@safe_api_call(retries=3)
@rate_limit_decorator(calls=API_RATE_LIMIT['calls'], period=API_RATE_LIMIT['period'])
def initialize_model():
    """Initialize the Gemini model with error handling."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except Exception as e:
        raise APIError(f"Failed to initialize Gemini model: {str(e)}")

def validate_topic(topic: str) -> tuple[bool, str]:
    """Validate the research topic."""
    if not topic or not topic.strip():
        return False, "Please enter a research topic."
    if len(topic) < MIN_TOPIC_LENGTH:
        return False, f"Topic must be at least {MIN_TOPIC_LENGTH} characters."
    if len(topic) > MAX_TOPIC_LENGTH:
        return False, f"Topic must be no more than {MAX_TOPIC_LENGTH} characters."
    return True, ""

def handle_topic_submission(topic: str, iterations: int) -> None:
    """Handle topic submission with error handling."""
    try:
        # Validate topic
        is_valid, error_message = validate_topic(topic)
        if not is_valid:
            st.error(error_message)
            return
            
        state = st.session_state.app_state
        state.last_topic = topic
        state.iterations = iterations
        state.stage = 'analysis'
        
        # Initialize model
        model = initialize_model()
        
        # Generate initial insights
        pre_analyst = PreAnalysisAgent(model)
        with st.spinner("Generating initial insights..."):
            insights = pre_analyst.generate_insights(topic)
            if insights:
                state.insights = insights
                
            focus_areas = pre_analyst.generate_focus_areas(topic)
            if focus_areas:
                state.focus_areas = focus_areas
                
        st.rerun()
        
    except APIError as e:
        st.error(f"API Error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

def handle_focus_selection(selected_areas: List[str]) -> None:
    """Handle focus area selection with validation."""
    state = st.session_state.app_state
    if len(selected_areas) > 5:
        st.error("Please select no more than 5 focus areas.")
        return
        
    state.selected_focus_areas = selected_areas
    state.stage = 'research'
    st.rerun()

def conduct_research() -> None:
    """Conduct progressive research analysis."""
    try:
        state = st.session_state.app_state
        model = initialize_model()
        analyst = ResearchAnalyst(model)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        analyses = []
        for i in range(state.iterations):
            iteration = i + 1
            status_text.text(f"Research Iteration {iteration}/{state.iterations}")
            
            # Get progressive configuration
            config = ProgressiveConfig.get_iteration_config(iteration)
            model.generation_config = genai.types.GenerationConfig(**config)
            
            # Conduct analysis
            analysis = analyst.analyze(
                state.last_topic,
                state.selected_focus_areas,
                '\n'.join(str(a) for a in analyses) if analyses else None
            )
            
            if analysis:
                analyses.append(analysis)
                
            progress = (i + 1) / state.iterations
            progress_bar.progress(progress)
            
        # Generate synthesis
        if analyses:
            synthesizer = SynthesisExpert(model)
            synthesis = synthesizer.synthesize(
                state.last_topic,
                state.selected_focus_areas,
                analyses
            )
            
            if synthesis:
                state.synthesis = synthesis
                
        state.stage = 'complete'
        st.rerun()
        
    except APIError as e:
        st.error(f"API Error: {str(e)}")
        state.stage = 'input'
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        state.stage = 'input'

def main():
    """Main application entry point."""
    initialize_state()
    state = st.session_state.app_state
    
    # Display logo
    display_logo()
    
    # Handle different application stages
    if state.stage == 'input':
        input_form(state, handle_topic_submission)
        
    elif state.stage == 'analysis':
        input_form(state, handle_topic_submission)
        display_insights(state.insights)
        display_focus_areas(state, handle_focus_selection, lambda: handle_focus_selection([]))
        
    elif state.stage == 'research':
        input_form(state, handle_topic_submission)
        display_insights(state.insights)
        conduct_research()
        
    elif state.stage == 'complete':
        input_form(state, handle_topic_submission)
        
        if state.synthesis:
            st.title(state.synthesis.get('title', 'Research Results'))
            st.markdown(clean_markdown_content(state.synthesis.get('content', '')))
            
            # Download button for report
            report_content = f"# {state.synthesis.get('title', 'Research Results')}\n\n"
            report_content += state.synthesis.get('content', '')
            
            st.download_button(
                "📥 Download Report",
                report_content,
                file_name="research_report.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main() 