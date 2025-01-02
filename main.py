"""Main application file for the MARA research assistant."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
import streamlit as st

from agents import (
    PreAnalysisAgent,
    PromptDesigner,
    FrameworkEngineer,
    ResearchAnalyst,
    SynthesisExpert
)
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_VISION_MODEL,
    MAX_ITERATIONS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="MARA Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.block-container {
    padding: 2rem 1rem;
    max-width: 1000px;
}

.element-container {
    margin-bottom: 1rem;
}

/* Input container styling */
div[data-testid="stForm"] {
    border: 1px solid rgba(250, 250, 250, 0.2);
    border-radius: 4px;
    padding: 1.5rem;
    background-color: rgba(28, 31, 34, 0.6);
    margin-bottom: 2rem;
}

/* Button styling */
.stButton > button {
    width: 100%;
    border-radius: 4px;
    padding: 0.75rem !important;
    font-weight: 500;
    font-size: 1.1rem !important;
    height: auto !important;
    min-height: 3rem !important;
    margin-top: 1.5rem;
    background-color: rgb(0, 102, 204) !important;
    border: none !important;
    color: white !important;
}

.stButton > button:hover {
    background-color: rgb(0, 122, 244) !important;
    border: none !important;
    color: white !important;
}

.stButton > button:active {
    background-color: rgb(0, 82, 164) !important;
    border: none !important;
    color: white !important;
}

/* Text input styling */
.stTextInput > div > div > input {
    font-size: 1.1rem;
    padding: 0.75rem;
    border-radius: 4px;
    background-color: rgba(28, 31, 34, 0.8);
    border: 1px solid rgba(250, 250, 250, 0.2);
    color: rgb(250, 250, 250);
    margin-bottom: 1rem;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(250, 250, 250, 0.6);
}

/* Slider styling */
.stSlider > div > div > div > div {
    background-color: rgb(0, 102, 204) !important;
}

.stSlider > div > div > div > div > div {
    background-color: white !important;
}

div[data-baseweb="select"] > div {
    background-color: rgba(28, 31, 34, 0.8) !important;
    border: 1px solid rgba(250, 250, 250, 0.2) !important;
    color: rgb(250, 250, 250) !important;
}

/* Help text styling */
.stMarkdown {
    font-size: 1rem;
    line-height: 1.6;
    color: rgb(250, 250, 250);
}

/* Expander styling */
div[data-testid="stExpander"] {
    border: none;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    border-radius: 4px;
    margin-bottom: 1rem;
    background-color: rgba(28, 31, 34, 0.6);
}

div[data-testid="stExpander"] > div[role="button"] {
    padding: 1rem;
    color: rgb(250, 250, 250);
}

div[data-testid="stExpander"] > div[data-testid="stExpanderContent"] {
    padding: 1rem;
    background-color: rgba(28, 31, 34, 0.4);
}

/* Select slider styling */
div[data-testid="stSelectSlider"] label {
    color: rgb(250, 250, 250) !important;
    font-size: 1rem !important;
}

div[data-testid="stSelectSlider"] div[role="slider"] {
    background-color: rgb(0, 102, 204) !important;
    border: 2px solid white !important;
}

div[data-testid="stSelectSlider"] div[data-testid="stTickBar"] > div {
    background-color: rgba(250, 250, 250, 0.2) !important;
}

div[data-testid="stSelectSlider"] div[data-testid="stTickBar"] > div:hover {
    background-color: rgba(250, 250, 250, 0.4) !important;
}

/* Header styling */
h1 {
    font-size: 2.5rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
    color: rgb(250, 250, 250) !important;
}

/* Title container */
div[data-testid="stTitle"] {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

/* Input label */
.stTextInput > label {
    font-size: 1.1rem !important;
    color: rgb(250, 250, 250) !important;
    margin-bottom: 0.5rem !important;
}

/* Help tooltip */
div[data-baseweb="tooltip"] {
    background-color: rgba(28, 31, 34, 0.95) !important;
    border: 1px solid rgba(250, 250, 250, 0.2) !important;
    border-radius: 4px !important;
    color: rgb(250, 250, 250) !important;
    padding: 0.5rem !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize or reset session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.topic = ""
        st.session_state.depth = 2  # Default depth
        st.session_state.insights = None
        st.session_state.focus_areas = None
        st.session_state.selected_focus_areas = None
        st.session_state.optimized_prompt = None
        st.session_state.framework = None
        st.session_state.analyses = []
        st.session_state.final_report = None
        st.session_state.focus_area_expanded = True
        st.session_state.did_you_know_expanded = True
        st.session_state.eli5_expanded = True

def initialize_models():
    """Initialize Gemini models."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        vision_model = genai.GenerativeModel(GEMINI_VISION_MODEL)
        return model, vision_model
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        st.error("Failed to initialize AI models. Please check your API key and try again.")
        return None, None

def display_header():
    """Display application header."""
    st.markdown('<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">'
                '<div style="font-size: 2.5rem;">🔬</div>'
                '<div>'
                '<h1 style="margin: 0;">MARA Research Assistant</h1>'
                '<p style="margin: 0; color: rgb(250, 250, 250);">Welcome to the <strong>Multi-Agent Research Assistant (MARA)</strong>! '
                'Enter your research topic or question below to begin the analysis process.</p>'
                '</div>'
                '</div>', unsafe_allow_html=True)

def get_topic_input() -> Tuple[str, int, bool]:
    """Get the research topic and analysis parameters from user input."""
    # Create a container for input fields
    with st.container():
        # Topic input
        topic = st.text_input(
            "Enter your research topic or question:",
            value=st.session_state.topic,
            key="topic_input",
            placeholder="e.g., What are the implications of quantum computing on cryptography?"
        ).strip()
        
        # Create two columns for depth selector and start button
        col1, col2 = st.columns([2, 1])
        
        # Depth selector in first column
        with col1:
            depth = st.select_slider(
                "Analysis Depth",
                options=[1, 2, 3, 4, 5],
                value=st.session_state.depth,
                key="depth_selector",
                help="Choose 1-5 iterations. More iterations = deeper analysis = longer processing time."
            )
        
        # Start button in second column
        with col2:
            start_clicked = st.button(
                "🚀 Start Analysis",
                type="primary",
                key="start_button",
                help="Begin the research analysis process"
            )
        
        return topic, depth, start_clicked

def display_insights(insights: Dict[str, str]):
    """Display quick insights about the topic."""
    if not insights:
        return
        
    with st.expander("💡 Did You Know?", expanded=st.session_state.did_you_know_expanded):
        st.markdown(insights.get('did_you_know', ''))
    
    with st.expander("⚡ ELI5", expanded=st.session_state.eli5_expanded):
        st.markdown(insights.get('eli5', ''))

def display_focus_selection(focus_areas: List[str]) -> Optional[List[str]]:
    """Display and handle focus area selection."""
    if not focus_areas:
        return None
        
    with st.expander("🎯 Focus Areas", expanded=st.session_state.focus_area_expanded):
        st.markdown("Select focus areas to emphasize in the analysis:")
        selected = st.multiselect(
            "",
            options=focus_areas,
            default=st.session_state.selected_focus_areas,
            key="focus_areas_select",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Continue", key="focus_continue", type="primary"):
                st.session_state.focus_area_expanded = False
                return selected
        with col2:
            if st.button("Skip", key="focus_skip"):
                st.session_state.focus_area_expanded = False
                return None
    
    return None

def display_optimized_prompt(prompt: str):
    """Display the optimized research prompt."""
    if not prompt:
        return
        
    with st.expander("🎯 Optimized Prompt", expanded=False):
        st.markdown(prompt)

def display_framework(framework: str):
    """Display the research framework."""
    if not framework:
        return
        
    with st.expander("📋 Research Framework", expanded=False):
        st.markdown(framework)

def display_analysis_results(analyses: List[Dict[str, str]]):
    """Display research analysis results."""
    if not analyses:
        return
        
    with st.expander("📊 Research Analysis", expanded=False):
        for i, analysis in enumerate(analyses, 1):
            if analysis.get('title'):
                st.markdown(f"### Analysis {i}: {analysis['title']}")
            if analysis.get('subtitle'):
                st.markdown(f"*{analysis['subtitle']}*")
            if analysis.get('content'):
                st.markdown(analysis['content'])
            if i < len(analyses):
                st.divider()

def display_final_report(report: str):
    """Display the final synthesized report."""
    if not report:
        return
        
    with st.expander("📑 Final Report", expanded=False):
        st.markdown(report)

def main():
    """Main application function."""
    initialize_session_state()
    model, vision_model = initialize_models()
    if not model or not vision_model:
        return
        
    display_header()
    topic, depth, start_clicked = get_topic_input()
    
    # Update session state
    if topic:
        st.session_state.topic = topic
        st.session_state.depth = depth
    
    # Only proceed if we have a topic and the start button was clicked
    if not (topic and start_clicked):
        return
    
    # Initialize agents
    pre_analysis = PreAnalysisAgent(model)
    prompt_designer = PromptDesigner(model)
    framework_engineer = FrameworkEngineer(model)
    research_analyst = ResearchAnalyst(model)
    synthesis_expert = SynthesisExpert(model)
    
    # Generate initial insights
    if not st.session_state.insights:
        with st.spinner("Generating quick insights..."):
            st.session_state.insights = pre_analysis.generate_insights(topic)
    
    if st.session_state.insights:
        display_insights(st.session_state.insights)
    
    # Generate focus areas
    if not st.session_state.focus_areas:
        with st.spinner("Identifying potential focus areas..."):
            st.session_state.focus_areas = prompt_designer.generate_focus_areas(topic)
    
    if st.session_state.focus_areas:
        selected = display_focus_selection(st.session_state.focus_areas)
        if selected is not None:
            st.session_state.selected_focus_areas = selected
            st.rerun()
    
    # Generate optimized prompt
    if not st.session_state.optimized_prompt and st.session_state.selected_focus_areas is not None:
        with st.spinner("Optimizing research prompt..."):
            st.session_state.optimized_prompt = prompt_designer.design_prompt(
                topic,
                st.session_state.selected_focus_areas
            )
            st.rerun()
    
    if st.session_state.optimized_prompt:
        display_optimized_prompt(st.session_state.optimized_prompt)
    
    # Generate research framework
    if not st.session_state.framework and st.session_state.optimized_prompt:
        with st.spinner("Creating research framework..."):
            st.session_state.framework = framework_engineer.create_framework(
                st.session_state.optimized_prompt
            )
    
    if st.session_state.framework:
        display_framework(st.session_state.framework)
    
    # Conduct research analysis
    if (st.session_state.framework and 
        len(st.session_state.analyses) < depth):  # Use depth parameter
        with st.spinner(f"Conducting research analysis (iteration {len(st.session_state.analyses) + 1} of {depth})..."):
            previous = st.session_state.analyses[-1] if st.session_state.analyses else None
            analysis = research_analyst.analyze(
                topic,
                st.session_state.framework,
                previous['content'] if previous else None
            )
            if analysis:
                st.session_state.analyses.append(analysis)
    
    if st.session_state.analyses:
        display_analysis_results(st.session_state.analyses)
    
    # Generate final report
    if (st.session_state.analyses and 
        len(st.session_state.analyses) >= depth and  # Use depth parameter
        not st.session_state.final_report):
        with st.spinner("Synthesizing final report..."):
            analyses_text = [a['content'] for a in st.session_state.analyses if a.get('content')]
            st.session_state.final_report = synthesis_expert.synthesize(topic, analyses_text)
    
    if st.session_state.final_report:
        display_final_report(st.session_state.final_report)

if __name__ == "__main__":
    main() 