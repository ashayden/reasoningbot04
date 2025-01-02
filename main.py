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

def initialize_session_state():
    """Initialize or reset session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.topic = ""
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
    st.title("🔬 MARA Research Assistant")
    st.markdown("""
    Welcome to the **Multi-Agent Research Assistant (MARA)**! 
    Enter your research topic or question below to begin the analysis process.
    """)

def get_topic_input() -> str:
    """Get the research topic from user input."""
    topic = st.text_input(
        "Enter your research topic or question:",
        value=st.session_state.topic,
        key="topic_input",
        placeholder="e.g., What are the implications of quantum computing on cryptography?"
    )
    return topic.strip()

def display_insights(insights: Dict[str, str]):
    """Display quick insights about the topic."""
    if not insights:
        return
        
    with st.expander("💡 Did You Know?", expanded=st.session_state.did_you_know_expanded):
        st.write(insights.get('did_you_know', ''))
    
    with st.expander("⚡ ELI5", expanded=st.session_state.eli5_expanded):
        st.write(insights.get('eli5', ''))

def display_focus_selection(focus_areas: List[str]) -> Optional[List[str]]:
    """Display and handle focus area selection."""
    if not focus_areas:
        return None
        
    with st.expander("🎯 Focus Areas", expanded=st.session_state.focus_area_expanded):
        selected = st.multiselect(
            "Select focus areas to emphasize in the analysis:",
            options=focus_areas,
            default=st.session_state.selected_focus_areas,
            key="focus_areas_select"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Continue", key="focus_continue"):
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
        st.write(prompt)

def display_framework(framework: str):
    """Display the research framework."""
    if not framework:
        return
        
    with st.expander("📋 Research Framework", expanded=False):
        st.write(framework)

def display_analysis_results(analyses: List[Dict[str, str]]):
    """Display research analysis results."""
    if not analyses:
        return
        
    with st.expander("📊 Research Analysis", expanded=False):
        for i, analysis in enumerate(analyses, 1):
            if analysis.get('title'):
                st.subheader(f"Analysis {i}: {analysis['title']}")
            if analysis.get('subtitle'):
                st.write(f"*{analysis['subtitle']}*")
            if analysis.get('content'):
                st.write(analysis['content'])
            if i < len(analyses):
                st.divider()

def display_final_report(report: str):
    """Display the final synthesized report."""
    if not report:
        return
        
    with st.expander("📑 Final Report", expanded=False):
        st.write(report)

def main():
    """Main application function."""
    initialize_session_state()
    model, vision_model = initialize_models()
    if not model or not vision_model:
        return
        
    display_header()
    topic = get_topic_input()
    
    if not topic:
        return
        
    st.session_state.topic = topic
    
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
        len(st.session_state.analyses) < MAX_ITERATIONS):
        with st.spinner(f"Conducting research analysis (iteration {len(st.session_state.analyses) + 1})..."):
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
        len(st.session_state.analyses) >= MAX_ITERATIONS and 
        not st.session_state.final_report):
        with st.spinner("Synthesizing final report..."):
            analyses_text = [a['content'] for a in st.session_state.analyses if a.get('content')]
            st.session_state.final_report = synthesis_expert.synthesize(topic, analyses_text)
    
    if st.session_state.final_report:
        display_final_report(st.session_state.final_report)

if __name__ == "__main__":
    main() 