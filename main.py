"""Main application file for the MARA research assistant."""

import logging
import streamlit as st
import google.generativeai as genai
from typing import List, Dict

from config import GEMINI_MODEL
from utils import validate_topic, sanitize_topic, QuotaExceededError
from agents import PreAnalysisAgent, ResearchAnalyst, SynthesisExpert
from state import AppState
from components import (
    display_logo, input_form, display_insights, display_focus_areas
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
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

[data-testid="baseButton-secondary"] {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    color: #2c3338;
    padding: 0.75rem;
    min-height: 3rem;
    transition: all 0.2s ease;
}

[data-testid="baseButton-secondary"]:hover {
    background-color: #e9ecef;
    border-color: #ced4da;
}

[data-testid="baseButton-primary"] {
    background-color: rgba(0, 102, 204, 0.1);
    border: 1px solid #0066cc;
    box-shadow: 0 0 0 1px #0066cc;
    color: #0066cc;
    font-weight: 500;
    padding: 0.75rem;
    min-height: 3rem;
    transition: all 0.2s ease;
}

[data-testid="baseButton-primary"]:hover {
    background-color: rgba(0, 102, 204, 0.2);
}

textarea {
    font-size: 1.1em;
    line-height: 1.5;
    padding: 0.5em;
    height: 150px;
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    color: #2c3338;
}
</style>
""", unsafe_allow_html=True)

def handle_topic_submission(topic: str, iterations: int):
    """Handle topic submission and validation."""
    if validate_topic(topic):
        state = AppState.load_state()
        state.last_topic = topic
        state.topic = sanitize_topic(topic)
        state.iterations = iterations
        state.stage = 'insights'
        state.save_state()
        st.rerun()

def handle_focus_continue(selected: List[str]):
    """Handle focus area continue action."""
    state = AppState.load_state()
    state.focus_container_expanded = False
    state.selected_focus_areas = selected
    state.stage = 'analysis'
    state.save_state()
    st.rerun()

def handle_focus_skip():
    """Handle focus area skip action."""
    state = AppState.load_state()
    state.focus_container_expanded = False
    state.selected_focus_areas = []
    state.stage = 'analysis'
    state.save_state()
    st.rerun()

def process_stage():
    """Process the current stage of research."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        state = AppState.load_state()
        
        # Always show logo
        display_logo()
        
        # Always show input form
        input_form(state, handle_topic_submission)
        
        # Display insights if available
        if state.insights:
            display_insights(state.insights)
        
        # Process stages
        if state.stage == 'insights':
            with st.spinner('Generating initial insights...'):
                pre_analysis = PreAnalysisAgent(model)
                insights = pre_analysis.generate_insights(state.topic)
                if insights:
                    state.insights = insights
                    focus_areas = pre_analysis.generate_focus_areas(state.topic)
                    if focus_areas:
                        state.focus_areas = focus_areas
                        state.stage = 'focus'
                        state.save_state()
                        st.rerun()
        
        # Always show focus areas if available
        if state.focus_areas:
            display_focus_areas(state, handle_focus_continue, handle_focus_skip)
        
        # Display analysis results if available
        if state.analysis_results:
            for index, analysis in enumerate(state.analysis_results):
                display_research_analysis(analysis, index)
        
        # Process analysis stage
        if state.stage == 'analysis':
            with st.spinner('Conducting research analysis...'):
                analyst = ResearchAnalyst(model)
                iterations_remaining = state.iterations - len(state.analysis_results)
                
                if iterations_remaining > 0:
                    previous = state.analysis_results[-1]['content'] if state.analysis_results else None
                    analysis = analyst.analyze(
                        state.topic,
                        state.selected_focus_areas,
                        previous
                    )
                    
                    if analysis:
                        state.analysis_results.append(analysis)
                        state.save_state()
                        st.rerun()
                else:
                    state.stage = 'synthesis'
                    state.save_state()
                    st.rerun()
        
        # Process synthesis stage
        elif state.stage == 'synthesis':
            if not state.synthesis:
                with st.spinner('Generating final synthesis...'):
                    synthesis_expert = SynthesisExpert(model)
                    synthesis = synthesis_expert.synthesize(
                        topic=state.topic,
                        focus_areas=state.selected_focus_areas,
                        analyses=state.analysis_results
                    )
                    if synthesis:
                        state.synthesis = synthesis
                        state.stage = "complete"
                        state.save_state()
                        display_synthesis(synthesis)
                    else:
                        logger.error("Failed to generate synthesis")
                        st.error("Failed to generate synthesis. Please try again.")
            else:
                display_synthesis(state.synthesis)
            
            if st.button("Start New Research", type="primary"):
                state.soft_reset()
                st.rerun()
    
    except QuotaExceededError:
        st.error("API quota exceeded. Please try again later.")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def display_synthesis(synthesis: Dict[str, str]) -> None:
    """Display the final synthesis report."""
    if not synthesis:
        return
        
    with st.expander("📊 Final Synthesis Report", expanded=False):
        # Display title and subtitle
        st.markdown(f"# {synthesis.get('title', '')}")
        st.markdown(f"*{synthesis.get('subtitle', '')}*")
        st.markdown("---")
        
        # Get the content
        content = synthesis.get('content', '')
        if content:
            # Process content to remove highlighted sections
            lines = content.split('\n')
            filtered_lines = []
            for line in lines:
                # Skip lines with background styling
                if any(bg in line.lower() for bg in ['background-color:', 'rgb', 'rgba', '#']):
                    continue
                # Skip lines with "Subtitle:" prefix
                if line.strip().startswith('Subtitle:'):
                    continue
                # Remove any HTML-style background coloring
                if '<span' in line.lower() and 'background' in line.lower():
                    continue
                # Remove any markdown-style highlighting
                if line.strip().startswith('==') and line.strip().endswith('=='):
                    continue
                filtered_lines.append(line)
            
            # Join and clean the content
            filtered_content = '\n'.join(filtered_lines)
            # Remove any remaining highlight markers
            filtered_content = filtered_content.replace('==', '')
            st.markdown(filtered_content)
        else:
            st.warning("No content available for this synthesis.")
        
        # Create download content
        synthesis_text = f"""# {synthesis.get('title', '')}

*{synthesis.get('subtitle', '')}*

---

{filtered_content}
"""
        # Use a unique key for the download button to prevent rerun
        st.download_button(
            label="Download Report",
            data=synthesis_text,
            file_name="synthesis_report.md",
            mime="text/markdown",
            key="synthesis_download"
        )

def display_research_analysis(analysis: Dict[str, str], index: int) -> None:
    """Display a research analysis result."""
    if not analysis:
        return
        
    # Create a brief summary from the title for the progress indicator
    brief_title = analysis.get('title', '').split(':')[-1].strip() if ':' in analysis.get('title', '') else analysis.get('title', '')
    
    with st.expander(f"📚 Research Analysis #{index + 1}: {brief_title}", expanded=False):
        # Display title and subtitle
        st.markdown(f"# {analysis.get('title', '')}")
        st.markdown(f"*{analysis.get('subtitle', '')}*")
        st.markdown("---")
        
        # Get the content
        content = analysis.get('content', '')
        if content:
            # Process content to remove highlighted sections
            lines = content.split('\n')
            filtered_lines = []
            for line in lines:
                # Skip lines with background styling
                if any(bg in line.lower() for bg in ['background-color:', 'rgb', 'rgba', '#']):
                    continue
                # Skip lines with "Subtitle:" prefix
                if line.strip().startswith('Subtitle:'):
                    continue
                # Remove any HTML-style background coloring
                if '<span' in line.lower() and 'background' in line.lower():
                    continue
                # Remove any markdown-style highlighting
                if line.strip().startswith('==') and line.strip().endswith('=='):
                    continue
                filtered_lines.append(line)
            
            # Join and clean the content
            filtered_content = '\n'.join(filtered_lines)
            # Remove any remaining highlight markers
            filtered_content = filtered_content.replace('==', '')
            st.markdown(filtered_content)
        else:
            st.warning("No content available for this analysis.")

def main():
    """Main application entry point."""
    state = AppState.load_state()
    process_stage()

if __name__ == "__main__":
    main() 