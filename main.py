"""Main application file for the MARA research assistant."""

import logging
import streamlit as st
import google.generativeai as genai

from config import GEMINI_MODEL
from utils import validate_topic, sanitize_topic, QuotaExceededError
from agents import PreAnalysisAgent, ResearchAnalyst, SynthesisExpert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables."""
    if 'topic' not in st.session_state:
        st.session_state.topic = ''
    if 'stage' not in st.session_state:
        st.session_state.stage = 'input'
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    if 'focus_areas' not in st.session_state:
        st.session_state.focus_areas = None
    if 'selected_focus_areas' not in st.session_state:
        st.session_state.selected_focus_areas = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'synthesis' not in st.session_state:
        st.session_state.synthesis = None
    if 'iterations' not in st.session_state:
        st.session_state.iterations = 3

def reset_state():
    """Reset all session state variables."""
    st.session_state.topic = ''
    st.session_state.stage = 'input'
    st.session_state.insights = None
    st.session_state.focus_areas = None
    st.session_state.selected_focus_areas = []
    st.session_state.analysis_results = []
    st.session_state.synthesis = None
    st.session_state.iterations = 3

def display_insights(insights):
    """Display research insights."""
    if insights:
        st.markdown(f"## {insights['title']}")
        st.markdown(f"*{insights['subtitle']}*")
        st.markdown(insights['content'])

def display_focus_areas(focus_areas):
    """Display and handle focus area selection."""
    if focus_areas:
        with st.expander("🎯 Select Focus Areas (Optional)", expanded=True):
            st.write("Choose 0-5 areas to focus the research:")
            selected = []
            for area in focus_areas:
                if st.checkbox(area, key=f"focus_{area}"):
                    selected.append(area)
                if len(selected) >= 5:
                    break
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Skip"):
                    st.session_state.selected_focus_areas = []
                    st.session_state.stage = 'analysis'
            with col2:
                if st.button("Continue"):
                    st.session_state.selected_focus_areas = selected
                    st.session_state.stage = 'analysis'

def display_analysis(analysis):
    """Display research analysis."""
    if analysis:
        with st.expander(f"📊 {analysis['title']}", expanded=True):
            st.markdown(f"*{analysis['subtitle']}*")
            st.markdown(analysis['content'])

def display_synthesis(synthesis):
    """Display research synthesis."""
    if synthesis:
        st.markdown("## 📑 Final Research Synthesis")
        st.markdown(synthesis)

def process_stage():
    """Process the current stage of research."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        if st.session_state.stage == 'input':
            st.title("🔍 Research Assistant")
            st.write("Enter a topic to begin the research process.")
            
            with st.form("topic_form"):
                topic = st.text_input("Research Topic:", key="topic_input")
                iterations = st.slider("Number of Research Iterations:", 1, 5, 3, key="iterations_input")
                submitted = st.form_submit_button("Start Analysis")
                
                if submitted and topic:
                    if validate_topic(topic):
                        st.session_state.topic = sanitize_topic(topic)
                        st.session_state.iterations = iterations
                        st.session_state.stage = 'insights'
                        st.rerun()
        
        elif st.session_state.stage == 'insights':
            pre_analysis = PreAnalysisAgent(model)
            insights = pre_analysis.generate_insights(st.session_state.topic)
            if insights:
                st.session_state.insights = insights
                focus_areas = pre_analysis.generate_focus_areas(st.session_state.topic)
                if focus_areas:
                    st.session_state.focus_areas = focus_areas
                    st.session_state.stage = 'focus'
                    st.rerun()
        
        elif st.session_state.stage == 'focus':
            display_insights(st.session_state.insights)
            display_focus_areas(st.session_state.focus_areas)
        
        elif st.session_state.stage == 'analysis':
            display_insights(st.session_state.insights)
            
            analyst = ResearchAnalyst(model)
            iterations_remaining = st.session_state.iterations - len(st.session_state.analysis_results)
            
            if iterations_remaining > 0:
                previous = st.session_state.analysis_results[-1]['content'] if st.session_state.analysis_results else None
                analysis = analyst.analyze(
                    st.session_state.topic,
                    st.session_state.selected_focus_areas,
                    previous
                )
                
                if analysis:
                    st.session_state.analysis_results.append(analysis)
                    st.rerun()
            else:
                st.session_state.stage = 'synthesis'
                st.rerun()
            
            for analysis in st.session_state.analysis_results:
                display_analysis(analysis)
        
        elif st.session_state.stage == 'synthesis':
            display_insights(st.session_state.insights)
            for analysis in st.session_state.analysis_results:
                display_analysis(analysis)
            
            if not st.session_state.synthesis:
                synthesis_expert = SynthesisExpert(model)
                analysis_texts = [a['content'] for a in st.session_state.analysis_results]
                synthesis = synthesis_expert.synthesize(st.session_state.topic, analysis_texts)
                if synthesis:
                    st.session_state.synthesis = synthesis
                    st.rerun()
            
            display_synthesis(st.session_state.synthesis)
            
            if st.button("Start New Research"):
                reset_state()
                st.rerun()
    
    except QuotaExceededError:
        st.error("API quota exceeded. Please try again later.")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def main():
    """Main application entry point."""
    initialize_session_state()
    process_stage()

if __name__ == "__main__":
    main() 