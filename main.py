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

textarea {
    font-size: 1.1em !important;
    line-height: 1.5 !important;
    padding: 0.5em !important;
    height: 150px !important;
    background-color: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    color: #2c3338 !important;
}
</style>
""", unsafe_allow_html=True)

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
        st.session_state.iterations = 2

def reset_state():
    """Reset all session state variables."""
    st.session_state.topic = ''
    st.session_state.stage = 'input'
    st.session_state.insights = None
    st.session_state.focus_areas = None
    st.session_state.selected_focus_areas = []
    st.session_state.analysis_results = []
    st.session_state.synthesis = None
    st.session_state.iterations = 2

def display_insights(insights):
    """Display insights in proper containers."""
    if insights:
        with st.container():
            with st.expander("💡 Did You Know?", expanded=True):
                st.markdown(insights['did_you_know'])
            
            with st.expander("⚡ Overview", expanded=True):
                st.markdown(insights['eli5'])

def display_focus_areas(focus_areas):
    """Display focus areas for selection."""
    if not focus_areas:
        st.error("Failed to load focus areas. Please try again.")
        return
    
    # Track container state in session state
    if 'focus_container_expanded' not in st.session_state:
        st.session_state.focus_container_expanded = True
    
    with st.expander("🎯 Focus Areas", expanded=st.session_state.focus_container_expanded):
        st.write("Choose up to 5 areas to focus your analysis on (optional):")
        
        # Initialize selected_areas if not present
        if 'selected_areas' not in st.session_state:
            st.session_state.selected_areas = []
        
        # Create columns for focus area selection
        cols = st.columns(2)
        for i, area in enumerate(focus_areas):
            col_idx = i % 2
            with cols[col_idx]:
                key = f"focus_area_{i}"
                is_selected = area in st.session_state.selected_areas
                if st.checkbox(area, value=is_selected, key=key):
                    if area not in st.session_state.selected_areas:
                        st.session_state.selected_areas.append(area)
                elif area in st.session_state.selected_areas:
                    st.session_state.selected_areas.remove(area)
        
        # Show selection status
        st.markdown("---")
        num_selected = len(st.session_state.selected_areas)
        
        if num_selected > 5:
            st.warning("⚠️ Please select no more than 5 focus areas")
        else:
            if num_selected > 0:
                st.success(f"✅ You have selected {num_selected} focus area{'s' if num_selected > 1 else ''}")
                st.write("Selected areas:")
                for area in st.session_state.selected_areas:
                    st.write(f"- {area}")
            
            # Add buttons side by side
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Skip", type="secondary", key="skip_button"):
                    st.session_state.focus_container_expanded = False
                    st.session_state.selected_focus_areas = []
                    st.session_state.stage = 'analysis'
                    st.rerun()
            with col2:
                if num_selected <= 5:
                    if st.button("Continue", type="primary", key="continue_button"):
                        st.session_state.focus_container_expanded = False
                        st.session_state.selected_focus_areas = st.session_state.selected_areas
                        st.session_state.stage = 'analysis'
                        st.rerun()

def display_analysis(analysis):
    """Display research analysis."""
    if analysis:
        with st.expander(f"🔄 {analysis['title']}", expanded=True):
            st.markdown(f"*{analysis['subtitle']}*")
            st.markdown(analysis['content'])

def display_synthesis(synthesis):
    """Display research synthesis."""
    if synthesis:
        with st.expander("📊 Final Research Synthesis", expanded=True):
            st.markdown(synthesis)

def process_stage():
    """Process the current stage of research."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        if st.session_state.stage == 'input':
            st.title("🔍 Research Assistant")
            st.write("Enter a topic to begin the research process.")
            
            with st.form("topic_form"):
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
                
                submitted = st.form_submit_button(
                    "🚀 Start Analysis",
                    use_container_width=True,
                    type="primary"
                )
                
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
            
            if st.button("Start New Research", type="primary"):
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