"""Main application file for the MARA research assistant."""

import logging
import streamlit as st
import google.generativeai as genai
from typing import Dict, Tuple
import re
import markdown

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
    if 'last_topic' not in st.session_state:
        st.session_state.last_topic = ''

def soft_reset_state():
    """Reset state while preserving the last topic."""
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
        selected = []  # Track current selections
        for i, area in enumerate(focus_areas):
            col_idx = i % 2
            with cols[col_idx]:
                key = f"focus_area_{i}"
                is_selected = area in st.session_state.selected_areas
                
                # Only allow selection if under limit or already selected
                if len(selected) < 5 or is_selected:
                    if st.checkbox(area, value=is_selected, key=key):
                        selected.append(area)
                else:
                    # Show disabled checkbox if at limit
                    st.checkbox(area, value=False, key=key, disabled=True)
        
        # Update session state with current selections
        st.session_state.selected_areas = selected
        
        # Show selection status
        st.markdown("---")
        num_selected = len(selected)
        
        if num_selected > 0:
            st.success(f"✅ Selected {num_selected} focus area{'s' if num_selected > 1 else ''}")
            st.write("Selected areas:")
            for area in selected:
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
            if st.button("Continue", type="primary", key="continue_button"):
                st.session_state.focus_container_expanded = False
                st.session_state.selected_focus_areas = selected
                st.session_state.stage = 'analysis'
                st.rerun()

def display_research_analysis(analysis_data: Dict[str, str]) -> None:
    """Display research analysis with prominent title."""
    if not analysis_data:
        return
        
    with st.expander("📚 " + analysis_data["title"], expanded=False):
        st.markdown(f"### {analysis_data['title']}")
        st.markdown(f"*{analysis_data['subtitle']}*")
        st.markdown("---")
        st.markdown(analysis_data["content"])

def generate_download_content(synthesis_data: Dict[str, str], format: str) -> Tuple[str, str, bytes]:
    """Generate downloadable content in the specified format."""
    if format == "markdown":
        content = f"# {synthesis_data['title']}\n\n"
        content += f"*{synthesis_data['subtitle']}*\n\n"
        content += synthesis_data['content']
        return "text/markdown", "synthesis_report.md", content.encode('utf-8')
    elif format == "txt":
        content = f"{synthesis_data['title']}\n"
        content += f"{synthesis_data['subtitle']}\n\n"
        content += re.sub(r'\*\*|•|\n###', '', synthesis_data['content'])  # Remove markdown
        return "text/plain", "synthesis_report.txt", content.encode('utf-8')
    else:  # PDF
        import pdfkit
        html_content = f"<h1>{synthesis_data['title']}</h1>"
        html_content += f"<p><em>{synthesis_data['subtitle']}</em></p>"
        html_content += markdown.markdown(synthesis_data['content'])
        pdf_content = pdfkit.from_string(html_content, False)
        return "application/pdf", "synthesis_report.pdf", pdf_content

def display_synthesis(synthesis_data: Dict[str, str]) -> None:
    """Display synthesis report with prominent title and download options."""
    if not synthesis_data:
        return
        
    with st.expander("📑 " + synthesis_data["title"], expanded=False):
        st.markdown(f"### {synthesis_data['title']}")
        st.markdown(f"*{synthesis_data['subtitle']}*")
        st.markdown("---")
        st.markdown(synthesis_data["content"])
        
        # Add download options
        st.markdown("---")
        st.markdown("### Download Report")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download PDF"):
                mime_type, filename, content = generate_download_content(synthesis_data, "pdf")
                st.download_button(
                    label="Download PDF",
                    data=content,
                    file_name=filename,
                    mime=mime_type
                )
        
        with col2:
            if st.button("📝 Download Markdown"):
                mime_type, filename, content = generate_download_content(synthesis_data, "markdown")
                st.download_button(
                    label="Download Markdown",
                    data=content,
                    file_name=filename,
                    mime=mime_type
                )
        
        with col3:
            if st.button("📄 Download Text"):
                mime_type, filename, content = generate_download_content(synthesis_data, "txt")
                st.download_button(
                    label="Download Text",
                    data=content,
                    file_name=filename,
                    mime=mime_type
                )

def process_stage():
    """Process the current stage of research."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Always show logo
        st.image("assets/mara-logo.png", use_container_width=True)
        
        # Always show input form, but modify based on state
        with st.form("topic_form", clear_on_submit=False):
            topic = st.text_area(
                "What would you like to explore?",
                value=st.session_state.get('last_topic', ''),
                help="Enter your research topic or question.",
                placeholder="e.g., 'Examine the impact of artificial intelligence on healthcare...'"
            )
            
            iterations = st.number_input(
                "Number of Analysis Iterations",
                min_value=1,
                max_value=5,
                value=st.session_state.get('iterations', 2),
                step=1,
                help="Choose 1-5 iterations. More iterations = deeper insights = longer wait."
            )
            
            # Show different buttons based on state
            if st.session_state.stage == 'input':
                submitted = st.form_submit_button(
                    "🚀 Start Analysis",
                    use_container_width=True,
                    type="primary"
                )
                
                if submitted and topic:
                    if validate_topic(topic):
                        st.session_state.last_topic = topic
                        st.session_state.topic = sanitize_topic(topic)
                        st.session_state.iterations = iterations
                        st.session_state.stage = 'insights'
                        st.rerun()
            else:
                if st.form_submit_button(
                    "❌ Cancel",
                    use_container_width=True,
                    type="secondary"
                ):
                    soft_reset_state()
                    st.rerun()
        
        # Display insights if available
        if st.session_state.insights:
            display_insights(st.session_state.insights)
        
        # Process stages
        if st.session_state.stage == 'insights':
            pre_analysis = PreAnalysisAgent(model)
            insights = pre_analysis.generate_insights(st.session_state.topic)
            if insights:
                st.session_state.insights = insights
                focus_areas = pre_analysis.generate_focus_areas(st.session_state.topic)
                if focus_areas:
                    st.session_state.focus_areas = focus_areas
                    st.session_state.stage = 'focus'
                    st.rerun()
        
        # Always show focus areas if available
        if st.session_state.focus_areas:
            display_focus_areas(st.session_state.focus_areas)
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            for analysis in st.session_state.analysis_results:
                display_research_analysis(analysis)
        
        # Process analysis stage
        if st.session_state.stage == 'analysis':
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
        
        # Process synthesis stage
        elif st.session_state.stage == 'synthesis':
            if not st.session_state.synthesis:
                synthesis_expert = SynthesisExpert(model)
                synthesis = synthesis_expert.synthesize(
                    topic=st.session_state.topic,
                    focus_areas=st.session_state.selected_focus_areas,
                    analyses=st.session_state.analysis_results
                )
                if synthesis:
                    display_synthesis(synthesis)
                    st.session_state.stage = "complete"
                else:
                    logger.error("Failed to generate synthesis")
                    st.error("Failed to generate synthesis. Please try again.")
            
            if st.button("Start New Research", type="primary"):
                soft_reset_state()
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