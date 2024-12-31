"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai

from config import GEMINI_MODEL, DEPTH_ITERATIONS
from utils import validate_topic, sanitize_topic
from agents import PromptDesigner, FrameworkEngineer, ResearchAnalyst, SynthesisExpert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="M.A.R.A. - Multi-Agent Reasoning Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS and Logo
st.markdown("""
<style>
.block-container { max-width: 800px; padding: 2rem 1rem; }
.stButton > button { width: 100%; }
div[data-testid="stImage"] { text-align: center; }
div[data-testid="stImage"] > img { max-width: 800px; width: 100%; }
textarea {
    font-size: 1.1em !important;
    line-height: 1.5 !important;
    padding: 0.5em !important;
    height: 150px !important;
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
    color: #fff !important;
}
div[data-baseweb="select-slider"] {
    padding: 1rem 0;
}
div[data-baseweb="select-slider"] > div {
    background-color: transparent !important;
}
div[data-baseweb="select-slider"] span {
    color: #fff !important;
    font-family: "Source Sans Pro", sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}
div[data-baseweb="slider"] {
    background: transparent !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background: #fff !important;
    border: none !important;
    width: 24px !important;
    height: 24px !important;
    margin-top: -11px !important;
    border-radius: 50% !important;
}
div[data-baseweb="slider"] div[data-testid="stSliderBar"] {
    background: rgba(255, 255, 255, 0.2) !important;
    height: 2px !important;
}
div[data-baseweb="slider"] div[data-testid="stSliderProgress"] {
    background: #fff !important;
    height: 2px !important;
}
div[data-baseweb="select-slider"] div[role="tablist"] span {
    background: transparent !important;
    padding: 4px 8px !important;
    border-radius: 4px !important;
}
div[data-baseweb="select-slider"] div[role="tablist"] span[aria-selected="true"] {
    color: #fff !important;
    background: transparent !important;
}
button[kind="primary"] {
    background-color: #0066cc !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

# Initialize session state
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = {
        'topic': None,
        'framework': None,
        'analysis': None,
        'summary': None
    }

# Initialize Gemini
@st.cache_resource
def initialize_gemini():
    """Initialize the Gemini model with caching."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {str(e)}")
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

def analyze_topic(model, topic: str, iterations: int = 1):
    """Perform multi-agent analysis of a topic."""
    try:
        # Validate and sanitize input
        is_valid, error_msg = validate_topic(topic)
        if not is_valid:
            st.error(error_msg)
            return None, None, None
            
        topic = sanitize_topic(topic)
        
        # Initialize agents
        prompt_designer = PromptDesigner(model)
        framework_engineer = FrameworkEngineer(model)
        research_analyst = ResearchAnalyst(model)
        synthesis_expert = SynthesisExpert(model)
        
        # Agent 0: Prompt Designer
        with st.status("✍️ Designing optimal prompt...") as status:
            prompt_design = prompt_designer.design_prompt(topic)
            if not prompt_design:
                return None, None, None
            st.markdown(prompt_design)
            status.update(label="✍️ Optimized Prompt")

        # Agent 1: Framework Engineer
        with st.status("🎯 Creating analysis framework...") as status:
            framework = framework_engineer.create_framework(prompt_design)
            if not framework:
                return None, None, None
            st.markdown(framework)
            status.update(label="🎯 Analysis Framework")
        
        # Agent 2: Research Analyst
        analysis_results = []
        previous_analysis = None
        
        for iteration_num in range(iterations):
            with st.status(f"🔄 Performing research analysis #{iteration_num + 1}...") as status:
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
                status.update(label=f"🔄 Research Analysis #{iteration_num + 1}")
        
        # Agent 3: Synthesis Expert
        with st.status("📊 Generating final report...") as status:
            summary = synthesis_expert.synthesize(topic, analysis_results)
            if not summary:
                return None, None, None
            st.markdown(summary)
            status.update(label="📊 Final Report")
            
        return framework, analysis_results, summary
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        st.error(f"Analysis error: {str(e)}")
        return None, None, None

# Initialize model
model = initialize_gemini()
if not model:
    st.stop()

# Input form
with st.form("analysis_form"):
    topic = st.text_area(
        "What would you like to explore?",
        help="Enter your research topic or question. Feel free to provide additional context or specific aspects you'd like to explore.",
        placeholder="e.g., 'Examine the impact of artificial intelligence on healthcare, focusing on diagnostic applications, ethical considerations, and future implications.'"
    )
    
    depth = st.select_slider(
        "Analysis Depth",
        options=list(DEPTH_ITERATIONS.keys()),
        value="Balanced"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.form_submit_button(
            "🚀 Start Analysis",
            use_container_width=True,
            help="Click to begin the multi-agent analysis process"
        )

# Analysis section
if submit and topic:
    # Reset state if topic changed
    if st.session_state.current_analysis['topic'] != topic:
        st.session_state.current_analysis = {
            'topic': topic,
            'framework': None,
            'analysis': None,
            'summary': None
        }
    
    # Run analysis
    iterations = DEPTH_ITERATIONS[depth]
    framework, analysis, summary = analyze_topic(model, topic, iterations)
    
    if framework and analysis and summary:
        st.session_state.current_analysis.update({
            'framework': framework,
            'analysis': analysis,
            'summary': summary
        })
        st.success("Analysis complete! Review the results above.") 