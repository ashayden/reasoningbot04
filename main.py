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
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

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
    """Perform multi-agent analysis of a topic.
    
    Args:
        model: The initialized Gemini model
        topic: The topic to analyze
        iterations: Number of analysis iterations to perform
        
    Returns:
        Tuple of (framework, analysis_results, summary) or (None, None, None) on error
    """
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
                with st.container():
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

# Main UI
with st.sidebar:
    st.markdown("""
    0. ✍️ Prompt Designer
    1. 🎯 Framework Engineer
    2. 🔄 Research Analyst
    3. 📊 Synthesis Expert
    """)

# Initialize model
model = initialize_gemini()
if not model:
    st.stop()

# Input form
with st.form("analysis_form"):
    topic = st.text_input(
        "What would you like to explore?",
        placeholder="e.g., 'Artificial Intelligence' or 'Climate Change'"
    )
    
    depth = st.select_slider(
        "Analysis Depth",
        options=list(DEPTH_ITERATIONS.keys()),
        value="Balanced"
    )
    
    submit = st.form_submit_button("🚀 Start Analysis")

if submit and topic:
    iterations = DEPTH_ITERATIONS[depth]
    framework, analysis, summary = analyze_topic(model, topic, iterations)
    
    if framework and analysis and summary:
        st.success("Analysis complete! Review the results above.") 