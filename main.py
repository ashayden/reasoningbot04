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
}
.input-description {
    font-size: 0.9em;
    color: #666;
    margin-bottom: 0.5em;
}
.agent-progress {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1rem 0;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
}
.agent-step {
    text-align: center;
    padding: 0.5rem;
    flex: 1;
}
.agent-step.active {
    font-weight: bold;
    color: #0066cc;
}
.depth-slider {
    padding: 1rem 0;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.depth-slider label {
    font-weight: bold;
    margin-bottom: 0.5rem;
}
.depth-description {
    font-size: 0.9em;
    color: #666;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

# Agent Progress Tracking
st.markdown("""
<div class="agent-progress">
    <div class="agent-step">✍️<br>Prompt<br>Designer</div>
    <div class="agent-step">🎯<br>Framework<br>Engineer</div>
    <div class="agent-step">🔄<br>Research<br>Analyst</div>
    <div class="agent-step">📊<br>Synthesis<br>Expert</div>
</div>
""", unsafe_allow_html=True)

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
        
        # Update progress indicators
        progress_container = st.empty()
        
        def update_progress(step):
            progress_html = f"""
            <div class="agent-progress">
                <div class="agent-step{'active' if step == 0 else ''}">
                    ✍️<br>Prompt<br>Designer
                </div>
                <div class="agent-step{'active' if step == 1 else ''}">
                    🎯<br>Framework<br>Engineer
                </div>
                <div class="agent-step{'active' if step == 2 else ''}">
                    🔄<br>Research<br>Analyst
                </div>
                <div class="agent-step{'active' if step == 3 else ''}">
                    📊<br>Synthesis<br>Expert
                </div>
            </div>
            """
            progress_container.markdown(progress_html, unsafe_allow_html=True)
        
        # Agent 0: Prompt Designer
        update_progress(0)
        with st.status("✍️ Designing optimal prompt...") as status:
            prompt_design = prompt_designer.design_prompt(topic)
            if not prompt_design:
                return None, None, None
            st.markdown(prompt_design)
            status.update(label="✍️ Optimized Prompt")

        # Agent 1: Framework Engineer
        update_progress(1)
        with st.status("🎯 Creating analysis framework...") as status:
            framework = framework_engineer.create_framework(prompt_design)
            if not framework:
                return None, None, None
            st.markdown(framework)
            status.update(label="🎯 Analysis Framework")
        
        # Agent 2: Research Analyst
        update_progress(2)
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
        update_progress(3)
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

# Input form
with st.form("analysis_form"):
    st.markdown("""
        <div class="input-description">
        Enter your research topic or question. Feel free to provide additional context or specific aspects you'd like to explore.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    topic = st.text_area(
        "What would you like to explore?",
        placeholder="e.g., 'Examine the impact of artificial intelligence on healthcare, focusing on diagnostic applications, ethical considerations, and future implications.'",
        help="You can provide a detailed description of your topic. Include specific aspects or questions you'd like to explore."
    )
    
    st.markdown('<div class="depth-slider">', unsafe_allow_html=True)
    depth = st.select_slider(
        "Analysis Depth",
        options=list(DEPTH_ITERATIONS.keys()),
        value="Balanced"
    )
    
    depth_descriptions = {
        "Quick": "Basic overview with 1 research iteration",
        "Balanced": "Moderate depth with 2 research iterations",
        "Deep": "Detailed analysis with 3 research iterations",
        "Comprehensive": "Exhaustive research with 4 iterations"
    }
    
    st.markdown(f"""
        <div class="depth-description">
        {depth_descriptions[depth]}
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
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