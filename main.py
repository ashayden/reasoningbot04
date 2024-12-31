"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai
import os

from config import GEMINI_MODEL
from utils import validate_topic, sanitize_topic
from agents import PreAnalysisAgent, PromptDesigner, FrameworkEngineer, ResearchAnalyst, SynthesisExpert

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
.block-container { 
    max-width: 800px; 
    padding: 2rem 1rem; 
}

.stButton > button { 
    width: 100%; 
}

/* Focus area buttons */
[data-testid="baseButton-secondary"] {
    background-color: rgba(30, 30, 30, 0.6) !important;
    border: 1px solid #333 !important;
    color: white !important;
    padding: 0.75rem !important;
    min-height: 3rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="baseButton-secondary"]:hover {
    background-color: rgba(42, 42, 42, 0.8) !important;
    border-color: #444 !important;
}

[data-testid="baseButton-primary"] {
    background-color: rgba(0, 102, 204, 0.2) !important;
    border: 1px solid #0066cc !important;
    box-shadow: 0 0 0 1px #0066cc !important;
    color: #0066cc !important;
    font-weight: 500 !important;
    padding: 0.75rem !important;
    min-height: 3rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="baseButton-primary"]:hover {
    background-color: rgba(0, 102, 204, 0.3) !important;
}

[data-testid="baseButton-primary"]:disabled {
    background-color: #1E1E1E !important;
    border-color: #333 !important;
    box-shadow: none !important;
    color: #4a4a4a !important;
    cursor: not-allowed !important;
}

div[data-testid="stImage"] { 
    text-align: center; 
}

div[data-testid="stImage"] > img { 
    max-width: 800px; 
    width: 100%; 
}

textarea {
    font-size: 1.1em !important;
    line-height: 1.5 !important;
    padding: 0.5em !important;
    height: 150px !important;
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
    color: #fff !important;
}

/* Number input styling */
div[data-testid="stNumberInput"] input {
    color: #fff !important;
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: #333 !important;
    border: none !important;
    color: #fff !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: #444 !important;
}
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

# Initialize analysis state
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = {
        'analysis': {
            'topic': None,
            'stage': 'start',
            'iterations': None,
            'completed_stages': set()
        },
        'outputs': {
            'insights': None,
            'prompt': None,
            'focus_areas': None,
            'selected_areas': [],
            'enhanced_prompt': None,
            'framework': None,
            'analysis_results': [],
            'summary': None
        }
    }

def display_completed_outputs():
    """Display outputs from completed stages."""
    state = st.session_state.analysis_state
    outputs = state['outputs']
    
    # Display insights if completed
    if outputs['insights']:
        st.markdown("### 💡 Quick Insights")
        with st.expander("Did You Know?"):
            st.markdown(outputs['insights']['did_you_know'])
        with st.expander("ELI5 (Explain Like I'm 5)"):
            st.markdown(outputs['insights']['eli5'])
    
    # Display prompt if completed
    if outputs['prompt']:
        with st.expander("✍️ Optimized Prompt"):
            st.markdown(outputs['prompt'])
    
    # Display framework if completed
    if outputs['framework']:
        with st.expander("🎯 Analysis Framework"):
            st.markdown(outputs['framework'])
    
    # Display analysis results if completed
    if outputs['analysis_results']:
        for i, result in enumerate(outputs['analysis_results']):
            with st.expander(f"🔄 Research Analysis #{i + 1}"):
                st.markdown(result)
    
    # Display summary if completed
    if outputs['summary']:
        with st.expander("📊 Final Report"):
            st.markdown(outputs['summary'])

def advance_stage(next_stage):
    """Advance to the next stage and update completed stages."""
    state = st.session_state.analysis_state
    current_stage = state['analysis']['stage']
    state['analysis']['completed_stages'].add(current_stage)
    state['analysis']['stage'] = next_stage

def process_insights(model, topic):
    """Process the insights stage."""
    state = st.session_state.analysis_state
    
    # Generate insights
    pre_analysis = PreAnalysisAgent(model)
    insights = pre_analysis.generate_insights(topic)
    
    if not insights:
        return False
    
    # Store insights
    state['outputs']['insights'] = insights
    
    # Display insights
    st.markdown("### 💡 Quick Insights")
    
    st.markdown("#### Did You Know?")
    st.markdown(insights['did_you_know'])
    
    st.markdown("#### ELI5 (Explain Like I'm 5)")
    st.markdown(insights['eli5'])
    
    return True

def process_prompt(model, topic):
    """Process the prompt optimization stage."""
    state = st.session_state.analysis_state
    
    # Generate prompt
    prompt_designer = PromptDesigner(model)
    prompt = prompt_designer.design_prompt(topic)
    
    if not prompt:
        return False
    
    # Store prompt
    state['outputs']['prompt'] = prompt
    
    # Display prompt
    st.markdown("### ✍️ Optimized Prompt")
    st.markdown(prompt)
    
    return True

def handle_focus_selection(model, topic):
    """Handle focus area selection stage."""
    state = st.session_state.analysis_state
    
    # Generate focus areas if needed
    if not state['outputs']['focus_areas']:
        prompt_designer = PromptDesigner(model)
        state['outputs']['focus_areas'] = prompt_designer.generate_focus_areas(topic)
    
    st.markdown("### 🎯 Select Focus Areas")
    st.markdown("Choose specific aspects you'd like the analysis to emphasize (optional):")
    
    # Display selection UI
    selected = st.multiselect(
        "Focus Areas",
        options=state['outputs']['focus_areas'],
        default=state['outputs']['selected_areas'],
        label_visibility="collapsed"
    )
    state['outputs']['selected_areas'] = selected
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Handle Skip button
    if col1.button(
        "Skip",
        key="skip_focus",
        help="Proceed with analysis using only the optimized prompt",
        use_container_width=True
    ):
        return True
    
    # Handle Continue button
    continue_disabled = len(selected) == 0
    if col2.button(
        "Continue",
        key="continue_focus",
        disabled=continue_disabled,
        help="Proceed with analysis using selected focus areas",
        type="primary",
        use_container_width=True
    ):
        prompt_designer = PromptDesigner(model)
        state['outputs']['enhanced_prompt'] = prompt_designer.design_prompt(topic, selected)
        return True
    
    return False

def process_stage(model, topic):
    """Process the current stage and manage transitions."""
    state = st.session_state.analysis_state
    
    # Display outputs from completed stages
    display_completed_outputs()
    
    # Process current stage
    if state['analysis']['stage'] == 'start':
        # Move directly to insights stage
        advance_stage('insights')
        st.experimental_rerun()
        
    elif state['analysis']['stage'] == 'insights':
        if process_insights(model, topic):
            advance_stage('prompt')
            st.experimental_rerun()
            
    elif state['analysis']['stage'] == 'prompt':
        if process_prompt(model, topic):
            advance_stage('focus')
            st.experimental_rerun()
            
    elif state['analysis']['stage'] == 'focus':
        if handle_focus_selection(model, topic):
            advance_stage('framework')
            st.experimental_rerun()
        st.stop()  # Pause for user input

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
    """Perform multi-agent analysis of a topic with proper state management."""
    try:
        # Validate and sanitize input
        is_valid, error_msg = validate_topic(topic)
        if not is_valid:
            st.error(error_msg)
            return None, None, None
            
        topic = sanitize_topic(topic)
        state = st.session_state.analysis_state
        
        # Initialize agents
        framework_engineer = FrameworkEngineer(model)
        research_analyst = ResearchAnalyst(model)
        synthesis_expert = SynthesisExpert(model)
        
        # Process stages up to framework
        process_stage(model, topic)
        
        # Stage: Framework Development
        if state['analysis']['stage'] == 'framework':
            with st.status("🎯 Creating analysis framework...") as status:
                framework = framework_engineer.create_framework(
                    state['outputs']['prompt'],
                    state['outputs']['enhanced_prompt']
                )
                if not framework:
                    return None, None, None
                
                state['outputs']['framework'] = framework
                st.markdown(framework)
                advance_stage('analysis')
                st.rerun()
        
        # Stage: Research Analysis
        if state['analysis']['stage'] == 'analysis':
            analysis_results = []
            previous_analysis = None
            
            for iteration_num in range(iterations):
                with st.status(f"🔄 Performing research analysis #{iteration_num + 1}...") as status:
                    result = research_analyst.analyze(
                        topic, 
                        state['outputs']['framework'],
                        previous_analysis
                    )
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
                    status.update(label=f"🔄 Research Analysis #{iteration_num + 1}")
            
            state['outputs']['analysis_results'] = analysis_results
            advance_stage('summary')
            st.rerun()
        
        # Stage: Final Synthesis
        if state['analysis']['stage'] == 'summary':
            with st.status("📊 Generating final report...") as status:
                summary = synthesis_expert.synthesize(
                    topic,
                    state['outputs']['analysis_results']
                )
                if not summary:
                    return None, None, None
                
                state['outputs']['summary'] = summary
                st.markdown(summary)
                advance_stage('complete')
                st.rerun()
        
        return (
            state['outputs']['framework'],
            state['outputs']['analysis_results'],
            state['outputs']['summary']
        )
        
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
    # Text input
    topic = st.text_area(
        "What would you like to explore?",
        help="Enter your research topic or question. Feel free to provide additional context or specific aspects you'd like to explore.",
        placeholder="e.g., 'Examine the impact of artificial intelligence on healthcare, focusing on diagnostic applications, ethical considerations, and future implications.'"
    )
    
    # Analysis iterations input
    iterations = st.number_input(
        "Number of Analysis Iterations",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Choose 1-5 iterations. More iterations = deeper insights = longer wait."
    )
    
    # Submit button
    submit = st.form_submit_button(
        "🚀 Start Analysis",
        use_container_width=True,
        help="Click to begin the multi-agent analysis process"
    )

# Analysis section
if submit and topic:
    state = st.session_state.analysis_state
    
    # Reset state if topic changed or starting new analysis
    if (state['analysis']['topic'] != topic or 
        state['analysis']['stage'] == 'complete'):
        
        # Reset analysis state
        st.session_state.analysis_state = {
            'analysis': {
                'topic': topic,
                'stage': 'start',
                'iterations': iterations,
                'completed_stages': set()
            },
            'outputs': {
                'insights': None,
                'prompt': None,
                'focus_areas': None,
                'selected_areas': [],
                'enhanced_prompt': None,
                'framework': None,
                'analysis_results': [],
                'summary': None
            }
        }
    
    # Display initial status
    if state['analysis']['stage'] == 'start':
        with st.status("🚀 Starting analysis...", expanded=True):
            st.write("Initializing analysis process...")
    
    # Run analysis
    try:
        framework, analysis, summary = analyze_topic(model, topic, iterations)
        
        # Handle completion
        if framework and analysis and summary:
            if state['analysis']['stage'] == 'complete':
                st.success("Analysis complete! Review the results above.")
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        logger.error(f"Analysis error: {str(e)}") 