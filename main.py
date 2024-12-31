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

# Initialize state
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
    st.session_state.displayed_outputs = set()

def process_stage_output(stage_name, content, expanded=True):
    """Handle stage output display consistently."""
    if stage_name not in st.session_state.displayed_outputs:
        # Show expanded for new content
        with st.expander(stage_name, expanded=expanded):
            st.markdown(content)
        st.session_state.displayed_outputs.add(stage_name)
    else:
        # Show collapsed for existing content
        with st.expander(stage_name, expanded=False):
            st.markdown(content)

def display_completed_outputs():
    """Display outputs from completed stages."""
    state = st.session_state.analysis_state
    outputs = state['outputs']
    
    # Display insights if completed
    if outputs['insights']:
        process_stage_output(
            "💡 Did You Know?",
            outputs['insights']['did_you_know'],
            expanded=False
        )
        process_stage_output(
            "⚡ ELI5 (Explain Like I'm 5)",
            outputs['insights']['eli5'],
            expanded=False
        )
    
    # Display prompt if completed
    if outputs['prompt']:
        process_stage_output(
            "✍️ Optimized Prompt",
            outputs['prompt'],
            expanded=False
        )
    
    # Display framework if completed
    if outputs['framework']:
        process_stage_output(
            "🎯 Analysis Framework",
            outputs['framework'],
            expanded=False
        )
    
    # Display analysis results if completed
    if outputs['analysis_results']:
        for i, result in enumerate(outputs['analysis_results']):
            process_stage_output(
                f"🔄 Research Analysis #{i + 1}",
                result,
                expanded=False
            )
    
    # Display summary if completed
    if outputs['summary']:
        process_stage_output(
            "📊 Final Report",
            outputs['summary'],
            expanded=False
        )

def advance_stage(next_stage):
    """Advance to the next stage and update completed stages."""
    state = st.session_state.analysis_state
    current_stage = state['analysis']['stage']
    state['analysis']['completed_stages'].add(current_stage)
    state['analysis']['stage'] = next_stage

def initialize_or_reset_state(topic, iterations, force=False):
    """Initialize or reset state with proper checks."""
    if force or st.session_state.analysis_state['analysis']['topic'] != topic:
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
        st.session_state.displayed_outputs = set()
        return True
    return False

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
    process_stage_output(
        "💡 Did You Know?",
        insights['did_you_know'],
        expanded=True
    )
    process_stage_output(
        "⚡ ELI5 (Explain Like I'm 5)",
        insights['eli5'],
        expanded=True
    )
    
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
    process_stage_output(
        "✍️ Optimized Prompt",
        prompt,
        expanded=True
    )
    
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
        advance_stage('insights')
        st.rerun()
        
    elif state['analysis']['stage'] == 'insights':
        if process_insights(model, topic):
            advance_stage('prompt')
            st.rerun()
            
    elif state['analysis']['stage'] == 'prompt':
        if process_prompt(model, topic):
            advance_stage('focus')
            st.rerun()
            
    elif state['analysis']['stage'] == 'focus':
        if handle_focus_selection(model, topic):
            advance_stage('framework')
            st.rerun()
        st.stop()  # Pause for user input
            
    elif state['analysis']['stage'] == 'framework':
        if process_framework(model, topic):
            advance_stage('analysis')
            st.rerun()
            
    elif state['analysis']['stage'] == 'analysis':
        if process_analysis_stage(model, topic, state['analysis']['iterations']):
            advance_stage('summary')
            st.rerun()
            
    elif state['analysis']['stage'] == 'summary':
        if process_summary(model, topic):
            advance_stage('complete')
            st.rerun()

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
        
        # Process all stages
        process_stage(model, topic)
        
        # Return final outputs
        return (
            state['outputs']['framework'],
            state['outputs']['analysis_results'],
            state['outputs']['summary']
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        st.error(f"Analysis error: {str(e)}")
        return None, None, None

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
    # Initialize or reset state if needed
    if st.session_state.analysis_state['analysis']['topic'] != topic:
        initialize_or_reset_state(topic, iterations)
        st.session_state.analysis_state['analysis']['topic'] = topic
        st.session_state.analysis_state['analysis']['iterations'] = iterations
    
    # Get current state
    state = st.session_state.analysis_state
    current_stage = state['analysis']['stage']
    
    try:
        # Process stages
        if current_stage == 'start':
            st.markdown("💡 Generating insights...")
            if process_insights(model, topic):
                advance_stage('prompt')
                st.rerun()
                
        elif current_stage == 'insights':
            st.markdown("✍️ Optimizing prompt...")
            if process_prompt(model, topic):
                advance_stage('focus')
                st.rerun()
                
        elif current_stage == 'prompt':
            st.markdown("🎯 Select focus areas")
            if handle_focus_selection(model, topic):
                advance_stage('framework')
                st.rerun()
                
        elif current_stage == 'focus':
            st.markdown("🔨 Building analysis framework...")
            if process_framework(model, topic):
                advance_stage('analysis')
                st.rerun()
                
        elif current_stage == 'analysis':
            st.markdown("🔄 Performing analysis...")
            if process_analysis_stage(model, topic, iterations):
                advance_stage('summary')
                st.rerun()
                
        elif current_stage == 'summary':
            st.markdown("📊 Generating final report...")
            if process_summary(model, topic):
                advance_stage('complete')
                st.success("✅ Analysis complete! Review the results above.")
                
        # Display completed outputs
        display_completed_outputs()
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        st.error(f"An error occurred during analysis: {str(e)}")
        
    # Log state for debugging
    logger.info(f"Stage: {current_stage}")
    logger.info(f"Completed stages: {state['analysis']['completed_stages']}")

def process_framework(model, topic):
    """Process the framework development stage."""
    state = st.session_state.analysis_state
    
    # Generate framework
    framework_engineer = FrameworkEngineer(model)
    framework = framework_engineer.create_framework(
        state['outputs']['prompt'],
        state['outputs']['enhanced_prompt']
    )
    
    if not framework:
        return False
    
    # Store framework
    state['outputs']['framework'] = framework
    
    # Display framework
    process_stage_output(
        "🎯 Analysis Framework",
        framework,
        expanded=True
    )
    
    return True

def process_analysis_stage(model, topic, iterations):
    """Process the analysis stage."""
    state = st.session_state.analysis_state
    
    research_analyst = ResearchAnalyst(model)
    analysis_results = []
    previous_analysis = None
    
    for iteration_num in range(iterations):
        result = research_analyst.analyze(
            topic, 
            state['outputs']['framework'],
            previous_analysis
        )
        
        if not result:
            return False
        
        content = ""
        if result['title']:
            content += f"# {result['title']}\n\n"
        if result['subtitle']:
            content += f"*{result['subtitle']}*\n\n"
        if result['content']:
            content += result['content']
        
        analysis_results.append(content)
        previous_analysis = result['content']
        
        # Display each iteration result
        process_stage_output(
            f"🔄 Research Analysis #{iteration_num + 1}",
            content,
            expanded=True
        )
    
    # Store analysis results
    state['outputs']['analysis_results'] = analysis_results
    return True

def process_summary(model, topic):
    """Process the summary stage."""
    state = st.session_state.analysis_state
    
    synthesis_expert = SynthesisExpert(model)
    summary = synthesis_expert.synthesize(
        topic,
        state['outputs']['analysis_results']
    )
    
    if not summary:
        return False
    
    # Store summary
    state['outputs']['summary'] = summary
    
    # Display summary
    process_stage_output(
        "📊 Final Report",
        summary,
        expanded=True
    )
    
    return True 