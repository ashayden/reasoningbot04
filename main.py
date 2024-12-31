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

# Initialize consolidated state
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'stage': 'start',
        'topic': None,
        'outputs': {
            'insights': None,
            'initial_prompt': None,
            'focus_areas': None,
            'selected_areas': [],
            'enhanced_prompt': None,
            'framework': None,
            'analysis_results': [],
            'summary': None
        },
        'ui_state': {
            'focus_selection_complete': False
        }
    }

def transition_to_stage(new_stage):
    """Handle stage transitions with proper state management."""
    st.session_state.app_state['stage'] = new_stage
    if new_stage == 'framework':
        st.session_state.app_state['ui_state']['focus_selection_complete'] = True

def handle_focus_area_selection(topic, prompt_designer):
    """Handle focus area selection with proper state management."""
    state = st.session_state.app_state
    
    # Generate areas if needed
    if not state['outputs']['focus_areas']:
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
        transition_to_stage('framework')
        st.rerun()
    
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
        # Generate enhanced prompt with selected areas
        state['outputs']['enhanced_prompt'] = prompt_designer.design_prompt(
            topic,
            selected
        )
        
        with st.status("✍️ Updated Prompt", expanded=False) as status:
            st.markdown(state['outputs']['enhanced_prompt'])
            status.update(label="✍️ Updated Prompt")
        
        transition_to_stage('framework')
        st.rerun()
    
    st.stop()

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
        state = st.session_state.app_state
        
        # Initialize agents
        pre_analysis = PreAnalysisAgent(model)
        prompt_designer = PromptDesigner(model)
        framework_engineer = FrameworkEngineer(model)
        research_analyst = ResearchAnalyst(model)
        synthesis_expert = SynthesisExpert(model)
        
        # Stage: Quick Insights
        if state['stage'] == 'start':
            insights = pre_analysis.generate_insights(topic)
            if not insights:
                return None, None, None
            
            state['outputs']['insights'] = insights
            
            with st.status("💡 Did You Know", expanded=True) as status:
                st.markdown(insights['did_you_know'])
                status.update(label="💡 Did You Know")
                
            with st.status("⚡ ELI5", expanded=True) as status:
                st.markdown(insights['eli5'])
                status.update(label="⚡ ELI5")
            
            transition_to_stage('prompt')
            st.rerun()
            
        # Display previous insights if they exist
        elif state['outputs']['insights']:
            with st.status("💡 Did You Know", expanded=False) as status:
                st.markdown(state['outputs']['insights']['did_you_know'])
            with st.status("⚡ ELI5", expanded=False) as status:
                st.markdown(state['outputs']['insights']['eli5'])
        
        # Stage: Initial Prompt Design
        if state['stage'] == 'prompt':
            with st.status("✍️ Designing optimal prompt...") as status:
                initial_prompt = prompt_designer.design_prompt(topic)
                if not initial_prompt:
                    return None, None, None
                
                state['outputs']['initial_prompt'] = initial_prompt
                st.markdown(initial_prompt)
                status.update(label="✍️ Optimized Prompt")
                
                transition_to_stage('focus_areas')
                st.rerun()
                
        # Display stored prompt if it exists
        elif state['outputs']['initial_prompt']:
            with st.status("✍️ Optimized Prompt", expanded=False) as status:
                st.markdown(state['outputs']['initial_prompt'])
        
        # Stage: Focus Area Selection
        if state['stage'] == 'focus_areas':
            handle_focus_area_selection(topic, prompt_designer)
        
        # Stage: Framework Development
        if state['stage'] == 'framework':
            with st.status("🎯 Creating analysis framework...") as status:
                # Pass both prompts to framework engineer
                framework = framework_engineer.create_framework(
                    state['outputs']['initial_prompt'],
                    state['outputs']['enhanced_prompt']  # Will be None if skipped
                )
                if not framework:
                    return None, None, None
                
                state['outputs']['framework'] = framework
                st.markdown(framework)
                status.update(label="🎯 Analysis Framework")
                
                transition_to_stage('analysis')
                st.rerun()
        else:
            # Display stored framework
            if state['outputs']['framework']:
                with st.status("🎯 Analysis Framework", expanded=False) as status:
                    st.markdown(state['outputs']['framework'])
        
        # Stage: Research Analysis
        if state['stage'] == 'analysis':
            analysis_results = []
            previous_analysis = None
            
            for iteration_num in range(iterations):
                with st.status(f"🔄 Performing research analysis #{iteration_num + 1}...") as status:
                    st.divider()
                    
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
                    st.divider()
                    status.update(label=f"🔄 Research Analysis #{iteration_num + 1}")
            
            state['outputs']['analysis_results'] = analysis_results
            transition_to_stage('summary')
            st.rerun()
        else:
            # Display stored analysis results
            if state['outputs']['analysis_results']:
                for i, content in enumerate(state['outputs']['analysis_results']):
                    with st.status(f"🔄 Research Analysis #{i + 1}", expanded=False) as status:
                        st.markdown(content)
        
        # Stage: Final Synthesis
        if state['stage'] == 'summary':
            with st.status("📊 Generating final report...") as status:
                summary = synthesis_expert.synthesize(
                    topic,
                    state['outputs']['analysis_results']
                )
                if not summary:
                    return None, None, None
                
                state['outputs']['summary'] = summary
                st.markdown(summary)
                status.update(label="📊 Final Report")
                
                transition_to_stage('complete')
                st.rerun()
        else:
            # Display stored summary
            if state['outputs']['summary']:
                with st.status("📊 Final Report", expanded=False) as status:
                    st.markdown(state['outputs']['summary'])
        
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
    # Reset state if topic changed or starting new analysis
    if st.session_state.app_state['topic'] != topic or st.session_state.app_state['stage'] == 'complete':
        st.session_state.app_state = {
            'stage': 'start',
            'topic': topic,
            'outputs': {
                'insights': None,
                'initial_prompt': None,
                'focus_areas': None,
                'selected_areas': [],
                'enhanced_prompt': None,
                'framework': None,
                'analysis_results': [],
                'summary': None
            },
            'ui_state': {
                'focus_selection_complete': False
            }
        }
        st.rerun()
    
    # Display progress indicator
    if st.session_state.app_state['stage'] == 'start':
        with st.status("🚀 Starting analysis...", expanded=True):
            st.write("Initializing analysis process...")
    
    # Run analysis
    framework, analysis, summary = analyze_topic(model, topic, iterations)
    
    # Handle completion
    if framework and analysis and summary:
        st.session_state.app_state.update({
            'framework': framework,
            'analysis': analysis,
            'summary': summary
        })
        if st.session_state.app_state['stage'] == 'complete':
            st.success("Analysis complete! Review the results above.") 