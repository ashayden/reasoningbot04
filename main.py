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
    background-color: #4a4a4a !important;
    border-color: #4a4a4a !important;
    color: white !important;
}

[data-testid="baseButton-secondary"]:hover {
    background-color: #5a5a5a !important;
    border-color: #5a5a5a !important;
}

[data-testid="baseButton-primary"] {
    background-color: #0066cc !important;
    border: none !important;
}

[data-testid="baseButton-primary"]:disabled {
    background-color: #1E1E1E !important;
    color: #4a4a4a !important;
    cursor: not-allowed;
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

/* Focus area styling */
div[data-testid="stCheckbox"] {
    background-color: rgba(70, 70, 70, 0.2);
    border-radius: 30px;
    padding: 0.5rem 1rem;
    margin: 0.25rem 0;
    transition: background-color 0.2s ease;
}

div[data-testid="stCheckbox"]:hover {
    background-color: rgba(70, 70, 70, 0.4);
}

div[data-testid="stCheckbox"] label {
    cursor: pointer;
}

div[data-testid="stCheckbox"] label span {
    font-size: 0.9em;
}

div[data-testid="stCheckbox"] label p {
    margin: 0;
    padding: 0;
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
    
# Initialize focus area state
if 'focus_state' not in st.session_state:
    st.session_state.focus_state = {
        'areas': None,
        'selected': set(),
        'proceed': False,
        'enhanced_prompt': None
    }

def reset_focus_state():
    """Reset the focus area state."""
    if 'focus_state' in st.session_state:
        del st.session_state.focus_state
    if 'checkbox_states' in st.session_state:
        del st.session_state.checkbox_states
    st.session_state.focus_state = {
        'areas': None,
        'selected': set(),
        'proceed': False,
        'enhanced_prompt': None
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
        pre_analysis = PreAnalysisAgent(model)
        prompt_designer = PromptDesigner(model)
        framework_engineer = FrameworkEngineer(model)
        research_analyst = ResearchAnalyst(model)
        synthesis_expert = SynthesisExpert(model)
        
        # Get quick insights
        insights = pre_analysis.generate_insights(topic)
        if not insights:
            return None, None, None
        
        # Display Did You Know section
        with st.status("💡 Did You Know", expanded=True) as status:
            st.markdown(insights['did_you_know'])
            status.update(label="💡 Did You Know")
            
        # Display ELI5 section
        with st.status("⚡ ELI5", expanded=True) as status:
            st.markdown(insights['eli5'])
            status.update(label="⚡ ELI5")
        
        # Agent 0: Prompt Designer - First generate the optimized prompt
        with st.status("✍️ Designing optimal prompt...") as status:
            initial_prompt = prompt_designer.design_prompt(topic)
            if not initial_prompt:
                return None, None, None
            st.markdown(initial_prompt)
            status.update(label="✍️ Optimized Prompt")
            
        # Generate focus areas if not already done
        if st.session_state.focus_state['areas'] is None:
            st.session_state.focus_state['areas'] = prompt_designer.generate_focus_areas(topic)
            
        # Display focus areas for selection
        if st.session_state.focus_state['areas']:
            st.markdown("### 🎯 Select Focus Areas")
            st.markdown("Choose specific aspects you'd like the analysis to emphasize (optional):")
            
            # Initialize checkbox states if not exists
            if 'checkbox_states' not in st.session_state:
                st.session_state.checkbox_states = {}
            
            # Create a form for focus area selection
            with st.form(key="focus_area_form"):
                # Create columns for better layout
                cols = st.columns(3)
                
                # Distribute focus areas across columns
                for i, area in enumerate(st.session_state.focus_state['areas']):
                    col_idx = i % 3
                    # Initialize checkbox state if not exists
                    if f'checkbox_{i}' not in st.session_state.checkbox_states:
                        st.session_state.checkbox_states[f'checkbox_{i}'] = area in st.session_state.focus_state['selected']
                    
                    # Display checkbox in appropriate column
                    st.session_state.checkbox_states[f'checkbox_{i}'] = cols[col_idx].checkbox(
                        area,
                        value=st.session_state.checkbox_states[f'checkbox_{i}'],
                        key=f'focus_area_{i}'
                    )
                
                # Add some spacing
                st.markdown("---")
                
                # Create two columns for buttons
                col1, col2 = st.columns(2)
                
                # Submit buttons
                submitted = st.form_submit_button("Continue", type="primary")
                skipped = st.form_submit_button("Skip")
                
            # Handle form submission
            if submitted or skipped:
                if submitted:
                    # Update selected areas based on checkbox states
                    st.session_state.focus_state['selected'] = {
                        area for i, area in enumerate(st.session_state.focus_state['areas'])
                        if st.session_state.checkbox_states[f'checkbox_{i}']
                    }
                    
                    # Generate enhanced prompt if areas are selected
                    if st.session_state.focus_state['selected']:
                        st.session_state.focus_state['enhanced_prompt'] = prompt_designer.design_prompt(
                            topic,
                            list(st.session_state.focus_state['selected'])
                        )
                        
                        # Show updated prompt in a collapsed section
                        with st.status("✍️ Updated Prompt", expanded=False) as status:
                            st.markdown(st.session_state.focus_state['enhanced_prompt'])
                            status.update(label="✍️ Updated Prompt")
                
                # Set proceed flag
                st.session_state.focus_state['proceed'] = True
            
            # Stop here if form hasn't been submitted
            if not st.session_state.focus_state['proceed']:
                st.stop()

        # Agent 1: Framework Engineer - Pass both prompts (only reached after user choice)
        with st.status("🎯 Creating analysis framework...") as status:
            framework = framework_engineer.create_framework(
                initial_prompt,
                st.session_state.focus_state['enhanced_prompt']
            )
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
    # Reset state if topic changed
    if st.session_state.current_analysis['topic'] != topic:
        st.session_state.current_analysis = {
            'topic': topic,
            'framework': None,
            'analysis': None,
            'summary': None
        }
        # Reset focus area state
        reset_focus_state()
    
    # Run analysis
    framework, analysis, summary = analyze_topic(model, topic, iterations)
    
    if framework and analysis and summary:
        st.session_state.current_analysis.update({
            'framework': framework,
            'analysis': analysis,
            'summary': summary
        })
        st.success("Analysis complete! Review the results above.") 