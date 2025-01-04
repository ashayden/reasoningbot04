"""Main application file for MARA."""

import logging
import streamlit as st
import google.generativeai as genai
import time
from typing import Optional, Dict, Any

from config import GEMINI_MODEL
from utils import validate_topic, sanitize_topic, QuotaExceededError, MARAError
from agents import PreAnalysisAgent, PromptDesigner, FrameworkEngineer, ResearchAnalyst, SynthesisExpert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

[data-testid="baseButton-primary"]:disabled {
    background-color: #f8f9fa !important;
    border-color: #dee2e6 !important;
    box-shadow: none !important;
    color: #6c757d !important;
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
    background-color: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    color: #2c3338 !important;
}

/* Number input styling */
div[data-testid="stNumberInput"] input {
    color: #2c3338 !important;
    background-color: #ffffff !important;
    border: 1px solid #dee2e6 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: #f8f9fa !important;
    border: 1px solid #dee2e6 !important;
    color: #2c3338 !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: #e9ecef !important;
}
</style>
""", unsafe_allow_html=True)

# Logo/Header
st.image("assets/mara-logo.png", use_container_width=True)

class StateManager:
    """Manages application state and transitions."""
    
    def __init__(self):
        if 'app_state' not in st.session_state:
            self.reset()
    
    def reset(self, topic: Optional[str] = None, iterations: int = 2):
        """Reset application state with optional new topic and iterations."""
        st.session_state.app_state = {
            'topic': topic,
            'iterations': iterations,
            'show_insights': bool(topic),
            'show_focus': False,
            'show_framework': False,
            'show_analysis': False,
            'show_summary': False,
            'insights': None,
            'focus_areas': None,
            'framework': None,
            'analysis_results': [],
            'focus_selection_complete': False,
            'selected_areas': [],
            'prompt': None
        }
        
        # Clear focus area states
        for key in ['focus_area_expanded', 'current_focus_areas']:
            if key in st.session_state:
                del st.session_state[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from app state."""
        return st.session_state.app_state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a value in app state."""
        st.session_state.app_state[key] = value
    
    def show_stage(self, stage: str):
        """Show a stage in the application."""
        self.set(f'show_{stage}', True)
    
    def hide_stage(self, stage: str):
        """Hide a stage in the application."""
        self.set(f'show_{stage}', False)
    
    def is_stage_visible(self, stage: str) -> bool:
        """Check if a stage is visible."""
        return self.get(f'show_{stage}', False)
    
    def can_proceed_to_framework(self) -> bool:
        """Check if we can proceed to framework stage."""
        return self.get('focus_selection_complete', False)
    
    def can_proceed_to_summary(self) -> bool:
        """Check if we can proceed to summary stage."""
        return len(self.get('analysis_results', [])) == self.get('iterations', 2)

class StageProcessor:
    """Handles processing of analysis stages."""
    
    def __init__(self, model, state_manager: StateManager):
        self.model = model
        self.state = state_manager
    
    def process_stage(self, stage_name: str, container, stage_fn, next_stage=None, 
                     spinner_text=None, display_fn=None, **kwargs):
        """Process a single stage of the analysis with simplified error handling."""
        try:
            # Check if we need to generate content
            if self._should_generate_content(stage_name):
                with container, st.spinner(spinner_text or f"Processing {stage_name}..."):
                    result = stage_fn(**kwargs)
                    if not result:
                        raise ValueError(f"Failed to generate content for {stage_name}")
                    
                    self.state.set(stage_name, result)
                    
                    # Automatically show next stage for insights
                    if stage_name == 'insights':
                        self.state.show_stage('focus')
                    
                    st.rerun()
            
            # Display existing content if available
            if display_fn and self.state.get(stage_name) is not None:
                self._display_content(stage_name, container, display_fn)
                
                # Special handling for insights stage
                if stage_name == 'insights' and not self.state.get('prompt'):
                    self._generate_optimized_prompt()
                
                # Special handling for framework stage
                if stage_name == 'framework':
                    self.state.show_stage('analysis')
            
        except Exception as e:
            self._handle_error(stage_name, str(e))
    
    def _should_generate_content(self, stage_name: str) -> bool:
        """Check if content needs to be generated for a stage."""
        return stage_name not in st.session_state.app_state or self.state.get(stage_name) is None
    
    def _display_content(self, stage_name: str, container, display_fn):
        """Display existing content for a stage."""
        try:
            with container:
                display_fn(self.state.get(stage_name))
        except Exception as e:
            self._handle_error(stage_name, f"Failed to display content: {str(e)}")
    
    def _generate_optimized_prompt(self):
        """Generate optimized prompt after insights are displayed."""
        try:
            optimized_prompt = PromptDesigner(self.model).design_prompt(self.state.get('topic'))
            if optimized_prompt:
                self.state.set('prompt', optimized_prompt)
        except Exception as e:
            logger.error(f"Error generating optimized prompt: {str(e)}")
    
    def _handle_error(self, stage_name: str, error_msg: str):
        """Handle errors during stage processing."""
        logger.error(f"Error in {stage_name} stage: {error_msg}")
        st.error(f"Failed to process {stage_name}. Please try again.")
        self.state.hide_stage(stage_name)
    
    def process_analysis(self, container):
        """Process the analysis stage with iteration handling."""
        with container:
            current_results = len(self.state.get('analysis_results', []))
            total_iterations = self.state.get('iterations', 2)
            
            if current_results < total_iterations:
                with st.spinner("🔄 Performing analysis..."):
                    result = ResearchAnalyst(self.model).analyze(
                        self.state.get('topic'),
                        self.state.get('framework', ''),
                        self.state.get('analysis_results', [])[-1] if self.state.get('analysis_results') else None
                    )
                    
                    if result:
                        content = format_analysis_result(result)
                        analysis_results = self.state.get('analysis_results', [])
                        analysis_results.append(content)
                        self.state.set('analysis_results', analysis_results)
                        
                        if len(analysis_results) == total_iterations:
                            self.state.show_stage('summary')
                        st.rerun()
            
            # Display existing results
            for i, result in enumerate(self.state.get('analysis_results', [])):
                with st.expander(f"🔄 Research Analysis #{i + 1}", expanded=False):
                    st.markdown(result)

def initialize_gemini():
    """Initialize the Gemini model with caching."""
    try:
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("Google API key not found in Streamlit secrets.")
            return None
            
        api_key = st.secrets["GOOGLE_API_KEY"]
        if not api_key:
            st.error("Google API key is empty. Please check your Streamlit secrets.")
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
                
    except Exception as e:
        if "429" in str(e):
            st.error("API quota exceeded. Please wait a few minutes and try again.")
        else:
            st.error(f"Failed to initialize Gemini model: {str(e)}")
        return None

@st.cache_resource
def get_model():
    return initialize_gemini()

model = get_model()
if not model:
    st.error("Failed to initialize the AI model. Please check your API key in Streamlit secrets and try again.")
    st.stop()

def validate_and_sanitize_input(topic: str) -> tuple[bool, str, str]:
    """Validate and sanitize user input."""
    logger.info(f"Validating topic: '{topic}'")
    
    # First check if topic is None or empty
    if topic is None:
        logger.error("Topic is None")
        return False, "Please enter a topic to analyze.", ""
    
    # Validate the topic
    is_valid, error_msg = validate_topic(topic)
    if not is_valid:
        logger.error(f"Topic validation failed: {error_msg}")
        return False, error_msg, ""
    
    # Sanitize the topic
    sanitized = sanitize_topic(topic)
    if not sanitized:
        logger.error("Sanitized topic is empty")
        return False, "Please enter a valid topic to analyze.", ""
    
    logger.info(f"Topic validated and sanitized successfully: '{sanitized}'")
    return True, "", sanitized

def handle_error(e: Exception, context: str):
    """Handle errors consistently."""
    error_msg = f"Error during {context}: {str(e)}"
    logger.error(error_msg)
    
    # Handle quota exceeded errors with clear user feedback
    if isinstance(e, QuotaExceededError):
        st.error("⚠️ API quota limit reached. Please wait 5 minutes before trying again.")
        st.info("💡 This helps ensure fair usage of the API for all users.")
        # Disable the form temporarily
        st.session_state.form_disabled = True
        # Schedule re-enable after 5 minutes
        st.session_state.quota_reset_time = time.time() + 300  # 5 minutes
        return
    
    # Provide user-friendly error message for other errors
    user_msg = {
        'insights': "Failed to generate initial insights. Please try again.",
        'prompt': "Failed to optimize the prompt. Please try again.",
        'focus': "Failed to generate focus areas. Please try again.",
        'framework': "Failed to build analysis framework. Please try again.",
        'analysis': "Failed during analysis. Please try again.",
        'summary': "Failed to generate final report. Please try again."
    }.get(context, "An unexpected error occurred. Please try again.")
    
    st.error(user_msg)
    
    # Reset state for the current stage
    if context in st.session_state.app_state:
        st.session_state.app_state[context] = None

def handle_form_submission(state_manager: StateManager, topic: str, iterations: int):
    """Handle form submission and validate input."""
    is_valid, error_msg, sanitized_topic = validate_and_sanitize_input(topic)
    if not is_valid:
        st.error(error_msg)
        return
    
    state_manager.reset(sanitized_topic, iterations)
    st.rerun()

def main():
    """Main application function."""
    state_manager = StateManager()
    stage_processor = StageProcessor(model, state_manager)
    
    # Check if quota timer has expired
    if hasattr(st.session_state, 'quota_reset_time'):
        if time.time() >= st.session_state.quota_reset_time:
            st.session_state.form_disabled = False
            del st.session_state.quota_reset_time
    
    # Input form
    with st.form("analysis_form"):
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
        
        submit = st.form_submit_button(
            "🚀 Start Analysis",
            use_container_width=True,
            disabled=st.session_state.get('form_disabled', False)
        )
    
    if submit:
        handle_form_submission(state_manager, topic, iterations)
    
    if not state_manager.get('topic'):
        return
    
    try:
        # Process insights
        if state_manager.is_stage_visible('insights'):
            stage_processor.process_stage(
                'insights',
                st.container(),
                lambda: PreAnalysisAgent(model).generate_insights(state_manager.get('topic')),
                'focus',
                "💡 Generating insights...",
                display_insights
            )
        
        # Process focus areas
        if state_manager.is_stage_visible('focus'):
            focus_container = st.container()
            stage_processor.process_stage(
                'focus',
                focus_container,
                lambda: PreAnalysisAgent(model).generate_focus_areas(state_manager.get('topic')),
                'framework',
                "🎯 Generating focus areas...",
                display_focus_areas
            )
        
        # Process framework
        if state_manager.is_stage_visible('framework') and state_manager.can_proceed_to_framework():
            framework_container = st.container()
            stage_processor.process_stage(
                'framework',
                framework_container,
                lambda: FrameworkEngineer(model).create_framework(
                    state_manager.get('prompt', ''),
                    state_manager.get('enhanced_prompt')
                ),
                'analysis',
                "🔨 Building analysis framework...",
                lambda x: st.expander("🎯 Research Framework", expanded=True).markdown(x)
            )
            
            # Ensure analysis stage is shown after framework
            if state_manager.get('framework'):
                state_manager.show_stage('analysis')
        
        # Process analysis
        if state_manager.is_stage_visible('analysis'):
            analysis_container = st.container()
            stage_processor.process_analysis(analysis_container)
        
        # Process summary
        if state_manager.is_stage_visible('summary'):
            stage_processor.process_stage(
                'summary',
                st.container(),
                lambda: SynthesisExpert(model).synthesize(
                    state_manager.get('topic'),
                    state_manager.get('analysis_results', [])
                ),
                spinner_text="📊 Generating final report...",
                display_fn=lambda x: st.expander("📊 Final Report", expanded=False).markdown(x)
            )
            
    except Exception as e:
        handle_error(e, "analysis")

def format_analysis_result(result: Dict[str, str]) -> str:
    """Format analysis result with consistent structure."""
    content = ""
    if result['title']:
        content += f"# {result['title']}\n\n"
    if result['subtitle']:
        content += f"*{result['subtitle']}*\n\n"
    if result['content']:
        content += result['content']
    return content

def display_insights(insights: dict):
    """Display insights in proper containers."""
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
        if 'selected_areas' not in st.session_state.app_state:
            st.session_state.app_state['selected_areas'] = []
        
        # Create columns for focus area selection
        cols = st.columns(2)
        for i, area in enumerate(focus_areas):
            col_idx = i % 2
            with cols[col_idx]:
                key = f"focus_area_{i}"
                is_selected = area in st.session_state.app_state['selected_areas']
                if st.checkbox(area, value=is_selected, key=key):
                    if area not in st.session_state.app_state['selected_areas']:
                        st.session_state.app_state['selected_areas'].append(area)
                elif area in st.session_state.app_state['selected_areas']:
                    st.session_state.app_state['selected_areas'].remove(area)
        
        # Show selection status
        st.markdown("---")
        num_selected = len(st.session_state.app_state['selected_areas'])
        
        if num_selected > 5:
            st.warning("⚠️ Please select no more than 5 focus areas")
        else:
            if num_selected > 0:
                st.success(f"✅ You have selected {num_selected} focus area{'s' if num_selected > 1 else ''}")
                st.write("Selected areas:")
                for area in st.session_state.app_state['selected_areas']:
                    st.write(f"- {area}")
            
            # Add buttons side by side
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Skip", type="secondary"):
                    st.session_state.focus_container_expanded = False
                    st.session_state.app_state['focus_selection_complete'] = True
                    st.session_state.app_state['show_framework'] = True
                    st.rerun()
            with col2:
                if num_selected <= 5:
                    if st.button("Continue", type="primary"):
                        st.session_state.focus_container_expanded = False
                        st.session_state.app_state['focus_selection_complete'] = True
                        st.session_state.app_state['show_framework'] = True
                        st.rerun()

if __name__ == "__main__":
    main() 