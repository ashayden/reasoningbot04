import streamlit as st
import google.generativeai as genai

# Configure API key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("Error accessing Google API key. Please check your Streamlit secrets configuration.")
    st.stop()

# Initialize model
def create_model():
    return genai.GenerativeModel("gemini-1.5-pro-latest")

# Custom CSS for better UI
st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
    max-width: 800px;
}

.stTextInput > div > div > input {
    padding: 0.5rem 1rem;
    font-size: 1rem;
    border-radius: 0.5rem;
}

.stButton > button {
    width: 100%;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    font-weight: 500;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.streamlit-expanderHeader {
    font-size: 1rem;
    font-weight: 600;
    padding: 0.75rem 0;
    border-radius: 0.5rem;
}

.element-container {
    margin-bottom: 1rem;
}

.main-title {
    font-size: 2.5rem !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Progress Steps
STEPS = ["Preparing", "Developing", "Researching", "Complete"]

def render_stepper(current_step: int) -> str:
    """Renders a progress stepper with proper styling."""
    html = """
        <style>
        .stepper-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 2rem auto;
            padding: 1rem 2rem;
            max-width: 700px;
            background: transparent;
            position: relative;
        }
        .step {
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            flex: 1;
        }
        .step-number {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.2);
            color: rgba(255, 255, 255, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-bottom: 8px;
            z-index: 2;
        }
        .step-label {
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
        }
        .step-line {
            position: absolute;
            top: 18px;
            left: calc(50% + 25px);
            right: calc(-50% + 25px);
            height: 2px;
            background-color: rgba(255, 255, 255, 0.2);
            z-index: 1;
        }
        .step.active .step-number {
            border-color: #2439f7;
            color: #2439f7;
            background-color: rgba(255, 255, 255, 0.9);
        }
        .step.active .step-label {
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
        }
        .step.complete .step-number {
            background-color: #28a745;
            border-color: #28a745;
            color: white;
        }
        .step.complete .step-line {
            background-color: #28a745;
        }
        .step:last-child .step-line {
            display: none;
        }
        </style>
    """
    
    steps_html = []
    for i, label in enumerate(STEPS):
        status = "complete" if i < current_step else "active" if i == current_step else ""
        steps_html.append(f'''
            <div class="step {status}">
                <div class="step-number">{i + 1}</div>
                <div class="step-label">{label}</div>
                <div class="step-line"></div>
            </div>
        ''')
    
    return html + f'<div class="stepper-container">{"".join(steps_html)}</div>'

# Main UI
st.markdown("<h1 class='main-title'>Gemini Reasoning Bot</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    ### About
    This application uses multiple AI agents to perform deep analysis on any topic:
    1. **🎯 Agent 1** creates a sophisticated analysis framework
    2. **🔄 Agent 2** performs multiple iterations of deep reasoning
    3. **📊 Agent 3** synthesizes the findings into a comprehensive report
    """)
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Enter your topic of interest
    2. Choose analysis depth
    3. Click 'Start Analysis' and watch the magic happen!
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    with st.form("analysis_form"):
        topic = st.text_input(
            "What topic would you like to explore?",
            placeholder="Enter any topic, e.g., 'Artificial Intelligence' or 'Climate Change'"
        )
        depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Balanced", "Deep", "Comprehensive"],
            value="Balanced",
            help="More depth = deeper analysis, but takes longer"
        )
        submit = st.form_submit_button("🚀 Start Analysis")

with col2:
    st.info("👈 Enter your topic and click 'Start Analysis' to begin!")

# Display progress stepper if analysis is in progress
if st.session_state.current_step > 0:
    st.markdown(render_stepper(st.session_state.current_step), unsafe_allow_html=True)

if submit and topic:
    # Test API connection
    try:
        model = create_model()
        test_response = model.generate_content("Hello, are you working?")
        st.success("API connection successful!")
        st.session_state.current_step = 1
        st.markdown(render_stepper(st.session_state.current_step), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error connecting to Gemini API: {str(e)}")
        st.stop() 