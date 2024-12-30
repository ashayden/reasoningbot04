import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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