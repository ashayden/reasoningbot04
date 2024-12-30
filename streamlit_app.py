import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug information
logger.info("Starting Streamlit app...")
st.set_page_config(
    page_title="Gemini Reasoning Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger.info("Page config set")

# Basic UI elements
try:
    st.title("Gemini Reasoning Bot")
    logger.info("Title rendered")
    
    st.write("Welcome to the Gemini Reasoning Bot!")
    logger.info("Welcome message rendered")
    
    # Simple input form
    with st.form("input_form"):
        topic = st.text_input("What topic would you like to explore?")
        submit = st.form_submit_button("Start Analysis")
        logger.info("Form rendered")
    
    if submit and topic:
        st.info(f"Starting analysis of: {topic}")
        logger.info(f"Form submitted with topic: {topic}")

except Exception as e:
    logger.error(f"Error in UI rendering: {str(e)}")
    st.error(f"An error occurred: {str(e)}") 