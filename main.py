import streamlit as st
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="M.A.R.A. - Multi-Agent Reasoning Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.block-container { max-width: 800px; padding: 2rem 1rem; }
.main-title { font-size: 2.5rem; text-align: center; margin-bottom: 2rem; font-weight: 700; }
.stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini
@st.cache_resource
def initialize_gemini():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-pro-latest")
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

def analyze_topic(model, topic, iterations=1):
    """Perform multi-agent analysis of a topic."""
    try:
        # Agent 1: Framework (Low temperature for structured, consistent output)
        with st.status("🎯 Agent 1: Creating analysis framework..."):
            framework = model.generate_content(
                f"""Create a refined analysis framework for '{topic}'. Include multiple perspectives and implications. Be specific but concise.
                Structure the framework with:
                1. Key Dimensions for Analysis
                2. Critical Questions to Address
                3. Potential Impact Areas
                4. Methodological Approach""",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Very low for structured, consistent output
                    top_p=0.95,
                    top_k=40
                )
            ).text
            st.markdown(framework)
        
        # Agent 2: Analysis (Higher temperature for creative insights)
        analysis = []
        with st.status("🔄 Agent 2: Performing analysis..."):
            for i in range(iterations):
                st.write(f"Iteration {i+1}/{iterations}")
                result = model.generate_content(
                    f"""{framework}\n\nAnalyze '{topic}' as a Nobel laureate, following the framework above. 
                    Previous context: {analysis[-1] if analysis else topic}
                    
                    Focus on:
                    1. Novel insights and perspectives
                    2. Critical analysis of assumptions
                    3. Interdisciplinary connections
                    4. Practical implications""",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.9,  # High for creative, diverse insights
                        top_p=0.95,
                        top_k=40
                    )
                ).text
                analysis.append(result)
                st.markdown(result)
        
        # Agent 3: Summary (Medium-low temperature for balanced synthesis)
        with st.status("📊 Agent 3: Generating final report..."):
            summary = model.generate_content(
                f"""Synthesize this analysis of '{topic}' into a Final Report with:
                1. Executive Summary (2-3 paragraphs)
                2. Key Insights (bullet points)
                3. Analysis
                4. Conclusion
                
                Analysis to synthesize: {' '.join(analysis)}
                
                Ensure the synthesis:
                1. Maintains objectivity
                2. Highlights key patterns
                3. Addresses contradictions
                4. Provides actionable insights""",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Medium-low for balanced synthesis
                    top_p=0.95,
                    top_k=40
                )
            ).text
            st.markdown(summary)
            
        return framework, analysis, summary
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        logger.error(f"Analysis error: {str(e)}")
        return None, None, None

# Main UI
st.markdown("<h1 class='main-title'>M.A.R.A. 🤖</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    ### About
    Multi-Agent Reasoning Assistant powered by:
    1. 🎯 Framework Engineer (Structured, T=0.1)
    2. 🔄 Research Analyst (Creative, T=0.9)
    3. 📊 Synthesis Expert (Balanced, T=0.3)
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
        options=["Quick", "Balanced", "Deep", "Comprehensive"],
        value="Balanced"
    )
    
    submit = st.form_submit_button("🚀 Start Analysis")

if submit and topic:
    depth_iterations = {"Quick": 1, "Balanced": 2, "Deep": 3, "Comprehensive": 4}
    iterations = depth_iterations[depth]
    
    framework, analysis, summary = analyze_topic(model, topic, iterations)
    
    if framework and analysis and summary:
        st.success("Analysis complete! Review the results above.") 