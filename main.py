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
        # Agent 0: Prompt Designer - Very low temperature for precise prompt engineering
        with st.status("✍️ Designing optimal prompt..."):
            prompt_design = model.generate_content(
                f"""As an expert prompt engineer, create a detailed and thorough prompt to analyze '{topic}'.
                The prompt should:
                1. Break down the topic into key components
                2. Specify the depth and breadth of analysis needed
                3. Include relevant context and constraints
                4. Define clear objectives and deliverables
                Be specific and structured in your prompt design.""",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Very low temperature for precise, consistent output
                    candidate_count=1,
                    max_output_tokens=1024
                )
            ).text
            st.markdown(prompt_design)

        # Agent 1: Framework - Lower temperature for more focused, structured output
        with st.status("🎯 Creating analysis framework..."):
            framework = model.generate_content(
                prompt_design,  # Using the optimized prompt from Agent 0
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for structured, consistent framework
                    candidate_count=1,
                    max_output_tokens=1024
                )
            ).text
            st.markdown(framework)
        
        # Agent 2: Analysis - Higher temperature for creative, diverse perspectives
        analysis = []
        with st.status("🔄 Performing analysis..."):
            for i in range(iterations):
                iteration_title = f"Analysis #{i+1}: {topic.title()} - Key Dimensions"
                st.write(f"### {iteration_title}")
                result = model.generate_content(
                    f"""{framework}\n\nAnalyze '{topic}' as a Nobel laureate, following the framework above. Previous context: {analysis[-1] if analysis else topic}""",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,  # Higher temperature for creative analysis
                        candidate_count=1,
                        max_output_tokens=2048
                    )
                ).text
                analysis.append(result)
                st.markdown(result)
        
        # Agent 3: Summary - Medium-low temperature for balanced, coherent synthesis
        with st.status("📊 Generating final report..."):
            summary = model.generate_content(
                f"""Synthesize this analysis of '{topic}' into a Final Report with:
                1. Executive Summary (2-3 paragraphs)
                2. Key Insights (bullet points)
                3. Analysis
                4. Conclusion
                
                Analysis to synthesize: {' '.join(analysis)}""",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Medium-low temperature for coherent synthesis
                    candidate_count=1,
                    max_output_tokens=4096
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
    1. 🎯 Framework Engineer
    2. 🔄 Research Analyst
    3. 📊 Synthesis Expert
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