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
.prompt-box { 
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 5px solid #1f77b4;
}
.framework-box {
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 5px solid #2ca02c;
}
.analysis-box {
    background-color: #fff;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #ddd;
}
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #eee;
}
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
        # Agent 1: Prompt Engineer & Framework Designer
        with st.status("🎯 Agent 1: Designing analysis approach..."):
            # First, create an optimized prompt as an AI prompt engineer
            prompt_design = model.generate_content(
                f"""As an expert AI Prompt Engineer, create a comprehensive and detailed prompt to analyze '{topic}'.
                
                Consider:
                1. Key aspects that need exploration
                2. Potential biases to address
                3. Scope and limitations
                4. Specific areas of focus
                5. Required expertise and perspectives
                
                Structure the prompt to ensure:
                1. Clear objectives and deliverables
                2. Multiple analytical dimensions
                3. Balanced perspective requirements
                4. Specific evaluation criteria
                5. Innovation and practical relevance
                
                Format the prompt professionally with clear sections and instructions.""",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Very low for precise, structured output
                    top_p=0.95,
                    top_k=40
                )
            ).text
            
            # Display prompt
            st.markdown('<div class="section-header">📝 Optimized Analysis Prompt</div>', unsafe_allow_html=True)
            st.markdown('<div class="prompt-box">' + prompt_design + '</div>', unsafe_allow_html=True)
            
            # Then, create the analysis framework based on the optimized prompt
            framework = model.generate_content(
                f"""Using the following prompt as a foundation:
                {prompt_design}
                
                Create a refined analysis framework for '{topic}'. Include multiple perspectives and implications. Be specific but concise.
                
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
            
            # Display framework
            st.markdown('<div class="section-header">🔍 Analysis Framework</div>', unsafe_allow_html=True)
            st.markdown('<div class="framework-box">' + framework + '</div>', unsafe_allow_html=True)
        
        # Agent 2: Analysis (Higher temperature for creative insights)
        analysis = []
        with st.status("🔄 Agent 2: Performing analysis..."):
            for i in range(iterations):
                st.write(f"Iteration {i+1}/{iterations}")
                result = model.generate_content(
                    f"""{prompt_design}\n\n{framework}\n\nAnalyze '{topic}' as a Nobel laureate, following the framework above. 
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
                st.markdown(f'<div class="section-header">📊 Analysis Iteration {i+1}</div>', unsafe_allow_html=True)
                st.markdown('<div class="analysis-box">' + result + '</div>', unsafe_allow_html=True)
        
        # Agent 3: Summary (Medium-low temperature for balanced synthesis)
        with st.status("📊 Agent 3: Generating final report..."):
            summary = model.generate_content(
                f"""Using the optimized prompt and framework:
                {prompt_design}
                
                {framework}
                
                Synthesize this analysis of '{topic}' into a Final Report with:
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
            st.markdown('<div class="section-header">📑 Final Report</div>', unsafe_allow_html=True)
            st.markdown('<div class="analysis-box">' + summary + '</div>', unsafe_allow_html=True)
            
        return prompt_design, framework, analysis, summary
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        logger.error(f"Analysis error: {str(e)}")
        return None, None, None, None

# Main UI
st.markdown("<h1 class='main-title'>M.A.R.A. 🤖</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    ### About
    Multi-Agent Reasoning Assistant powered by:
    1. 🎯 Framework Engineer & Prompt Expert (Structured, T=0.1)
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
    
    prompt_design, framework, analysis, summary = analyze_topic(model, topic, iterations)
    
    if all([prompt_design, framework, analysis, summary]):
        st.success("Analysis complete! Review the results above.") 