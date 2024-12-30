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
.output-box { 
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 0.5rem 0;
    border-left: 5px solid #1f77b4;
}
.agent-status {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 1rem;
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
        results = {}
        
        # Agent 1: Prompt Engineer & Framework Designer
        with st.status("🎯 Agent 1: Designing analysis approach...") as status:
            # First, create an optimized prompt as an AI prompt engineer
            status.write("Creating optimized analysis prompt...")
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
            results['prompt'] = prompt_design
            
            # Then, create the analysis framework based on the optimized prompt
            status.write("Developing analysis framework...")
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
            results['framework'] = framework
            status.update(label="✅ Agent 1: Analysis approach designed", state="complete")
        
        # Agent 2: Analysis (Higher temperature for creative insights)
        analysis = []
        with st.status("🔄 Agent 2: Performing analysis...") as status:
            for i in range(iterations):
                status.write(f"Conducting analysis iteration {i+1}/{iterations}...")
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
            results['analysis'] = analysis
            status.update(label="✅ Agent 2: Analysis complete", state="complete")
        
        # Agent 3: Summary (Medium-low temperature for balanced synthesis)
        with st.status("📊 Agent 3: Generating final report...") as status:
            status.write("Synthesizing findings...")
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
            results['summary'] = summary
            status.update(label="✅ Agent 3: Final report generated", state="complete")
            
        return results
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        logger.error(f"Analysis error: {str(e)}")
        return None

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
    
    results = analyze_topic(model, topic, iterations)
    
    if results:
        # Debug logging
        logger.info(f"Results received: {list(results.keys())}")
        
        try:
            # Display Optimized Prompt
            st.write("### Results:")
            
            with st.expander("📝 Optimized Prompt", expanded=False):
                if results.get('prompt'):
                    st.markdown('<div class="output-box">' + results['prompt'] + '</div>', unsafe_allow_html=True)
                else:
                    st.warning("No prompt generated")
            
            # Display Framework
            with st.expander("🔍 Analysis Framework", expanded=False):
                if results.get('framework'):
                    st.markdown('<div class="output-box">' + results['framework'] + '</div>', unsafe_allow_html=True)
                else:
                    st.warning("No framework generated")
            
            # Display Analysis Iterations
            if results.get('analysis'):
                for i, analysis in enumerate(results['analysis'], 1):
                    with st.expander(f"🔄 Research Iteration {i}", expanded=False):
                        if analysis:
                            st.markdown('<div class="output-box">' + analysis + '</div>', unsafe_allow_html=True)
                        else:
                            st.warning(f"No analysis for iteration {i}")
            
            # Display Final Report
            with st.expander("📊 Final Report", expanded=False):
                if results.get('summary'):
                    st.markdown('<div class="output-box">' + results['summary'] + '</div>', unsafe_allow_html=True)
                else:
                    st.warning("No final report generated")
            
            st.success("Analysis complete! Explore the results in the sections above.")
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")
            logger.error(f"Error displaying results: {str(e)}")
    else:
        st.error("No results generated. Please try again.") 