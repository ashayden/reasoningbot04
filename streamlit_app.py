import os
import time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def create_model():
    return genai.GenerativeModel("gemini-1.5-pro-latest")

def generate_with_retry(model, prompt, temperature=0.5, max_retries=3, initial_delay=1):
    placeholder = st.empty()
    result = ""
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                ),
                stream=True
            )
            
            for chunk in response:
                result += chunk.text
                placeholder.markdown(result)
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error: Failed after {max_retries} attempts: {str(e)}")
                raise
            st.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            model = create_model()

def run_analysis(topic, num_iterations):
    try:
        # Initialize model
        model = create_model()
        
        # Agent 1: Framework Creation
        st.subheader("Agent 1: Creating Analysis Framework")
        system_prompt = generate_with_retry(
            model,
            f"""You are an expert prompt engineer. Your task is to take the '{topic}' and create a refined system prompt for a reasoning agent and a structured framework for analysis.
            Include instructions for examining multiple perspectives, potential implications, and interconnected aspects.
            Be specific but concise.""",
            temperature=0.3
        )

        # Agent 2: Reasoning Loops
        full_analysis = []
        context = topic
        
        for i in range(num_iterations):
            st.subheader(f"Agent 2: Reasoning Loop {i+1}/{num_iterations}")
            loop_content = generate_with_retry(
                model,
                f"""{system_prompt}
                
                Previous context: {context}
                Analyze this topic as if you were a Nobel Prize winner in the relevant field, drawing upon deep expertise and groundbreaking insights. Provide fresh analysis following the framework above.""",
                temperature=1.0
            )
            context = loop_content
            full_analysis.append(loop_content)
            st.divider()

        # Agent 3: Final Summary
        st.subheader("Agent 3: Final Summary")
        summary = generate_with_retry(
            model,
            f"""You are an expert analyst. Provide a comprehensive final report with the following structure: 
            Synthesize the findings from the analysis loops about '{topic}' into a Final Report with the following structure: 

            1. Executive Summary
            - A concise overview of the investigation and key findings (2-3 paragraphs)

            2. Key Insights
            - Bullet-pointed list of the most important discoveries and conclusions
            - Focus on actionable and noteworthy findings
            - Include surprising or counter-intuitive insights

            3. Analysis
            [Scale analysis depth based on research loops]
            - Synthesize major concepts and themes from the research
            - Examine relationships between different aspects
            - Support claims with evidence from the research
            - Address any contradictions or nuances found

            4. Supplementary Synthesis
            [Dynamic section based on topic and research depth]
            Choose relevant elements from:
            - Recommendations for action or further investigation
            - Implications of the findings
            - Counter-arguments or alternative perspectives
            - Significance and broader impact
            - Limitations of current understanding
            - Future trends or developments

            5. Conclusion
            - Summarize the most important takeaways
            - Place findings in broader context
            - Highlight remaining questions or areas for future research

            6. Further Learning
            - List key sources referenced in the analysis
            - Recommend additional reading materials
            - Finally, suggest 3 follow-up questions:
                - 1 that digs deeper into a key aspect of the topic
                - 1 that explores a related topic
                - 1 that investigates surprising or unexpected connections to the topic

            Write in a clear, authoritative tone. Support all major claims with evidence from the research.
                
            Analysis to synthesize:
            {' '.join(full_analysis)}""",
            temperature=0.1
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.warning("Please try again. If the error persists, try fewer iterations.")
        return False

# Streamlit UI
st.title("Gemini Reasoning Bot")
st.markdown("""
This application uses multiple AI agents to perform deep analysis on any topic:
1. **Agent 1** creates a sophisticated analysis framework
2. **Agent 2** performs multiple iterations of deep reasoning
3. **Agent 3** synthesizes the findings into a comprehensive report
""")

# Input form
with st.form("analysis_form"):
    topic = st.text_input("What topic would you like to explore?")
    num_iterations = st.slider("Number of reasoning iterations", min_value=1, max_value=5, value=2)
    submit = st.form_submit_button("Start Analysis")

if submit:
    if not topic:
        st.error("Please enter a topic to analyze.")
    else:
        with st.spinner("Analysis in progress..."):
            success = run_analysis(topic, num_iterations)
            if success:
                st.success("Analysis completed successfully!") 