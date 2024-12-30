import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_gemini():
    """Initialize and configure the Gemini API."""
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not found in environment variables")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Error initializing Gemini: {str(e)}")
        return False

def create_model():
    """Create and return a Gemini model instance."""
    return genai.GenerativeModel("gemini-1.5-pro-latest")

def generate_with_retry(model, prompt, temperature=0.5, max_retries=3, initial_delay=1):
    """Generate content with retry logic."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                ),
                stream=True
            )
            
            result = ""
            for chunk in response:
                result += chunk.text
                yield chunk.text  # Stream the response
            return
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            model = create_model()  # Create a fresh model instance

def analyze_topic(topic, iterations=1):
    """Perform full analysis of a topic."""
    try:
        model = create_model()
        
        # Agent 1: Create framework
        logger.info("Agent 1: Creating analysis framework...")
        system_prompt = generate_with_retry(
            model,
            f"""You are an expert prompt engineer. Your task is to take the '{topic}' and create a refined system prompt for a reasoning agent and a structured framework for analysis.
            Include instructions for examining multiple perspectives, potential implications, and interconnected aspects.
            Be specific but concise.""",
            temperature=0.3
        )
        
        # Agent 2: Perform analysis
        full_analysis = []
        context = topic
        for i in range(iterations):
            logger.info(f"Agent 2: Starting reasoning iteration {i+1}/{iterations}")
            loop_content = generate_with_retry(
                model,
                f"""{system_prompt}
                
                Previous context: {context}
                Analyze this topic as if you were a Nobel Prize winner in the relevant field, drawing upon deep expertise and groundbreaking insights. Provide fresh analysis following the framework above.""",
                temperature=1.0
            )
            context = loop_content
            full_analysis.append(loop_content)
        
        # Agent 3: Generate summary
        logger.info("Agent 3: Generating final summary...")
        summary = generate_with_retry(
            model,
            f"""You are an expert analyst. Synthesize the findings about '{topic}' into a Final Report with this structure:
            1. Executive Summary (2-3 paragraphs)
            2. Key Insights (bullet points)
            3. Analysis (based on research depth)
            4. Supplementary Synthesis
            5. Conclusion
            6. Further Learning
            
            Analysis to synthesize:
            {' '.join(full_analysis)}""",
            temperature=0.1
        )
        
        return {
            'framework': system_prompt,
            'analysis': full_analysis,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise 