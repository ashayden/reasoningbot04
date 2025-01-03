"""Agent implementations for the MARA application."""

import logging
from typing import Any, Dict, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from config import (
    PROMPT_DESIGN_CONFIG,
    FRAMEWORK_CONFIG,
    ANALYSIS_CONFIG,
    SYNTHESIS_CONFIG
)
from utils import rate_limit_decorator

logger = logging.getLogger(__name__)

class PreAnalysisAgent:
    """Quick insights agent."""
    
    def __init__(self, model: Any):
        self.model = model
    
    @rate_limit_decorator
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate quick insights about the topic."""
        try:
            # Generate fun fact
            fact_prompt = (
                f"Generate a single, fascinating, and unexpected fact about {topic}, presented in one sentence with emojis. "
                "The fact needs to be surprising, unique, and counter-intuitive, revealing a lesser-known aspect with "
                "vivid language and potentially statistics. It should challenge common assumptions."
            )
            
            fact_response = self.model.generate_content(
                fact_prompt,
                generation_config=GenerationConfig(**PROMPT_DESIGN_CONFIG)
            )
            if not fact_response or not fact_response.text:
                return None
            
            # Generate ELI5
            eli5_prompt = (
                f"Write a very short, engaging explanation of {topic} for a general audience. Use simple language, "
                "a fun analogy, and emojis to make it memorable. Focus on a key, aspect "
                "of the topic. Make it 2-3 sentences maximum."
            )
            
            eli5_response = self.model.generate_content(
                eli5_prompt,
                generation_config=GenerationConfig(**PROMPT_DESIGN_CONFIG)
            )
            if not eli5_response or not eli5_response.text:
                return None
            
            return {
                'did_you_know': fact_response.text,
                'eli5': eli5_response.text
            }
            
        except Exception as e:
            logger.error(f"PreAnalysis generation failed: {str(e)}")
            return None

class PromptDesigner:
    """Research framework designer."""
    
    def __init__(self, model: Any):
        self.model = model
    
    @rate_limit_decorator
    def generate_framework(self, topic: str, optimized_prompt: str, focus_areas: Optional[list] = None) -> Optional[str]:
        """Generate research framework using optimized prompt and focus areas."""
        try:
            # Log the configuration being used
            logger.info(f"Using configuration: {FRAMEWORK_CONFIG}")
            
            # Extract key themes from focus areas
            areas_text = ", ".join(focus_areas[:3]) if focus_areas else ""  # Limit to top 3 focus areas
            
            # Create a more focused prompt
            prompt = (
                f"Create a brief research framework for {topic} focusing on: {areas_text}\n\n"
                "Format the response in 4 short sections:\n"
                "1. Key Questions (2-3 bullet points)\n"
                "2. Main Topics (3-4 bullet points)\n"
                "3. Methods (2-3 bullet points)\n"
                "4. Expected Insights (2-3 bullet points)\n\n"
                "Keep each bullet point to one line. Total response should be under 500 words."
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**FRAMEWORK_CONFIG)
            )
            
            if not response or not response.text:
                logger.error("Empty response from model")
                return None
                
            # Clean up the response
            framework = response.text.strip()
            if len(framework) > 2000:  # Add a safety limit
                framework = framework[:2000] + "..."
                
            return framework
            
        except Exception as e:
            logger.error(f"Framework generation failed: {str(e)}")
            return None
    
    @rate_limit_decorator
    def generate_focus_areas(self, topic: str) -> Optional[list]:
        """Generate focus areas for the topic."""
        try:
            prompt = (
                f"Generate 8 key research areas for {topic}. "
                "Format as a simple list with one topic per line. "
                "Do not include numbers, bullets, or any other formatting."
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**PROMPT_DESIGN_CONFIG)
            )
            
            if not response or not response.text:
                logger.error("Empty response from model")
                return None
            
            # Split by newlines and clean up each line
            areas = [
                line.strip()
                for line in response.text.split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'))
            ]
            
            # Take first 8 valid areas
            valid_areas = areas[:8]
            
            if not valid_areas:
                logger.error("No valid focus areas found in response")
                return None
                
            logger.info(f"Generated {len(valid_areas)} focus areas")
            return valid_areas
            
        except Exception as e:
            logger.error(f"Focus areas generation failed: {str(e)}")
            return None
    
    @rate_limit_decorator
    def design_prompt(self, topic: str, focus_areas: Optional[list] = None) -> Optional[str]:
        """Design research prompt."""
        try:
            if focus_areas:
                prompt = f"Create a research outline for {topic} focusing on: {', '.join(focus_areas)}"
            else:
                prompt = f"Create a research outline for {topic}"
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**PROMPT_DESIGN_CONFIG)
            )
            return response.text if response and response.text else None
            
        except Exception as e:
            logger.error(f"Prompt design failed: {str(e)}")
            return None

class ResearchAnalyst:
    """Research analyst."""
    
    def __init__(self, model: Any):
        self.model = model
    
    @rate_limit_decorator
    def analyze(self, topic: str, framework: str, previous: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Analyze a specific aspect of the topic."""
        try:
            if previous:
                prompt = f"Continue the research on {topic}, building on: {previous}"
            else:
                prompt = f"Research {topic} using this framework: {framework}"
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**ANALYSIS_CONFIG)
            )
            if not response or not response.text:
                return None
            
            # Split into title and content
            lines = response.text.split('\n', 1)
            title = lines[0].strip() if len(lines) > 0 else "Research Analysis"
            content = lines[1].strip() if len(lines) > 1 else response.text
            
            return {
                'title': title,
                'content': content
            }
            
        except Exception as e:
            logger.error(f"Research analysis failed: {str(e)}")
            return None

class SynthesisExpert:
    """Research synthesizer."""
    
    def __init__(self, model: Any):
        self.model = model
    
    @rate_limit_decorator
    def synthesize(self, topic: str, research_results: list) -> Optional[str]:
        """Create final synthesis."""
        try:
            # Limit the input size by extracting key points
            summary_points = []
            for result in research_results:
                # Extract first paragraph and any bullet points
                lines = result.split('\n')
                summary = lines[0]  # Always include first line
                bullets = [line for line in lines if line.strip().startswith('•') or line.strip().startswith('-')]
                if bullets:
                    summary += '\n' + '\n'.join(bullets[:3])  # Include up to 3 bullet points
                summary_points.append(summary)
            
            # Create a focused synthesis prompt
            prompt = (
                f"Create a concise synthesis of this research about {topic}. "
                "Format the response in these sections:\n"
                "1. Key Findings (3-4 bullet points)\n"
                "2. Analysis (2-3 paragraphs)\n"
                "3. Conclusion (1 paragraph)\n\n"
                "Research points to synthesize:\n"
                f"{' '.join(summary_points)}\n\n"
                "Keep the response focused and under 1000 words."
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**{
                    **SYNTHESIS_CONFIG,
                    'max_output_tokens': 2048  # Reduce token limit
                })
            )
            
            if not response or not response.text:
                logger.error("Empty response from synthesis")
                return None
            
            # Clean up and format the response
            synthesis = response.text.strip()
            if len(synthesis) > 2000:  # Add a safety limit
                synthesis = synthesis[:2000] + "..."
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Research synthesis failed: {str(e)}")
            return None 