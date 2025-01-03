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
                f"Write a very short, engaging explanation of {topic} for a child. Use simple language, "
                "a fun analogy, and emojis to make it memorable. Focus on a key, child-friendly aspect "
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
            # Combine focus areas and base framework
            areas_text = ", ".join(focus_areas) if focus_areas else ""
            
            prompt = (
                f"Using this research outline:\n{optimized_prompt}\n\n"
                f"And focusing on these aspects: {areas_text}\n\n"
                f"Create a detailed research framework for analyzing {topic}. Include:\n\n"
                "1. Key Research Questions\n"
                "2. Primary Areas of Investigation\n"
                "3. Methodological Approach\n"
                "4. Expected Outcomes\n\n"
                "Structure each section with clear points and supporting details."
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**FRAMEWORK_CONFIG)
            )
            return response.text if response and response.text else None
            
        except Exception as e:
            logger.error(f"Framework generation failed: {str(e)}")
            return None
    
    @rate_limit_decorator
    def generate_focus_areas(self, topic: str) -> Optional[list]:
        """Generate focus areas for the topic."""
        try:
            prompt = f"List 8 key aspects of {topic} that should be researched. Format as bullet points."
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**PROMPT_DESIGN_CONFIG)
            )
            
            if not response or not response.text:
                return None
            
            # Extract bullet points
            lines = [line.strip().lstrip('-*•').strip() for line in response.text.split('\n')]
            return [line for line in lines if line][:8]
            
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
            prompt = (
                f"Synthesize this research about {topic}:\n\n"
                f"{' '.join(research_results)}\n\n"
                "Create a comprehensive report with:\n"
                "1. Executive Summary (2-3 paragraphs)\n"
                "2. Key Findings (bullet points)\n"
                "3. Analysis (detailed discussion)\n"
                "4. Conclusion (clear takeaways)"
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**SYNTHESIS_CONFIG)
            )
            return response.text if response and response.text else None
            
        except Exception as e:
            logger.error(f"Research synthesis failed: {str(e)}")
            return None 