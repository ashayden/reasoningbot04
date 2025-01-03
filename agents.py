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
from utils import handle_error

logger = logging.getLogger(__name__)

class PreAnalysisAgent:
    """Quick insights agent."""
    
    def __init__(self, model: Any):
        self.model = model
        self._cached_insights = {}  # Cache for insights to prevent regeneration
    
    @handle_error
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate quick insights about the topic."""
        # Check cache first
        cache_key = topic.lower().strip()
        if cache_key in self._cached_insights:
            logger.info(f"Using cached insights for topic: {topic}")
            return self._cached_insights[cache_key]
        
        logger.info(f"Generating new insights for topic: {topic}")
        
        # Generate fun fact
        fact_prompt = (
            f"Share one fascinating and unexpected fact about {topic} in a single sentence. "
            "Include relevant emojis. Focus on a surprising or counter-intuitive aspect."
        )
        
        fact_response = self.model.generate_content(
            fact_prompt,
            generation_config=GenerationConfig(**PROMPT_DESIGN_CONFIG)
        )
        if not fact_response or not fact_response.text:
            logger.error("Failed to generate fun fact")
            return None
        
        # Generate ELI5
        eli5_prompt = (
            f"Explain {topic} in 2-3 simple sentences for a general audience. "
            "Use basic language and add relevant emojis."
        )
        
        eli5_response = self.model.generate_content(
            eli5_prompt,
            generation_config=GenerationConfig(**PROMPT_DESIGN_CONFIG)
        )
        if not eli5_response or not eli5_response.text:
            logger.error("Failed to generate ELI5 explanation")
            return None
        
        insights = {
            'did_you_know': fact_response.text.strip(),
            'eli5': eli5_response.text.strip()
        }
        
        # Cache the results
        self._cached_insights[cache_key] = insights
        logger.info(f"Successfully generated and cached insights for: {topic}")
        return insights

class PromptDesigner:
    """Research framework designer."""
    
    def __init__(self, model: Any):
        self.model = model
        self._cached_framework = None  # Cache for framework to prevent regeneration
    
    @handle_error
    def generate_focus_areas(self, topic: str) -> Optional[list]:
        """Generate focus areas for the topic."""
        prompt = (
            f"List 8 key research areas for {topic}. "
            "Return only the area names, one per line. "
            "No additional formatting, comments, or descriptions."
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(**{
                **PROMPT_DESIGN_CONFIG,
                'temperature': 0.1  # Lower temperature for more focused output
            })
        )
        
        if not response or not response.text:
            logger.error("Empty response from model")
            return None
        
        # Clean up the response
        text = response.text.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text[text.find('\n')+1:text.rfind('\n')]
        
        # Split by newlines and clean up each line
        areas = []
        for line in text.split('\n'):
            line = line.strip().strip('[],"\'')
            if '#' in line:
                line = line[:line.find('#')].strip()
            line = line.strip('"\'[] ,')
            if line and not line.startswith(('#', '-', '*', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')):
                areas.append(line)
        
        # Take first 8 valid areas
        valid_areas = areas[:8]
        
        if not valid_areas:
            logger.error("No valid focus areas found in response")
            return None
            
        logger.info(f"Generated {len(valid_areas)} focus areas")
        return valid_areas
    
    @handle_error
    def generate_framework(self, topic: str, optimized_prompt: str, focus_areas: Optional[list] = None) -> Optional[str]:
        """Generate research framework using optimized prompt and focus areas."""
        # Return cached framework if available
        if self._cached_framework:
            logger.info("Using cached framework")
            return self._cached_framework
        
        # Extract key themes from focus areas
        areas_text = ", ".join(focus_areas[:3]) if focus_areas else ""  # Limit to top 3 focus areas
        
        # Create a detailed prompt
        prompt = (
            f"Create a comprehensive research framework for {topic} focusing on: {areas_text}\n\n"
            "Format the response in 4 sections:\n"
            "1. Key Questions (2-3 bullet points)\n"
            "2. Main Topics (3-4 bullet points)\n"
            "3. Methods (2-3 bullet points)\n"
            "4. Expected Insights (2-3 bullet points)\n\n"
            "Keep each bullet point detailed but focused."
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
        
        # Cache the framework
        self._cached_framework = framework
        return framework
    
    @handle_error
    def design_prompt(self, topic: str, focus_areas: Optional[list] = None) -> Optional[str]:
        """Design research prompt."""
        # Create a structured prompt
        if focus_areas:
            prompt = (
                f"Create a focused research framework for analyzing {topic}, "
                f"specifically examining: {', '.join(focus_areas[:3])}.\n\n"
                "Structure the response in these sections:\n"
                "1. Research Questions (2-3 clear, focused questions)\n"
                "2. Key Areas to Investigate (3-4 main topics)\n"
                "3. Methodology (2-3 specific research methods)\n"
                "4. Expected Outcomes (2-3 anticipated findings)\n\n"
                "Keep each section concise but informative."
            )
        else:
            prompt = (
                f"Create a focused research framework for analyzing {topic}.\n\n"
                "Structure the response in these sections:\n"
                "1. Research Questions (2-3 clear, focused questions)\n"
                "2. Key Areas to Investigate (3-4 main topics)\n"
                "3. Methodology (2-3 specific research methods)\n"
                "4. Expected Outcomes (2-3 anticipated findings)\n\n"
                "Keep each section concise but informative."
            )
        
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(**{
                **PROMPT_DESIGN_CONFIG,
                'temperature': 0.1  # Lower temperature for more focused output
            })
        )
        
        if not response or not response.text:
            logger.error("Empty response from model")
            return None
        
        return response.text.strip()

class ResearchAnalyst:
    """Research analyst."""
    
    def __init__(self, model: Any):
        self.model = model
    
    @handle_error
    def analyze(self, topic: str, framework: str, previous: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Analyze a specific aspect of the topic."""
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

class SynthesisExpert:
    """Research synthesizer."""
    
    def __init__(self, model: Any):
        self.model = model
    
    @handle_error
    def synthesize(self, topic: str, research_results: list) -> Optional[str]:
        """Create final synthesis."""
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
            "Keep the response focused."
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(**SYNTHESIS_CONFIG)
        )
        
        if not response or not response.text:
            logger.error("Empty response from synthesis")
            return None
        
        return response.text.strip() 