"""Agent implementations for the MARA application."""

import logging
from typing import Any, Dict, Optional

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

from config import (
    PROMPT_DESIGN_CONFIG,
    FRAMEWORK_CONFIG,
    ANALYSIS_CONFIG,
    SYNTHESIS_CONFIG
)
from utils import rate_limit_decorator

logger = logging.getLogger(__name__)

def extract_response_text(response: Any) -> Optional[str]:
    """Extract text from a Gemini model response."""
    try:
        if hasattr(response, "parts"):
            return "".join(part.text for part in response.parts if hasattr(part, "text")).strip()
        elif hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "candidates"):
            for candidate in response.candidates:
                if hasattr(candidate.content, "parts"):
                    return "".join(part.text for part in candidate.content.parts if hasattr(part, "text")).strip()
        return None
    except Exception as e:
        logger.error(f"Failed to extract response text: {str(e)}")
        return None

class PreAnalysisAgent:
    """Quick insights agent."""
    
    def __init__(self, model):
        """Initialize the agent with a model."""
        self.model = model
        
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate quick insights about the topic."""
        try:
            # Generate fun fact
            fact_prompt = (
                f"Generate a single, fascinating, and unexpected fact about {topic}, presented in one sentence with emojis. "
                "The fact needs to be surprising, unique, and counter-intuitive, revealing a lesser-known aspect with "
                "vivid language and potentially statistics. It should challenge common assumptions."
            )
            
            fact_response = self.model.generate_content(fact_prompt)
            fact_text = extract_response_text(fact_response)
            if not fact_text:
                return None
            
            # Generate ELI5
            eli5_prompt = (
                f"Write a very short, engaging explanation of {topic} for a child. Use simple language, "
                "a fun analogy, and emojis to make it memorable. Focus on a key, child-friendly aspect "
                "of the topic. Make it 2-3 sentences maximum."
            )
            
            eli5_response = self.model.generate_content(eli5_prompt)
            eli5_text = extract_response_text(eli5_response)
            if not eli5_text:
                return None
            
            return {
                'did_you_know': fact_text,
                'eli5': eli5_text
            }
            
        except Exception as e:
            logger.error(f"PreAnalysis generation failed: {str(e)}")
            return None

class PromptDesigner:
    """Research framework designer."""
    
    def __init__(self, model):
        """Initialize the agent with a model."""
        self.model = model
        
    def generate_framework(self, topic: str) -> Optional[str]:
        """Generate research framework."""
        try:
            prompt = (
                f"Create a detailed research framework for analyzing {topic}. Include:\n\n"
                "1. Culture (music, food, traditions)\n"
                "2. History (colonialism, key events)\n"
                "3. Economy (tourism, industry)\n"
                "4. Environment (geography, climate)\n"
                "5. Society (demographics, community)\n"
                "6. Politics (governance, issues)\n\n"
                "Structure each section with clear points and supporting details."
            )
            
            response = self.model.generate_content(prompt)
            return extract_response_text(response)
            
        except Exception as e:
            logger.error(f"Framework generation failed: {str(e)}")
            return None

class ResearchAnalyst:
    """Research analyst."""
    
    def __init__(self, model):
        """Initialize the agent with a model."""
        self.model = model
        
    def analyze(self, topic: str, framework: str, aspect: str) -> Optional[Dict[str, str]]:
        """Analyze a specific aspect of the topic."""
        try:
            prompt = f"Research this aspect of {topic}: {aspect}\n\nFramework context:\n{framework}"
            
            response = self.model.generate_content(prompt)
            result = extract_response_text(response)
            if not result:
                return None
            
            # Split into title and content
            lines = result.split('\n', 1)
            title = lines[0].strip() if len(lines) > 0 else "Research Analysis"
            content = lines[1].strip() if len(lines) > 1 else result
            
            return {
                'title': title,
                'content': content
            }
            
        except Exception as e:
            logger.error(f"Research analysis failed: {str(e)}")
            return None

class SynthesisExpert:
    """Research synthesizer."""
    
    def __init__(self, model):
        """Initialize the agent with a model."""
        self.model = model
        
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
            
            response = self.model.generate_content(prompt)
            return extract_response_text(response)
            
        except Exception as e:
            logger.error(f"Research synthesis failed: {str(e)}")
            return None 