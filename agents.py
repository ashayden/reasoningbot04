"""Agent implementations for the MARA application."""

import logging
from typing import Dict, Any, Optional

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

from config import (
    PROMPT_DESIGN_CONFIG,
    FRAMEWORK_CONFIG,
    ANALYSIS_CONFIG,
    SYNTHESIS_CONFIG,
    REQUEST_TIMEOUT
)
from utils import rate_limit_decorator, parse_title_content

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, model: Any):
        self.model = model
    
    @rate_limit_decorator
    def generate_content(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Generate content with rate limiting and error handling."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**config),
                timeout=REQUEST_TIMEOUT
            )
            return response.text
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            st.error(f"Error generating content: {str(e)}")
            return None

class PromptDesigner(BaseAgent):
    """Agent responsible for designing optimal prompts."""
    
    def design_prompt(self, topic: str) -> Optional[str]:
        """Design an optimal prompt for the given topic."""
        prompt = f"""As an expert prompt engineer, create a concise one-paragraph prompt that will guide the development 
        of a research framework for analyzing '{topic}'. Focus on the essential aspects that need to be 
        investigated while maintaining analytical rigor and academic standards."""
        
        return self.generate_content(prompt, PROMPT_DESIGN_CONFIG)

class FrameworkEngineer(BaseAgent):
    """Agent responsible for creating analysis frameworks."""
    
    def create_framework(self, prompt_design: str) -> Optional[str]:
        """Create a research framework based on the prompt design."""
        prompt = f"""{prompt_design}

        Based on this prompt, create a detailed research framework that:
        1. Outlines the key areas of investigation
        2. Specifies methodological approaches
        3. Defines evaluation criteria
        4. Sets clear milestones for the analysis process"""
        
        return self.generate_content(prompt, FRAMEWORK_CONFIG)

class ResearchAnalyst(BaseAgent):
    """Agent responsible for conducting research analysis."""
    
    def analyze(self, topic: str, framework: str, previous_analysis: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Conduct research analysis."""
        if previous_analysis is None:
            prompt = f"""Acting as a leading expert in topic-related field: Based on the framework above, conduct an initial research analysis of '{topic}'. 
            Follow the methodological approaches and evaluation criteria specified in the framework.
            Provide detailed findings for each key area of investigation outlined.
            
            Start your response with a title in this exact format (including the newlines):
            Title: Your Main Title Here
            Subtitle: Your Descriptive Subtitle Here

            Then continue with your analysis content."""
        else:
            prompt = f"""Review the previous research iteration:
            {previous_analysis}
            
            Based on this previous analysis and the original framework, expand and deepen the research by:
            1. Identifying gaps or areas needing more depth
            2. Exploring new connections and implications
            3. Refining and strengthening key arguments
            4. Adding new supporting evidence or perspectives
            
            Start your response with a title in this exact format (including the newlines):
            Title: Your Main Title Here
            Subtitle: Your Descriptive Subtitle Here

            Then continue with your analysis content."""
        
        result = self.generate_content(prompt, ANALYSIS_CONFIG)
        if result:
            return parse_title_content(result)
        return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing research findings."""
    
    def synthesize(self, topic: str, analyses: list) -> Optional[str]:
        """Synthesize all research analyses into a final report."""
        prompt = f"""Synthesize all research from agent 2 on '{topic}' into a Final Report with:
        1. Executive Summary (2-3 paragraphs)
        2. Key Insights (bullet points)
        3. Analysis
        4. Conclusion
        5. Further Considerations & Counter-Arguments (where applicable)
        6. Recommended Readings and Resources
        
        Analysis to synthesize: {' '.join(analyses)}"""
        
        return self.generate_content(prompt, SYNTHESIS_CONFIG) 