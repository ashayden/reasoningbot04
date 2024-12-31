"""Agent implementations for the MARA application."""

import logging
from typing import Dict, Any, Optional, Tuple

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

from config import (
    PROMPT_DESIGN_CONFIG,
    FRAMEWORK_CONFIG,
    ANALYSIS_CONFIG,
    SYNTHESIS_CONFIG
)
from utils import rate_limit_decorator, parse_title_content
from templates import (
    CITATION_REQUIREMENTS,
    REFERENCES_SECTION,
    FRAMEWORK_TEMPLATE,
    INITIAL_ANALYSIS_STRUCTURE,
    ITERATION_ANALYSIS_STRUCTURE,
    SYNTHESIS_STRUCTURE
)

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, model: Any):
        self.model = model
        self._last_thoughts = None
    
    @property
    def last_thoughts(self) -> Optional[str]:
        """Get the last model's thoughts for debugging or chaining."""
        return self._last_thoughts
    
    @rate_limit_decorator
    def generate_content(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Generate content with rate limiting and error handling."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**config)
            )
            
            if not response or not response.text:
                logger.error("Empty response from model")
                return None
                
            logger.info(f"Raw response length: {len(response.text)}")
            
            # Split response into thoughts and actual content
            parts = response.text.split("\n\n", 1)
            
            if len(parts) > 1 and "Thoughts" in parts[0]:
                self._last_thoughts = parts[0]
                content = parts[1].strip()
                logger.info(f"Extracted content length: {len(content)}")
                return content
            else:
                logger.info("No thoughts section found, returning full response")
                return response.text.strip()
                
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            st.error(f"Error generating content: {str(e)}")
            return None

class FrameworkEngineer(BaseAgent):
    """Agent responsible for creating analysis frameworks."""
    
    def create_framework(self, prompt_design: str) -> Optional[str]:
        """Create a research framework based on the prompt design."""
        prompt = f"""Based on this prompt design:
        {prompt_design}

        Create a comprehensive research framework following this structure:
        {FRAMEWORK_TEMPLATE}

        For each section and subsection, provide detailed and specific content relevant to the topic.
        Ensure each point is thoroughly explained and contextually appropriate.
        Use clear, academic language while maintaining accessibility.
        
        Previous thought process (if available):
        {self._last_thoughts if self._last_thoughts else 'Not available'}"""
        
        return self.generate_content(prompt, FRAMEWORK_CONFIG)

class ResearchAnalyst(BaseAgent):
    """Agent responsible for conducting research analysis."""
    
    def __init__(self, model: Any):
        super().__init__(model)
        self.iteration_count = 0
        
    def _get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration with dynamic temperature scaling."""
        from config import (
            ANALYSIS_CONFIG, 
            ANALYSIS_BASE_TEMP, 
            ANALYSIS_TEMP_INCREMENT,
            ANALYSIS_MAX_TEMP
        )
        
        config = ANALYSIS_CONFIG.copy()
        temp = min(
            ANALYSIS_BASE_TEMP + (self.iteration_count * ANALYSIS_TEMP_INCREMENT),
            ANALYSIS_MAX_TEMP
        )
        config["temperature"] = temp
        return config
    
    def analyze(self, topic: str, framework: str, previous_analysis: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Conduct research analysis."""
        if previous_analysis is None:
            self.iteration_count = 0  # Reset counter for new analysis
            prompt = f"""Acting as a leading expert in topic-related field: Based on the framework above, conduct an initial research analysis of '{topic}'. 
            Follow the methodological approaches and evaluation criteria specified in the framework.
            
            Framework context:
            {framework}
            
            Structure your analysis using this format:

            Start with:
            Title: [Descriptive title reflecting the main focus]
            Subtitle: [Specific aspect or approach being analyzed]

            Then provide a comprehensive analysis following this structure:
            {INITIAL_ANALYSIS_STRUCTURE}

            {CITATION_REQUIREMENTS}
            {REFERENCES_SECTION}

            Ensure each section is thoroughly developed with specific examples and evidence."""
        else:
            self.iteration_count += 1
            previous_context = f"""Previous analysis context:
            {previous_analysis}
            
            Previous agent's thought process:
            {self._last_thoughts if self._last_thoughts else 'Not available'}"""
            
            prompt = f"""Review the previous research iteration and expand the analysis.
            
            {previous_context}
            
            For this iteration #{self.iteration_count + 1}, focus on:
            1. Identifying gaps or areas needing more depth
            2. Exploring new connections and implications
            3. Refining and strengthening key arguments
            4. Adding new supporting evidence or perspectives
            
            Structure your analysis using this format:

            Start with:
            Title: [Descriptive title reflecting the new focus]
            Subtitle: [Specific aspect being expanded upon]

            Then provide:
            {ITERATION_ANALYSIS_STRUCTURE}

            {CITATION_REQUIREMENTS}
            {REFERENCES_SECTION}

            Note: As this is iteration {self.iteration_count + 1}, be more explorative and creative 
            while maintaining academic rigor. Push the boundaries of conventional analysis while 
            ensuring all claims are well-supported."""
        
        result = self.generate_content(prompt, self._get_analysis_config())
        if result:
            return parse_title_content(result)
        return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing research findings."""
    
    def synthesize(self, topic: str, analyses: list) -> Optional[str]:
        """Synthesize all research analyses into a final report."""
        prompt = f"""Synthesize all research from previous analyses on '{topic}' into a Final Report.
        
        {SYNTHESIS_STRUCTURE}

        {CITATION_REQUIREMENTS}
        
        Analysis to synthesize: {' '.join(analyses)}"""
        
        return self.generate_content(prompt, SYNTHESIS_CONFIG) 