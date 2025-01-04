"""Agent implementations for the MARA application."""

import logging
from typing import Dict, Any, Optional, List

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

from config import (
    PROMPT_DESIGN_CONFIG,
    FRAMEWORK_CONFIG,
    ANALYSIS_CONFIG,
    SYNTHESIS_CONFIG,
    PREANALYSIS_CONFIG,
    ANALYSIS_BASE_TEMP,
    ANALYSIS_TEMP_INCREMENT,
    ANALYSIS_MAX_TEMP
)
from utils import rate_limit_decorator

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, model):
        self.model = model
    
    @rate_limit_decorator
    def generate_content(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Generate content with error handling and rate limiting."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**config)
            )
            return response.text if response else None
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return None

class PreAnalysisAgent(BaseAgent):
    """Agent responsible for initial analysis and insights."""
    
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate initial insights about the topic."""
        prompt = f"""Analyze '{topic}' and provide two insights:

1. Did You Know: Share one fascinating, lesser-known fact about the topic. Keep it to a single clear sentence.
2. Overview: Provide a clear, accessible 2-3 sentence explanation for a general audience. Focus on key points and avoid technical jargon.

Format your response as a Python dictionary with 'did_you_know' and 'eli5' as keys. Example:
{{'did_you_know': 'The fact...', 'eli5': 'The overview...'}}"""
        
        result = self.generate_content(prompt, PREANALYSIS_CONFIG)
        return eval(result) if result else None
    
    def generate_focus_areas(self, topic: str) -> Optional[List[str]]:
        """Generate potential focus areas for research."""
        prompt = f"""For '{topic}', suggest 8-10 diverse research focus areas that:
1. Cover different aspects and perspectives
2. Include both obvious and non-obvious angles
3. Span theoretical and practical implications

Return as a Python list of strings."""
        
        result = self.generate_content(prompt, PREANALYSIS_CONFIG)
        return eval(result) if result else None

class PromptDesigner(BaseAgent):
    """Agent responsible for optimizing research prompts."""
    
    def design_prompt(self, topic: str) -> Optional[str]:
        """Design an optimized research prompt."""
        prompt = f"""Create a research prompt for '{topic}' that:
1. Encourages comprehensive analysis
2. Highlights key areas of investigation
3. Suggests innovative approaches
4. Maintains academic rigor

Return the prompt as a clear, focused paragraph."""
        
        return self.generate_content(prompt, PROMPT_DESIGN_CONFIG)

class FrameworkEngineer(BaseAgent):
    """Agent responsible for creating analysis frameworks."""
    
    def create_framework(self, initial_prompt: str, enhanced_prompt: Optional[str] = None) -> Optional[str]:
        """Create a research framework based on the prompt design."""
        prompt_context = enhanced_prompt or initial_prompt
        
        prompt = f"""Based on: {prompt_context}

Create a research framework that:

1. Research Objectives
   - Primary and secondary questions
   - Expected outcomes
   - Key hypotheses

2. Methodology
   - Research methods
   - Data collection
   - Analysis techniques

3. Investigation Areas
   - Core topics
   - Subtopics
   - Cross-cutting themes

4. Theoretical Framework
   - Key concepts
   - Relationships
   - Integration points

5. Critical Perspectives
   - Assumptions
   - Limitations
   - Alternative views

Format as a clear, structured markdown document."""
        
        return self.generate_content(prompt, FRAMEWORK_CONFIG)

class ResearchAnalyst(BaseAgent):
    """Agent responsible for conducting deep analysis."""
    
    def analyze(self, topic: str, framework: str, previous_analysis: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Conduct analysis based on framework and previous findings."""
        context = f"Previous Analysis:\n{previous_analysis}\n\n" if previous_analysis else ""
        
        prompt = f"""Topic: {topic}
Framework: {framework}
{context}
Conduct a deep analysis that:
1. Builds on previous findings (if any)
2. Reveals new insights and patterns
3. Challenges assumptions
4. Integrates multiple perspectives

Format as a dictionary with 'title', 'subtitle', and 'content' keys."""
        
        # Adjust temperature based on iteration
        temp = min(ANALYSIS_BASE_TEMP + (len(context) > 0) * ANALYSIS_TEMP_INCREMENT, ANALYSIS_MAX_TEMP)
        config = {**ANALYSIS_CONFIG, 'temperature': temp}
        
        result = self.generate_content(prompt, config)
        return eval(result) if result else None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing findings."""
    
    def synthesize(self, topic: str, analysis_results: List[str]) -> Optional[str]:
        """Synthesize all analysis results into a final report."""
        prompt = f"""Topic: {topic}
Analysis Results: {analysis_results}

Create a comprehensive research report that:
1. Synthesizes all findings
2. Highlights key insights
3. Draws meaningful conclusions
4. Suggests future directions

Format as a structured markdown document with:
- Executive Summary
- Key Findings
- Analysis
- Conclusions
- References"""
        
        return self.generate_content(prompt, SYNTHESIS_CONFIG) 