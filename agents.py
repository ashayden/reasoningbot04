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

Format your response EXACTLY as a Python dictionary like this:
{{"did_you_know": "Your fact here", "eli5": "Your overview here"}}

Important: Use only straight quotes (") not curly quotes, and ensure the response is a valid Python dictionary."""
        
        try:
            result = self.generate_content(prompt, PREANALYSIS_CONFIG)
            if not result:
                return None
                
            # Clean the response to ensure valid Python syntax
            result = result.strip()
            result = result.replace('"', '"').replace('"', '"')  # Replace curly quotes
            result = result.replace("'", "'").replace("'", "'")  # Replace curly single quotes
            
            # Safely evaluate the string as a Python dictionary
            insights = eval(result)
            
            # Validate the dictionary structure
            if not isinstance(insights, dict):
                logger.error("Response is not a dictionary")
                return None
                
            required_keys = {'did_you_know', 'eli5'}
            if not all(key in insights for key in required_keys):
                logger.error("Response missing required keys")
                return None
                
            return insights
            
        except Exception as e:
            logger.error(f"Error parsing insights response: {str(e)}")
            return None
    
    def generate_focus_areas(self, topic: str) -> Optional[List[str]]:
        """Generate potential focus areas for research."""
        prompt = f"""For '{topic}', suggest 8-10 diverse research focus areas that:
1. Cover different aspects and perspectives
2. Include both obvious and non-obvious angles
3. Span theoretical and practical implications

Format your response EXACTLY as a Python list of strings, one per line, like this:
[
    "First focus area",
    "Second focus area",
    "Third focus area"
]

Important: 
- Use only straight quotes (")
- Each focus area should be on its own line
- Do not include any explanations or additional text
- Keep each focus area concise (3-7 words)"""
        
        try:
            result = self.generate_content(prompt, PREANALYSIS_CONFIG)
            if not result:
                return None
                
            # Clean the response to ensure valid Python syntax
            result = result.strip()
            
            # Try parsing as a direct Python list first
            try:
                result = result.replace('"', '"').replace('"', '"')  # Replace curly quotes
                result = result.replace("'", "'").replace("'", "'")  # Replace curly single quotes
                focus_areas = eval(result)
            except:
                # Fallback: try to parse line by line
                logger.info("Falling back to line-by-line parsing")
                lines = [line.strip() for line in result.split('\n')]
                focus_areas = []
                for line in lines:
                    # Remove common prefixes and clean up
                    line = line.strip('[]," \t-•*')
                    if line and not line.startswith(('Format', 'Important', 'Use', 'Each', 'Do not')):
                        focus_areas.append(line)
            
            # Validate the list structure
            if not isinstance(focus_areas, list):
                logger.error("Response is not a list")
                return None
            
            # Clean up and validate each focus area
            cleaned_areas = []
            for area in focus_areas:
                if isinstance(area, str) and area.strip():
                    # Remove any remaining quotes and clean up whitespace
                    cleaned = area.strip().strip('"\'').strip()
                    if cleaned:
                        cleaned_areas.append(cleaned)
            
            # Validate final list
            if not (8 <= len(cleaned_areas) <= 10):
                logger.error(f"Invalid number of focus areas: {len(cleaned_areas)}")
                return None
            
            return cleaned_areas
            
        except Exception as e:
            logger.error(f"Error parsing focus areas response: {str(e)}")
            return None

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

Format your response EXACTLY as a Python dictionary like this:
{{"title": "Your title here", "subtitle": "Your subtitle here", "content": "Your detailed analysis here"}}

Important: Use only straight quotes (") not curly quotes, and ensure the response is a valid Python dictionary."""
        
        try:
            # Adjust temperature based on iteration
            temp = min(ANALYSIS_BASE_TEMP + (len(context) > 0) * ANALYSIS_TEMP_INCREMENT, ANALYSIS_MAX_TEMP)
            config = {**ANALYSIS_CONFIG, 'temperature': temp}
            
            result = self.generate_content(prompt, config)
            if not result:
                return None
                
            # Clean the response to ensure valid Python syntax
            result = result.strip()
            result = result.replace('"', '"').replace('"', '"')  # Replace curly quotes
            result = result.replace("'", "'").replace("'", "'")  # Replace curly single quotes
            
            # Safely evaluate the string as a Python dictionary
            analysis = eval(result)
            
            # Validate the dictionary structure
            if not isinstance(analysis, dict):
                logger.error("Response is not a dictionary")
                return None
                
            required_keys = {'title', 'subtitle', 'content'}
            if not all(key in analysis for key in required_keys):
                logger.error("Response missing required keys")
                return None
                
            # Ensure all values are strings
            if not all(isinstance(analysis[key], str) for key in required_keys):
                logger.error("Not all values are strings")
                return None
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return None

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