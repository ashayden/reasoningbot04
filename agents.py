"""Agent implementations for the MARA application."""

import logging
from typing import Dict, Any, Optional, List

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

from config import (
    PREANALYSIS_CONFIG,
    ANALYSIS_CONFIG,
    SYNTHESIS_CONFIG,
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

Format your response EXACTLY as a Python dictionary with these two keys:
{{"did_you_know": "Your fact here", "eli5": "Your overview here"}}

Important:
- Use only straight quotes (")
- No line breaks in the dictionary
- Keep the exact keys: did_you_know, eli5
- Ensure proper dictionary formatting
- Avoid nested quotes or special characters"""
        
        try:
            result = self.generate_content(prompt, PREANALYSIS_CONFIG)
            if not result:
                return None
                
            # Clean and parse the response
            result = result.strip()
            result = result.replace('"', '"').replace('"', '"')  # Replace curly quotes
            result = result.replace("'", "'").replace("'", "'")  # Replace curly single quotes
            result = result.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
            
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
                
            # Clean up values
            for key in required_keys:
                if key in insights:
                    insights[key] = insights[key].strip().strip('"\'').strip()
            
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

Format your response as a Python list of strings, one per line:
[
    "First focus area",
    "Second focus area",
    "Third focus area"
]

Important:
- Use only straight quotes (")
- Each focus area should be concise (3-7 words)
- Make each area distinct and specific
- Ensure areas are relevant to the topic"""
        
        try:
            result = self.generate_content(prompt, PREANALYSIS_CONFIG)
            if not result:
                return None
                
            # Clean and parse the response
            result = result.strip()
            result = result.replace('"', '"').replace('"', '"')
            result = result.replace("'", "'").replace("'", "'")
            
            focus_areas = eval(result)
            
            if not isinstance(focus_areas, list) or not (8 <= len(focus_areas) <= 10):
                logger.error("Invalid response format")
                return None
                
            return [area.strip().strip('"\'').strip() for area in focus_areas if area.strip()]
            
        except Exception as e:
            logger.error(f"Error parsing focus areas response: {str(e)}")
            return None

class ResearchAnalyst(BaseAgent):
    """Agent responsible for conducting iterative research analysis."""
    
    def analyze(self, topic: str, focus_areas: Optional[List[str]], previous_analysis: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Conduct analysis based on topic, focus areas, and previous findings."""
        context = f"Previous Analysis:\n{previous_analysis}\n\n" if previous_analysis else ""
        focus_context = f"\nSelected Focus Areas:\n{', '.join(focus_areas)}" if focus_areas else ""
        
        base_prompt = f"""Topic: {topic}{focus_context}
{context}
You are an expert academic researcher conducting an in-depth analysis."""

        if not previous_analysis:
            # First research loop - Initial comprehensive analysis
            prompt = base_prompt + """
Conduct a thorough initial analysis that:
1. Examines the core aspects of the topic
2. Identifies key patterns and relationships
3. Explores fundamental concepts
4. Establishes a strong analytical foundation
5. Considers multiple perspectives"""
        else:
            # Subsequent research loops - Deeper analysis
            prompt = base_prompt + """
Building upon previous findings, conduct a deeper analysis that:
1. Uncovers hidden connections and patterns
2. Explores nuanced relationships
3. Challenges assumptions
4. Proposes creative interpretations
5. Synthesizes insights into novel perspectives
6. Examines unconventional angles
7. Identifies emerging implications"""

        prompt += """

Format your response EXACTLY as a Python dictionary with these keys:
{{"title": "Research Analysis", "subtitle": "Key Findings", "content": "Your analysis here"}}

Important:
- Use only straight quotes (")
- No line breaks in the dictionary
- Keep the exact keys: title, subtitle, content
- Ensure proper dictionary formatting
- Avoid nested quotes or special characters"""
        
        try:
            # Adjust temperature based on iteration
            temp = min(ANALYSIS_BASE_TEMP + (len(context) > 0) * ANALYSIS_TEMP_INCREMENT, ANALYSIS_MAX_TEMP)
            config = {**ANALYSIS_CONFIG, 'temperature': temp}
            
            result = self.generate_content(prompt, config)
            if not result:
                return None
                
            # Clean and parse the response
            result = result.strip()
            result = result.replace('"', '"').replace('"', '"')  # Replace curly quotes
            result = result.replace("'", "'").replace("'", "'")  # Replace curly single quotes
            result = result.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
            
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
                
            # Clean up values
            for key in required_keys:
                if key in analysis:
                    analysis[key] = analysis[key].strip().strip('"\'').strip()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing findings into a thesis-driven report."""
    
    def synthesize(self, topic: str, analysis_results: List[str]) -> Optional[str]:
        """Synthesize analysis results into a thesis-driven report."""
        prompt = f"""Topic: {topic}
Analysis Results: {analysis_results}

Create a comprehensive thesis-driven synthesis that includes:

1. Thesis Statement
- Clear, arguable main claim about the topic
- Based on research findings
- Specific and focused

2. Executive Summary
- Overview of key findings
- Major patterns and themes
- Significance of conclusions

3. Evidence Analysis
- Detailed examination of supporting evidence
- Connections between findings
- Strength of evidence
- Counter-arguments addressed

4. Argumentation
- Logical flow of ideas
- Clear reasoning
- Evidence-based claims
- Alternative viewpoints considered

5. Implications
- Theoretical significance
- Practical applications
- Future directions
- Broader impact

Format as a structured markdown document with clear sections and subsections.
Ensure strong connections between evidence and thesis.
Address potential counterarguments.
Maintain academic rigor while being accessible.
"""
        
        return self.generate_content(prompt, SYNTHESIS_CONFIG) 