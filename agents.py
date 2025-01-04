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
        focus_context = f"\nSelected Focus Areas:\n{', '.join(focus_areas) if focus_areas else []}"
        
        base_prompt = f"""Topic: {topic}{focus_context}
{context}
You are an expert academic researcher conducting an in-depth analysis. Your goal is to provide specific, evidence-based insights that build upon each other.

Format your analysis using the following structure:

1. Main Sections (use clear headings)
- Each section should focus on a key theme or finding
- Use descriptive headings that indicate the content
- Maintain consistent heading style

2. Evidence and Examples
- Each point should be supported by specific evidence
- Use bullet points for clarity
- Start each bullet with a clear topic statement in bold
- Follow with supporting evidence and explanation

3. Connections and Significance
- After each section, explain connections to other findings
- Evaluate the significance of observations
- Draw clear conclusions

4. Final Section: Implications
- Conclude with clear implications
- Provide actionable recommendations
- Consider future impacts"""

        if not previous_analysis:
            # First research loop - Initial comprehensive analysis
            prompt = base_prompt + """

Focus your analysis on:
1. Makes specific, concrete observations about the topic
2. Provides clear evidence to support each observation
3. Identifies key patterns and relationships
4. Establishes clear connections between different aspects
5. Evaluates the significance of your findings"""
        else:
            # Subsequent research loops - Deeper analysis
            prompt = base_prompt + """

Focus your deeper analysis on:
1. Uncovers more nuanced connections
2. Challenges or validates previous observations with new evidence
3. Identifies emerging patterns and trends
4. Explores complex relationships between factors
5. Evaluates long-term implications
6. Proposes new interpretative frameworks
7. Synthesizes insights into novel perspectives"""

        prompt += """

Format your response EXACTLY as shown below:
{
    "title": "Research Analysis",
    "subtitle": "Key Findings and Evidence",
    "content": "Your detailed analysis here"
}

Important Formatting Rules:
1. Use clear section headings with proper spacing
2. Format bullet points consistently:
   • Start with a bold statement: **Key Point:** followed by evidence
   • Use proper bullet points (•)
   • Maintain consistent indentation
3. Use markdown formatting:
   - Bold for emphasis (**text**)
   - Italics for subtitles (*text*)
   - Clear section breaks (use blank lines)
4. Maintain consistent structure:
   - Main sections with headings
   - Bulleted evidence points
   - Connection/significance paragraphs
   - Clear implications section
5. Ensure readability:
   - Use clear topic sentences
   - Provide specific evidence
   - Create logical flow
   - Maintain professional tone"""
        
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
            try:
                analysis = eval(result)
            except:
                # Try parsing with modified string if eval fails
                result = result.replace('"{', '{').replace('}"', '}')  # Remove extra quotes around dict
                analysis = eval(result)
            
            # Validate the dictionary structure
            if not isinstance(analysis, dict):
                logger.error("Response is not a dictionary")
                return None
                
            required_keys = {'title', 'subtitle', 'content'}
            if not all(key in analysis for key in required_keys):
                logger.error("Response missing required keys")
                return None
                
            # Clean up values and ensure they're strings
            cleaned_analysis = {}
            for key in required_keys:
                if key in analysis:
                    value = analysis[key]
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    cleaned_analysis[key] = str(value).strip().strip('"\'').strip()
            
            return cleaned_analysis
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing findings into a thesis-driven report."""
    
    def synthesize(self, topic: str, analysis_results: List[str]) -> Optional[str]:
        """Synthesize analysis results into a thesis-driven report."""
        prompt = f"""Topic: {topic}
Analysis Results: {analysis_results}

Create a compelling, narrative-driven synthesis that weaves together the research findings into a cohesive argument. Your synthesis should be both academically rigorous and engaging to read.

Structure your synthesis as follows:

1. Executive Summary (2-3 paragraphs)
- Open with an engaging hook that captures the significance of the topic
- Present your main thesis clearly and compellingly
- Preview the key themes and findings
- Highlight the broader implications

2. Key Findings and Analysis (Multiple sections with clear headings)
- Present each major finding with supporting evidence
- Use clear topic sentences and transitions
- Build connections between different findings
- Evaluate the significance of each point
- Use specific examples and evidence
- Create a clear narrative progression

3. Synthesis of Insights
- Weave together the major themes
- Identify patterns and relationships
- Draw meaningful connections
- Present novel interpretations
- Evaluate broader implications

4. Implications and Future Directions
- Discuss practical applications
- Identify areas for further research
- Consider long-term impacts
- Address potential challenges
- Propose next steps

Format Guidelines:
- Use clear section headings (H2 with ##)
- Include subsections where appropriate (H3 with ###)
- Use bullet points sparingly and purposefully
- Create clear paragraph breaks
- Use markdown formatting for emphasis
- Maintain a narrative flow throughout
- Write in an engaging, professional style
- Balance academic rigor with readability

Remember to:
- Make explicit connections between ideas
- Support claims with specific evidence
- Evaluate significance of findings
- Consider multiple perspectives
- Address potential counterarguments
- Maintain focus on the main thesis
"""
        
        return self.generate_content(prompt, SYNTHESIS_CONFIG) 