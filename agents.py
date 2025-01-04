"""Agent implementations for the MARA application."""

import logging
from typing import Dict, Any, Optional, List
import time

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
    
    def _generate_with_backoff(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate content with exponential backoff for retries."""
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                retry_count += 1
            except Exception as e:
                logger.error(f"Generation error (attempt {retry_count + 1}): {str(e)}")
                retry_count += 1
                time.sleep(2 ** retry_count)  # Exponential backoff
        return None

    def generate_content(self, prompt: str, config: Optional[Dict] = None) -> Optional[str]:
        """Generate content with the specified configuration."""
        try:
            if config:
                self.model.generation_config = genai.types.GenerationConfig(**config)
            response = self._generate_with_backoff(prompt)
            if response:
                # Clean up the response
                response = response.replace('\\"', '"')  # Fix escaped quotes
                response = response.replace('\\n', '\n')  # Fix escaped newlines
                response = response.strip()
                return response
            return None
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
    
    def analyze(self, topic: str, focus_areas: List[str], previous_analysis: Optional[str] = None) -> Dict[str, str]:
        """Generate research analysis for the given topic and focus areas."""
        try:
            prompt = f'''Analyze the topic "{topic}" focusing on recent developments and key insights.
            
Previous analysis (if any): {previous_analysis if previous_analysis else "None"}
Focus areas: {", ".join(focus_areas) if focus_areas else "General analysis"}

Important notes:
1. Create a unique, specific title that captures the essence of your analysis
2. Write a subtitle that previews your key findings
3. Structure your analysis with clear sections and bullet points
4. Use markdown formatting for headings and emphasis
5. Return your response in this exact format:
{{
    "title": "Your Unique Title Here",
    "subtitle": "Your Subtitle Here",
    "content": "Your Analysis Content Here"
}}

Remember:
- Make titles specific and informative
- Use bullet points for key findings
- Include evidence and examples
- Build on previous analysis if provided
- Focus on selected areas if specified'''

            response = self._generate_with_backoff(prompt)
            
            # Clean and parse the response
            cleaned_response = response.strip()
            if not cleaned_response.startswith('{'):
                cleaned_response = '{' + cleaned_response
            if not cleaned_response.endswith('}'):
                cleaned_response = cleaned_response + '}'
                
            # Safely evaluate the string as a dictionary
            import ast
            try:
                result = ast.literal_eval(cleaned_response)
            except:
                # Fallback parsing if ast.literal_eval fails
                import json
                try:
                    result = json.loads(cleaned_response)
                except:
                    # Last resort: basic string manipulation
                    parts = cleaned_response.split('",')
                    title = parts[0].split('"title": "')[1]
                    subtitle = parts[1].split('"subtitle": "')[1]
                    content = parts[2].split('"content": "')[1].rstrip('"}')
                    result = {
                        "title": title,
                        "subtitle": subtitle,
                        "content": content
                    }
            
            # Validate the result
            required_keys = ['title', 'subtitle', 'content']
            if not all(key in result for key in required_keys):
                raise ValueError("Missing required keys in analysis response")
                
            return result

        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing findings into a thesis-driven report."""
    
    def synthesize(self, topic: str, focus_areas: Optional[List[str]], analyses: List[str]) -> Optional[Dict[str, str]]:
        """Synthesize multiple analyses into a cohesive report."""
        # Convert analyses list to formatted string, extracting content from each analysis dict
        analyses_text = ""
        for analysis in analyses:
            try:
                if isinstance(analysis, dict) and 'content' in analysis:
                    analyses_text += analysis['content'] + "\n\n"
                else:
                    analyses_text += str(analysis) + "\n\n"
            except Exception as e:
                logger.error(f"Error processing analysis: {str(e)}")
                continue
                
        focus_context = f"\nSelected Focus Areas:\n{', '.join(focus_areas)}" if focus_areas else ""
        
        prompt = f"""Topic: {topic}{focus_context}

Previous Analyses:
{analyses_text}

You are an expert synthesis writer tasked with creating a compelling, thesis-driven report that weaves together all research findings into a cohesive narrative. Begin with a creative, specific title that captures your synthesis's main argument, followed by an engaging subtitle that previews your key insights.

For example, instead of generic titles like "Research Synthesis" or "Final Report", create titles like:
"Preserving Soul: A Blueprint for Cultural Sustainability"
"Urban Evolution: Navigating Change and Tradition"
"Community Crossroads: Charting a Path Forward"

Your synthesis should:
1. Present a clear, compelling thesis that emerges from the research
2. Weave together key findings into a cohesive narrative
3. Address all primary and secondary research questions
4. Evaluate the significance of major findings
5. Draw meaningful connections between different analyses
6. Provide specific evidence and examples
7. Offer actionable recommendations
8. Consider broader implications

Structure your synthesis with:
1. Title and Subtitle
- Create a unique, specific title that captures your main argument
- Add an engaging subtitle that previews key insights
- Avoid generic labels like "Research Synthesis" or "Final Report"

2. Executive Summary
- Present your main thesis
- Preview key findings
- Highlight major implications

3. Key Findings and Analysis
- Organize findings thematically
- Support with specific evidence
- Draw clear connections

4. Synthesis and Implications
- Weave findings into a cohesive narrative
- Evaluate significance
- Consider broader context

5. Recommendations
- Provide actionable insights
- Consider different stakeholders
- Address key challenges

Format your response as a dictionary with these exact keys:
{
    "title": "Your creative, specific title here",
    "subtitle": "Your engaging, preview subtitle here",
    "content": "Your detailed synthesis here"
}

Important Formatting Rules:
1. Use clear section headings
2. Format consistently:
   - Bold for emphasis (**text**)
   - Italics for subtitles (*text*)
   - Clear section breaks
3. Ensure readability:
   - Clear topic sentences
   - Logical flow
   - Professional tone
4. Support all claims with evidence
5. Make explicit connections between ideas"""
        
        try:
            result = self.generate_content(prompt, SYNTHESIS_CONFIG)
            if not result:
                return None
                
            # Clean and parse the response
            result = result.strip()
            result = result.replace('"', '"').replace('"', '"')
            result = result.replace("'", "'").replace("'", "'")
            result = result.replace('\n', ' ').replace('\r', ' ')
            
            try:
                synthesis = eval(result)
            except:
                result = result.replace('"{', '{').replace('}"', '}')
                synthesis = eval(result)
            
            if not isinstance(synthesis, dict):
                logger.error("Response is not a dictionary")
                return None
                
            required_keys = {'title', 'subtitle', 'content'}
            if not all(key in synthesis for key in required_keys):
                logger.error("Response missing required keys")
                return None
                
            cleaned_synthesis = {}
            for key in required_keys:
                if key in synthesis:
                    value = synthesis[key]
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    cleaned_synthesis[key] = str(value).strip().strip('"\'').strip()
            
            return cleaned_synthesis
            
        except Exception as e:
            logger.error(f"Error parsing synthesis response: {str(e)}")
            return None 