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

FORMATTING REQUIREMENTS:

1. Title Structure
   - Create a unique, specific title that captures the main theme
   - Add an engaging subtitle that previews key insights
   - Example: "Cultural Evolution: Mapping Urban Identity in Transition"

2. Section Organization
   Each analysis must have these sections in order:
   a) Key Findings and Evidence
   b) Detailed Analysis
   c) Connections and Significance
   d) Implications

3. Bullet Point Format
   Each bullet point must follow this exact structure:
   • **Key Statement:** Your main point here. Follow with 2-3 sentences of supporting evidence or explanation.

4. Section Content Requirements:

   a) Key Findings and Evidence
   - Start with 3-4 bullet points
   - Each bullet presents one clear finding
   - Include specific evidence or examples
   - Use concrete numbers or statistics when available

   b) Detailed Analysis
   - Examine relationships between findings
   - Provide deeper context
   - Use specific examples
   - Connect to broader themes

   c) Connections and Significance
   - Draw explicit connections between findings
   - Evaluate importance of patterns
   - Consider broader implications
   - Identify emerging trends

   d) Implications
   - Present 2-3 clear implications
   - Offer specific recommendations
   - Consider different stakeholders
   - Address potential challenges

5. Markdown Formatting
   - Use ** for bold text
   - Use * for italics
   - Use ### for section headings
   - Leave one blank line between sections
   - Use • for bullet points
   - Maintain consistent indentation

6. Writing Style
   - Use clear topic sentences
   - Provide specific evidence
   - Create logical flow
   - Maintain professional tone
   - Use active voice
   - Keep paragraphs focused

Format your response as a dictionary with these exact keys:
{
    "title": "Your creative, specific title here",
    "subtitle": "Your engaging, preview subtitle here",
    "content": "Your analysis following the format above"
}"""

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
            
            # Restore newlines in content
            if 'content' in cleaned_analysis:
                content = cleaned_analysis['content']
                # Add newlines after section headings
                content = content.replace('###', '\n###')
                # Add newlines between bullet points
                content = content.replace('•', '\n•')
                # Add newlines between sections
                content = content.replace('Detailed Analysis', '\n\nDetailed Analysis')
                content = content.replace('Connections and Significance', '\n\nConnections and Significance')
                content = content.replace('Implications', '\n\nImplications')
                # Clean up any multiple newlines
                content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
                cleaned_analysis['content'] = content
            
            return cleaned_analysis
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
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