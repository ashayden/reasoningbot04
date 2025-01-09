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

1. Did You Know: Share one fascinating, lesser-known fact about the topic. Keep it to a single clear sentence. Include 1-3 relevant emojis placed naturally within the text where they are most contextually relevant (not grouped at the start).
2. Overview: If '{topic}' is a question, provide a clear, direct answer. Otherwise, provide a clear, accessible 2-3 sentence explanation for a general audience. Focus on key points and avoid technical jargon. Include 1-3 relevant emojis placed naturally within the text where they are most contextually relevant (not grouped at the start).

Format your response EXACTLY as shown below, including the comma between key-value pairs:
{{"did_you_know": "Your fact here with contextual emojis", "eli5": "Your overview here with contextual emojis"}}

Important:
- Place emojis naturally within the text where they are most relevant
- Do not group emojis together at the start or end
- Use only straight quotes (")
- No line breaks in the dictionary
- Keep the exact keys: did_you_know, eli5
- Ensure proper dictionary formatting with comma between key-value pairs
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
            
            # Try multiple parsing approaches
            try:
                # First try ast.literal_eval for safety
                import ast
                insights = ast.literal_eval(result)
            except:
                try:
                    # Try json.loads as fallback
                    import json
                    insights = json.loads(result)
                except:
                    # Last resort: basic string manipulation
                    # Extract content between curly braces
                    content = result[result.find('{'): result.rfind('}') + 1]
                    # Split by comma and extract key-value pairs
                    pairs = content.strip('{}').split('",')
                    insights = {}
                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip().strip('"').strip()
                            value = value.strip().strip('"').strip()
                            insights[key] = value
            
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

Format your response as a Python list of strings:
[
    "First focus area",
    "Second focus area",
    "Third focus area"
]

Important:
- Use only straight quotes (")
- Each focus area should be concise (3-7 words)
- Make each area distinct and specific
- Ensure areas are relevant to the topic
- Return ONLY the list, no additional text"""
        
        try:
            result = self.generate_content(prompt, PREANALYSIS_CONFIG)
            if not result:
                return None
                
            # Clean and parse the response
            result = result.strip()
            
            # Remove any text before the first [ and after the last ]
            start_idx = result.find('[')
            end_idx = result.rfind(']')
            if start_idx == -1 or end_idx == -1:
                logger.error("Could not find list brackets in response")
                return None
            
            result = result[start_idx:end_idx + 1]
            
            # Clean up the string
            result = result.replace('"', '"').replace('"', '"')  # Replace curly quotes
            result = result.replace("'", "'").replace("'", "'")  # Replace curly single quotes
            result = result.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
            
            # Try multiple parsing approaches
            try:
                # First try ast.literal_eval for safety
                import ast
                focus_areas = ast.literal_eval(result)
            except:
                try:
                    # Try json.loads as fallback
                    import json
                    focus_areas = json.loads(result)
                except:
                    # Last resort: basic string manipulation
                    # Remove brackets and split by commas
                    items = result.strip('[]').split('",')
                    focus_areas = [item.strip().strip('"').strip() for item in items if item.strip()]
            
            # Validate the result
            if not isinstance(focus_areas, list):
                logger.error("Response is not a list")
                return None
                
            # Clean up and validate each focus area
            cleaned_areas = []
            for area in focus_areas:
                if isinstance(area, str) and area.strip():
                    cleaned_areas.append(area.strip().strip('"\'').strip())
            
            # Ensure we have enough valid focus areas
            if not (8 <= len(cleaned_areas) <= 10):
                logger.error(f"Invalid number of focus areas: {len(cleaned_areas)}")
                return None
                
            return cleaned_areas
            
        except Exception as e:
            logger.error(f"Error parsing focus areas response: {str(e)}")
            return None

class ResearchAnalyst(BaseAgent):
    """Agent responsible for conducting iterative research analysis."""
    
    def analyze(self, topic: str, focus_areas: List[str], previous_analysis: Optional[str] = None) -> Dict[str, str]:
        """Generate research analysis for the given topic and focus areas."""
        try:
            # Calculate iteration number based on previous analysis
            iteration = 1
            if previous_analysis:
                iteration = len([a for a in previous_analysis.split('\n') if a.startswith('Title:')]) + 1

            # Adjust temperature based on iteration
            current_temp = min(
                ANALYSIS_BASE_TEMP + (ANALYSIS_TEMP_INCREMENT * (iteration - 1)),
                ANALYSIS_MAX_TEMP
            )
            
            # Create config for this iteration
            iteration_config = ANALYSIS_CONFIG.copy()
            iteration_config['temperature'] = current_temp
            # Increase max tokens with each iteration
            iteration_config['max_output_tokens'] = min(
                ANALYSIS_CONFIG['max_output_tokens'] + (256 * (iteration - 1)),
                4096  # Maximum safe token limit
            )
            
            prompt = f'''Analyze the topic "{topic}" focusing on recent developments and key insights.
            
Previous analysis (if any): {previous_analysis if previous_analysis else "None"}
Focus areas: {", ".join(focus_areas) if focus_areas else "General analysis"}

Important notes:
1. Create a unique, specific title that captures the essence of your analysis
2. Write a subtitle that previews your key findings
3. Structure your analysis with clear sections and bullet points
4. Use markdown formatting for headings and emphasis
5. As this is iteration {iteration}, {
    "focus on foundational aspects and key concepts" if iteration == 1 else
    "build upon previous findings and explore deeper connections" if iteration == 2 else
    "delve into nuanced implications and complex relationships" if iteration == 3 else
    "synthesize insights and explore innovative perspectives" if iteration == 4 else
    "push boundaries and explore transformative implications"
}
6. Return your response in this exact format:
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
            if not response:
                return None
            
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
                        "content": content.replace('\\n', '\n')  # Convert literal \n to newlines
                    }
            
            # Validate the result
            required_keys = ['title', 'subtitle', 'content']
            if not all(key in result for key in required_keys):
                raise ValueError("Missing required keys in analysis response")
                
            # Clean up content formatting
            if 'content' in result:
                result['content'] = result['content'].replace('\\n', '\n')
                
            return result

        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing findings into a comprehensive, expert-level report."""

    def _format_references(self, content: str) -> str:
        """Format references according to APA 7th edition standards."""
        # Split content to isolate references section
        sections = content.split("References")
        if len(sections) < 2:
            return content
            
        main_content, references = sections[0], sections[1]
        
        # Process references
        ref_lines = [line.strip() for line in references.split('\n') if line.strip()]
        formatted_refs = []
        
        for ref in ref_lines:
            # Skip lines that don't look like references
            if not ref or ref.startswith('#') or ref.startswith('-'):
                continue
                
            # Format Research Analysis references
            if "Research Analysis" in ref:
                try:
                    analysis_num = ref.split("Research Analysis")[1].split('.')[0].strip()
                    title = ref.split(').')[1].strip() if ').' in ref else ref
                    formatted_refs.append(f"Research Analysis {analysis_num}. ({time.strftime('%Y')}). {title}.")
                except:
                    formatted_refs.append(ref)
            # Format standard references
            else:
                # Ensure proper punctuation
                if not ref.endswith('.'):
                    ref += '.'
                formatted_refs.append(ref)
        
        # Sort references
        formatted_refs.sort(key=lambda x: (
            'Research Analysis' not in x,  # Research Analyses first
            x.lower()  # Then alphabetically
        ))
        
        # Combine content
        formatted_content = main_content + "\n\n## References\n\n" + '\n'.join(formatted_refs)
        return formatted_content

    def synthesize(self, topic: str, focus_areas: Optional[List[str]], analyses: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Synthesize multiple analyses into a cohesive, expert-level report with clear organization and recommendations."""
        # Cache key for persistence
        cache_key = f"synthesis_{topic}_{'-'.join(focus_areas) if focus_areas else 'all'}"
        
        # Check cache first
        @st.cache_data(ttl=3600)
        def get_cached_synthesis(key: str):
            return st.session_state.get(key)
        
        cached_result = get_cached_synthesis(cache_key)
        if cached_result:
            return cached_result

        # Convert analyses list to formatted string with improved structure
        analyses_text = self._format_analyses(analyses)
        
        focus_context = f"\nSelected Focus Areas:\n{', '.join(focus_areas)}" if focus_areas else ""
        
        prompt = f'''Topic: {topic}{focus_context}

Previous Analyses:
{analyses_text}

As an expert in fields relevant to {topic} and an engaging writer, create a comprehensive synthesis of the research findings. Adopt the perspective of a subject matter expert and skilled communicator to make complex ideas accessible while maintaining intellectual rigor.

Return your response in this exact format:
{{
    "title": "Your creative, specific title that captures the key insight",
    "subtitle": "Your engaging subtitle that previews the main findings",
    "content": "Your detailed synthesis here"
}}

Required sections and formatting:
1. Executive Summary
   - Begin with a clear, engaging overview of key findings
   - Present main thesis and supporting points
   - Highlight significance and implications

2. Detailed Analysis
   - Organize findings into clear thematic sections
   - Support claims with evidence from analyses
   - Explain complex concepts clearly
   - Address relationships between key ideas
   
3. Discussion & Implications
   - Examine broader significance
   - Address counter-arguments or limitations
   - Discuss real-world applications
   
4. Recommendations
   - Actionable next steps
   - Areas for further investigation
   
5. Further Reading
   - Curated list of high-quality sources
   - Brief annotations explaining relevance
   
6. References
   - Format all citations in APA style (7th edition)
   - Each reference must be on a new line
   - Include only sources directly referenced in the analyses
   - Format: Author, A. A. (Year). Title of work. Publisher/Source.
   - For research analyses, use this format: Research Analysis [Number]. (Year). [Title of Analysis].
   - Remove any placeholder text or example references
   - Do not include "References" as a heading - it will be added automatically
   - Do not include any explanatory text or notes
   - Do not include any empty lines between references
   - Do not include any quotation marks or special characters
   - Sort references alphabetically by author's last name or analysis number

Use markdown formatting:
- Maintain authoritative but accessible tone
- Define technical terms when introduced
- Use clear topic sentences and transitions
- Provide concrete examples
- Balance depth with clarity'''

        try:
            response = self._generate_with_backoff(prompt)
            if not response:
                return None
                
            # Clean and parse the response
            cleaned_response = response.strip()
            if not cleaned_response.startswith('{'):
                cleaned_response = '{' + cleaned_response
            if not cleaned_response.endswith('}'):
                cleaned_response = cleaned_response + '}'
            
            # Parse response
            try:
                import ast
                result = ast.literal_eval(cleaned_response)
            except:
                try:
                    import json
                    result = json.loads(cleaned_response)
                except:
                    # Last resort parsing
                    parts = cleaned_response.split('",')
                    title = parts[0].split('"title": "')[1]
                    subtitle = parts[1].split('"subtitle": "')[1]
                    content = parts[2].split('"content": "')[1].rstrip('"}')
                    result = {
                        "title": title,
                        "subtitle": subtitle,
                        "content": content.replace('\\n', '\n')
                    }
            
            # Validate and clean result
            required_keys = ['title', 'subtitle', 'content']
            if not all(key in result for key in required_keys):
                raise ValueError("Missing required keys in synthesis response")
            
            # Clean up content formatting
            if 'content' in result:
                result['content'] = result['content'].replace('\\n', '\n')
            
            # Format references in the result
            if result and 'content' in result:
                result['content'] = self._format_references(result['content'])
            
            # Cache the result
            st.session_state[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error generating synthesis: {str(e)}")
            return None

    def _format_analyses(self, analyses: List[Dict[str, str]]) -> str:
        """Format analyses for synthesis input with improved structure."""
        formatted_text = ""
        for i, analysis in enumerate(analyses, 1):
            try:
                if isinstance(analysis, dict):
                    formatted_text += f"\n## Research Analysis {i}\n"
                    formatted_text += f"### {analysis.get('title', '')}\n"
                    if 'subtitle' in analysis:
                        formatted_text += f"#### {analysis['subtitle']}\n"
                    formatted_text += f"{analysis.get('content', '')}\n\n"
                else:
                    formatted_text += f"Analysis {i}: {str(analysis)}\n\n"
            except Exception as e:
                logger.error(f"Error formatting analysis {i}: {str(e)}")
                continue
        return formatted_text 