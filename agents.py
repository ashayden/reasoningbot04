"""Agent implementations for the MARA application."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

from config import (
    PROMPT_DESIGN_CONFIG,
    FRAMEWORK_CONFIG,
    ANALYSIS_CONFIG,
    ANALYSIS_BASE_TEMP,
    ANALYSIS_TEMP_INCREMENT,
    ANALYSIS_MAX_TEMP,
    SYNTHESIS_CONFIG
)
from utils import (
    rate_limit_decorator,
    parse_title_content,
    CitationFormatter,
    MarkdownFormatter
)

logger = logging.getLogger(__name__)

class PreAnalysisAgent:
    """Agent responsible for generating quick insights before main analysis."""
    
    def __init__(self, model):
        """Initialize the agent with a model."""
        self.model = model
        
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate quick insights about the topic.
        
        Args:
            topic: The topic to analyze
            
        Returns:
            Dictionary containing 'did_you_know' and 'eli5' sections,
            or None if generation fails
        """
        try:
            # Generate fun fact with higher creativity
            fact_config = {
                "temperature": 0.9,  # Higher temperature for more creative facts
                "top_p": 0.9,
                "top_k": 40,
                "candidate_count": 1
            }
            
            fact_prompt = (
                "Generate a fascinating and unexpected fact about this topic: "
                f"'{topic}'\n\n"
                "The fact should:\n"
                "- Be surprising, unique, or counter-intuitive\n"
                "- Reveal an interesting connection or lesser-known aspect\n"
                "- Use vivid, engaging language\n"
                "- Include relevant statistics or specific details when possible\n"
                "- Add 1-2 relevant emojis that enhance understanding\n"
                "- Be exactly one sentence that hooks the reader\n"
                "- Challenge common assumptions or expectations\n\n"
                "Make it memorable and thought-provoking. Respond with just the fact, no additional text."
            )
            
            fact_response = self.model.generate_content(
                fact_prompt,
                generation_config=GenerationConfig(**fact_config)
            )
            if not fact_response:
                return None
            
            # Generate ELI5 with balanced creativity
            eli5_config = {
                "temperature": 0.7,  # Balanced temperature for creative yet clear explanations
                "top_p": 0.8,
                "top_k": 30,
                "candidate_count": 1
            }
            
            eli5_prompt = (
                "Explain this topic as if teaching a curious child: "
                f"'{topic}'\n\n"
                "Your explanation should:\n"
                "- Use creative analogies or metaphors\n"
                "- Connect to everyday experiences\n"
                "- Include memorable examples\n"
                "- Be engaging and fun\n"
                "- Use simple but vivid language\n"
                "- Be 2-3 sentences maximum\n"
                "- Add 1-3 relevant emojis that help visualization\n"
                "- If it's a question, answer directly\n"
                "- If it's a topic, give an interesting overview\n\n"
                "Make it engaging and memorable. Respond with just the explanation, no additional text."
            )
            
            eli5_response = self.model.generate_content(
                eli5_prompt,
                generation_config=GenerationConfig(**eli5_config)
            )
            if not eli5_response:
                return None
            
            return {
                'did_you_know': fact_response.text.strip(),
                'eli5': eli5_response.text.strip()
            }
            
        except Exception as e:
            logger.error(f"PreAnalysis generation failed: {str(e)}")
            return None

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
            
            if not response:
                logger.error("Empty response from model")
                return None
            
            # Handle the response based on its type
            try:
                # For responses with parts
                if hasattr(response, 'parts'):
                    content = []
                    for part in response.parts:
                        if hasattr(part, 'text'):
                            content.append(part.text)
                    return '\n'.join(content).strip()
                # For responses with candidates
                elif hasattr(response, 'candidates'):
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content'):
                            if hasattr(candidate.content, 'parts'):
                                content = []
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text'):
                                        content.append(part.text)
                                return '\n'.join(content).strip()
                # For simple responses with text
                elif hasattr(response, 'text'):
                    return response.text.strip()
                
                logger.error("Unable to extract text from response")
                return None
                
            except Exception as e:
                logger.error(f"Error extracting content from response: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            st.error(f"Error generating content: {str(e)}")
            return None

class PromptDesigner(BaseAgent):
    """Agent responsible for designing optimal prompts."""
    
    def generate_focus_areas(self, topic: str) -> Optional[list]:
        """Generate potential focus areas for the topic.
        
        Args:
            topic: The topic to analyze
            
        Returns:
            List of potential focus areas, or None if generation fails
        """
        try:
            prompt = f"""Analyze this topic/question and generate a list of 8-12 potential focus areas: '{topic}'

            The focus areas should include:
            - Core sub-topics within the main topic
            - Related fields or disciplines
            - Specific aspects or angles
            - Relevant issues or challenges
            - Important considerations
            - Key applications or implications
            
            Each focus area should be:
            - Concise (2-5 words)
            - Specific and meaningful
            - Relevant to the topic
            - Distinct from other areas
            
            Format: Return only a Python list of strings, one focus area per item.
            Example: ["Machine Learning Applications", "Ethical Implications", "Data Privacy", ...]"""
            
            response = self.model.generate_content(prompt)
            if not response or not response.text:
                return None
                
            # Extract list from response and clean up
            try:
                # Remove any markdown code block syntax
                text = response.text.replace("```python", "").replace("```", "").strip()
                # Safely evaluate the string as a Python list
                focus_areas = eval(text)
                if not isinstance(focus_areas, list):
                    return None
                return focus_areas
            except:
                logger.error("Failed to parse focus areas response")
                return None
            
        except Exception as e:
            logger.error(f"Focus areas generation failed: {str(e)}")
            return None
    
    def design_prompt(self, topic: str, selected_focus_areas: Optional[list] = None) -> Optional[str]:
        """Design an optimal prompt for the given topic.
        
        Args:
            topic: The topic to analyze
            selected_focus_areas: Optional list of focus areas to emphasize
        """
        base_prompt = f"""As an expert prompt engineer, create a detailed prompt that will guide the development 
        of a research framework for analyzing '{topic}'."""
        
        if selected_focus_areas:
            focus_areas_str = "\n".join(f"- {area}" for area in selected_focus_areas)
            base_prompt += f"\n\nPay special attention to these selected focus areas:\n{focus_areas_str}"
        
        base_prompt += "\nFocus on the essential aspects that need to be investigated while maintaining analytical rigor and academic standards."
        
        return self.generate_content(base_prompt, PROMPT_DESIGN_CONFIG)

class FrameworkEngineer(BaseAgent):
    """Agent responsible for creating analysis frameworks."""
    
    def create_framework(self, initial_prompt: str, enhanced_prompt: Optional[str] = None) -> Optional[str]:
        """Create a research framework based on the prompt design.
        
        Args:
            initial_prompt: The initial optimized prompt
            enhanced_prompt: Optional prompt enhanced with selected focus areas
        """
        # Combine prompts if enhanced prompt exists
        prompt_context = initial_prompt
        if enhanced_prompt:
            prompt_context = f"""Initial Prompt:
            {initial_prompt}
            
            Enhanced Prompt with Selected Focus Areas:
            {enhanced_prompt}"""
        
        prompt = f"""{prompt_context}

        Based on these prompts, create a comprehensive research framework that follows this exact structure:

        # [Create a concise, descriptive title that directly states the analysis focus]

        A. Research Objectives:
           1. Primary Research Questions
           2. Secondary Research Questions
           3. Expected Outcomes

        B. Methodological Approach:
           1. Research Methods
           2. Data Collection Strategies
           3. Analysis Techniques

        C. Investigation Areas:
           1. Core Topics
           2. Subtopics
           3. Cross-cutting Themes

        D. Ethical Considerations:
           1. Key Ethical Issues
           2. Stakeholder Analysis
           3. Risk Assessment

        E. Evaluation Framework:
           1. Success Metrics
           2. Quality Indicators
           3. Validation Methods

        F. Timeline and Milestones:
           1. Research Phases
           2. Key Deliverables
           3. Review Points

        For each section and subsection:
        - Provide detailed and specific content relevant to the topic
        - Ensure each point is thoroughly explained
        - Use clear, academic language while maintaining accessibility
        - Focus on methodological and structural aspects
        - Outline concrete steps and criteria
        
        Note: This framework should focus on the research structure and methodology. 
        Citations and references are not needed in this outline phase."""
        
        return self.generate_content(prompt, FRAMEWORK_CONFIG)

class ResearchAnalyst(BaseAgent):
    """Agent responsible for conducting research analysis."""
    
    def __init__(self, model: Any):
        super().__init__(model)
        self.iteration_count = 0
    
    def _get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration with dynamic temperature scaling."""
        config = ANALYSIS_CONFIG.copy()
        # Increase temperature with each iteration, but cap at max
        temp = min(
            ANALYSIS_BASE_TEMP + (self.iteration_count * ANALYSIS_TEMP_INCREMENT),
            ANALYSIS_MAX_TEMP
        )
        config["temperature"] = temp
        return config
    
    def _format_analysis(self, result: Dict[str, str]) -> Dict[str, str]:
        """Format the analysis result with proper markdown."""
        formatted = {
            'title': '',
            'subtitle': '',
            'content': ''
        }
        
        if result.get('title'):
            # Remove any existing # symbols and format properly
            clean_title = result['title'].lstrip('#').strip()
            # Remove any asterisks from title
            clean_title = clean_title.replace('*', '')
            formatted['title'] = clean_title
        
        if result.get('subtitle'):
            # Remove any existing * symbols and format properly
            clean_subtitle = result['subtitle'].strip('*').strip()
            # Format as italic without asterisks
            formatted['subtitle'] = f"_{clean_subtitle}_"
        
        if result.get('content'):
            sections = result['content'].split('\n')
            formatted_sections = []
            current_section = []
            in_references = False
            reference_entries = []
            
            for line in sections:
                line = line.strip()
                if not line:
                    continue
                
                # Remove any stray asterisks that aren't part of bullet points
                if not line.startswith('*'):
                    line = line.replace('*', '')
                
                # Handle section headers (numbered sections)
                if any(line.startswith(f"{i}.") for i in range(1, 8)):
                    if current_section:
                        if in_references and reference_entries:
                            # Add collected references with proper formatting
                            current_section.extend(reference_entries)
                        formatted_sections.append('\n'.join(current_section))
                        current_section = []
                        reference_entries = []
                    
                    section_title = line.split('.', 1)[1].strip()
                    if "References" in section_title:
                        in_references = True
                        current_section.append(f"\n## {section_title}")
                    else:
                        in_references = False
                        current_section.append(f"\n## {section_title}")
                    continue
                
                # Handle subsection headers (bullet points that look like headers)
                if not in_references and line.startswith('-') and ':' in line and len(line) < 50:
                    subsection = line.lstrip('- ').strip()
                    current_section.append(f"\n### {subsection}")
                    continue
                
                # Handle bullet points and references
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    cleaned_line = line.lstrip('-•* ').strip()
                    # Remove any asterisks from bullet point content
                    cleaned_line = cleaned_line.replace('*', '')
                    if in_references:
                        # Format reference entries consistently
                        citation = cleaned_line
                        if not citation.endswith('.'):
                            citation += '.'
                        reference_entries.append(f"* {citation}")
                    else:
                        # Regular bullet points
                        current_section.append(f"* {cleaned_line}")
                    continue
                
                # Handle regular text
                if not in_references:
                    current_section.append(line)
            
            # Add final section
            if current_section:
                if in_references and reference_entries:
                    # Add collected references with proper formatting
                    current_section.extend(reference_entries)
                formatted_sections.append('\n'.join(current_section))
            
            # Join sections with proper spacing and clean up
            content = '\n\n'.join(formatted_sections)
            
            # Clean up extra whitespace and ensure proper markdown spacing
            content = content.replace('\n\n\n', '\n\n')  # Remove triple line breaks
            content = content.replace('\n##', '\n\n##')  # Ensure space before headers
            content = content.replace('\n###', '\n\n###')  # Ensure space before subheaders
            content = content.replace('\n*', '\n\n*')  # Ensure space before lists
            
            # Special handling for references section spacing
            content = content.replace('* \n', '*\n')  # Remove extra space after reference bullets
            content = content.replace('.\n*', '.\n\n*')  # Add space between reference entries
            
            formatted['content'] = content.strip()
        
        return formatted
    
    def analyze(self, topic: str, framework: str, previous_analysis: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Conduct research analysis."""
        if previous_analysis is None:
            self.iteration_count = 0  # Reset counter for new analysis
            prompt = f"""Acting as a leading expert in topic-related field: Based on the framework above, conduct an initial research analysis of '{topic}'. 
            Follow the methodological approaches and evaluation criteria specified in the framework.
            Provide detailed findings for each key area of investigation outlined.
            
            Framework context:
            {framework}
            
            Structure your analysis using this format:

            Start with:
            Title: [Descriptive title reflecting the main focus of topic analysis]
            Subtitle: [Specific aspect of analysis and/or approach being analyzed]

            Then provide a comprehensive analysis following this structure:

            1. Introduction
               - Context and background
               - Scope of analysis
               - Key objectives

            2. Methodology Overview
               - Approach used
               - Data sources
               - Analytical methods

            3. Key Findings
               - Primary discoveries (with citations)
               - Supporting evidence (with citations)
               - Critical insights

            4. Analysis
               - Detailed examination of findings (with citations)
               - Interpretation of results
               - Connections and patterns

            5. Implications
               - Theoretical implications
               - Practical applications
               - Future considerations

            6. Limitations and Gaps
               - Current limitations
               - Areas needing further research
               - Potential biases

            7. Provide full bibliography:
                - Use APA 7th edition format
                - Include all in-text citations
                - Add DOIs where available
                - List primary sources first
                - Organize alphabetically

            Important:
            - Use proper APA in-text citations (Author, Year)
            - Each section should have at least 2-3 relevant citations
            - Ensure citations are from reputable academic sources
            - Include a mix of seminal works and recent research (last 5 years)
            - All citations must have corresponding entries in References

            Ensure each section is thoroughly developed with specific examples and evidence."""
        else:
            self.iteration_count += 1
            prompt = f"""Review the previous research iteration and expand the analysis.
            
            Previous analysis:
            {previous_analysis}
            
            For iteration #{self.iteration_count + 1}, focus on:
            1. Identifying gaps or areas needing more depth
            2. Exploring new connections and implications
            3. Refining and strengthening key arguments
            4. Adding new supporting evidence or perspectives
            
            Structure your analysis using this format:

            Start with:
            Title: [Descriptive title reflecting the new focus]
            Subtitle: [Specific aspect being expanded upon]

            Then provide:

            1. Previous Analysis Review
               - Key points from previous iteration
               - Areas identified for expansion
               - New perspectives to explore

            2. Expanded Analysis
               - Deeper investigation of key themes (with citations)
               - New evidence and insights (with citations)
               - Advanced interpretations

            3. Novel Connections
               - Cross-cutting themes (with citations)
               - Interdisciplinary insights
               - Emerging patterns

            4. Critical Evaluation
               - Strengthened arguments (with citations)
               - Counter-arguments addressed
               - Enhanced evidence base

            5. Synthesis and Integration
               - Integration with previous findings
               - Enhanced understanding
               - Refined conclusions

            6. Provide full bibliography:
                - Use APA 7th edition format
                - Include all in-text citations
                - Add DOIs where available
                - List primary sources first
                - Organize alphabetically

            Important:
            - Use proper APA in-text citations (Author, Year)
            - Each section should have at least 2-3 relevant citations
            - Ensure citations are from reputable academic sources
            - Include a mix of seminal works and recent research (last 5 years)
            - All citations must have corresponding entries in References

            Note: As this is iteration {self.iteration_count + 1}, be more explorative and creative 
            while maintaining academic rigor."""
        
        result = self.generate_content(prompt, self._get_analysis_config())
        if result:
            parsed = parse_title_content(result)
            return self._format_analysis(parsed)
        return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing research findings."""
    
    def _format_report(self, text: str) -> str:
        """Format the report with proper markdown and section organization."""
        if not text:
            return ""
        
        # Split into lines and initialize containers
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        formatted_lines = []
        in_references = False
        
        # Extract title and subtitle from the content
        title = None
        subtitle = None
        content_start = 0
        
        # Look for title and subtitle in the first few lines
        for i, line in enumerate(lines):
            if line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
                content_start = i + 1
            elif line.startswith('Subtitle:'):
                subtitle = line.replace('Subtitle:', '').strip()
                content_start = i + 1
        
        # If no explicit title/subtitle found, use first non-numbered line as title
        if not title:
            for i, line in enumerate(lines):
                if not any(line.startswith(f"{n}.") for n in range(1, 10)):
                    title = line
                    content_start = i + 1
                    break
        
        # Add title and subtitle to formatted lines
        if title:
            formatted_lines.append(f"# {title}")
        if subtitle:
            formatted_lines.append(f"_{subtitle}_")
        formatted_lines.append("")
        
        # Process remaining content
        for line in lines[content_start:]:
            # Clean up any raw markdown characters
            line = line.replace('**', '')  # Remove raw bold markers
            line = line.replace('__', '')  # Remove raw underline markers
            
            # Handle main section headers (numbered sections)
            if any(line.startswith(f"{i}.") for i in range(1, 8)):
                # Add extra spacing before new sections
                if formatted_lines:
                    formatted_lines.append("")
                
                # Extract and format section title
                section_title = line.split('.', 1)[1].strip()
                
                # Special handling for references section
                if "Works Cited" in section_title or "References" in section_title:
                    in_references = True
                
                # Add formatted header (using ## for section headers)
                formatted_lines.append(f"## {section_title}")
                continue
            
            # Handle references section
            if in_references:
                if not line.startswith('*'):
                    # Format as bullet point if not already
                    line = f"* {line}"
                if not line.endswith('.'):
                    line = f"{line}."
                formatted_lines.append(line)
                continue
            
            # Handle bullet points
            if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                cleaned_line = line.lstrip('-*• ').strip()
                # Clean up any markdown in bullet points
                cleaned_line = cleaned_line.replace('**', '').replace('__', '')
                formatted_lines.append(f"* {cleaned_line}")
                continue
            
            # Handle regular text
            formatted_lines.append(line)
        
        # Join lines with proper spacing and clean up
        text = "\n".join(formatted_lines)
        
        # Ensure proper spacing between sections
        text = text.replace("\n## ", "\n\n## ")  # Double space before section headers
        text = text.replace("\n* ", "\n\n* ")    # Double space before lists
        text = text.replace("\n\n\n", "\n\n")    # Remove triple spacing
        
        # Final cleanup of any remaining raw markdown
        text = text.replace('****', '')  # Remove any double bold markers
        text = text.replace('____', '')  # Remove any double underline markers
        
        return text.strip()
    
    def synthesize(self, topic: str, analyses: list) -> Optional[str]:
        """Synthesize all research analyses into a final report."""
        prompt = f"""Create a comprehensive research synthesis on '{topic}' following this exact structure:

        Title: [Descriptive title reflecting the main focus of topic analysis]
        Subtitle: [Specific aspect of analysis]

        1. Executive Summary
        Provide a 2-3 paragraph overview that:
        - Synthesizes key findings with citations
        - Highlights major discoveries
        - Summarizes methodology
        
        2. Key Insights
        Present 4-6 major insights that:
        - Include specific citations
        - Focus on significant findings
        - Connect to methodology
        
        3. Analysis
        Develop a thorough analysis that:
        - Synthesizes all findings
        - Integrates perspectives
        - Evaluates evidence
        - Organizes by themes
        
        4. Conclusion
        Provide research implications:
        - Summarize key findings
        - Discuss impacts
        - Suggest future directions
        - Make recommendations
        
        5. Further Considerations
        Address complexities:
        - Present counter-arguments
        - Discuss limitations
        - Note uncertainties
        - Identify challenges
        
        6. Recommended Readings
        List essential sources:
        - Note seminal works
        - Include recent research
        - Add methodology guides
        - List digital resources
        
        7. Works Cited
        Provide full bibliography:
        - Use APA 7th edition format
        - Include all in-text citations
        - Add DOIs where available
        - List primary sources first
        - Organize alphabetically
        - Each entry should be on a new line
        - Each entry should end with a period
        - Each entry should start with a bullet point (*)

        Important Guidelines:
        - Use proper APA in-text citations (Author, Year)
        - Ensure every citation has a reference
        - Include both classic and recent works
        - Maintain academic tone
        - Cross-reference analyses
        - Format citations consistently
        - Include DOIs for recent works

        Format Guidelines:
        - Use numbered sections (1., 2., etc.)
        - Use bullet points for lists (-)
        - Include proper spacing between sections
        - Format references with bullet points
        - End each reference with a period
        
        Analysis to synthesize: {' '.join(analyses)}"""
        
        result = self.generate_content(prompt, SYNTHESIS_CONFIG)
        if result:
            return self._format_report(result)
        return None 