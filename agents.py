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
    PREANALYSIS_CONFIG
)
from utils import rate_limit_decorator, parse_title_content

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
    
    def _extract_content(self, response: Any) -> Optional[str]:
        """Extract content from response."""
        try:
            if not response:
                logger.error("Empty response from model")
                return None
            
            # Get the full response text
            full_text = None
            
            # Try parts accessor first since it's recommended for complex responses
            try:
                if hasattr(response, 'parts'):
                    full_text = "".join(part.text for part in response.parts)
                elif hasattr(response, 'candidates'):
                    full_text = response.candidates[0].content.parts[0].text
                else:
                    full_text = response.text
            except (AttributeError, IndexError) as e:
                logger.error(f"Failed to extract text from response: {str(e)}")
                return None
            
            if not full_text or not full_text.strip():
                logger.error("Extracted empty text from response")
                return None
                
            logger.info(f"Raw response text: {full_text}")
            
            # Just clean up whitespace and return the content
            final_content = full_text.strip()
            if not final_content:
                logger.error("No content extracted")
                return None
                
            logger.info(f"Extracted content: {final_content}")
            return final_content
            
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return None
    
    @rate_limit_decorator
    def generate_content(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Generate content with rate limiting and error handling."""
        try:
            logger.info("Generating content with Gemini API...")
            logger.info(f"Configuration: {config}")
            
            if not prompt or not prompt.strip():
                logger.error("Empty prompt provided")
                raise ValueError("Empty prompt provided")
                
            if not self.model:
                logger.error("Model not initialized")
                raise ValueError("Model not initialized")
            
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(**config)
                )
                logger.info("Successfully received response from Gemini API")
                
                if not response:
                    logger.error("Received empty response from Gemini API")
                    return None
                
                content = self._extract_content(response)
                if content:
                    logger.info("Successfully extracted content from response")
                    return content
                else:
                    logger.error("Failed to extract content from response")
                    return None
                    
            except Exception as e:
                logger.error(f"Error calling Gemini API: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            raise  # Re-raise to be handled by the calling function

class PreAnalysisAgent(BaseAgent):
    """Agent responsible for generating quick insights before main analysis."""
    
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate quick insights about the topic."""
        try:
            logger.info(f"Generating insights for topic: {topic}")
            
            # Generate fun fact
            fact_prompt = (
                f"Generate a single interesting fact about {topic}. "
                "Make it surprising and include 1-2 relevant emojis. "
                "Keep it to one sentence."
            )
            
            logger.info("Generating fun fact...")
            fact_text = self.generate_content(fact_prompt, PREANALYSIS_CONFIG)
            if not fact_text:
                logger.error("Failed to generate fun fact")
                return None
            logger.info("Fun fact generated successfully")
            
            # Generate ELI5
            eli5_prompt = (
                f"Explain {topic} in extremely simple terms. "
                "Use basic words, 1-3 sentences, and 1-3 emojis. "
                "Example: New York is a big city with tall buildings and lots of people 🌆. It's famous for its pizza 🍕 and busy streets."
            )
            
            logger.info("Generating ELI5 explanation...")
            eli5_text = self.generate_content(eli5_prompt, PREANALYSIS_CONFIG)
            if not eli5_text:
                logger.error("Failed to generate ELI5 explanation")
                return None
            logger.info("ELI5 explanation generated successfully")
            
            insights = {
                'did_you_know': fact_text,
                'eli5': eli5_text
            }
            logger.info("Successfully generated both insights")
            return insights
            
        except Exception as e:
            logger.error(f"PreAnalysis generation failed: {str(e)}")
            raise  # Re-raise the exception to be handled by the process_stage function

class PromptDesigner(BaseAgent):
    """Agent responsible for designing optimal prompts."""
    
    def generate_focus_areas(self, topic: str) -> Optional[list]:
        """Generate potential focus areas for the topic."""
        try:
            prompt = f"""Generate 8-12 focus areas for analyzing {topic}.
            Include core sub-topics, related fields, specific aspects, key issues, and important considerations.
            Each focus area should be 2-5 words, specific, and distinct.
            Format as a Python list of strings.
            Example: ["Machine Learning Applications", "Ethical Implications", "Data Privacy"]"""
            
            response_text = self.generate_content(prompt, PROMPT_DESIGN_CONFIG)
            if not response_text:
                return None
                
            # Extract list from response and clean up
            try:
                # Remove any markdown code block syntax
                text = response_text.replace("```python", "").replace("```", "").strip()
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
        """Design an optimal prompt for the given topic."""
        base_prompt = f"""Create a detailed research framework prompt for analyzing {topic}."""
        
        if selected_focus_areas:
            focus_areas_str = "\n".join(f"- {area}" for area in selected_focus_areas)
            base_prompt += f"\nFocus on these areas:\n{focus_areas_str}"
        
        base_prompt += "\nEnsure the framework covers essential aspects while maintaining academic rigor."
        
        return self.generate_content(base_prompt, PROMPT_DESIGN_CONFIG)

class FrameworkEngineer(BaseAgent):
    """Agent responsible for creating analysis frameworks."""
    
    def create_framework(self, initial_prompt: str, enhanced_prompt: Optional[str] = None) -> Optional[str]:
        """Create a research framework based on the prompt design."""
        # Combine prompts if enhanced prompt exists
        prompt_context = initial_prompt
        if enhanced_prompt:
            prompt_context = f"""Initial Prompt:
            {initial_prompt}
            
            Enhanced Prompt with Selected Focus Areas:
            {enhanced_prompt}"""
        
        prompt = f"""{prompt_context}

        Based on these prompts, create a comprehensive research framework that follows this exact structure:

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

        For each section and subsection, provide detailed and specific content relevant to the topic.
        Ensure each point is thoroughly explained and contextually appropriate.
        Use clear, academic language while maintaining accessibility."""
        
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
        # Increase temperature with each iteration, but cap at max
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
            Provide detailed findings for each key area of investigation outlined.
            
            Framework context:
            {framework}
            
            Structure your analysis using this format:

            Start with:
            Title: [Descriptive title reflecting the main focus]
            Subtitle: [Specific aspect or approach being analyzed]

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

            7. References
               - List all cited works in APA format
               - Include DOIs where available
               - Ensure all citations in the text have corresponding references

            Important:
            - Use in-text citations in APA format (Author, Year) for all major claims and findings
            - Each section should have at least 2-3 relevant citations
            - Ensure citations are from reputable academic sources
            - Include a mix of seminal works and recent research (last 5 years)
            - All citations must have corresponding entries in the References section

            Ensure each section is thoroughly developed with specific examples and evidence."""
        else:
            self.iteration_count += 1  # Increment counter for subsequent iterations
            prompt = f"""Review the previous research iteration and expand the analysis.
            
            Previous analysis:
            {previous_analysis}
            
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

            6. References
               - List all new citations in APA format
               - Include DOIs where available
               - Ensure all citations have corresponding references

            Important:
            - Use in-text citations in APA format (Author, Year) for all major claims and findings
            - Each section should have at least 2-3 relevant citations
            - Ensure citations are from reputable academic sources
            - Include a mix of seminal works and recent research (last 5 years)
            - All citations must have corresponding entries in the References section

            Note: As this is iteration {self.iteration_count + 1}, be more explorative and creative 
            while maintaining academic rigor. Push the boundaries of conventional analysis while 
            ensuring all claims are well-supported."""
        
        result = self.generate_content(prompt, self._get_analysis_config())
        if result:
            return parse_title_content(result)
        return None

class SynthesisExpert(BaseAgent):
    """Agent responsible for synthesizing research findings."""
    
    def _format_report(self, text: str) -> str:
        """Format the report with proper markdown and section organization."""
        if not text:
            return ""
            
        sections = text.split('\n')
        formatted_sections = []
        current_section = []
        in_references = False
        
        for line in sections:
            line = line.strip()
            if not line:
                continue
                
            # Handle main section headers
            if any(line.startswith(f"{i}.") for i in range(1, 8)):
                if current_section:
                    formatted_sections.append('\n'.join(current_section))
                    current_section = []
                # Convert numbered sections to markdown headers
                section_title = line.split('.', 1)[1].strip()
                current_section.append(f"## {section_title}")
                continue
            
            # Special handling for references section
            if "Works Cited" in line or "References" in line:
                in_references = True
                if current_section:
                    formatted_sections.append('\n'.join(current_section))
                current_section = [f"## {line}"]
                continue
            
            # Format bullet points
            if line.startswith('-'):
                line = line.replace('-', '•', 1)
            
            # Format citations in references section
            if in_references and line.strip() and not line.startswith('##'):
                # Ensure proper spacing and formatting for references
                if not line.endswith('.'):
                    line += '.'
                line = '- ' + line
            
            current_section.append(line)
        
        if current_section:
            formatted_sections.append('\n'.join(current_section))
        
        return '\n\n'.join(formatted_sections)
    
    def synthesize(self, topic: str, analyses: list) -> Optional[str]:
        """Synthesize all research analyses into a final report."""
        prompt = f"""Synthesize all research from agent 2 on '{topic}' into a Final Report with:
        
        1. Executive Summary (2-3 paragraphs)
           - Include key citations for major findings
           - Highlight most significant discoveries
        
        2. Key Insights (bullet points)
           - Support each insight with relevant citations
           - Include methodology used to derive insights
        
        3. Analysis
           - Comprehensive synthesis of findings with citations
           - Integration of multiple perspectives
           - Critical evaluation of evidence
        
        4. Conclusion
           - Summary of main findings
           - Implications for theory and practice
           - Future research directions
        
        5. Further Considerations & Counter-Arguments
           - Alternative viewpoints with citations
           - Limitations of current research
           - Areas of uncertainty or debate
        
        6. Recommended Readings and Resources
           - Key papers and their main contributions
           - Seminal works in the field
           - Recent significant publications
        
        7. Works Cited
           - Comprehensive bibliography in APA format
           - Include all sources cited in the report
           - Organize by primary sources, secondary sources, and additional resources
           - Include DOIs where available
        
        Important:
        - Use in-text citations in APA format (Author, Year)
        - Ensure all citations have corresponding entries in Works Cited
        - Include both seminal works and recent research
        - Maintain academic rigor while being accessible
        - Cross-reference findings from different analyses
        
        Analysis to synthesize: {' '.join(analyses)}"""
        
        result = self.generate_content(prompt, SYNTHESIS_CONFIG)
        if result:
            return self._format_report(result)
        return None 