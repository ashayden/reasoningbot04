"""Agent implementations for the MARA application."""

import logging
from typing import Dict, Any, Optional, Tuple
import os
import sys

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

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
            
            if not response:
                logger.error("Empty response from model")
                return None
            
            # Handle the new response format
            try:
                # Try to get text from parts
                text = ""
                for part in response.parts:
                    if hasattr(part, 'text'):
                        text += part.text
                
                if not text:
                    logger.error("No text content in response parts")
                    return None
                
                logger.info(f"Raw response length: {len(text)}")
                
                # Split response into thoughts and actual content
                parts = text.split("\n\n", 1)
                
                if len(parts) > 1 and "Thoughts" in parts[0]:
                    self._last_thoughts = parts[0]
                    content = parts[1].strip()
                    logger.info(f"Extracted content length: {len(content)}")
                    return content
                else:
                    logger.info("No thoughts section found, returning full response")
                    return text.strip()
                    
            except Exception as e:
                logger.error(f"Error processing response parts: {str(e)}")
                # Fallback to candidates if parts access fails
                if hasattr(response, 'candidates') and response.candidates:
                    text = response.candidates[0].content.text
                    return text.strip()
                return None
                
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            st.error(f"Error generating content: {str(e)}")
            return None

class PromptDesigner(BaseAgent):
    """Agent responsible for designing optimal prompts."""
    
    def design_prompt(self, topic: str) -> Optional[str]:
        """Design an optimal prompt for the given topic."""
        prompt = f"""As an expert prompt engineer, design an optimal prompt to analyze this topic: '{topic}'

        Your response MUST contain these EXACT section headers:

        Desired Output:
        [Write a clear, specific description of what the analysis should accomplish and deliver]

        Avoid:
        [List specific approaches, biases, and limitations that should be avoided]

        Emphasize:
        [List key aspects, methods, and perspectives that should be emphasized]

        Consider these aspects in your response:
        1. Key aspects that need investigation
        2. Potential research angles
        3. Important contextual factors
        4. Relevant academic disciplines
        5. Methodological approaches

        IMPORTANT: Each section MUST start with the exact header (e.g., "Desired Output:", "Avoid:", "Emphasize:").
        Make each section clear, specific, and actionable.
        
        Previous thought process (if available):
        {self._last_thoughts if self._last_thoughts else 'Not available'}"""
        
        result = self.generate_content(prompt, PROMPT_DESIGN_CONFIG)
        if not result:
            logger.error("Empty result from generate_content")
            return None
            
        # Extract only the desired sections
        sections = {}
        current_section = None
        section_content = []
        
        # Split into lines and clean up
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        
        # Log the raw response for debugging
        logger.info(f"Raw response:\n{result}")
        
        for line in lines:
            # Check for section headers with exact matches
            if "Desired Output:" in line:
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'Desired Output'
                section_content = []
                # If there's content after the header on the same line
                content_after_header = line.split("Desired Output:", 1)[1].strip()
                if content_after_header:
                    section_content.append(content_after_header)
            elif "Avoid:" in line:
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'Avoid'
                section_content = []
                # If there's content after the header on the same line
                content_after_header = line.split("Avoid:", 1)[1].strip()
                if content_after_header:
                    section_content.append(content_after_header)
            elif "Emphasize:" in line:
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                current_section = 'Emphasize'
                section_content = []
                # If there's content after the header on the same line
                content_after_header = line.split("Emphasize:", 1)[1].strip()
                if content_after_header:
                    section_content.append(content_after_header)
            elif current_section and not any(header in line for header in ["Consider:", "Previous thought process:"]):
                section_content.append(line)
                
        # Add the last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
            
        # Log found sections for debugging
        logger.info(f"Found sections: {list(sections.keys())}")
        logger.info("Section contents:")
        for section, content in sections.items():
            logger.info(f"{section}:\n{content}\n")
            
        # Verify we have all required sections
        required_sections = {'Desired Output', 'Avoid', 'Emphasize'}
        if not all(section in sections for section in required_sections):
            logger.error(f"Missing required sections. Found: {list(sections.keys())}")
            return None
            
        # Format the final output
        formatted_output = []
        for section in ['Desired Output', 'Avoid', 'Emphasize']:
            if section in sections and sections[section].strip():
                formatted_output.append(f"{section}:\n{sections[section].strip()}")
                
        final_output = '\n\n'.join(formatted_output)
        if not final_output.strip():
            logger.error("Empty formatted output")
            return None
            
        # Log final output for debugging
        logger.info(f"Final formatted output:\n{final_output}")
            
        return final_output

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