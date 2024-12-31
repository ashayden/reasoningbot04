"""Agent implementations for the MARA application."""

import logging
from typing import Dict, Any, Optional, Tuple

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import GenerationConfig

from config import (
    PROMPT_DESIGN_CONFIG,
    FRAMEWORK_CONFIG,
    ANALYSIS_CONFIG,
    SYNTHESIS_CONFIG
)
from utils import rate_limit_decorator, parse_title_content

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
            # Generate fun fact
            fact_prompt = (
                "Generate a single sentence interesting fact related to this topic: "
                f"'{topic}'\n\n"
                "The fact should:\n"
                "- Be surprising or fascinating\n"
                "- Be true and accurate\n"
                "- Not directly answer or summarize the topic\n"
                "- Include 1-2 relevant emojis when appropriate\n"
                "- Be exactly one sentence\n\n"
                "Respond with just the fact, no additional text."
            )
            
            fact_response = self.model.generate_content(fact_prompt)
            if not fact_response:
                return None
            
            # Generate ELI5
            eli5_prompt = (
                "Explain this topic in extremely simple terms: "
                f"'{topic}'\n\n"
                "The explanation should:\n"
                "- Use very simple, straightforward language\n"
                "- Be 1-3 sentences maximum\n"
                "- Include 1-3 relevant emojis when appropriate\n"
                "- If it's a question, answer it directly\n"
                "- If it's a topic, give a high-level overview\n\n"
                "Respond with just the explanation, no additional text."
            )
            
            eli5_response = self.model.generate_content(eli5_prompt)
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
        """Generate content with rate limiting and error handling.
        
        The new Gemini 2.0 model returns both thoughts and response.
        This method extracts only the response for user display while
        storing the thoughts for internal use.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**config)
            )
            
            if not response or not response.text:
                logger.error("Empty response from model")
                return None
                
            logger.info(f"Raw response length: {len(response.text)}")
            
            # Split response into thoughts and actual content
            parts = response.text.split("\n\n", 1)
            
            if len(parts) > 1 and "Thoughts" in parts[0]:
                self._last_thoughts = parts[0]
                content = parts[1].strip()
                logger.info(f"Extracted content length: {len(content)}")
                return content
            else:
                # If no clear separation, return the whole response
                logger.info("No thoughts section found, returning full response")
                return response.text.strip()
                
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            st.error(f"Error generating content: {str(e)}")
            return None

class PromptDesigner(BaseAgent):
    """Agent responsible for designing optimal prompts."""
    
    def design_prompt(self, topic: str) -> Optional[str]:
        """Design an optimal prompt for the given topic."""
        prompt = f"""As an expert prompt engineer, create a detailed prompt that will guide the development 
        of a research framework for analyzing '{topic}'. Focus on the essential aspects that need to be 
        investigated while maintaining analytical rigor and academic standards."""
        
        return self.generate_content(prompt, PROMPT_DESIGN_CONFIG)

class FrameworkEngineer(BaseAgent):
    """Agent responsible for creating analysis frameworks."""
    
    def create_framework(self, prompt_design: str) -> Optional[str]:
        """Create a research framework based on the prompt design."""
        prompt = f"""{prompt_design}

        Based on this prompt, create a comprehensive research framework that follows this exact structure:

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
            # Include previous agent's thoughts if available
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
        
        return self.generate_content(prompt, SYNTHESIS_CONFIG) 