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

def parse_title_content(text: str) -> Dict[str, str]:
    """Parse title and content from analysis text."""
    lines = text.split('\n')
    result = {'title': '', 'subtitle': '', 'content': []}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if not result['title'] and line.lower().startswith('title:'):
            result['title'] = line.split(':', 1)[1].strip()
        elif result['title'] and not result['subtitle'] and line.lower().startswith('subtitle:'):
            result['subtitle'] = line.split(':', 1)[1].strip()
        elif result['title']:  # Only collect content after finding title
            result['content'].append(line)
    
    result['content'] = '\n'.join(result['content']).strip()
    return result

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
            if not prompt or not prompt.strip():
                logger.error("Empty prompt provided")
                return None
                
            if not self.model:
                logger.error("Model not initialized")
                return None
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**config)
            )
            
            if not response:
                return None
            
            # Extract text from response
            try:
                if hasattr(response, 'parts'):
                    text = "".join(part.text for part in response.parts)
                elif hasattr(response, 'candidates'):
                    text = response.candidates[0].content.parts[0].text
                else:
                    text = response.text
                    
                return text.strip() if text else None
                
            except Exception as e:
                logger.error(f"Failed to extract text: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            raise

class PreAnalysisAgent(BaseAgent):
    """Agent responsible for initial analysis and insights."""
    
    @rate_limit_decorator
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate initial insights about the topic."""
        try:
            logger.info(f"Generating insights for topic: '{topic}'")
            
            if not topic:
                logger.error("Empty topic provided to generate_insights")
                return None
                
            if not self.model:
                logger.error("Model not initialized in PreAnalysisAgent")
                return None
            
            # Generate the "Did You Know" insight
            dyk_prompt = f"Provide a clear, direct overview of {topic} in 1-3 sentences. Include 1-3 relevant emojis naturally within the text. Focus on key points and avoid phrases like 'The question is about'."
            logger.info(f"Sending DYK prompt to model: {dyk_prompt}")
            
            try:
                logger.info("Attempting to generate DYK content")
                generation_config = GenerationConfig(**PREANALYSIS_CONFIG)
                logger.info(f"Using generation config: {generation_config}")
                
                dyk_response = self.model.generate_content(
                    dyk_prompt,
                    generation_config=generation_config
                )
                logger.info("Received DYK response from model")
                
                if not dyk_response or not hasattr(dyk_response, 'text'):
                    logger.error("Invalid DYK response from model")
                    return None
                
                dyk_text = dyk_response.text.strip()
                logger.info(f"DYK text: {dyk_text}")
                
                # Generate the ELI5 insight
                eli5_prompt = f"Explain {topic} in simple terms that a 5-year-old would understand. Keep it to 1-3 sentences and include 1-2 relevant emojis."
                logger.info(f"Sending ELI5 prompt to model: {eli5_prompt}")
                
                eli5_response = self.model.generate_content(
                    eli5_prompt,
                    generation_config=generation_config
                )
                logger.info("Received ELI5 response from model")
                
                if not eli5_response or not hasattr(eli5_response, 'text'):
                    logger.error("Invalid ELI5 response from model")
                    return None
                
                eli5_text = eli5_response.text.strip()
                logger.info(f"ELI5 text: {eli5_text}")
                
                insights = {
                    'did_you_know': dyk_text,
                    'eli5': eli5_text
                }
                logger.info("Successfully generated insights")
                return insights
                
            except Exception as e:
                logger.error(f"Error during model generation: {str(e)}", exc_info=True)
                return None
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            return None
            
    @rate_limit_decorator
    def generate_focus_areas(self, topic: str) -> Optional[List[str]]:
        """Generate focus areas for analysis."""
        try:
            logger.info(f"Generating focus areas for topic: '{topic}'")
            
            if not topic:
                logger.error("Empty topic provided to generate_focus_areas")
                return None
                
            if not self.model:
                logger.error("Model not initialized in PreAnalysisAgent")
                return None
            
            # Generate focus areas
            prompt = (
                f"Generate 8-12 specific focus areas for analyzing {topic}. "
                "Each focus area should be a concise phrase (3-5 words) that captures a key aspect to analyze. "
                "Format the response as a simple bullet point list with one focus area per line, starting with '- '. "
                "Do not include any other text, numbering, or explanations."
            )
            logger.info(f"Sending focus areas prompt to model: {prompt}")
            
            try:
                logger.info("Attempting to generate focus areas")
                generation_config = GenerationConfig(**PREANALYSIS_CONFIG)
                logger.info(f"Using generation config: {generation_config}")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                logger.info("Received focus areas response from model")
                
                if not response or not hasattr(response, 'text'):
                    logger.error("Invalid focus areas response from model")
                    return None
                
                # Parse the response
                text = response.text.strip()
                logger.info(f"Focus areas text: {text}")
                
                # Split into lines and clean up
                focus_areas = [
                    line.strip('- ').strip()
                    for line in text.split('\n')
                    if line.strip() and line.strip().startswith('-')
                ]
                
                # Validate the results
                if not focus_areas:
                    logger.error("No valid focus areas found in response")
                    return None
                    
                if len(focus_areas) < 8:
                    logger.error(f"Too few focus areas generated: {len(focus_areas)}")
                    return None
                    
                if len(focus_areas) > 12:
                    logger.info(f"Trimming focus areas from {len(focus_areas)} to 12")
                    focus_areas = focus_areas[:12]
                
                logger.info(f"Successfully generated {len(focus_areas)} focus areas: {focus_areas}")
                return focus_areas
                
            except Exception as e:
                logger.error(f"Error during model generation: {str(e)}", exc_info=True)
                return None
            
        except Exception as e:
            logger.error(f"Error generating focus areas: {str(e)}", exc_info=True)
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

        As a distinguished research methodologist, create an innovative yet rigorous research framework that transcends conventional boundaries while maintaining academic integrity. Structure the framework as follows:

        A. Research Objectives:
           1. Primary Research Questions
              - Core inquiries that challenge existing paradigms
              - Questions that bridge multiple disciplines
              - Investigations that reveal hidden connections
           2. Secondary Research Questions
              - Supporting inquiries that illuminate nuances
              - Questions that explore unexpected relationships
              - Probes into underlying mechanisms
           3. Expected Outcomes
              - Anticipated theoretical contributions
              - Potential paradigm shifts
              - Novel interpretative frameworks

        B. Methodological Approach:
           1. Research Methods
              - Integration of complementary methodologies
              - Innovative analytical approaches
              - Cross-disciplinary techniques
           2. Data Collection Strategies
              - Multi-modal evidence gathering
              - Triangulation approaches
              - Novel data source identification
           3. Analysis Techniques
              - Advanced interpretative methods
              - Pattern recognition strategies
              - Synthesis of disparate findings

        C. Investigation Areas:
           1. Core Topics
              - Central theoretical constructs
              - Key phenomenological aspects
              - Fundamental relationships
           2. Subtopics
              - Emergent themes and patterns
              - Interconnected elements
              - Hidden variables
           3. Cross-cutting Themes
              - Meta-level patterns
              - Systemic relationships
              - Unexpected correlations

        D. Theoretical Integration:
           1. Conceptual Frameworks
              - Synthesis of competing theories
              - Novel theoretical propositions
              - Integration points
           2. Interdisciplinary Connections
              - Cross-domain implications
              - Theoretical bridges
              - Paradigm intersections
           3. Knowledge Gaps
              - Theoretical blind spots
              - Unexplored territories
              - Potential breakthroughs

        E. Critical Perspectives:
           1. Epistemological Considerations
              - Underlying assumptions
              - Knowledge construction
              - Theoretical limitations
           2. Methodological Tensions
              - Competing approaches
              - Validity challenges
              - Integration difficulties
           3. Alternative Viewpoints
              - Contrasting frameworks
              - Opposing perspectives
              - Novel interpretations

        F. Research Impact:
           1. Theoretical Implications
              - Paradigm advancement
              - Knowledge expansion
              - Conceptual innovation
           2. Practical Applications
              - Real-world relevance
              - Implementation pathways
              - Societal impact
           3. Future Directions
              - Emerging questions
              - Research trajectories
              - Potential developments

        For each section and subsection:
        - Develop sophisticated, multi-layered analyses
        - Identify unexpected connections and relationships
        - Challenge conventional wisdom while maintaining rigor
        - Consider both obvious and subtle implications
        - Integrate diverse theoretical perspectives
        - Highlight potential paradigm shifts
        - Emphasize novel interpretative frameworks

        Use precise academic language while ensuring clarity and accessibility.
        Focus on creating a framework that reveals hidden patterns and unexpected insights."""
        
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
            prompt = f"""As a Nobel laureate-level expert in the field, conduct a groundbreaking initial research analysis of '{topic}'. 
            Leverage the provided framework to reveal profound insights and unexpected connections while maintaining rigorous academic standards.
            
            Framework context:
            {framework}
            
            Structure your analysis using this format:

            Start with:
            Title: [Compelling title that captures the essence of your discoveries]
            Subtitle: [Intriguing aspect that challenges conventional understanding]

            Then provide a comprehensive analysis following this structure:

            1. Introduction
               - Contextual groundwork that challenges existing paradigms
               - Novel framing of the research scope
               - Ambitious yet achievable objectives that push boundaries
               - Unexpected angles or perspectives that merit exploration

            2. Theoretical Foundation
               - Synthesis of competing theoretical frameworks
               - Identification of hidden assumptions and biases
               - Novel theoretical connections across disciplines
               - Emerging paradigms and their implications

            3. Methodological Innovation
               - Advanced analytical approaches
               - Integration of complementary methods
               - Novel data triangulation strategies
               - Innovative analytical frameworks

            4. Key Discoveries
               - Groundbreaking findings with robust evidence (with citations)
               - Unexpected patterns and relationships
               - Counter-intuitive insights
               - Paradigm-shifting implications

            5. Critical Analysis
               - Deep examination of complex relationships (with citations)
               - Multi-level interpretation of findings
               - Integration of competing perspectives
               - Identification of emergent patterns
               - Exploration of paradoxes and tensions

            6. Theoretical Implications
               - Contributions to existing theories
               - Novel theoretical propositions
               - Cross-disciplinary implications
               - Potential paradigm shifts
               - Future theoretical directions

            7. Practical Significance
               - Real-world applications and impact
               - Implementation challenges and opportunities
               - Societal implications
               - Future possibilities

            8. Research Frontiers
               - Emerging questions and paradoxes
               - Unexplored territories
               - Methodological innovations needed
               - Future research trajectories

            9. References
               - Comprehensive bibliography in APA format
               - Include DOIs where available
               - Balance seminal works with cutting-edge research
               - Include cross-disciplinary sources

            Critical Requirements:
            - Challenge conventional wisdom while maintaining academic rigor
            - Identify unexpected connections across disciplines
            - Support all claims with robust evidence and citations
            - Include 3-4 citations per major section
            - Balance theoretical depth with practical implications
            - Emphasize novel interpretations and insights
            - Consider counter-intuitive findings
            - Explore paradoxes and tensions in the field
            - Integrate competing theoretical perspectives
            - Highlight potential paradigm shifts

            Your analysis should not merely summarize existing knowledge but should push the boundaries of understanding while maintaining scholarly excellence."""
        else:
            self.iteration_count += 1  # Increment counter for subsequent iterations
            prompt = f"""As a Nobel laureate building upon previous research, conduct a deeper analysis that reveals new layers of understanding and unexpected connections.
            
            Previous analysis:
            {previous_analysis}
            
            For iteration #{self.iteration_count + 1}, transcend conventional boundaries by:
            1. Identifying subtle patterns and hidden relationships
            2. Exploring paradoxes and apparent contradictions
            3. Challenging assumptions and established paradigms
            4. Synthesizing disparate findings into novel frameworks
            5. Revealing unexpected implications and applications
            
            Structure your analysis using this format:

            Start with:
            Title: [Compelling title that captures your novel insights]
            Subtitle: [Intriguing aspect that challenges current understanding]

            Then provide:

            1. Meta-Analysis
               - Critical evaluation of previous findings
               - Identification of hidden patterns
               - Emerging questions and paradoxes
               - Novel interpretative frameworks

            2. Theoretical Advancement
               - Integration of competing perspectives
               - Novel theoretical propositions
               - Cross-disciplinary implications
               - Paradigm-shifting insights

            3. Methodological Innovation
               - Advanced analytical approaches
               - Novel data interpretation strategies
               - Integration of diverse methods
               - Innovative frameworks

            4. Unexpected Connections
               - Cross-domain relationships
               - Counter-intuitive findings
               - Emergent patterns
               - Novel synthesis of ideas

            5. Critical Implications
               - Theoretical breakthroughs
               - Practical applications
               - Societal impact
               - Future directions

            6. References
               - Comprehensive bibliography in APA format
               - Include DOIs where available
               - Emphasize cutting-edge research
               - Include cross-disciplinary sources

            Critical Requirements:
            - Push beyond conventional analysis while maintaining rigor
            - Identify subtle patterns and relationships
            - Support novel insights with robust evidence
            - Include 3-4 citations per major section
            - Emphasize unexpected connections
            - Challenge existing paradigms
            - Explore paradoxes and tensions
            - Propose innovative frameworks
            - Consider counter-intuitive implications

            Note: As iteration {self.iteration_count + 1}, strive to reveal deeper layers of understanding and unexpected connections that challenge and expand current knowledge."""
        
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

class PromptDesigner(BaseAgent):
    """Agent responsible for designing optimal prompts."""
    
    def design_prompt(self, topic: str) -> Optional[str]:
        """Design an optimized prompt for the given topic."""
        try:
            logger.info(f"Designing prompt for topic: '{topic}'")
            
            if not topic:
                logger.error("Empty topic provided to design_prompt")
                return None
                
            if not self.model:
                logger.error("Model not initialized in PromptDesigner")
                return None
            
            # Generate the prompt
            prompt = (
                f"Create a comprehensive research framework for analyzing {topic}. "
                "Structure it as a markdown document with:\n"
                "1. Major sections (using level 2 headers)\n"
                "2. Subsections as bullet points\n"
                "3. Include methodological considerations\n"
                "4. Focus on academic rigor and practical relevance\n"
                "Format as clean markdown without explanatory text."
            )
            logger.info(f"Sending prompt design request to model: {prompt}")
            
            try:
                logger.info("Attempting to generate optimized prompt")
                generation_config = GenerationConfig(**PREANALYSIS_CONFIG)
                logger.info(f"Using generation config: {generation_config}")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                logger.info("Received prompt design response from model")
                
                if not response or not hasattr(response, 'text'):
                    logger.error("Invalid prompt design response from model")
                    return None
                
                # Clean up the response
                optimized_prompt = response.text.strip()
                logger.info("Successfully generated optimized prompt")
                return optimized_prompt
                
            except Exception as e:
                logger.error(f"Error during model generation: {str(e)}", exc_info=True)
                return None
            
        except Exception as e:
            logger.error(f"Error designing prompt: {str(e)}", exc_info=True)
            return None 