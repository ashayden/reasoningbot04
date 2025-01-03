"""Agent implementations for the MARA application."""

import logging
import sqlite3
import json
import hashlib
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from config import config
from utils import handle_error, ProcessingError

logger = logging.getLogger(__name__)

class Cache:
    """SQLite-based persistent cache."""
    
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key."""
        data_str = json.dumps(data, sort_keys=True)
        return f"{prefix}:{hashlib.sha256(data_str.encode()).hexdigest()}"
    
    def get(self, prefix: str, data: Any, max_age: Optional[timedelta] = None) -> Optional[Any]:
        """Retrieve value from cache."""
        key = self._generate_key(prefix, data)
        with sqlite3.connect(self.db_path) as conn:
            if max_age:
                cutoff = datetime.now() - max_age
                result = conn.execute(
                    "SELECT value FROM cache WHERE key = ? AND timestamp > ?",
                    (key, cutoff)
                ).fetchone()
            else:
                result = conn.execute(
                    "SELECT value FROM cache WHERE key = ?",
                    (key,)
                ).fetchone()
        
        return json.loads(result[0]) if result else None
    
    def set(self, prefix: str, data: Any, value: Any):
        """Store value in cache."""
        key = self._generate_key(prefix, data)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value))
            )

# Global cache instance
cache = Cache()

@dataclass
class ResearchResult:
    """Data class for research results."""
    title: str
    subtitle: Optional[str] = None
    content: str = ""

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, model: Any):
        if not isinstance(model, genai.GenerativeModel):
            raise ValueError("model must be an instance of GenerativeModel")
        self.model = model
    
    def _generate_content(self, prompt: str, config: GenerationConfig) -> Optional[str]:
        """Generate content with error handling."""
        try:
            generation_config = config.model_dump() if hasattr(config, 'model_dump') else config
            response = self.model.generate_content(prompt, generation_config=generation_config)
            
            if not response.text:
                if hasattr(response, 'prompt_feedback'):
                    logger.error(f"Content blocked due to safety settings: {response.prompt_feedback}")
                    raise ProcessingError("Content generation blocked by safety settings. Please try rephrasing your request.")
                else:
                    logger.error("Empty response from model with no feedback")
                    raise ProcessingError("Empty response from model")
            
            return response.text.strip()
            
        except Exception as e:
            logger.exception(f"Error generating content: {str(e)}")
            if "safety" in str(e).lower():
                raise ProcessingError("Content generation was blocked by safety settings. Please try rephrasing your request.")
            raise ProcessingError(f"Failed to generate content: {str(e)}")

class PreAnalysisAgent(BaseAgent):
    """Quick insights agent."""
    
    @handle_error
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate quick insights about the topic."""
        cached = cache.get("insights", topic, max_age=timedelta(days=1))
        if cached:
            logger.info(f"Using cached insights for topic: {topic}")
            return cached
        
        logger.info(f"Generating new insights for topic: {topic}")
        
        fact_prompt = (
            f"Share one fascinating and unexpected fact about {topic}. "
            "Focus on a surprising or counter-intuitive aspect that most people wouldn't know. "
            "Format as a complete, well-structured sentence. "
            "\nEmoji guidelines:\n"
            "- Use 1-2 emojis that naturally relate to key concepts\n"
            "- Place emojis next to the concepts they represent\n"
            "- Don't cluster emojis together\n"
            "- Ensure the text makes sense if emojis were removed"
        )
        fact = self._generate_content(fact_prompt, config.PROMPT_DESIGN_CONFIG)
        
        eli5_prompt = (
            f"If '{topic}' is a question, answer it simply. "
            "Otherwise, explain {topic} in a way that's easy to understand. "
            "Use simple language and give a 2-3 sentence response. "
            "\nEmoji guidelines:\n"
            "- Use 2-4 emojis that naturally relate to key concepts\n"
            "- Place emojis next to the concepts they represent\n"
            "- Use emojis to highlight key points or transitions\n"
            "- Don't cluster emojis together\n"
            "- Ensure the text makes sense if emojis were removed"
        )
        eli5 = self._generate_content(eli5_prompt, config.PROMPT_DESIGN_CONFIG)
        
        insights = {
            'did_you_know': fact,
            'eli5': eli5
        }
        
        cache.set("insights", topic, insights)
        return insights

class PromptDesigner(BaseAgent):
    """Research framework designer."""
    
    @handle_error
    def generate_focus_areas(self, topic: str) -> Optional[List[str]]:
        """Generate focus areas for the topic."""
        cached = cache.get("focus_areas", topic, max_age=timedelta(days=1))
        if cached:
            logger.info(f"Using cached focus areas for topic: {topic}")
            return cached
            
        logger.info(f"Generating focus areas for topic: {topic}")
        
        prompt = (
            f"Generate 8 key research areas for analyzing {topic}.\n\n"
            "Requirements:\n"
            "1. Each area should be a clear, specific aspect of the topic\n"
            "2. Write each area on a new line\n"
            "3. Use simple, clear phrases (3-7 words each)\n"
            "4. No numbering, bullets, or special characters\n"
            "5. No explanations or additional text\n\n"
            "Example format:\n"
            "Historical Development and Origins\n"
            "Economic Impact and Market Trends\n"
            "Social and Cultural Implications\n"
            "..."
        )
        
        try:
            response = self._generate_content(prompt, config.PROMPT_DESIGN_CONFIG)
            areas = []
            for line in response.split('\n'):
                cleaned = line.strip()
                cleaned = cleaned.strip('•-*[]()#').strip()
                cleaned = cleaned.strip('1234567890.').strip()
                cleaned = cleaned.strip('"\'').strip()
                
                if (cleaned and 
                    len(cleaned.split()) >= 2 and
                    len(cleaned.split()) <= 7 and
                    not any(cleaned.startswith(x) for x in ['•', '-', '*', '#', '>', '•']) and
                    not cleaned.lower().startswith(('example', 'note:', 'format'))):
                    areas.append(cleaned)
            
            valid_areas = areas[:8]
            
            if len(valid_areas) < 3:
                logger.error(f"Not enough valid focus areas generated. Got {len(valid_areas)}: {valid_areas}")
                raise ProcessingError("Failed to generate enough valid focus areas")
            
            logger.info(f"Successfully generated {len(valid_areas)} focus areas")
            cache.set("focus_areas", topic, valid_areas)
            return valid_areas
            
        except Exception as e:
            logger.error(f"Error generating focus areas: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise ProcessingError("Failed to generate valid focus areas") from e
    
    @handle_error
    def design_prompt(self, topic: str, focus_areas: Optional[List[str]] = None) -> Optional[str]:
        """Design research prompt."""
        cache_key = {"topic": topic, "areas": focus_areas}
        cached = cache.get("prompt", cache_key, max_age=timedelta(days=1))
        if cached:
            return cached
            
        if focus_areas:
            prompt = (
                f"Create a focused research framework for analyzing {topic}, "
                f"specifically examining: {', '.join(focus_areas[:3])}.\n\n"
                "Structure the response in these sections:\n"
                "1. Research Questions (2-3 clear, focused questions)\n"
                "2. Key Areas to Investigate (3-4 main topics)\n"
                "3. Methodology (2-3 specific research methods)\n"
                "4. Expected Outcomes (2-3 anticipated findings)\n\n"
                "Keep each section concise but informative."
            )
        else:
            prompt = (
                f"Create a focused research framework for analyzing {topic}.\n\n"
                "Structure the response in these sections:\n"
                "1. Research Questions (2-3 clear, focused questions)\n"
                "2. Key Areas to Investigate (3-4 main topics)\n"
                "3. Methodology (2-3 specific research methods)\n"
                "4. Expected Outcomes (2-3 anticipated findings)\n\n"
                "Keep each section concise but informative."
            )
        
        result = self._generate_content(prompt, config.PROMPT_DESIGN_CONFIG)
        cache.set("prompt", cache_key, result)
        return result
    
    @handle_error
    def create_optimized_prompt(
        self,
        topic: str,
        focus_areas: Optional[List[str]] = None
    ) -> Optional[str]:
        """Create an optimized prompt for the Framework Engineer."""
        cache_key = {"topic": topic, "areas": focus_areas}
        cached = cache.get("optimized_prompt", cache_key, max_age=timedelta(days=1))
        if cached:
            return cached
            
        focus_text = ", ".join(focus_areas) if focus_areas else "all relevant aspects"
        
        prompt = (
            f"As a Framework Engineer, develop a comprehensive research framework for analyzing {topic}.\n\n"
            f"Focus Areas: {focus_text}\n\n"
            "Your framework should:\n"
            "1. Be specifically tailored to this topic and focus areas\n"
            "2. Define clear research boundaries and scope\n"
            "3. Identify key research objectives and questions\n"
            "4. Specify methodological approaches\n"
            "5. Outline data collection and analysis strategies\n"
            "6. Consider potential challenges and limitations\n"
            "7. Suggest evaluation criteria for findings\n\n"
            "This framework will guide a Research Analyst in conducting a thorough, "
            "systematic investigation of the topic. Ensure the framework provides clear "
            "direction while allowing for discovery of unexpected insights.\n\n"
            "The framework should enable progressive deepening of analysis across multiple "
            "research iterations, with each iteration building upon previous findings."
        )
        
        result = self._generate_content(prompt, config.PROMPT_DESIGN_CONFIG)
        cache.set("optimized_prompt", cache_key, result)
        return result

class FrameworkEngineer(BaseAgent):
    """Research framework engineer."""
    
    @handle_error
    def create_framework(
        self,
        topic: str,
        optimized_prompt: str,
        focus_areas: Optional[List[str]] = None
    ) -> Optional[str]:
        """Create a comprehensive research framework."""
        cache_key = {
            "topic": topic,
            "prompt": optimized_prompt,
            "areas": focus_areas
        }
        cached = cache.get("framework", cache_key, max_age=timedelta(days=1))
        if cached:
            return cached
        
        prompt = (
            f"Create a comprehensive research framework based on this prompt:\n\n"
            f"{optimized_prompt}\n\n"
            "Structure the framework with these sections:\n\n"
            "1. Research Objectives\n"
            "   - Primary objective\n"
            "   - Secondary objectives\n"
            "   - Specific goals\n\n"
            "2. Methodology\n"
            "   - Research approach\n"
            "   - Data collection methods\n"
            "   - Analysis techniques\n\n"
            "3. Key Areas of Investigation\n"
            "   - Primary focus areas\n"
            "   - Secondary themes\n"
            "   - Cross-cutting issues\n\n"
            "4. Expected Outcomes\n"
            "   - Anticipated findings\n"
            "   - Potential insights\n"
            "   - Success criteria\n\n"
            "5. Research Parameters\n"
            "   - Scope boundaries\n"
            "   - Limitations\n"
            "   - Assumptions\n\n"
            "Format with clear headings and bullet points."
        )
        
        framework = self._generate_content(prompt, config.FRAMEWORK_CONFIG)
        cache.set("framework", cache_key, framework)
        return framework

class ResearchAnalyst(BaseAgent):
    """Research analyst with Nobel laureate expertise."""
    
    @handle_error
    def analyze(
        self,
        topic: str,
        framework: str,
        optimized_prompt: str,
        iteration: int,
        total_iterations: int,
        previous: Optional[str] = None
    ) -> Optional[ResearchResult]:
        """Analyze a specific aspect of the topic with Nobel-level expertise."""
        # Adjust temperature based on iteration
        base_temp = 0.6
        temp_increment = 0.1
        current_temp = min(base_temp + (iteration * temp_increment), 0.9)
        
        analysis_config = config.ANALYSIS_CONFIG.model_dump()
        analysis_config['temperature'] = current_temp
        
        # Expert perspective prompt
        expert_prompt = (
            "You are a Nobel laureate with deep expertise in fields relevant to this topic. "
            "Your analysis should reflect this level of understanding while remaining accessible. "
            "Focus exclusively on the topic and specified focus areas, providing unique insights "
            "and uncovering meaningful connections within this scope.\n\n"
        )
        
        if previous:
            prompt = (
                f"{expert_prompt}"
                f"Iteration {iteration + 1} of {total_iterations} - Advanced Research Analysis\n\n"
                f"Topic: {topic}\n"
                f"Previous Analysis:\n{previous}\n\n"
                f"Framework:\n{framework}\n\n"
                f"Optimized Prompt:\n{optimized_prompt}\n\n"
                "Building upon the previous analysis:\n"
                "1. Identify a key aspect from the previous analysis that warrants deeper exploration\n"
                "2. Uncover novel connections and insights within the topic scope\n"
                "3. Apply your expertise to reveal hidden patterns and relationships\n\n"
                "Format your response with:\n"
                "1. A concise main title that captures the key insight or finding\n"
                "2. A descriptive subtitle on the next line that provides context\n"
                "3. Detailed analysis incorporating:\n"
                "   - Key theoretical insights about the topic\n"
                "   - Relevant empirical evidence and data\n"
                "   - Novel perspectives and interpretations\n"
                "   - Important connections within the topic scope\n"
                "4. At least 3 significant findings or implications\n"
                "5. Relevant academic citations and references\n\n"
                "Title format example:\n"
                "Environmental Impact Assessment\n"
                "Analyzing Long-term Ecological Changes and Conservation Strategies\n\n"
                "Stay focused on the topic and its direct implications.\n"
                f"Note: This is iteration {iteration + 1} of {total_iterations}, "
                f"with analysis temperature set to {current_temp} for increased insight generation."
            )
        else:
            prompt = (
                f"{expert_prompt}"
                f"Iteration 1 of {total_iterations} - Initial Advanced Research Analysis\n\n"
                f"Topic: {topic}\n"
                f"Framework:\n{framework}\n\n"
                f"Optimized Prompt:\n{optimized_prompt}\n\n"
                "Establish the foundation for progressive analysis:\n"
                "1. Identify the core concepts and principles of the topic\n"
                "2. Map the current understanding and key debates\n"
                "3. Highlight critical aspects needing deeper investigation\n\n"
                "Format your response with:\n"
                "1. A concise main title that captures the key insight or finding\n"
                "2. A descriptive subtitle on the next line that provides context\n"
                "3. Detailed analysis incorporating:\n"
                "   - Key theoretical insights about the topic\n"
                "   - Relevant empirical evidence and data\n"
                "   - Novel perspectives and interpretations\n"
                "   - Important connections within the topic scope\n"
                "4. At least 3 significant findings or implications\n"
                "5. Relevant academic citations and references\n\n"
                "Title format example:\n"
                "Environmental Impact Assessment\n"
                "Analyzing Long-term Ecological Changes and Conservation Strategies\n\n"
                "Stay focused on the topic and its direct implications.\n"
                f"Note: This is iteration 1 of {total_iterations}, establishing the foundation "
                f"with analysis temperature set to {current_temp}."
            )
        
        response = self._generate_content(prompt, GenerationConfig(**analysis_config))
        
        # Split response into title, subtitle, and content
        lines = response.split('\n', 2)
        if len(lines) >= 2:
            title = lines[0].strip()
            subtitle = lines[1].strip()
            content = lines[2].strip() if len(lines) > 2 else ""
            
            # If title contains a colon, split into main title and subtitle
            if ':' in title:
                main_title, subtitle_part = title.split(':', 1)
                title = main_title.strip()
                subtitle = subtitle_part.strip()
        else:
            title = lines[0].strip()
            subtitle = None
            content = ""
        
        result = ResearchResult(
            title=title,
            subtitle=subtitle,
            content=content
        )
        
        return result

class SynthesisExpert(BaseAgent):
    """Academic research synthesizer."""
    
    @handle_error
    def synthesize(self, topic: str, research_results: List[ResearchResult]) -> Optional[str]:
        """Create final synthesis report with academic rigor."""
        summary_points = []
        for result in research_results:
            summary = [
                f"Title: {result.title}",
                f"Focus: {result.subtitle or 'N/A'}",
                "Key Points:",
                result.content
            ]
            summary_points.append('\n'.join(summary))
        
        prompt = (
            "As an academic synthesis expert, create a comprehensive final report that addresses "
            f"the core question or topic: '{topic}'. Focus exclusively on synthesizing the research "
            "findings and their direct implications. Maintain scholarly precision while ensuring "
            "the content is engaging and accessible.\n\n"
            "Research Findings:\n"
            + "\n\n---\n\n".join(summary_points)
            + "\n\nCreate a detailed academic report with this structure:\n\n"
            "1. Format the title as:\n"
            "   - Main title (concise, without 'Title:' prefix)\n"
            "   - Descriptive subtitle on next line (without 'Subtitle:' prefix)\n\n"
            "2. Begin with Executive Summary (One focused paragraph unless volume requires two)\n"
            "   - Open with the central research question\n"
            "   - State the primary research objective\n"
            "   - Summarize key methodological approach\n"
            "   - Present 2-3 most significant discoveries\n"
            "   - Highlight critical implications\n"
            "   - End with the research's broader impact\n\n"
            "3. Key Findings\n"
            "   - Present 4-6 major discoveries\n"
            "   - Support each with specific evidence\n"
            "   - Connect findings to each other\n"
            "   - Identify emerging patterns\n\n"
            "4. Detailed Analysis\n"
            "   - Begin each paragraph with a clear topic sentence\n"
            "   - Support claims with specific examples\n"
            "   - Connect related findings\n"
            "   - Build logical progression of ideas\n"
            "   - Address important gaps in understanding\n\n"
            "5. Practical Implications\n"
            "   - Present concrete applications\n"
            "   - Specify actionable insights\n"
            "   - Identify areas needing attention\n"
            "   - Describe potential impact\n\n"
            "6. Limitations and Considerations\n"
            "   - Address key uncertainties\n"
            "   - Discuss knowledge gaps\n"
            "   - Identify areas for future investigation\n"
            "   - Suggest next steps\n\n"
            "7. Conclusion (One focused paragraph unless volume requires two)\n"
            "   - Begin with research significance\n"
            "   - Synthesize 2-3 most impactful discoveries\n"
            "   - Connect findings to broader context\n"
            "   - Address practical implications\n"
            "   - Present forward-looking perspective\n"
            "   - End with compelling final insight\n\n"
            "8. Essential References\n"
            "   - List key sources cited\n"
            "   - Include relevant works\n"
            "   - Focus on topic-specific resources\n\n"
            "9. Bibliography\n"
            "   - Follow APA 7th edition format\n"
            "   - Prioritize peer-reviewed sources\n"
            "   - Include DOIs when available\n"
            "   - List primary sources first\n"
            "   - Each entry on new line with bullet (*)\n\n"
            "Writing Guidelines:\n"
            "- Begin each section with a strong topic sentence\n"
            "- Use specific examples from the research\n"
            "- Avoid vague references ('this shows', 'it appears')\n"
            "- Replace weak transitions ('in conclusion', 'this report')\n"
            "- Write with precision and clarity\n"
            "- Define technical terms when introduced\n"
            "- Maintain active voice\n"
            "- Use concrete, specific language\n"
            "- Create logical flow through specific connections\n"
            "- Keep focus on the topic and its implications"
        )
        
        return self._generate_content(prompt, config.SYNTHESIS_CONFIG) 