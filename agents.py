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
            # Convert Pydantic config to dict for Gemini
            generation_config = config.model_dump() if hasattr(config, 'model_dump') else config
            response = self.model.generate_content(prompt, generation_config=generation_config)
            if not response or not response.text:
                raise ProcessingError("Empty response from model")
            return response.text.strip()
        except Exception as e:
            logger.exception(f"Error generating content: {str(e)}")
            raise ProcessingError(f"Failed to generate content: {str(e)}")

class PreAnalysisAgent(BaseAgent):
    """Quick insights agent."""
    
    @handle_error
    def generate_insights(self, topic: str) -> Optional[Dict[str, str]]:
        """Generate quick insights about the topic."""
        # Try cache first
        cached = cache.get("insights", topic, max_age=timedelta(days=1))
        if cached:
            logger.info(f"Using cached insights for topic: {topic}")
            return cached
        
        logger.info(f"Generating new insights for topic: {topic}")
        
        # Generate fun fact
        fact_prompt = (
            f"Share one fascinating and unexpected fact about {topic}. "
            "Focus on a surprising or counter-intuitive aspect that most people wouldn't know. "
            "Include relevant emojis and ensure the fact is both engaging and educational. "
            "Format as a complete, well-structured sentence."
        )
        fact = self._generate_content(fact_prompt, config.PROMPT_DESIGN_CONFIG)
        
        # Generate ELI5
        eli5_prompt = (
            f"Explain {topic} as if explaining to a curious 10-year-old. "
            "Use simple language but don't oversimplify the concepts. "
            "Include 2-3 clear, engaging sentences with relevant emojis. "
            "Make it both educational and memorable."
        )
        eli5 = self._generate_content(eli5_prompt, config.PROMPT_DESIGN_CONFIG)
        
        insights = {
            'did_you_know': fact,
            'eli5': eli5
        }
        
        # Cache the results
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
            
            # Clean up and validate each line
            areas = []
            for line in response.split('\n'):
                # Clean the line
                cleaned = line.strip()
                cleaned = cleaned.strip('•-*[]()#').strip()
                cleaned = cleaned.strip('1234567890.').strip()
                cleaned = cleaned.strip('"\'').strip()
                
                # Validate the line
                if (cleaned and 
                    len(cleaned.split()) >= 2 and  # At least 2 words
                    len(cleaned.split()) <= 7 and  # At most 7 words
                    not any(cleaned.startswith(x) for x in ['•', '-', '*', '#', '>', '•']) and
                    not cleaned.lower().startswith(('example', 'note:', 'format'))):
                    areas.append(cleaned)
            
            # Take first 8 valid areas
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
    def generate_framework(
        self,
        topic: str,
        optimized_prompt: str,
        focus_areas: Optional[List[str]] = None
    ) -> Optional[str]:
        """Generate research framework using optimized prompt and focus areas."""
        cache_key = {
            "topic": topic,
            "prompt": optimized_prompt,
            "areas": focus_areas
        }
        cached = cache.get("framework", cache_key, max_age=timedelta(days=1))
        if cached:
            return cached
        
        areas_text = ", ".join(focus_areas[:3]) if focus_areas else ""
        
        prompt = (
            f"Create a comprehensive research framework for {topic} focusing on: {areas_text}\n\n"
            "Format the response in 4 sections:\n"
            "1. Key Questions (2-3 bullet points)\n"
            "2. Main Topics (3-4 bullet points)\n"
            "3. Methods (2-3 bullet points)\n"
            "4. Expected Insights (2-3 bullet points)\n\n"
            "Keep each bullet point detailed but focused."
        )
        
        framework = self._generate_content(prompt, config.FRAMEWORK_CONFIG)
        cache.set("framework", cache_key, framework)
        return framework
    
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

class ResearchAnalyst(BaseAgent):
    """Research analyst."""
    
    @handle_error
    def analyze(
        self,
        topic: str,
        framework: str,
        previous: Optional[str] = None
    ) -> Optional[ResearchResult]:
        """Analyze a specific aspect of the topic."""
        if previous:
            prompt = (
                f"Continue the research analysis on {topic}, building on this previous insight:\n\n"
                f"{previous}\n\n"
                "Structure your response with:\n"
                "1. A clear, informative title that reflects this iteration's focus\n"
                "2. A brief subtitle that captures the key focus area\n"
                "3. Detailed analysis with specific examples and evidence\n"
                "4. At least 3 key findings or implications\n"
                "Format with clear sections and bullet points where appropriate.\n"
                "Ensure this analysis builds upon and doesn't repeat the previous insights."
            )
        else:
            prompt = (
                f"Conduct a thorough research analysis on {topic} using this framework:\n\n"
                f"{framework}\n\n"
                "Structure your response with:\n"
                "1. A clear, informative title that reflects the initial analysis\n"
                "2. A brief subtitle that captures the key focus area\n"
                "3. Detailed analysis with specific examples and evidence\n"
                "4. At least 3 key findings or implications\n"
                "Format with clear sections and bullet points where appropriate."
            )
        
        response = self._generate_content(prompt, config.ANALYSIS_CONFIG)
        
        # Split into title, subtitle, and content
        lines = response.split('\n', 2)
        result = ResearchResult(
            title=lines[0].strip(),
            subtitle=lines[1].strip() if len(lines) > 2 else None,
            content=lines[-1].strip()
        )
        
        return result

class SynthesisExpert(BaseAgent):
    """Research synthesizer."""
    
    @handle_error
    def synthesize(self, topic: str, research_results: List[ResearchResult]) -> Optional[str]:
        """Create final synthesis."""
        # Extract key points efficiently
        summary_points = []
        for result in research_results:
            summary = [result.title]  # Start with the title
            if result.subtitle:
                summary.append(result.subtitle)
            
            # Add content with bullet points
            content_lines = result.content.split('\n')
            bullets = []
            for line in content_lines:
                if line.strip().startswith(('•', '-', '*')):
                    bullets.append(line)
                    if len(bullets) == 3:
                        break
            
            if bullets:
                summary.extend(bullets)
            summary_points.append('\n'.join(summary))
        
        prompt = (
            f"Create a comprehensive synthesis of these research findings about {topic}:\n\n"
            + "\n---\n".join(summary_points)
            + "\n\nStructure your synthesis with:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Key Findings (3-4 bullet points)\n"
            "3. Analysis (2-3 paragraphs)\n"
            "4. Implications & Recommendations (2-3 bullet points)\n"
            "Ensure each section is detailed and well-supported by the research."
        )
        
        return self._generate_content(prompt, config.SYNTHESIS_CONFIG) 