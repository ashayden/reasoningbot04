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
            f"Share one fascinating and unexpected fact about {topic} in a single sentence. "
            "Include relevant emojis. Focus on a surprising or counter-intuitive aspect."
        )
        fact = self._generate_content(fact_prompt, config.PROMPT_DESIGN_CONFIG)
        
        # Generate ELI5
        eli5_prompt = (
            f"Explain {topic} in 2-3 simple sentences for a general audience. "
            "Use basic language and add relevant emojis."
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
            return cached
            
        prompt = (
            f"List 8 key research areas for {topic}. "
            "Return only the area names, one per line. "
            "No additional formatting, comments, or descriptions."
        )
        
        response = self._generate_content(prompt, config.PROMPT_DESIGN_CONFIG)
        
        # Clean up the response
        areas = [
            line.strip().strip('[],"\'')
            for line in response.split('\n')
            if line.strip() and not line.strip()[0].isdigit()
        ][:8]  # Take first 8 valid areas
        
        if not areas:
            raise ProcessingError("No valid focus areas generated")
        
        cache.set("focus_areas", topic, areas)
        return areas
    
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
            prompt = f"Continue the research on {topic}, building on: {previous}"
        else:
            prompt = f"Research {topic} using this framework: {framework}"
        
        response = self._generate_content(prompt, config.ANALYSIS_CONFIG)
        
        # Split into title and content
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
    def synthesize(self, topic: str, research_results: List[str]) -> Optional[str]:
        """Create final synthesis."""
        # Extract key points efficiently
        summary_points = []
        for result in research_results:
            lines = result.split('\n')
            summary = [lines[0]]  # Always include first line
            
            # Add up to 3 bullet points efficiently
            bullets = []
            for line in lines[1:]:
                if line.strip().startswith(('•', '-')):
                    bullets.append(line)
                    if len(bullets) == 3:
                        break
            
            if bullets:
                summary.extend(bullets)
            summary_points.append('\n'.join(summary))
        
        prompt = (
            f"Synthesize these research findings about {topic}:\n\n"
            + "\n---\n".join(summary_points)
        )
        
        return self._generate_content(prompt, config.SYNTHESIS_CONFIG) 