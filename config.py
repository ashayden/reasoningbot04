"""Configuration settings for the MARA application."""

from pydantic import BaseModel, Field
from google.generativeai.types import GenerationConfig as GeminiConfig

class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

class AppConfig(BaseModel):
    """Main application configuration."""
    # Model configurations
    GEMINI_MODEL: str = "gemini-pro"
    
    # Generation configurations with specific tuning for each stage
    PROMPT_DESIGN_CONFIG: GenerationConfig = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=1024
    )
    
    FRAMEWORK_CONFIG: GenerationConfig = GenerationConfig(
        temperature=0.6,
        top_p=0.8,
        max_output_tokens=2048
    )
    
    ANALYSIS_CONFIG: GenerationConfig = GenerationConfig(
        temperature=0.8,
        top_p=0.9,
        max_output_tokens=4096
    )
    
    SYNTHESIS_CONFIG: GenerationConfig = GenerationConfig(
        temperature=0.6,
        top_p=0.8,
        max_output_tokens=3072
    )
    
    # Input validation
    MIN_TOPIC_LENGTH: int = Field(default=3, ge=1)
    MAX_TOPIC_LENGTH: int = Field(default=200, le=1000)

# Create a global config instance
config = AppConfig()

# For backward compatibility
GEMINI_MODEL = config.GEMINI_MODEL
PROMPT_DESIGN_CONFIG = config.PROMPT_DESIGN_CONFIG.model_dump()
FRAMEWORK_CONFIG = config.FRAMEWORK_CONFIG.model_dump()
ANALYSIS_CONFIG = config.ANALYSIS_CONFIG.model_dump()
SYNTHESIS_CONFIG = config.SYNTHESIS_CONFIG.model_dump()
MIN_TOPIC_LENGTH = config.MIN_TOPIC_LENGTH
MAX_TOPIC_LENGTH = config.MAX_TOPIC_LENGTH 