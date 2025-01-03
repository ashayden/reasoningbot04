"""Configuration settings for the MARA application."""

from pydantic import BaseModel, Field

class GenerationConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

class AppConfig(BaseModel):
    # Model configurations
    GEMINI_MODEL: str = "gemini-exp-1206"
    
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