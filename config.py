"""Configuration settings for the MARA application."""

from typing import Dict, Any

# Model Configuration
GEMINI_MODEL = "learnlm-1.5-pro-experimental"

# Topic Validation
MIN_TOPIC_LENGTH = 10
MAX_TOPIC_LENGTH = 500

# Progressive Research Configuration
class ProgressiveConfig:
    """Dynamic configuration based on iteration depth."""
    
    @staticmethod
    def get_iteration_config(iteration: int) -> Dict[str, Any]:
        """Get configuration for specific iteration depth."""
        # Base configuration
        config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048
        }
        
        # Progressive adjustments
        if iteration > 1:
            # Temperature increases with depth (0.7 -> 0.9)
            config['temperature'] = min(0.7 + (0.05 * (iteration - 1)), 0.9)
            
            # Top_p increases slightly (0.9 -> 0.95)
            config['top_p'] = min(0.9 + (0.0125 * (iteration - 1)), 0.95)
            
            # Token limit increases with depth
            config['max_output_tokens'] = min(
                2048 + (512 * (iteration - 1)),  # More aggressive token increase
                4096  # Maximum safe limit
            )
            
        return config

# Agent Configurations
PREANALYSIS_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 1024,
}

ANALYSIS_CONFIG = ProgressiveConfig.get_iteration_config(1)  # Base configuration

SYNTHESIS_CONFIG = {
    'temperature': 0.8,  # Slightly higher for creative synthesis
    'top_p': 0.9,
    'top_k': 40,
    'max_output_tokens': 4096,
}

# Cache Settings
CACHE_TTL = 3600  # 1 hour in seconds

# Rate Limiting
API_RATE_LIMIT = {
    'calls': 60,    # Maximum calls
    'period': 60.0  # Time period in seconds
}

# Error Handling
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0

# Content Processing
MAX_FOCUS_AREAS = 5
MIN_FOCUS_AREAS = 2 