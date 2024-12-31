"""Custom exceptions for the MARA application."""

class MARAException(Exception):
    """Base exception for MARA application."""
    pass

class RateLimitExceeded(MARAException):
    """Raised when rate limit is exceeded."""
    pass

class ModelError(MARAException):
    """Raised when there's an error with the model."""
    pass

class EmptyResponseError(ModelError):
    """Raised when model returns empty response."""
    pass

class InvalidInputError(MARAException):
    """Raised when input validation fails."""
    pass

class ParseError(MARAException):
    """Raised when parsing model output fails."""
    pass

class ConfigurationError(MARAException):
    """Raised when there's a configuration error."""
    pass 