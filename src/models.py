"""
Pydantic Models for Keynote MCP Server

This module defines all Pydantic models for input validation, output serialization,
and data transfer objects used throughout the MCP server.

Author: Senior AI Engineer Portfolio Project
License: MIT
"""

from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import math


# =============================================================================
# ENUMS
# =============================================================================

class MouseButton(str, Enum):
    """Valid mouse button options for automation."""
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class LogLevel(str, Enum):
    """Valid logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# =============================================================================
# MATHEMATICAL INPUT MODELS
# =============================================================================

class BasicArithmeticInput(BaseModel):
    """Input model for basic arithmetic operations."""
    a: int = Field(..., description="First operand", ge=-10**10, le=10**10)
    b: int = Field(..., description="Second operand", ge=-10**10, le=10**10)
    
    @validator('b')
    def validate_division_by_zero(cls, v, values):
        """Validate that division by zero is not attempted."""
        if 'a' in values and v == 0:
            raise ValueError("Cannot divide by zero")
        return v


class NumberListInput(BaseModel):
    """Input model for operations on lists of numbers."""
    numbers: List[int] = Field(..., description="List of numbers to process", min_items=1, max_items=1000)
    
    @validator('numbers')
    def validate_number_range(cls, v):
        """Validate that all numbers are within acceptable range."""
        for num in v:
            if abs(num) > 10**10:
                raise ValueError(f"Number {num} is too large. Maximum allowed: ±10^10")
        return v


class SingleNumberInput(BaseModel):
    """Input model for single number operations."""
    number: Union[int, float] = Field(..., description="Number to process", ge=-10**10, le=10**10)
    
    @validator('number')
    def validate_positive_for_sqrt(cls, v):
        """Validate that square root operations use positive numbers."""
        if v < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return v


class StringInput(BaseModel):
    """Input model for string processing operations."""
    text: str = Field(..., description="Text to process", min_length=1, max_length=1000)
    
    @validator('text')
    def validate_text_content(cls, v):
        """Validate that text contains only printable characters."""
        if not v.isprintable():
            raise ValueError("Text contains non-printable characters")
        return v


class AngleInput(BaseModel):
    """Input model for trigonometric operations."""
    angle_radians: float = Field(..., description="Angle in radians", ge=-2*math.pi, le=2*math.pi)


class FactorialInput(BaseModel):
    """Input model for factorial operations."""
    n: int = Field(..., description="Number to compute factorial for", ge=0, le=20)
    
    @validator('n')
    def validate_factorial_range(cls, v):
        """Validate factorial input range."""
        if v < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if v > 20:
            raise ValueError("Factorial input too large. Maximum allowed: 20")
        return v


class FibonacciInput(BaseModel):
    """Input model for Fibonacci sequence generation."""
    n: int = Field(..., description="Number of Fibonacci numbers to generate", ge=0, le=100)


# =============================================================================
# KEYNOTE AUTOMATION INPUT MODELS
# =============================================================================

class KeynoteRectangleInput(BaseModel):
    """Input model for drawing rectangles in Keynote."""
    button: MouseButton = Field(default=MouseButton.LEFT, description="Mouse button to use for drawing")


class KeynoteTextInput(BaseModel):
    """Input model for adding text to Keynote."""
    text: str = Field(..., description="Text to add to presentation", min_length=1, max_length=500)
    
    @validator('text')
    def validate_text_length(cls, v):
        """Validate text length for presentation display."""
        if len(v) > 500:
            raise ValueError("Text too long for presentation. Maximum: 500 characters")
        return v


class ImagePathInput(BaseModel):
    """Input model for image processing operations."""
    image_path: str = Field(..., description="Path to the image file")
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate image file path and extension."""
        import os
        if not os.path.exists(v):
            raise ValueError(f"Image file not found: {v}")
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        _, ext = os.path.splitext(v.lower())
        if ext not in valid_extensions:
            raise ValueError(f"Unsupported image format: {ext}. Supported: {valid_extensions}")
        return v


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class OperationResult(BaseModel):
    """Base model for operation results."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    timestamp: Optional[str] = Field(default=None, description="Operation timestamp")


class MathematicalResult(OperationResult):
    """Result model for mathematical operations."""
    result: Union[int, float, List[int], List[float]] = Field(..., description="Mathematical computation result")
    operation_type: str = Field(..., description="Type of mathematical operation performed")
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 10) if isinstance(v, float) else v
        }


class KeynoteOperationResult(OperationResult):
    """Result model for Keynote automation operations."""
    operation_type: str = Field(..., description="Type of Keynote operation performed")
    coordinates: Optional[Dict[str, int]] = Field(default=None, description="Screen coordinates used")
    duration: Optional[float] = Field(default=None, description="Operation duration in seconds")


class ImageResult(OperationResult):
    """Result model for image processing operations."""
    image_data: Optional[str] = Field(default=None, description="Base64 encoded image data")
    format: str = Field(..., description="Image format")
    dimensions: Optional[Dict[str, int]] = Field(default=None, description="Image dimensions")


# =============================================================================
# ERROR MODELS
# =============================================================================

class ValidationError(BaseModel):
    """Model for validation errors."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(..., description="Value that failed validation")


class OperationError(BaseModel):
    """Model for operation errors."""
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class ServerConfig(BaseModel):
    """Configuration model for the MCP server."""
    server_name: str = Field(default="Keynote-AI-Automation", description="Server name")
    version: str = Field(default="1.0.0", description="Server version")
    max_iterations: int = Field(default=6, ge=1, le=20, description="Maximum iterations for AI workflows")
    llm_timeout: int = Field(default=10, ge=1, le=60, description="LLM response timeout in seconds")
    keynote_delay: float = Field(default=2.0, ge=0.1, le=10.0, description="Delay between Keynote operations")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    debug_mode: bool = Field(default=False, description="Enable debug mode")


class ClientConfig(BaseModel):
    """Configuration model for the AI client."""
    api_key: str = Field(..., description="Google Gemini API key")
    model_name: str = Field(default="gemini-2.0-flash", description="LLM model to use")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    timeout: int = Field(default=10, ge=1, le=60, description="Request timeout in seconds")


# =============================================================================
# WORKFLOW MODELS
# =============================================================================

class WorkflowStep(BaseModel):
    """Model for individual workflow steps."""
    step_number: int = Field(..., description="Step number in workflow")
    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    status: str = Field(default="pending", description="Step status")
    result: Optional[Any] = Field(default=None, description="Step result")


class WorkflowResult(BaseModel):
    """Model for complete workflow results."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    total_steps: int = Field(..., description="Total number of steps")
    completed_steps: int = Field(..., description="Number of completed steps")
    success: bool = Field(..., description="Whether workflow completed successfully")
    final_result: Optional[Any] = Field(default=None, description="Final workflow result")
    steps: List[WorkflowStep] = Field(default_factory=list, description="Individual workflow steps")
    duration: Optional[float] = Field(default=None, description="Total workflow duration")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_result(error_type: str, message: str, details: Optional[Dict] = None) -> OperationError:
    """Create a standardized error result."""
    import datetime
    return OperationError(
        error_type=error_type,
        message=message,
        details=details or {},
        timestamp=datetime.datetime.now().isoformat()
    )


def create_success_result(message: str, result: Any = None, operation_type: str = "unknown") -> OperationResult:
    """Create a standardized success result."""
    import datetime
    return OperationResult(
        success=True,
        message=message,
        timestamp=datetime.datetime.now().isoformat()
    )


# =============================================================================
# MODEL REGISTRY
# =============================================================================

# Registry of all input models for easy access
INPUT_MODELS = {
    "add": BasicArithmeticInput,
    "subtract": BasicArithmeticInput,
    "multiply": BasicArithmeticInput,
    "divide": BasicArithmeticInput,
    "power": BasicArithmeticInput,
    "modulo": BasicArithmeticInput,
    "add_list": NumberListInput,
    "exponential_sum": NumberListInput,
    "square_root": SingleNumberInput,
    "cube_root": SingleNumberInput,
    "natural_log": SingleNumberInput,
    "factorial": FactorialInput,
    "fibonacci_sequence": FibonacciInput,
    "string_to_ascii_values": StringInput,
    "sine": AngleInput,
    "cosine": AngleInput,
    "tangent": AngleInput,
    "draw_rectangle_in_keynote": KeynoteRectangleInput,
    "add_text_to_keynote": KeynoteTextInput,
    "create_thumbnail": ImagePathInput,
}

# Registry of output models
OUTPUT_MODELS = {
    "mathematical": MathematicalResult,
    "keynote": KeynoteOperationResult,
    "image": ImageResult,
    "general": OperationResult,
}