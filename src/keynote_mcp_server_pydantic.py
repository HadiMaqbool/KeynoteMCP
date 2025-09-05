"""
Keynote MCP Server with Pydantic Integration - AI-Powered Presentation Automation

This MCP (Model Context Protocol) server provides AI agents with the ability to:
- Perform mathematical computations with robust input validation
- Automate Keynote presentation creation with type-safe operations
- Generate visual content programmatically with validated inputs
- Integrate LLM capabilities with desktop applications using Pydantic models

Key Improvements with Pydantic:
- Automatic input validation with detailed error messages
- Type-safe operations with runtime validation
- Standardized error handling and response formats
- Auto-generated schemas and documentation
- Performance optimizations with compiled models

Author: Senior AI Engineer Portfolio Project
License: MIT
"""

import math
import sys
import time
import logging
from typing import Dict, List, Union, Any
from datetime import datetime

# MCP Framework imports
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types

# Image processing
from PIL import Image as PILImage

# macOS automation
import pyautogui
import applescript
import subprocess
import screeninfo

# Pydantic imports
from pydantic import ValidationError

# Import our custom models
from models import (
    # Input models
    BasicArithmeticInput, NumberListInput, SingleNumberInput, StringInput,
    AngleInput, FactorialInput, FibonacciInput, KeynoteRectangleInput,
    KeynoteTextInput, ImagePathInput,
    # Output models
    MathematicalResult, KeynoteOperationResult, ImageResult, OperationResult,
    OperationError, create_error_result, create_success_result,
    # Enums
    MouseButton
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Keynote-AI-Automation-Pydantic")

# =============================================================================
# VALIDATION DECORATORS
# =============================================================================

def validate_input(input_model_class):
    """Decorator to validate tool inputs using Pydantic models."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Create input model instance for validation
                if args:
                    # Handle positional arguments
                    if len(args) == 1 and isinstance(args[0], dict):
                        # Single dictionary argument
                        validated_input = input_model_class(**args[0])
                    else:
                        # Multiple positional arguments
                        validated_input = input_model_class(*args)
                else:
                    # Handle keyword arguments
                    validated_input = input_model_class(**kwargs)
                
                # Call the original function with validated input
                return func(validated_input)
                
            except ValidationError as e:
                error_msg = f"Validation error in {func.__name__}: {e}"
                logger.error(error_msg)
                return create_error_result("validation_error", error_msg, {"errors": e.errors()})
            except Exception as e:
                error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
                logger.error(error_msg)
                return create_error_result("unexpected_error", error_msg)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

# =============================================================================
# MATHEMATICAL COMPUTATION TOOLS WITH PYDANTIC
# =============================================================================

@mcp.tool()
@validate_input(BasicArithmeticInput)
def add(input_data: BasicArithmeticInput) -> MathematicalResult:
    """
    Add two integers with Pydantic validation.
    
    Args:
        input_data: Validated input containing two integers
        
    Returns:
        MathematicalResult with the sum
    """
    logger.info(f"Computing addition: {input_data.a} + {input_data.b}")
    result = input_data.a + input_data.b
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed {input_data.a} + {input_data.b} = {result}",
        result=result,
        operation_type="addition",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(NumberListInput)
def add_list(input_data: NumberListInput) -> MathematicalResult:
    """
    Calculate the sum of all numbers in a list with Pydantic validation.
    
    Args:
        input_data: Validated input containing list of numbers
        
    Returns:
        MathematicalResult with the sum
    """
    logger.info(f"Computing sum of list: {input_data.numbers}")
    result = sum(input_data.numbers)
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed sum of {len(input_data.numbers)} numbers = {result}",
        result=result,
        operation_type="list_addition",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(BasicArithmeticInput)
def subtract(input_data: BasicArithmeticInput) -> MathematicalResult:
    """Subtract b from a with Pydantic validation."""
    logger.info(f"Computing subtraction: {input_data.a} - {input_data.b}")
    result = input_data.a - input_data.b
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed {input_data.a} - {input_data.b} = {result}",
        result=result,
        operation_type="subtraction",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(BasicArithmeticInput)
def multiply(input_data: BasicArithmeticInput) -> MathematicalResult:
    """Multiply two integers with Pydantic validation."""
    logger.info(f"Computing multiplication: {input_data.a} * {input_data.b}")
    result = input_data.a * input_data.b
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed {input_data.a} * {input_data.b} = {result}",
        result=result,
        operation_type="multiplication",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(BasicArithmeticInput)
def divide(input_data: BasicArithmeticInput) -> MathematicalResult:
    """
    Divide a by b with Pydantic validation.
    
    Note: Division by zero is prevented by BasicArithmeticInput validator
    """
    logger.info(f"Computing division: {input_data.a} / {input_data.b}")
    result = float(input_data.a / input_data.b)
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed {input_data.a} / {input_data.b} = {result}",
        result=result,
        operation_type="division",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(BasicArithmeticInput)
def power(input_data: BasicArithmeticInput) -> MathematicalResult:
    """Calculate base raised to the power of exponent with Pydantic validation."""
    logger.info(f"Computing power: {input_data.a}^{input_data.b}")
    result = input_data.a ** input_data.b
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed {input_data.a}^{input_data.b} = {result}",
        result=result,
        operation_type="power",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(SingleNumberInput)
def square_root(input_data: SingleNumberInput) -> MathematicalResult:
    """
    Calculate the square root of a number with Pydantic validation.
    
    Note: Negative numbers are prevented by SingleNumberInput validator
    """
    logger.info(f"Computing square root of: {input_data.number}")
    result = float(input_data.number ** 0.5)
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed √{input_data.number} = {result}",
        result=result,
        operation_type="square_root",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(SingleNumberInput)
def cube_root(input_data: SingleNumberInput) -> MathematicalResult:
    """Calculate the cube root of a number with Pydantic validation."""
    logger.info(f"Computing cube root of: {input_data.number}")
    result = float(input_data.number ** (1/3))
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed ∛{input_data.number} = {result}",
        result=result,
        operation_type="cube_root",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(FactorialInput)
def factorial(input_data: FactorialInput) -> MathematicalResult:
    """
    Calculate the factorial of a number with Pydantic validation.
    
    Note: Negative numbers and large values are prevented by FactorialInput validator
    """
    logger.info(f"Computing factorial of: {input_data.n}")
    result = int(math.factorial(input_data.n))
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed {input_data.n}! = {result}",
        result=result,
        operation_type="factorial",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(SingleNumberInput)
def natural_log(input_data: SingleNumberInput) -> MathematicalResult:
    """
    Calculate the natural logarithm of a number with Pydantic validation.
    
    Note: Non-positive numbers are prevented by SingleNumberInput validator
    """
    logger.info(f"Computing natural log of: {input_data.number}")
    result = float(math.log(input_data.number))
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed ln({input_data.number}) = {result}",
        result=result,
        operation_type="natural_logarithm",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(BasicArithmeticInput)
def modulo(input_data: BasicArithmeticInput) -> MathematicalResult:
    """
    Calculate the remainder when a is divided by b with Pydantic validation.
    
    Note: Division by zero is prevented by BasicArithmeticInput validator
    """
    logger.info(f"Computing modulo: {input_data.a} % {input_data.b}")
    result = input_data.a % input_data.b
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed {input_data.a} % {input_data.b} = {result}",
        result=result,
        operation_type="modulo",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(AngleInput)
def sine(input_data: AngleInput) -> MathematicalResult:
    """Calculate the sine of an angle in radians with Pydantic validation."""
    logger.info(f"Computing sine of: {input_data.angle_radians} radians")
    result = float(math.sin(input_data.angle_radians))
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed sin({input_data.angle_radians}) = {result}",
        result=result,
        operation_type="sine",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(AngleInput)
def cosine(input_data: AngleInput) -> MathematicalResult:
    """Calculate the cosine of an angle in radians with Pydantic validation."""
    logger.info(f"Computing cosine of: {input_data.angle_radians} radians")
    result = float(math.cos(input_data.angle_radians))
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed cos({input_data.angle_radians}) = {result}",
        result=result,
        operation_type="cosine",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(AngleInput)
def tangent(input_data: AngleInput) -> MathematicalResult:
    """Calculate the tangent of an angle in radians with Pydantic validation."""
    logger.info(f"Computing tangent of: {input_data.angle_radians} radians")
    result = float(math.tan(input_data.angle_radians))
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed tan({input_data.angle_radians}) = {result}",
        result=result,
        operation_type="tangent",
        timestamp=datetime.now().isoformat()
    )

# =============================================================================
# ADVANCED MATHEMATICAL TOOLS WITH PYDANTIC
# =============================================================================

@mcp.tool()
@validate_input(StringInput)
def string_to_ascii_values(input_data: StringInput) -> MathematicalResult:
    """
    Convert each character in a string to its ASCII value with Pydantic validation.
    
    Args:
        input_data: Validated input containing text string
        
    Returns:
        MathematicalResult with list of ASCII values
    """
    logger.info(f"Converting string '{input_data.text}' to ASCII values")
    result = [ord(char) for char in input_data.text]
    
    return MathematicalResult(
        success=True,
        message=f"Successfully converted '{input_data.text}' to ASCII values",
        result=result,
        operation_type="string_to_ascii",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(NumberListInput)
def exponential_sum(input_data: NumberListInput) -> MathematicalResult:
    """
    Calculate the sum of exponentials of numbers in a list with Pydantic validation.
    
    Args:
        input_data: Validated input containing list of numbers
        
    Returns:
        MathematicalResult with exponential sum
    """
    logger.info(f"Computing exponential sum for: {input_data.numbers}")
    result = sum(math.exp(num) for num in input_data.numbers)
    
    return MathematicalResult(
        success=True,
        message=f"Successfully computed exponential sum of {len(input_data.numbers)} numbers",
        result=result,
        operation_type="exponential_sum",
        timestamp=datetime.now().isoformat()
    )

@mcp.tool()
@validate_input(FibonacciInput)
def fibonacci_sequence(input_data: FibonacciInput) -> MathematicalResult:
    """
    Generate the first n numbers in the Fibonacci sequence with Pydantic validation.
    
    Args:
        input_data: Validated input containing number of Fibonacci numbers to generate
        
    Returns:
        MathematicalResult with Fibonacci sequence
    """
    logger.info(f"Generating first {input_data.n} Fibonacci numbers")
    
    if input_data.n == 0:
        result = []
    elif input_data.n == 1:
        result = [0]
    else:
        sequence = [0, 1]
        for i in range(2, input_data.n):
            sequence.append(sequence[i-1] + sequence[i-2])
        result = sequence
    
    return MathematicalResult(
        success=True,
        message=f"Successfully generated first {input_data.n} Fibonacci numbers",
        result=result,
        operation_type="fibonacci_sequence",
        timestamp=datetime.now().isoformat()
    )

# =============================================================================
# IMAGE PROCESSING TOOLS WITH PYDANTIC
# =============================================================================

@mcp.tool()
@validate_input(ImagePathInput)
def create_thumbnail(input_data: ImagePathInput) -> ImageResult:
    """
    Create a thumbnail from an image file with Pydantic validation.
    
    Args:
        input_data: Validated input containing image file path
        
    Returns:
        ImageResult with thumbnail data
    """
    try:
        logger.info(f"Creating thumbnail from: {input_data.image_path}")
        img = PILImage.open(input_data.image_path)
        img.thumbnail((100, 100), PILImage.Resampling.LANCZOS)
        
        # Get image dimensions
        width, height = img.size
        
        return ImageResult(
            success=True,
            message=f"Successfully created thumbnail from {input_data.image_path}",
            format="PNG",
            dimensions={"width": width, "height": height},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logger.error(error_msg)
        return ImageResult(
            success=False,
            message=error_msg,
            format="unknown",
            timestamp=datetime.now().isoformat()
        )

# =============================================================================
# KEYNOTE AUTOMATION TOOLS WITH PYDANTIC
# =============================================================================

@mcp.tool()
def open_keynote_presentation() -> KeynoteOperationResult:
    """
    Open Keynote application and create a new presentation.
    
    Returns:
        KeynoteOperationResult with operation status
    """
    try:
        logger.info("Opening Keynote and creating new presentation")
        start_time = time.time()
        
        # AppleScript to open Keynote and create new document
        script = '''
        tell application "Keynote"
            activate
            make new document
        end tell
        '''
        
        applescript.run(script)
        time.sleep(2)  # Wait for Keynote to fully load
        
        duration = time.time() - start_time
        
        return KeynoteOperationResult(
            success=True,
            message="Keynote opened successfully with new presentation",
            operation_type="open_presentation",
            duration=duration,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_msg = f"Error opening Keynote: {str(e)}"
        logger.error(error_msg)
        return KeynoteOperationResult(
            success=False,
            message=error_msg,
            operation_type="open_presentation",
            timestamp=datetime.now().isoformat()
        )

@mcp.tool()
@validate_input(KeynoteRectangleInput)
def draw_rectangle_in_keynote(input_data: KeynoteRectangleInput) -> KeynoteOperationResult:
    """
    Draw a rectangle shape in the active Keynote presentation with Pydantic validation.
    
    Args:
        input_data: Validated input containing mouse button preference
        
    Returns:
        KeynoteOperationResult with operation status
    """
    logger.info(f"Drawing rectangle in Keynote using {input_data.button} button")
    start_time = time.time()
    
    try:
        # Get screen information for multi-monitor support
        screens = screeninfo.get_monitors()
        logger.info(f"Detected {len(screens)} monitor(s)")
        
        # Find the screen where Keynote is running
        keynote_screen = None
        for screen in screens:
            logger.debug(f"Screen: {screen.name} - {screen.width}x{screen.height} at ({screen.x}, {screen.y})")
            # Simple heuristic: assume Keynote is on the leftmost screen
            if screen.x < 0:
                keynote_screen = screen
                break
        
        if not keynote_screen:
            logger.warning("Could not determine Keynote screen, using primary screen")
            keynote_screen = screens[0]
        
        # Calculate coordinates for Keynote UI elements
        shapes_button_x = keynote_screen.x + 1045
        shapes_button_y = keynote_screen.y + 40
        rectangle_tool_x = keynote_screen.x + 1025
        rectangle_tool_y = keynote_screen.y + 225
        
        # Rectangle drawing coordinates (center of slide)
        drag_start_x = keynote_screen.x + 940
        drag_start_y = keynote_screen.y + 605
        drag_end_x = keynote_screen.x + 1442
        drag_end_y = keynote_screen.y + 900
        
        coordinates = {
            "shapes_button": {"x": shapes_button_x, "y": shapes_button_y},
            "rectangle_tool": {"x": rectangle_tool_x, "y": rectangle_tool_y},
            "drag_start": {"x": drag_start_x, "y": drag_start_y},
            "drag_end": {"x": drag_end_x, "y": drag_end_y}
        }
        
        # Execute the drawing sequence
        pyautogui.click(shapes_button_x, shapes_button_y, button=input_data.button.value)
        time.sleep(1)
        
        pyautogui.click(rectangle_tool_x, rectangle_tool_y, button=input_data.button.value)
        time.sleep(1)
        
        pyautogui.mouseDown(x=drag_start_x, y=drag_start_y, button=input_data.button.value)
        pyautogui.dragTo(drag_end_x, drag_end_y, duration=0.5, button=input_data.button.value)
        time.sleep(2)
        
        duration = time.time() - start_time
        
        return KeynoteOperationResult(
            success=True,
            message="Rectangle drawn successfully in Keynote",
            operation_type="draw_rectangle",
            coordinates=coordinates,
            duration=duration,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_msg = f"Error drawing rectangle: {str(e)}"
        logger.error(error_msg)
        return KeynoteOperationResult(
            success=False,
            message=error_msg,
            operation_type="draw_rectangle",
            timestamp=datetime.now().isoformat()
        )

@mcp.tool()
@validate_input(KeynoteTextInput)
def add_text_to_keynote(input_data: KeynoteTextInput) -> KeynoteOperationResult:
    """
    Add text to the active Keynote presentation with Pydantic validation.
    
    Args:
        input_data: Validated input containing text content
        
    Returns:
        KeynoteOperationResult with operation status
    """
    logger.info(f"Adding text to Keynote: '{input_data.text}'")
    start_time = time.time()
    
    try:
        # Get screen information
        screens = screeninfo.get_monitors()
        keynote_screen = None
        
        for screen in screens:
            if screen.x < 0:  # Assuming Keynote is on the left screen
                keynote_screen = screen
                break
        
        if not keynote_screen:
            keynote_screen = screens[0]
        
        # Calculate text position (center of the rectangle area)
        text_x = keynote_screen.x + 1191  # Middle point between 940 and 1442
        text_y = keynote_screen.y + 762   # Middle point between 625 and 900
        
        coordinates = {"text_position": {"x": text_x, "y": text_y}}
        
        # Execute text addition sequence
        pyautogui.click(200, 100)  # Click Text button
        time.sleep(1)
        
        pyautogui.click(text_x, text_y)
        time.sleep(1)
        
        pyautogui.write(input_data.text)
        time.sleep(1)
        
        duration = time.time() - start_time
        
        return KeynoteOperationResult(
            success=True,
            message=f"Text '{input_data.text}' added successfully to Keynote",
            operation_type="add_text",
            coordinates=coordinates,
            duration=duration,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_msg = f"Error adding text: {str(e)}"
        logger.error(error_msg)
        return KeynoteOperationResult(
            success=False,
            message=error_msg,
            operation_type="add_text",
            timestamp=datetime.now().isoformat()
        )

# =============================================================================
# MCP RESOURCES
# =============================================================================

@mcp.resource("greeting://{name}")
def get_personalized_greeting(name: str) -> str:
    """
    Generate a personalized greeting message.
    
    Args:
        name: Name to include in the greeting
        
    Returns:
        Personalized greeting string
    """
    logger.info(f"Generating greeting for: {name}")
    return f"Hello, {name}! Welcome to the Keynote AI Automation system with Pydantic validation."

# =============================================================================
# MCP PROMPTS
# =============================================================================

@mcp.prompt()
def code_review_prompt(code: str) -> str:
    """
    Generate a prompt for code review.
    
    Args:
        code: Code to be reviewed
        
    Returns:
        Formatted code review prompt
    """
    return f"""Please review the following code for:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance optimizations
4. Security considerations
5. Documentation completeness
6. Pydantic model usage and validation

Code to review:
```python
{code}
```

Please provide detailed feedback with specific suggestions for improvement."""

@mcp.prompt()
def debug_assistance_prompt(error: str) -> list[base.Message]:
    """
    Generate a structured prompt for debugging assistance.
    
    Args:
        error: Error message or description
        
    Returns:
        List of formatted messages for debugging conversation
    """
    return [
        base.UserMessage("I'm encountering an error and need debugging assistance."),
        base.UserMessage(f"Error details: {error}"),
        base.AssistantMessage("I'll help you debug this issue. Let me ask a few questions to better understand the problem:"),
        base.AssistantMessage("1. What were you trying to accomplish when this error occurred?"),
        base.AssistantMessage("2. What steps did you take before encountering this error?"),
        base.AssistantMessage("3. Are there any error logs or additional context you can share?"),
        base.AssistantMessage("4. What environment are you running this in (OS, Python version, etc.)?"),
        base.AssistantMessage("5. Are you using Pydantic validation? If so, what validation errors are you seeing?")
    ]

# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the MCP server with Pydantic integration.
    
    Supports two modes:
    1. Development mode: Run with 'dev' argument for local testing
    2. Production mode: Run with stdio transport for MCP client integration
    """
    logger.info("Starting Keynote MCP Server with Pydantic Integration")
    
    # Check for development mode
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        logger.info("Running in development mode")
        mcp.run()  # Run without transport for dev server
    else:
        logger.info("Running in production mode with stdio transport")
        mcp.run(transport="stdio")  # Run with stdio for MCP client integration