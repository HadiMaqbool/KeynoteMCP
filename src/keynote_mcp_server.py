"""
Keynote MCP Server - AI-Powered Presentation Automation

This MCP (Model Context Protocol) server provides AI agents with the ability to:
- Perform mathematical computations
- Automate Keynote presentation creation
- Generate visual content programmatically
- Integrate LLM capabilities with desktop applications

Author: Senior AI Engineer Portfolio Project
License: MIT
"""

import math
import sys
import time
from typing import Dict, List, Union, Any
import logging

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Keynote-AI-Automation")

# =============================================================================
# MATHEMATICAL COMPUTATION TOOLS
# =============================================================================

@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two integers.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Sum of a and b
    """
    logger.info(f"Computing addition: {a} + {b}")
    return int(a + b)

@mcp.tool()
def add_list(numbers: List[int]) -> int:
    """
    Calculate the sum of all numbers in a list.
    
    Args:
        numbers: List of integers to sum
        
    Returns:
        Sum of all numbers in the list
    """
    logger.info(f"Computing sum of list: {numbers}")
    return sum(numbers)

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    logger.info(f"Computing subtraction: {a} - {b}")
    return int(a - b)

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    logger.info(f"Computing multiplication: {a} * {b}")
    return int(a * b)

@mcp.tool()
def divide(a: int, b: int) -> float:
    """
    Divide a by b.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Result of division as float
        
    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    logger.info(f"Computing division: {a} / {b}")
    return float(a / b)

@mcp.tool()
def power(base: int, exponent: int) -> int:
    """Calculate base raised to the power of exponent."""
    logger.info(f"Computing power: {base}^{exponent}")
    return int(base ** exponent)

@mcp.tool()
def square_root(number: int) -> float:
    """Calculate the square root of a number."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    logger.info(f"Computing square root of: {number}")
    return float(number ** 0.5)

@mcp.tool()
def cube_root(number: int) -> float:
    """Calculate the cube root of a number."""
    logger.info(f"Computing cube root of: {number}")
    return float(number ** (1/3))

@mcp.tool()
def factorial(n: int) -> int:
    """
    Calculate the factorial of a number.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    logger.info(f"Computing factorial of: {n}")
    return int(math.factorial(n))

@mcp.tool()
def natural_log(number: int) -> float:
    """
    Calculate the natural logarithm of a number.
    
    Args:
        number: Positive number
        
    Returns:
        Natural logarithm of the number
        
    Raises:
        ValueError: If number is not positive
    """
    if number <= 0:
        raise ValueError("Logarithm is not defined for non-positive numbers")
    logger.info(f"Computing natural log of: {number}")
    return float(math.log(number))

@mcp.tool()
def modulo(a: int, b: int) -> int:
    """Calculate the remainder when a is divided by b."""
    if b == 0:
        raise ZeroDivisionError("Cannot compute modulo with zero divisor")
    logger.info(f"Computing modulo: {a} % {b}")
    return int(a % b)

@mcp.tool()
def sine(angle_radians: float) -> float:
    """Calculate the sine of an angle in radians."""
    logger.info(f"Computing sine of: {angle_radians} radians")
    return float(math.sin(angle_radians))

@mcp.tool()
def cosine(angle_radians: float) -> float:
    """Calculate the cosine of an angle in radians."""
    logger.info(f"Computing cosine of: {angle_radians} radians")
    return float(math.cos(angle_radians))

@mcp.tool()
def tangent(angle_radians: float) -> float:
    """Calculate the tangent of an angle in radians."""
    logger.info(f"Computing tangent of: {angle_radians} radians")
    return float(math.tan(angle_radians))

# =============================================================================
# ADVANCED MATHEMATICAL TOOLS
# =============================================================================

@mcp.tool()
def string_to_ascii_values(text: str) -> List[int]:
    """
    Convert each character in a string to its ASCII value.
    
    Args:
        text: Input string
        
    Returns:
        List of ASCII values for each character
    """
    logger.info(f"Converting string '{text}' to ASCII values")
    return [ord(char) for char in text]

@mcp.tool()
def exponential_sum(numbers: List[int]) -> float:
    """
    Calculate the sum of exponentials of numbers in a list.
    
    Args:
        numbers: List of numbers to compute exponentials for
        
    Returns:
        Sum of e^number for each number in the list
    """
    logger.info(f"Computing exponential sum for: {numbers}")
    return sum(math.exp(num) for num in numbers)

@mcp.tool()
def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate the first n numbers in the Fibonacci sequence.
    
    Args:
        n: Number of Fibonacci numbers to generate
        
    Returns:
        List of first n Fibonacci numbers
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Cannot generate negative number of Fibonacci numbers")
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    logger.info(f"Generating first {n} Fibonacci numbers")
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence

# =============================================================================
# IMAGE PROCESSING TOOLS
# =============================================================================

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """
    Create a thumbnail from an image file.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Thumbnail image as MCP Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    try:
        logger.info(f"Creating thumbnail from: {image_path}")
        img = PILImage.open(image_path)
        img.thumbnail((100, 100), PILImage.Resampling.LANCZOS)
        return Image(data=img.tobytes(), format="PNG")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# =============================================================================
# KEYNOTE AUTOMATION TOOLS
# =============================================================================

@mcp.tool()
def open_keynote_presentation() -> Dict[str, Any]:
    """
    Open Keynote application and create a new presentation.
    
    Returns:
        Dictionary with operation status and message
        
    Note:
        Requires Keynote to be installed on macOS
    """
    try:
        logger.info("Opening Keynote and creating new presentation")
        
        # AppleScript to open Keynote and create new document
        script = '''
        tell application "Keynote"
            activate
            make new document
        end tell
        '''
        
        applescript.run(script)
        time.sleep(2)  # Wait for Keynote to fully load
        
        return {
            "success": True,
            "message": "Keynote opened successfully with new presentation",
            "content": [
                TextContent(
                    type="text",
                    text="Keynote opened successfully with new presentation"
                )
            ]
        }
    except Exception as e:
        error_msg = f"Error opening Keynote: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "content": [
                TextContent(
                    type="text",
                    text=error_msg
                )
            ]
        }

@mcp.tool()
def draw_rectangle_in_keynote(button: str = "left") -> Dict[str, Any]:
    """
    Draw a rectangle shape in the active Keynote presentation.
    
    Args:
        button: Mouse button to use ("left", "middle", "right")
        
    Returns:
        Dictionary with operation status and message
        
    Note:
        Requires Keynote to be open and active
    """
    logger.info(f"Drawing rectangle in Keynote using {button} button")
    
    try:
        # Validate button parameter
        button = button.strip('"\'')
        if button not in ["left", "middle", "right"]:
            raise ValueError(f"Invalid button: {button}. Must be 'left', 'middle', or 'right'")
        
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
        # These coordinates are calibrated for Keynote's default interface
        shapes_button_x = keynote_screen.x + 1045
        shapes_button_y = keynote_screen.y + 40
        rectangle_tool_x = keynote_screen.x + 1025
        rectangle_tool_y = keynote_screen.y + 225
        
        # Rectangle drawing coordinates (center of slide)
        drag_start_x = keynote_screen.x + 940
        drag_start_y = keynote_screen.y + 605
        drag_end_x = keynote_screen.x + 1442
        drag_end_y = keynote_screen.y + 900
        
        logger.debug(f"Using screen: {keynote_screen.name}")
        logger.debug(f"Shapes button: ({shapes_button_x}, {shapes_button_y})")
        logger.debug(f"Rectangle tool: ({rectangle_tool_x}, {rectangle_tool_y})")
        logger.debug(f"Drawing rectangle from ({drag_start_x}, {drag_start_y}) to ({drag_end_x}, {drag_end_y})")
        
        # Execute the drawing sequence
        # Step 1: Click Shapes button
        pyautogui.click(shapes_button_x, shapes_button_y, button=button)
        time.sleep(1)
        
        # Step 2: Click Rectangle tool
        pyautogui.click(rectangle_tool_x, rectangle_tool_y, button=button)
        time.sleep(1)
        
        # Step 3: Draw rectangle by dragging
        pyautogui.mouseDown(x=drag_start_x, y=drag_start_y, button=button)
        pyautogui.dragTo(drag_end_x, drag_end_y, duration=0.5, button=button)
        time.sleep(2)  # Wait for rectangle to be created
        
        return {
            "success": True,
            "message": "Rectangle drawn successfully in Keynote",
            "content": [
                TextContent(
                    type="text",
                    text="Rectangle drawn successfully in Keynote"
                )
            ]
        }
        
    except Exception as e:
        error_msg = f"Error drawing rectangle: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

@mcp.tool()
def add_text_to_keynote(text: str) -> Dict[str, Any]:
    """
    Add text to the active Keynote presentation at the rectangle's position.
    
    Args:
        text: Text content to add
        
    Returns:
        Dictionary with operation status and message
        
    Note:
        Requires Keynote to be open and a rectangle to be present
    """
    logger.info(f"Adding text to Keynote: '{text}'")
    
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
        
        logger.debug(f"Adding text at position: ({text_x}, {text_y})")
        
        # Execute text addition sequence
        # Step 1: Click Text button (assuming it's at a standard position)
        pyautogui.click(200, 100)
        time.sleep(1)
        
        # Step 2: Click at the calculated position
        pyautogui.click(text_x, text_y)
        time.sleep(1)
        
        # Step 3: Type the text
        pyautogui.write(text)
        time.sleep(1)
        
        return {
            "success": True,
            "message": f"Text '{text}' added successfully to Keynote",
            "content": [
                TextContent(
                    type="text",
                    text=f"Text '{text}' added successfully to Keynote"
                )
            ]
        }
        
    except Exception as e:
        error_msg = f"Error adding text: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

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
    return f"Hello, {name}! Welcome to the Keynote AI Automation system."

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

Code to review:
```python
{code}
```

Please provide detailed feedback with specific suggestions for improvement."""

@mcp.prompt()
def debug_assistance_prompt(error: str) -> List[base.Message]:
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
        base.AssistantMessage("4. What environment are you running this in (OS, Python version, etc.)?")
    ]

# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the MCP server.
    
    Supports two modes:
    1. Development mode: Run with 'dev' argument for local testing
    2. Production mode: Run with stdio transport for MCP client integration
    """
    logger.info("Starting Keynote MCP Server")
    
    # Check for development mode
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        logger.info("Running in development mode")
        mcp.run()  # Run without transport for dev server
    else:
        logger.info("Running in production mode with stdio transport")
        mcp.run(transport="stdio")  # Run with stdio for MCP client integration