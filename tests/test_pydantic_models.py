"""
Comprehensive tests for Pydantic models in the Keynote MCP Server.

This module tests all Pydantic models for input validation, output serialization,
and data transfer objects used throughout the MCP server.

Author: Senior AI Engineer Portfolio Project
License: MIT
"""

import pytest
import math
import os
import tempfile
from typing import List, Dict, Any
from pydantic import ValidationError

# Import the models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    # Input models
    BasicArithmeticInput, NumberListInput, SingleNumberInput, StringInput,
    AngleInput, FactorialInput, FibonacciInput, KeynoteRectangleInput,
    KeynoteTextInput, ImagePathInput,
    # Output models
    MathematicalResult, KeynoteOperationResult, ImageResult, OperationResult,
    OperationError, create_error_result, create_success_result,
    # Enums
    MouseButton, LogLevel,
    # Configuration models
    ServerConfig, ClientConfig,
    # Workflow models
    WorkflowStep, WorkflowResult
)


class TestBasicArithmeticInput:
    """Test BasicArithmeticInput model validation."""
    
    def test_valid_inputs(self):
        """Test valid arithmetic inputs."""
        # Valid addition
        input_data = BasicArithmeticInput(a=5, b=3)
        assert input_data.a == 5
        assert input_data.b == 3
        
        # Valid with negative numbers
        input_data = BasicArithmeticInput(a=-5, b=3)
        assert input_data.a == -5
        assert input_data.b == 3
    
    def test_division_by_zero_validation(self):
        """Test that division by zero is caught during validation."""
        with pytest.raises(ValidationError) as exc_info:
            BasicArithmeticInput(a=5, b=0)
        
        errors = exc_info.value.errors()
        assert any(error['type'] == 'value_error' for error in errors)
        assert any('Cannot divide by zero' in str(error['msg']) for error in errors)
    
    def test_range_validation(self):
        """Test that numbers outside valid range are rejected."""
        # Test upper bound
        with pytest.raises(ValidationError):
            BasicArithmeticInput(a=10**11, b=5)
        
        # Test lower bound
        with pytest.raises(ValidationError):
            BasicArithmeticInput(a=-10**11, b=5)


class TestNumberListInput:
    """Test NumberListInput model validation."""
    
    def test_valid_lists(self):
        """Test valid number lists."""
        # Valid list
        input_data = NumberListInput(numbers=[1, 2, 3, 4, 5])
        assert input_data.numbers == [1, 2, 3, 4, 5]
        
        # Single number
        input_data = NumberListInput(numbers=[42])
        assert input_data.numbers == [42]
    
    def test_empty_list_validation(self):
        """Test that empty lists are rejected."""
        with pytest.raises(ValidationError):
            NumberListInput(numbers=[])
    
    def test_list_size_validation(self):
        """Test that lists that are too large are rejected."""
        large_list = list(range(1001))  # 1001 items
        with pytest.raises(ValidationError):
            NumberListInput(numbers=large_list)
    
    def test_number_range_validation(self):
        """Test that numbers outside valid range are rejected."""
        with pytest.raises(ValidationError):
            NumberListInput(numbers=[10**11, 5, 3])


class TestSingleNumberInput:
    """Test SingleNumberInput model validation."""
    
    def test_valid_numbers(self):
        """Test valid single numbers."""
        # Integer
        input_data = SingleNumberInput(number=42)
        assert input_data.number == 42
        
        # Float
        input_data = SingleNumberInput(number=3.14)
        assert input_data.number == 3.14
        
        # Negative number
        input_data = SingleNumberInput(number=-5)
        assert input_data.number == -5
    
    def test_range_validation(self):
        """Test that numbers outside valid range are rejected."""
        with pytest.raises(ValidationError):
            SingleNumberInput(number=10**11)
        
        with pytest.raises(ValidationError):
            SingleNumberInput(number=-10**11)


class TestStringInput:
    """Test StringInput model validation."""
    
    def test_valid_strings(self):
        """Test valid string inputs."""
        input_data = StringInput(text="Hello World")
        assert input_data.text == "Hello World"
        
        input_data = StringInput(text="AI")
        assert input_data.text == "AI"
    
    def test_empty_string_validation(self):
        """Test that empty strings are rejected."""
        with pytest.raises(ValidationError):
            StringInput(text="")
    
    def test_long_string_validation(self):
        """Test that strings that are too long are rejected."""
        long_string = "a" * 1001
        with pytest.raises(ValidationError):
            StringInput(text=long_string)
    
    def test_non_printable_character_validation(self):
        """Test that non-printable characters are rejected."""
        with pytest.raises(ValidationError):
            StringInput(text="Hello\x00World")  # Contains null character


class TestAngleInput:
    """Test AngleInput model validation."""
    
    def test_valid_angles(self):
        """Test valid angle inputs."""
        input_data = AngleInput(angle_radians=0)
        assert input_data.angle_radians == 0
        
        input_data = AngleInput(angle_radians=math.pi/2)
        assert input_data.angle_radians == math.pi/2
        
        input_data = AngleInput(angle_radians=-math.pi)
        assert input_data.angle_radians == -math.pi
    
    def test_angle_range_validation(self):
        """Test that angles outside valid range are rejected."""
        with pytest.raises(ValidationError):
            AngleInput(angle_radians=3*math.pi)
        
        with pytest.raises(ValidationError):
            AngleInput(angle_radians=-3*math.pi)


class TestFactorialInput:
    """Test FactorialInput model validation."""
    
    def test_valid_factorial_inputs(self):
        """Test valid factorial inputs."""
        input_data = FactorialInput(n=0)
        assert input_data.n == 0
        
        input_data = FactorialInput(n=5)
        assert input_data.n == 5
        
        input_data = FactorialInput(n=20)
        assert input_data.n == 20
    
    def test_negative_factorial_validation(self):
        """Test that negative factorial inputs are rejected."""
        with pytest.raises(ValidationError):
            FactorialInput(n=-1)
    
    def test_large_factorial_validation(self):
        """Test that large factorial inputs are rejected."""
        with pytest.raises(ValidationError):
            FactorialInput(n=21)


class TestFibonacciInput:
    """Test FibonacciInput model validation."""
    
    def test_valid_fibonacci_inputs(self):
        """Test valid Fibonacci inputs."""
        input_data = FibonacciInput(n=0)
        assert input_data.n == 0
        
        input_data = FibonacciInput(n=10)
        assert input_data.n == 10
        
        input_data = FibonacciInput(n=100)
        assert input_data.n == 100
    
    def test_negative_fibonacci_validation(self):
        """Test that negative Fibonacci inputs are rejected."""
        with pytest.raises(ValidationError):
            FibonacciInput(n=-1)
    
    def test_large_fibonacci_validation(self):
        """Test that large Fibonacci inputs are rejected."""
        with pytest.raises(ValidationError):
            FibonacciInput(n=101)


class TestKeynoteInputs:
    """Test Keynote automation input models."""
    
    def test_keynote_rectangle_input(self):
        """Test KeynoteRectangleInput validation."""
        input_data = KeynoteRectangleInput(button=MouseButton.LEFT)
        assert input_data.button == MouseButton.LEFT
        
        input_data = KeynoteRectangleInput(button=MouseButton.RIGHT)
        assert input_data.button == MouseButton.RIGHT
    
    def test_keynote_text_input(self):
        """Test KeynoteTextInput validation."""
        input_data = KeynoteTextInput(text="Hello World")
        assert input_data.text == "Hello World"
        
        # Test long text
        long_text = "a" * 500
        input_data = KeynoteTextInput(text=long_text)
        assert input_data.text == long_text
    
    def test_keynote_text_length_validation(self):
        """Test that text that is too long is rejected."""
        long_text = "a" * 501
        with pytest.raises(ValidationError):
            KeynoteTextInput(text=long_text)


class TestImagePathInput:
    """Test ImagePathInput model validation."""
    
    def test_valid_image_path(self):
        """Test valid image path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'fake image data')
            tmp_path = tmp.name
        
        try:
            input_data = ImagePathInput(image_path=tmp_path)
            assert input_data.image_path == tmp_path
        finally:
            os.unlink(tmp_path)
    
    def test_nonexistent_file_validation(self):
        """Test that nonexistent files are rejected."""
        with pytest.raises(ValidationError):
            ImagePathInput(image_path="/nonexistent/file.png")
    
    def test_invalid_extension_validation(self):
        """Test that invalid file extensions are rejected."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'not an image')
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValidationError):
                ImagePathInput(image_path=tmp_path)
        finally:
            os.unlink(tmp_path)


class TestOutputModels:
    """Test output model serialization and validation."""
    
    def test_mathematical_result(self):
        """Test MathematicalResult model."""
        result = MathematicalResult(
            success=True,
            message="Test operation completed",
            result=42,
            operation_type="addition"
        )
        
        assert result.success is True
        assert result.message == "Test operation completed"
        assert result.result == 42
        assert result.operation_type == "addition"
        assert result.timestamp is not None
    
    def test_keynote_operation_result(self):
        """Test KeynoteOperationResult model."""
        result = KeynoteOperationResult(
            success=True,
            message="Rectangle drawn successfully",
            operation_type="draw_rectangle",
            coordinates={"x": 100, "y": 200},
            duration=1.5
        )
        
        assert result.success is True
        assert result.message == "Rectangle drawn successfully"
        assert result.operation_type == "draw_rectangle"
        assert result.coordinates == {"x": 100, "y": 200}
        assert result.duration == 1.5
    
    def test_image_result(self):
        """Test ImageResult model."""
        result = ImageResult(
            success=True,
            message="Thumbnail created successfully",
            format="PNG",
            dimensions={"width": 100, "height": 100}
        )
        
        assert result.success is True
        assert result.message == "Thumbnail created successfully"
        assert result.format == "PNG"
        assert result.dimensions == {"width": 100, "height": 100}


class TestErrorModels:
    """Test error model creation and validation."""
    
    def test_operation_error(self):
        """Test OperationError model."""
        error = OperationError(
            error_type="validation_error",
            message="Invalid input provided",
            details={"field": "number", "value": -1}
        )
        
        assert error.error_type == "validation_error"
        assert error.message == "Invalid input provided"
        assert error.details == {"field": "number", "value": -1}
        assert error.timestamp is not None
    
    def test_create_error_result(self):
        """Test create_error_result utility function."""
        error = create_error_result(
            error_type="test_error",
            message="Test error message",
            details={"test": "data"}
        )
        
        assert error.error_type == "test_error"
        assert error.message == "Test error message"
        assert error.details == {"test": "data"}
        assert error.timestamp is not None
    
    def test_create_success_result(self):
        """Test create_success_result utility function."""
        result = create_success_result("Test operation completed")
        
        assert result.success is True
        assert result.message == "Test operation completed"
        assert result.timestamp is not None


class TestConfigurationModels:
    """Test configuration model validation."""
    
    def test_server_config(self):
        """Test ServerConfig model."""
        config = ServerConfig(
            server_name="Test Server",
            version="2.0.0",
            max_iterations=10,
            llm_timeout=15,
            keynote_delay=3.0,
            log_level=LogLevel.DEBUG,
            debug_mode=True
        )
        
        assert config.server_name == "Test Server"
        assert config.version == "2.0.0"
        assert config.max_iterations == 10
        assert config.llm_timeout == 15
        assert config.keynote_delay == 3.0
        assert config.log_level == LogLevel.DEBUG
        assert config.debug_mode is True
    
    def test_server_config_validation(self):
        """Test ServerConfig validation."""
        # Test invalid max_iterations
        with pytest.raises(ValidationError):
            ServerConfig(max_iterations=0)
        
        with pytest.raises(ValidationError):
            ServerConfig(max_iterations=25)
        
        # Test invalid llm_timeout
        with pytest.raises(ValidationError):
            ServerConfig(llm_timeout=0)
        
        with pytest.raises(ValidationError):
            ServerConfig(llm_timeout=65)
    
    def test_client_config(self):
        """Test ClientConfig model."""
        config = ClientConfig(
            api_key="test-api-key",
            model_name="gemini-2.0-flash",
            max_retries=5,
            timeout=20
        )
        
        assert config.api_key == "test-api-key"
        assert config.model_name == "gemini-2.0-flash"
        assert config.max_retries == 5
        assert config.timeout == 20


class TestWorkflowModels:
    """Test workflow model validation."""
    
    def test_workflow_step(self):
        """Test WorkflowStep model."""
        step = WorkflowStep(
            step_number=1,
            operation="add",
            parameters={"a": 5, "b": 3},
            status="completed",
            result=8
        )
        
        assert step.step_number == 1
        assert step.operation == "add"
        assert step.parameters == {"a": 5, "b": 3}
        assert step.status == "completed"
        assert step.result == 8
    
    def test_workflow_result(self):
        """Test WorkflowResult model."""
        steps = [
            WorkflowStep(step_number=1, operation="add", parameters={"a": 5, "b": 3}, result=8),
            WorkflowStep(step_number=2, operation="multiply", parameters={"a": 8, "b": 2}, result=16)
        ]
        
        workflow = WorkflowResult(
            workflow_id="test-workflow-123",
            total_steps=2,
            completed_steps=2,
            success=True,
            final_result=16,
            steps=steps,
            duration=2.5
        )
        
        assert workflow.workflow_id == "test-workflow-123"
        assert workflow.total_steps == 2
        assert workflow.completed_steps == 2
        assert workflow.success is True
        assert workflow.final_result == 16
        assert len(workflow.steps) == 2
        assert workflow.duration == 2.5


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_json_serialization(self):
        """Test that models can be serialized to JSON."""
        input_data = BasicArithmeticInput(a=5, b=3)
        json_data = input_data.model_dump()
        
        assert json_data["a"] == 5
        assert json_data["b"] == 3
    
    def test_json_deserialization(self):
        """Test that models can be deserialized from JSON."""
        json_data = {"a": 5, "b": 3}
        input_data = BasicArithmeticInput(**json_data)
        
        assert input_data.a == 5
        assert input_data.b == 3
    
    def test_result_serialization(self):
        """Test that result models serialize correctly."""
        result = MathematicalResult(
            success=True,
            message="Test completed",
            result=42.123456789,
            operation_type="test"
        )
        
        json_data = result.model_dump()
        assert json_data["success"] is True
        assert json_data["message"] == "Test completed"
        assert json_data["result"] == 42.123456789  # Should be rounded by custom encoder
        assert json_data["operation_type"] == "test"


class TestModelRegistry:
    """Test the model registry functionality."""
    
    def test_input_models_registry(self):
        """Test that input models are properly registered."""
        from models import INPUT_MODELS
        
        # Test that key models are registered
        assert "add" in INPUT_MODELS
        assert "subtract" in INPUT_MODELS
        assert "multiply" in INPUT_MODELS
        assert "divide" in INPUT_MODELS
        assert "string_to_ascii_values" in INPUT_MODELS
        assert "exponential_sum" in INPUT_MODELS
        assert "draw_rectangle_in_keynote" in INPUT_MODELS
        assert "add_text_to_keynote" in INPUT_MODELS
    
    def test_output_models_registry(self):
        """Test that output models are properly registered."""
        from models import OUTPUT_MODELS
        
        # Test that key models are registered
        assert "mathematical" in OUTPUT_MODELS
        assert "keynote" in OUTPUT_MODELS
        assert "image" in OUTPUT_MODELS
        assert "general" in OUTPUT_MODELS


if __name__ == "__main__":
    pytest.main([__file__])