"""
Unit tests for mathematical computation tools in the Keynote MCP Server.

This module tests the mathematical functionality provided by the MCP server,
ensuring accuracy and proper error handling for all mathematical operations.
"""

import pytest
import math
from unittest.mock import patch, MagicMock

# Import the MCP server module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from keynote_mcp_server import (
    add, subtract, multiply, divide, power, square_root, cube_root,
    factorial, natural_log, modulo, sine, cosine, tangent,
    string_to_ascii_values, exponential_sum, fibonacci_sequence
)


class TestBasicArithmetic:
    """Test basic arithmetic operations."""
    
    def test_add_positive_numbers(self):
        """Test addition of positive numbers."""
        assert add(5, 3) == 8
        assert add(0, 0) == 0
        assert add(100, 200) == 300
    
    def test_add_negative_numbers(self):
        """Test addition with negative numbers."""
        assert add(-5, -3) == -8
        assert add(-5, 3) == -2
        assert add(5, -3) == 2
    
    def test_subtract_numbers(self):
        """Test subtraction operations."""
        assert subtract(10, 3) == 7
        assert subtract(0, 5) == -5
        assert subtract(-5, -3) == -2
    
    def test_multiply_numbers(self):
        """Test multiplication operations."""
        assert multiply(4, 5) == 20
        assert multiply(0, 100) == 0
        assert multiply(-3, 4) == -12
    
    def test_divide_numbers(self):
        """Test division operations."""
        assert divide(10, 2) == 5.0
        assert divide(7, 3) == pytest.approx(2.3333333333333335)
        assert divide(-10, 2) == -5.0
    
    def test_divide_by_zero(self):
        """Test division by zero raises appropriate error."""
        with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
            divide(5, 0)


class TestAdvancedMathematical:
    """Test advanced mathematical operations."""
    
    def test_power_operations(self):
        """Test power calculations."""
        assert power(2, 3) == 8
        assert power(5, 0) == 1
        assert power(3, 2) == 9
    
    def test_square_root(self):
        """Test square root calculations."""
        assert square_root(16) == 4.0
        assert square_root(25) == 5.0
        assert square_root(0) == 0.0
    
    def test_square_root_negative(self):
        """Test square root of negative number raises error."""
        with pytest.raises(ValueError, match="Cannot calculate square root of negative number"):
            square_root(-4)
    
    def test_cube_root(self):
        """Test cube root calculations."""
        assert cube_root(8) == pytest.approx(2.0)
        assert cube_root(27) == pytest.approx(3.0)
        assert cube_root(0) == 0.0
    
    def test_factorial(self):
        """Test factorial calculations."""
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(10) == 3628800
    
    def test_factorial_negative(self):
        """Test factorial of negative number raises error."""
        with pytest.raises(ValueError, match="Factorial is not defined for negative numbers"):
            factorial(-1)
    
    def test_natural_log(self):
        """Test natural logarithm calculations."""
        assert natural_log(1) == 0.0
        assert natural_log(math.e) == pytest.approx(1.0)
        assert natural_log(10) == pytest.approx(2.302585092994046)
    
    def test_natural_log_non_positive(self):
        """Test natural log of non-positive number raises error."""
        with pytest.raises(ValueError, match="Logarithm is not defined for non-positive numbers"):
            natural_log(0)
        
        with pytest.raises(ValueError, match="Logarithm is not defined for non-positive numbers"):
            natural_log(-5)
    
    def test_modulo(self):
        """Test modulo operations."""
        assert modulo(10, 3) == 1
        assert modulo(15, 5) == 0
        assert modulo(7, 2) == 1
    
    def test_modulo_by_zero(self):
        """Test modulo by zero raises error."""
        with pytest.raises(ZeroDivisionError, match="Cannot compute modulo with zero divisor"):
            modulo(5, 0)


class TestTrigonometric:
    """Test trigonometric functions."""
    
    def test_sine(self):
        """Test sine calculations."""
        assert sine(0) == 0.0
        assert sine(math.pi/2) == pytest.approx(1.0)
        assert sine(math.pi) == pytest.approx(0.0)
    
    def test_cosine(self):
        """Test cosine calculations."""
        assert cosine(0) == 1.0
        assert cosine(math.pi/2) == pytest.approx(0.0)
        assert cosine(math.pi) == pytest.approx(-1.0)
    
    def test_tangent(self):
        """Test tangent calculations."""
        assert tangent(0) == 0.0
        assert tangent(math.pi/4) == pytest.approx(1.0)


class TestStringProcessing:
    """Test string processing functions."""
    
    def test_string_to_ascii_values(self):
        """Test string to ASCII conversion."""
        assert string_to_ascii_values("A") == [65]
        assert string_to_ascii_values("AB") == [65, 66]
        assert string_to_ascii_values("Hello") == [72, 101, 108, 108, 111]
        assert string_to_ascii_values("") == []
    
    def test_exponential_sum(self):
        """Test exponential sum calculations."""
        assert exponential_sum([0]) == pytest.approx(1.0)
        assert exponential_sum([1]) == pytest.approx(math.e)
        assert exponential_sum([0, 1]) == pytest.approx(1.0 + math.e)
        assert exponential_sum([1, 2, 3]) == pytest.approx(math.e + math.e**2 + math.e**3)


class TestFibonacci:
    """Test Fibonacci sequence generation."""
    
    def test_fibonacci_sequence(self):
        """Test Fibonacci sequence generation."""
        assert fibonacci_sequence(0) == []
        assert fibonacci_sequence(1) == [0]
        assert fibonacci_sequence(2) == [0, 1]
        assert fibonacci_sequence(5) == [0, 1, 1, 2, 3]
        assert fibonacci_sequence(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    
    def test_fibonacci_negative(self):
        """Test Fibonacci with negative input raises error."""
        with pytest.raises(ValueError, match="Cannot generate negative number of Fibonacci numbers"):
            fibonacci_sequence(-1)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_large_numbers(self):
        """Test operations with large numbers."""
        # Test that large numbers don't cause overflow issues
        assert add(1000000, 2000000) == 3000000
        assert multiply(1000, 1000) == 1000000
    
    def test_floating_point_precision(self):
        """Test floating point precision handling."""
        result = divide(1, 3)
        assert isinstance(result, float)
        assert result == pytest.approx(0.3333333333333333)
    
    def test_type_coercion(self):
        """Test that functions handle type coercion properly."""
        # These should work with the type hints, but test the actual behavior
        assert add(5, 3) == 8
        assert multiply(4, 5) == 20


if __name__ == "__main__":
    pytest.main([__file__])