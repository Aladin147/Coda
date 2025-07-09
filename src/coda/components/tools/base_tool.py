"""
Base tool implementation for Coda.

This module provides the BaseTool class that all tools should inherit from,
along with common utilities and exceptions.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from abc import abstractmethod

from .interfaces import ToolInterface
from .models import ToolDefinition, ToolParameter, ToolCategory, ParameterType

logger = logging.getLogger("coda.tools.base")


class ToolError(Exception):
    """Base exception for tool errors."""
    
    def __init__(self, message: str, tool_name: str = "", recoverable: bool = True):
        super().__init__(message)
        self.tool_name = tool_name
        self.recoverable = recoverable


class ToolValidationError(ToolError):
    """Exception for parameter validation errors."""
    
    def __init__(self, message: str, parameter_name: str = "", tool_name: str = ""):
        super().__init__(message, tool_name, recoverable=True)
        self.parameter_name = parameter_name


class ToolExecutionError(ToolError):
    """Exception for tool execution errors."""
    
    def __init__(self, message: str, tool_name: str = "", recoverable: bool = False):
        super().__init__(message, tool_name, recoverable)


class ToolTimeoutError(ToolError):
    """Exception for tool execution timeouts."""
    
    def __init__(self, message: str, tool_name: str = "", timeout_seconds: float = 0):
        super().__init__(message, tool_name, recoverable=True)
        self.timeout_seconds = timeout_seconds


class BaseTool(ToolInterface):
    """
    Base class for all tools.
    
    Provides common functionality including:
    - Parameter validation
    - Error handling
    - Logging
    - Availability checking
    - Resource cleanup
    """
    
    def __init__(self):
        """Initialize the base tool."""
        self._definition: Optional[ToolDefinition] = None
        self._is_available = True
        self._execution_count = 0
        self._last_error: Optional[str] = None
        
        # Initialize definition
        self._definition = self._create_definition()
        
        logger.info(f"Initialized tool: {self._definition.name}")
    
    @abstractmethod
    def _create_definition(self) -> ToolDefinition:
        """Create the tool definition. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _execute_impl(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the tool implementation. Must be implemented by subclasses."""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """Get the tool definition."""
        return self._definition
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Tool parameters
            context: Optional execution context
            
        Returns:
            Tool execution result
            
        Raises:
            ToolValidationError: If parameters are invalid
            ToolExecutionError: If execution fails
            ToolTimeoutError: If execution times out
        """
        if not self.is_available():
            raise ToolExecutionError(f"Tool {self._definition.name} is not available", self._definition.name)
        
        try:
            # Validate parameters
            validated_params = await self.validate_parameters(parameters)
            
            # Execute with timeout
            timeout = self._definition.timeout_seconds
            result = await asyncio.wait_for(
                self._execute_impl(validated_params, context),
                timeout=timeout
            )
            
            self._execution_count += 1
            self._last_error = None
            
            logger.debug(f"Tool {self._definition.name} executed successfully")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {timeout} seconds"
            self._last_error = error_msg
            raise ToolTimeoutError(error_msg, self._definition.name, timeout)
            
        except ToolError:
            # Re-raise tool-specific errors
            raise
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            self._last_error = error_msg
            logger.error(f"Tool {self._definition.name} execution failed: {e}")
            raise ToolExecutionError(error_msg, self._definition.name)
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize parameters.
        
        Args:
            parameters: Raw parameters
            
        Returns:
            Validated and normalized parameters
            
        Raises:
            ToolValidationError: If validation fails
        """
        try:
            return self._definition.validate_parameters(parameters)
        except ValueError as e:
            raise ToolValidationError(str(e), tool_name=self._definition.name)
    
    def is_available(self) -> bool:
        """Check if the tool is currently available."""
        return self._is_available
    
    def set_available(self, available: bool) -> None:
        """Set tool availability."""
        self._is_available = available
        if not available:
            logger.warning(f"Tool {self._definition.name} marked as unavailable")
    
    async def cleanup(self) -> None:
        """Cleanup resources used by the tool."""
        logger.debug(f"Cleaning up tool: {self._definition.name}")
        # Base implementation does nothing
        pass
    
    def get_execution_count(self) -> int:
        """Get the number of times this tool has been executed."""
        return self._execution_count
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_count = 0
        self._last_error = None
    
    # Utility methods for creating common parameter types
    
    @staticmethod
    def create_string_parameter(name: str, description: str, required: bool = True,
                              default: Optional[str] = None, enum: Optional[List[str]] = None,
                              pattern: Optional[str] = None, examples: Optional[List[str]] = None) -> ToolParameter:
        """Create a string parameter."""
        return ToolParameter(
            name=name,
            type=ParameterType.STRING,
            description=description,
            required=required,
            default=default,
            enum=enum,
            pattern=pattern,
            examples=examples or []
        )
    
    @staticmethod
    def create_integer_parameter(name: str, description: str, required: bool = True,
                               default: Optional[int] = None, minimum: Optional[int] = None,
                               maximum: Optional[int] = None, examples: Optional[List[int]] = None) -> ToolParameter:
        """Create an integer parameter."""
        return ToolParameter(
            name=name,
            type=ParameterType.INTEGER,
            description=description,
            required=required,
            default=default,
            minimum=minimum,
            maximum=maximum,
            examples=examples or []
        )
    
    @staticmethod
    def create_float_parameter(name: str, description: str, required: bool = True,
                             default: Optional[float] = None, minimum: Optional[float] = None,
                             maximum: Optional[float] = None, examples: Optional[List[float]] = None) -> ToolParameter:
        """Create a float parameter."""
        return ToolParameter(
            name=name,
            type=ParameterType.FLOAT,
            description=description,
            required=required,
            default=default,
            minimum=minimum,
            maximum=maximum,
            examples=examples or []
        )
    
    @staticmethod
    def create_boolean_parameter(name: str, description: str, required: bool = True,
                               default: Optional[bool] = None, examples: Optional[List[bool]] = None) -> ToolParameter:
        """Create a boolean parameter."""
        return ToolParameter(
            name=name,
            type=ParameterType.BOOLEAN,
            description=description,
            required=required,
            default=default,
            examples=examples or []
        )
    
    @staticmethod
    def create_array_parameter(name: str, description: str, required: bool = True,
                             default: Optional[List[Any]] = None, examples: Optional[List[List[Any]]] = None) -> ToolParameter:
        """Create an array parameter."""
        return ToolParameter(
            name=name,
            type=ParameterType.ARRAY,
            description=description,
            required=required,
            default=default,
            examples=examples or []
        )
    
    @staticmethod
    def create_object_parameter(name: str, description: str, required: bool = True,
                              default: Optional[Dict[str, Any]] = None, examples: Optional[List[Dict[str, Any]]] = None) -> ToolParameter:
        """Create an object parameter."""
        return ToolParameter(
            name=name,
            type=ParameterType.OBJECT,
            description=description,
            required=required,
            default=default,
            examples=examples or []
        )
    
    # Utility methods for common validation patterns
    
    def validate_string_not_empty(self, value: str, parameter_name: str) -> str:
        """Validate that a string is not empty."""
        if not value or not value.strip():
            raise ToolValidationError(f"Parameter '{parameter_name}' cannot be empty", parameter_name, self._definition.name)
        return value.strip()
    
    def validate_positive_number(self, value, parameter_name: str):
        """Validate that a number is positive."""
        if value <= 0:
            raise ToolValidationError(f"Parameter '{parameter_name}' must be positive", parameter_name, self._definition.name)
        return value
    
    def validate_range(self, value, min_val, max_val, parameter_name: str):
        """Validate that a number is within a range."""
        if not (min_val <= value <= max_val):
            raise ToolValidationError(
                f"Parameter '{parameter_name}' must be between {min_val} and {max_val}",
                parameter_name, self._definition.name
            )
        return value
    
    def validate_list_not_empty(self, value: List[Any], parameter_name: str) -> List[Any]:
        """Validate that a list is not empty."""
        if not value:
            raise ToolValidationError(f"Parameter '{parameter_name}' cannot be empty", parameter_name, self._definition.name)
        return value
    
    # Context utilities
    
    def get_context_value(self, context: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
        """Get a value from execution context."""
        if context is None:
            return default
        return context.get(key, default)
    
    def has_context_value(self, context: Optional[Dict[str, Any]], key: str) -> bool:
        """Check if context has a specific key."""
        if context is None:
            return False
        return key in context
    
    # Logging utilities
    
    def log_info(self, message: str) -> None:
        """Log an info message with tool name."""
        logger.info(f"[{self._definition.name}] {message}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message with tool name."""
        logger.warning(f"[{self._definition.name}] {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error message with tool name."""
        logger.error(f"[{self._definition.name}] {message}")
    
    def log_debug(self, message: str) -> None:
        """Log a debug message with tool name."""
        logger.debug(f"[{self._definition.name}] {message}")


# Utility functions for tool development

def create_simple_tool_definition(name: str, description: str, category: ToolCategory = ToolCategory.UTILITY,
                                parameters: Optional[List[ToolParameter]] = None,
                                examples: Optional[List[Dict[str, Any]]] = None,
                                tags: Optional[List[str]] = None,
                                timeout_seconds: float = 30.0) -> ToolDefinition:
    """Create a simple tool definition with common defaults."""
    return ToolDefinition(
        name=name,
        description=description,
        category=category,
        parameters=parameters or [],
        examples=examples or [],
        tags=tags or [],
        timeout_seconds=timeout_seconds
    )


def validate_tool_name(name: str) -> bool:
    """Validate that a tool name follows naming conventions."""
    import re
    return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name))


def sanitize_tool_result(result: Any) -> Any:
    """Sanitize tool result for safe serialization."""
    if isinstance(result, (str, int, float, bool, type(None))):
        return result
    elif isinstance(result, dict):
        return {k: sanitize_tool_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [sanitize_tool_result(item) for item in result]
    else:
        return str(result)
