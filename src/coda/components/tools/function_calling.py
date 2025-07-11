"""
Function calling management for Coda.

This module provides the FunctionCallingManager class for handling
LLM function calling integration and tool result formatting.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .interfaces import FunctionCallingInterface
from .models import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger("coda.tools.function_calling")


class FunctionCallingManager(FunctionCallingInterface):
    """
    Manages function calling integration with LLMs.

    Features:
    - Parse LLM function call outputs
    - Generate OpenAI-compatible function schemas
    - Format tool results for LLM consumption
    - Validate function calls
    - Handle function calling errors
    """

    def __init__(self):
        """Initialize the function calling manager."""
        logger.info("FunctionCallingManager initialized")

    def parse_function_call(self, function_call_data: Dict[str, Any]) -> ToolCall:
        """
        Parse function call data into ToolCall.

        Args:
            function_call_data: Function call data from LLM

        Returns:
            Parsed ToolCall

        Raises:
            ValueError: If function call data is invalid
        """
        try:
            # Handle different function call formats
            if "name" in function_call_data and "arguments" in function_call_data:
                # OpenAI format
                tool_name = function_call_data["name"]
                arguments = function_call_data["arguments"]

                # Parse arguments if they're a string
                if isinstance(arguments, str):
                    try:
                        parameters = json.loads(arguments)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in function arguments: {e}")
                else:
                    parameters = arguments

            elif "function" in function_call_data:
                # Alternative format with nested function object
                function_obj = function_call_data["function"]
                tool_name = function_obj["name"]
                arguments = function_obj["arguments"]

                if isinstance(arguments, str):
                    parameters = json.loads(arguments)
                else:
                    parameters = arguments

            else:
                raise ValueError(
                    "Invalid function call format: missing 'name' and 'arguments' or 'function'"
                )

            # Generate call ID
            call_id = function_call_data.get("id", str(uuid.uuid4()))

            # Create ToolCall
            return ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                parameters=parameters or {},
                timestamp=datetime.now(),
                context=function_call_data.get("context", {}),
            )

        except Exception as e:
            logger.error(f"Failed to parse function call: {e}")
            raise ValueError(f"Invalid function call data: {e}")

    def format_tool_result(self, result: ToolResult, for_llm: bool = True) -> Dict[str, Any]:
        """
        Format tool result for LLM consumption.

        Args:
            result: Tool result to format
            for_llm: Whether to format for LLM (vs. human display)

        Returns:
            Formatted result
        """
        if for_llm:
            # Format for LLM consumption
            formatted = {
                "tool_call_id": result.call_id,
                "role": "tool",
                "name": result.tool_name,
                "content": self._format_result_content(result),
            }

            # Add metadata if execution failed
            if not result.success:
                formatted["error"] = True
                formatted["error_message"] = result.error

            return formatted
        else:
            # Format for human display
            return {
                "call_id": result.call_id,
                "tool_name": result.tool_name,
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
                "timestamp": result.timestamp.isoformat(),
            }

    def generate_function_schemas(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """
        Generate OpenAI function calling schemas.

        Args:
            tools: List of tool definitions

        Returns:
            List of function schemas
        """
        schemas = []

        for tool in tools:
            try:
                schema = tool.to_function_schema()
                schemas.append(schema)
            except Exception as e:
                logger.warning(f"Failed to generate schema for tool {tool.name}: {e}")

        return schemas

    def validate_function_call(
        self, function_call: Dict[str, Any], available_tools: List[str]
    ) -> bool:
        """
        Validate a function call against available tools.

        Args:
            function_call: Function call to validate
            available_tools: List of available tool names

        Returns:
            True if valid, False otherwise
        """
        try:
            # Parse the function call
            tool_call = self.parse_function_call(function_call)

            # Check if tool is available
            if tool_call.tool_name not in available_tools:
                logger.warning(
                    f"Function call validation failed: tool {tool_call.tool_name} not available"
                )
                return False

            return True

        except Exception as e:
            logger.warning(f"Function call validation failed: {e}")
            return False

    def create_function_call_message(self, tool_call: ToolCall) -> Dict[str, Any]:
        """
        Create a function call message for LLM.

        Args:
            tool_call: Tool call to format

        Returns:
            Function call message
        """
        return {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": tool_call.tool_name,
                "arguments": json.dumps(tool_call.parameters),
            },
        }

    def create_tool_response_message(self, result: ToolResult) -> Dict[str, Any]:
        """
        Create a tool response message for LLM.

        Args:
            result: Tool result to format

        Returns:
            Tool response message
        """
        return {
            "role": "function",
            "name": result.tool_name,
            "content": self._format_result_content(result),
        }

    def parse_multiple_function_calls(
        self, function_calls_data: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        """
        Parse multiple function calls.

        Args:
            function_calls_data: List of function call data

        Returns:
            List of parsed ToolCalls
        """
        tool_calls = []

        for i, call_data in enumerate(function_calls_data):
            try:
                tool_call = self.parse_function_call(call_data)
                tool_calls.append(tool_call)
            except Exception as e:
                logger.error(f"Failed to parse function call {i}: {e}")
                # Create error tool call
                error_call = ToolCall(
                    call_id=str(uuid.uuid4()),
                    tool_name="error",
                    parameters={"error": str(e)},
                    timestamp=datetime.now(),
                )
                tool_calls.append(error_call)

        return tool_calls

    def format_multiple_tool_results(
        self, results: List[ToolResult], for_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Format multiple tool results.

        Args:
            results: List of tool results
            for_llm: Whether to format for LLM

        Returns:
            List of formatted results
        """
        return [self.format_tool_result(result, for_llm) for result in results]

    def _format_result_content(self, result: ToolResult) -> str:
        """Format result content for LLM consumption."""
        if not result.success:
            return f"Error: {result.error}"

        if result.result is None:
            return "Operation completed successfully"

        if isinstance(result.result, str):
            return result.result
        elif isinstance(result.result, (dict, list)):
            try:
                return json.dumps(result.result, indent=2)
            except (TypeError, ValueError):
                return str(result.result)
        else:
            return str(result.result)

    def create_error_result(self, call_id: str, tool_name: str, error_message: str) -> ToolResult:
        """
        Create an error result for failed function calls.

        Args:
            call_id: Call ID
            tool_name: Tool name
            error_message: Error message

        Returns:
            Error ToolResult
        """
        return ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            success=False,
            error=error_message,
            execution_time_ms=0,
            timestamp=datetime.now(),
        )

    def extract_function_calls_from_message(self, message: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract function calls from an LLM message.

        Args:
            message: LLM message that may contain function calls

        Returns:
            List of extracted ToolCalls
        """
        tool_calls = []

        # Check for function_call (single call)
        if "function_call" in message:
            try:
                tool_call = self.parse_function_call(message["function_call"])
                tool_calls.append(tool_call)
            except Exception as e:
                logger.error(f"Failed to extract function call: {e}")

        # Check for tool_calls (multiple calls)
        if "tool_calls" in message:
            for call_data in message["tool_calls"]:
                try:
                    # Handle different formats
                    if "function" in call_data:
                        function_data = call_data["function"]
                        function_data["id"] = call_data.get("id")
                        tool_call = self.parse_function_call(function_data)
                        tool_calls.append(tool_call)
                    else:
                        tool_call = self.parse_function_call(call_data)
                        tool_calls.append(tool_call)
                except Exception as e:
                    logger.error(f"Failed to extract tool call: {e}")

        return tool_calls

    def validate_function_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Validate a function schema.

        Args:
            schema: Function schema to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if "name" not in schema:
                return False

            if "description" not in schema:
                return False

            if "parameters" not in schema:
                return False

            parameters = schema["parameters"]
            if not isinstance(parameters, dict):
                return False

            if "type" not in parameters or parameters["type"] != "object":
                return False

            if "properties" not in parameters:
                return False

            # Validate parameter properties
            properties = parameters["properties"]
            if not isinstance(properties, dict):
                return False

            # Check required array if present
            if "required" in parameters:
                required = parameters["required"]
                if not isinstance(required, list):
                    return False

                # All required parameters must be in properties
                for req_param in required:
                    if req_param not in properties:
                        return False

            return True

        except Exception:
            return False

    def get_function_calling_stats(self) -> Dict[str, Any]:
        """Get function calling statistics."""
        return {
            "manager_initialized": True,
            "supported_formats": ["openai", "anthropic", "generic"],
            "validation_enabled": True,
        }
