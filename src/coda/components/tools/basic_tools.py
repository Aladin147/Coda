"""
Basic tools for Coda.

This module provides fundamental utility tools like time, date, calculations, etc.
"""

import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool, create_simple_tool_definition
from .models import ParameterType, ToolCategory, ToolDefinition

logger = logging.getLogger("coda.tools.basic")


class GetTimeTool(BaseTool):
    """Tool to get current time."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="get_time",
            description="Get the current time in various formats",
            category=ToolCategory.UTILITY,
            parameters=[
                self.create_string_parameter(
                    "format",
                    "Time format (12h, 24h, iso, timestamp)",
                    required=False,
                    default="24h",
                    enum=["12h", "24h", "iso", "timestamp"],
                    examples=["24h", "12h", "iso"],
                ),
                self.create_string_parameter(
                    "timezone",
                    "Timezone (UTC, local, or timezone name)",
                    required=False,
                    default="local",
                    examples=["UTC", "local", "America/New_York"],
                ),
            ],
            examples=[
                {"format": "24h", "timezone": "local"},
                {"format": "12h", "timezone": "UTC"},
                {"format": "iso"},
            ],
            tags=["time", "utility", "datetime"],
            timeout_seconds=5.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the tool."""
        format_type = parameters.get("format", "24h")
        timezone_param = parameters.get("timezone", "local")

        # Get current time
        if timezone_param.lower() == "utc":
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now()  # Local time

        # Format based on requested format
        if format_type == "12h":
            return now.strftime("%I:%M:%S %p")
        elif format_type == "24h":
            return now.strftime("%H:%M:%S")
        elif format_type == "iso":
            return now.isoformat()
        elif format_type == "timestamp":
            return str(int(now.timestamp()))
        else:
            return now.strftime("%H:%M:%S")


class GetDateTool(BaseTool):
    """Tool to get current date."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="get_date",
            description="Get the current date in various formats",
            category=ToolCategory.UTILITY,
            parameters=[
                self.create_string_parameter(
                    "format",
                    "Date format (short, long, iso, custom)",
                    required=False,
                    default="short",
                    enum=["short", "long", "iso", "custom"],
                    examples=["short", "long", "iso"],
                ),
                self.create_string_parameter(
                    "custom_format",
                    "Custom date format string (when format=custom)",
                    required=False,
                    examples=["%Y-%m-%d", "%B %d, %Y", "%d/%m/%Y"],
                ),
            ],
            examples=[
                {"format": "short"},
                {"format": "long"},
                {"format": "custom", "custom_format": "%B %d, %Y"},
            ],
            tags=["date", "utility", "datetime"],
            timeout_seconds=5.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the tool."""
        format_type = parameters.get("format", "short")
        custom_format = parameters.get("custom_format")

        now = datetime.now()

        if format_type == "short":
            return now.strftime("%Y-%m-%d")
        elif format_type == "long":
            return now.strftime("%A, %B %d, %Y")
        elif format_type == "iso":
            return now.date().isoformat()
        elif format_type == "custom" and custom_format:
            try:
                return now.strftime(custom_format)
            except ValueError as e:
                raise ValueError(f"Invalid custom format: {e}")
        else:
            return now.strftime("%Y-%m-%d")


class CalculatorTool(BaseTool):
    """Tool for basic mathematical calculations."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="calculator",
            description="Perform basic mathematical calculations",
            category=ToolCategory.CALCULATION,
            parameters=[
                self.create_string_parameter(
                    "expression",
                    "Mathematical expression to evaluate (supports +, -, *, /, **, (), basic functions)",
                    required=True,
                    examples=["2 + 3", "10 * (5 + 3)", "sqrt(16)", "sin(3.14159/2)"],
                ),
                self.create_integer_parameter(
                    "precision",
                    "Number of decimal places for the result",
                    required=False,
                    default=6,
                    minimum=0,
                    maximum=15,
                ),
            ],
            examples=[
                {"expression": "2 + 3 * 4"},
                {"expression": "sqrt(25)", "precision": 2},
                {"expression": "sin(pi/2)"},
            ],
            tags=["math", "calculation", "utility"],
            timeout_seconds=10.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the tool."""
        expression = self.validate_string_not_empty(parameters["expression"], "expression")
        precision = parameters.get("precision", 6)

        # Safe evaluation with limited functions
        import math

        # Define safe functions
        safe_functions = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        # Create safe evaluation environment
        safe_dict = {"__builtins__": {}}
        safe_dict.update(safe_functions)

        try:
            # Evaluate the expression
            result = eval(expression, safe_dict)

            # Format result with specified precision
            if isinstance(result, float):
                return f"{result:.{precision}f}".rstrip("0").rstrip(".")
            else:
                return str(result)

        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {e}")


class RandomNumberTool(BaseTool):
    """Tool to generate random numbers."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="random_number",
            description="Generate random numbers (integers or floats)",
            category=ToolCategory.UTILITY,
            parameters=[
                self.create_string_parameter(
                    "type",
                    "Type of random number (integer, float)",
                    required=False,
                    default="integer",
                    enum=["integer", "float"],
                    examples=["integer", "float"],
                ),
                self.create_float_parameter(
                    "min_value",
                    "Minimum value (inclusive)",
                    required=False,
                    default=1,
                    examples=[1, 0, -10],
                ),
                self.create_float_parameter(
                    "max_value",
                    "Maximum value (inclusive for integers, exclusive for floats)",
                    required=False,
                    default=100,
                    examples=[100, 1000, 10],
                ),
                self.create_integer_parameter(
                    "count",
                    "Number of random numbers to generate",
                    required=False,
                    default=1,
                    minimum=1,
                    maximum=100,
                ),
            ],
            examples=[
                {"type": "integer", "min_value": 1, "max_value": 10},
                {"type": "float", "min_value": 0, "max_value": 1, "count": 5},
                {"type": "integer", "min_value": -50, "max_value": 50},
            ],
            tags=["random", "utility", "number"],
            timeout_seconds=5.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the tool."""
        number_type = parameters.get("type", "integer")
        min_value = parameters.get("min_value", 1)
        max_value = parameters.get("max_value", 100)
        count = parameters.get("count", 1)

        # Validate range
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")

        results = []

        for _ in range(count):
            if number_type == "integer":
                result = random.randint(int(min_value), int(max_value))
            else:  # float
                result = random.uniform(min_value, max_value)

            results.append(result)

        # Return single value or list
        if count == 1:
            return results[0]
        else:
            return results


class UUIDGeneratorTool(BaseTool):
    """Tool to generate UUIDs."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="generate_uuid",
            description="Generate UUID (Universally Unique Identifier)",
            category=ToolCategory.UTILITY,
            parameters=[
                self.create_string_parameter(
                    "version",
                    "UUID version (4 for random, 1 for time-based)",
                    required=False,
                    default="4",
                    enum=["1", "4"],
                    examples=["4", "1"],
                ),
                self.create_integer_parameter(
                    "count",
                    "Number of UUIDs to generate",
                    required=False,
                    default=1,
                    minimum=1,
                    maximum=50,
                ),
                self.create_boolean_parameter(
                    "uppercase", "Return UUID in uppercase", required=False, default=False
                ),
            ],
            examples=[
                {"version": "4"},
                {"version": "4", "count": 5},
                {"version": "1", "uppercase": True},
            ],
            tags=["uuid", "utility", "identifier"],
            timeout_seconds=5.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the tool."""
        version = parameters.get("version", "4")
        count = parameters.get("count", 1)
        uppercase = parameters.get("uppercase", False)

        results = []

        for _ in range(count):
            if version == "1":
                generated_uuid = str(uuid.uuid1())
            else:  # version 4 (default)
                generated_uuid = str(uuid.uuid4())

            if uppercase:
                generated_uuid = generated_uuid.upper()

            results.append(generated_uuid)

        # Return single value or list
        if count == 1:
            return results[0]
        else:
            return results


class TimestampTool(BaseTool):
    """Tool to work with timestamps."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="timestamp",
            description="Convert between timestamps and human-readable dates",
            category=ToolCategory.UTILITY,
            parameters=[
                self.create_string_parameter(
                    "operation",
                    "Operation to perform (current, to_timestamp, from_timestamp)",
                    required=True,
                    enum=["current", "to_timestamp", "from_timestamp"],
                    examples=["current", "to_timestamp", "from_timestamp"],
                ),
                self.create_string_parameter(
                    "datetime_string",
                    "Date/time string to convert (for to_timestamp operation)",
                    required=False,
                    examples=["2023-12-25 15:30:00", "December 25, 2023"],
                ),
                self.create_integer_parameter(
                    "timestamp",
                    "Unix timestamp to convert (for from_timestamp operation)",
                    required=False,
                    examples=[1703520600, 1640995200],
                ),
                self.create_string_parameter(
                    "format",
                    "Output format for from_timestamp operation",
                    required=False,
                    default="iso",
                    enum=["iso", "readable", "custom"],
                    examples=["iso", "readable"],
                ),
            ],
            examples=[
                {"operation": "current"},
                {"operation": "to_timestamp", "datetime_string": "2023-12-25 15:30:00"},
                {"operation": "from_timestamp", "timestamp": 1703520600, "format": "readable"},
            ],
            tags=["timestamp", "utility", "datetime", "conversion"],
            timeout_seconds=5.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the tool."""
        operation = parameters["operation"]

        if operation == "current":
            return int(time.time())

        elif operation == "to_timestamp":
            datetime_string = parameters.get("datetime_string")
            if not datetime_string:
                raise ValueError("datetime_string is required for to_timestamp operation")

            try:
                # Try common formats
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%m/%d/%Y %H:%M:%S",
                    "%m/%d/%Y",
                    "%B %d, %Y",
                    "%B %d, %Y %H:%M:%S",
                ]

                for fmt in formats:
                    try:
                        dt = datetime.strptime(datetime_string, fmt)
                        return int(dt.timestamp())
                    except ValueError:
                        continue

                # If no format worked, try ISO format
                dt = datetime.fromisoformat(datetime_string.replace("Z", "+00:00"))
                return int(dt.timestamp())

            except Exception as e:
                raise ValueError(f"Could not parse datetime string: {e}")

        elif operation == "from_timestamp":
            timestamp_value = parameters.get("timestamp")
            if timestamp_value is None:
                raise ValueError("timestamp is required for from_timestamp operation")

            format_type = parameters.get("format", "iso")

            try:
                dt = datetime.fromtimestamp(timestamp_value)

                if format_type == "iso":
                    return dt.isoformat()
                elif format_type == "readable":
                    return dt.strftime("%A, %B %d, %Y at %I:%M:%S %p")
                else:
                    return dt.isoformat()

            except Exception as e:
                raise ValueError(f"Invalid timestamp: {e}")

        else:
            raise ValueError(f"Unknown operation: {operation}")
