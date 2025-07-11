"""
Tool system for Coda.

This package provides comprehensive tool management functionality including:
- Dynamic tool registration and discovery
- Function calling with LLM integration
- Plugin architecture for extensible tools
- Real-time tool execution monitoring
- Integration with memory and personality systems

Features:
- Type-safe tool definitions with Pydantic models
- Async tool execution with error handling
- WebSocket integration for real-time events
- Plugin loading and management
- Tool analytics and performance monitoring
- Integration with Coda's memory and personality systems
"""

from .base_tool import BaseTool, ToolError, ToolValidationError

# Built-in tools
from .basic_tools import (
    CalculatorTool,
    GetDateTool,
    GetTimeTool,
    RandomNumberTool,
    UUIDGeneratorTool,
)
from .executor import ToolExecutor
from .function_calling import FunctionCallingManager
from .interfaces import (
    ToolExecutorInterface,
    ToolInterface,
    ToolManagerInterface,
    ToolRegistryInterface,
    WebSocketToolInterface,
)
from .manager import ToolManager
from .memory_tools import (
    GetMemoryStatsTool,
    SearchMemoryTool,
    StoreMemoryTool,
)
from .models import (
    ToolCall,
    ToolCategory,
    ToolConfig,
    ToolDefinition,
    ToolEvent,
    ToolExecution,
    ToolParameter,
    ToolResult,
)
from .plugin_loader import PluginLoader
from .registry import ToolRegistry
from .system_tools import (
    ExecuteCommandTool,
    GetEnvironmentTool,
    GetSystemInfoTool,
)
from .websocket_integration import WebSocketToolManager

__all__ = [
    # Interfaces
    "ToolInterface",
    "ToolRegistryInterface",
    "ToolExecutorInterface",
    "ToolManagerInterface",
    "WebSocketToolInterface",
    # Models
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "ToolCall",
    "ToolExecution",
    "ToolCategory",
    "ToolEvent",
    "ToolConfig",
    # Core Components
    "BaseTool",
    "ToolError",
    "ToolValidationError",
    "ToolRegistry",
    "ToolExecutor",
    "FunctionCallingManager",
    "ToolManager",
    "WebSocketToolManager",
    "PluginLoader",
    # Built-in Tools
    "GetTimeTool",
    "GetDateTool",
    "CalculatorTool",
    "RandomNumberTool",
    "UUIDGeneratorTool",
    "StoreMemoryTool",
    "SearchMemoryTool",
    "GetMemoryStatsTool",
    "GetSystemInfoTool",
    "GetEnvironmentTool",
    "ExecuteCommandTool",
]
