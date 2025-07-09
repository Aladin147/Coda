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

from .interfaces import (
    ToolInterface,
    ToolRegistryInterface,
    ToolExecutorInterface,
    ToolManagerInterface,
    WebSocketToolInterface,
)
from .models import (
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolCall,
    ToolExecution,
    ToolCategory,
    ToolEvent,
    ToolConfig,
)
from .base_tool import BaseTool, ToolError, ToolValidationError
from .registry import ToolRegistry
from .executor import ToolExecutor
from .function_calling import FunctionCallingManager
from .manager import ToolManager
from .websocket_integration import WebSocketToolManager
from .plugin_loader import PluginLoader

# Built-in tools
from .basic_tools import (
    GetTimeTool,
    GetDateTool,
    CalculatorTool,
    RandomNumberTool,
    UUIDGeneratorTool,
)
from .memory_tools import (
    StoreMemoryTool,
    SearchMemoryTool,
    GetMemoryStatsTool,
)
from .system_tools import (
    GetSystemInfoTool,
    GetEnvironmentTool,
    ExecuteCommandTool,
)

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
