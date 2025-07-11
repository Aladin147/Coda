"""
Interfaces for the tool system.

This module defines abstract base classes for tool components to ensure
consistent APIs and enable easy testing with mocks.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from .models import (
    ToolCall,
    ToolCategory,
    ToolDefinition,
    ToolEvent,
    ToolExecution,
    ToolResult,
    ToolStats,
)


class ToolInterface(ABC):
    """Interface for individual tools."""

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get the tool definition."""
        pass

    @abstractmethod
    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the tool is currently available."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources used by the tool."""
        pass


class ToolRegistryInterface(ABC):
    """Interface for tool registry."""

    @abstractmethod
    def register_tool(self, tool: ToolInterface) -> bool:
        """Register a tool."""
        pass

    @abstractmethod
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        pass

    @abstractmethod
    def get_tool(self, tool_name: str) -> Optional[ToolInterface]:
        """Get a tool by name."""
        pass

    @abstractmethod
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """List all registered tools, optionally filtered by category."""
        pass

    @abstractmethod
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by name, description, or tags."""
        pass

    @abstractmethod
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        pass

    @abstractmethod
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schemas for all tools."""
        pass

    @abstractmethod
    def validate_tool_call(self, tool_call: ToolCall) -> bool:
        """Validate a tool call against registered tools."""
        pass


class ToolExecutorInterface(ABC):
    """Interface for tool executor."""

    @abstractmethod
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        pass

    @abstractmethod
    async def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls."""
        pass

    @abstractmethod
    async def execute_tool_stream(self, tool_call: ToolCall) -> AsyncGenerator[ToolResult, None]:
        """Execute a tool with streaming results."""
        pass

    @abstractmethod
    def get_execution_status(self, call_id: str) -> Optional[ToolExecution]:
        """Get execution status for a call."""
        pass

    @abstractmethod
    async def cancel_execution(self, call_id: str) -> bool:
        """Cancel a running execution."""
        pass

    @abstractmethod
    def get_active_executions(self) -> List[ToolExecution]:
        """Get list of currently active executions."""
        pass

    @abstractmethod
    def get_execution_stats(self) -> ToolStats:
        """Get execution statistics."""
        pass


class ToolManagerInterface(ABC):
    """Interface for the main tool manager."""

    @abstractmethod
    async def process_function_call(self, function_call: Dict[str, Any]) -> ToolResult:
        """Process a function call from LLM."""
        pass

    @abstractmethod
    async def process_function_calls(
        self, function_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """Process multiple function calls from LLM."""
        pass

    @abstractmethod
    def get_available_tools(self, context: Optional[Dict[str, Any]] = None) -> List[ToolDefinition]:
        """Get available tools for current context."""
        pass

    @abstractmethod
    def get_function_schemas(
        self, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get function schemas for LLM."""
        pass

    @abstractmethod
    async def register_tool(self, tool: ToolInterface) -> bool:
        """Register a new tool."""
        pass

    @abstractmethod
    async def load_plugin(self, plugin_path: str) -> bool:
        """Load a tool plugin."""
        pass

    @abstractmethod
    def get_tool_analytics(self) -> Dict[str, Any]:
        """Get comprehensive tool analytics."""
        pass


class WebSocketToolInterface(ToolManagerInterface):
    """Interface for tool manager with WebSocket integration."""

    @abstractmethod
    async def set_websocket_integration(self, integration: Any) -> None:
        """Set WebSocket integration for event broadcasting."""
        pass

    @abstractmethod
    async def broadcast_tool_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a tool event via WebSocket."""
        pass

    @abstractmethod
    async def broadcast_tool_execution(self, execution: ToolExecution, result: ToolResult) -> None:
        """Broadcast tool execution event."""
        pass

    @abstractmethod
    async def broadcast_tool_registration(self, tool_definition: ToolDefinition) -> None:
        """Broadcast tool registration event."""
        pass


class PluginLoaderInterface(ABC):
    """Interface for plugin loader."""

    @abstractmethod
    async def load_plugin(self, plugin_path: str) -> List[ToolInterface]:
        """Load tools from a plugin file."""
        pass

    @abstractmethod
    async def discover_plugins(self, directory: str) -> List[str]:
        """Discover plugin files in a directory."""
        pass

    @abstractmethod
    async def reload_plugin(self, plugin_path: str) -> List[ToolInterface]:
        """Reload a plugin."""
        pass

    @abstractmethod
    def get_loaded_plugins(self) -> Dict[str, List[str]]:
        """Get list of loaded plugins and their tools."""
        pass

    @abstractmethod
    async def unload_plugin(self, plugin_path: str) -> bool:
        """Unload a plugin."""
        pass


class FunctionCallingInterface(ABC):
    """Interface for function calling management."""

    @abstractmethod
    def parse_function_call(self, function_call_data: Dict[str, Any]) -> ToolCall:
        """Parse function call data into ToolCall."""
        pass

    @abstractmethod
    def format_tool_result(self, result: ToolResult, for_llm: bool = True) -> Dict[str, Any]:
        """Format tool result for LLM consumption."""
        pass

    @abstractmethod
    def generate_function_schemas(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Generate OpenAI function calling schemas."""
        pass

    @abstractmethod
    def validate_function_call(
        self, function_call: Dict[str, Any], available_tools: List[str]
    ) -> bool:
        """Validate a function call against available tools."""
        pass


# Utility interfaces


class ToolAnalyticsInterface(ABC):
    """Interface for tool analytics."""

    @abstractmethod
    def record_execution(self, execution: ToolExecution, result: ToolResult) -> None:
        """Record a tool execution for analytics."""
        pass

    @abstractmethod
    def get_usage_stats(self, time_range_hours: int = 24) -> ToolStats:
        """Get usage statistics for a time range."""
        pass

    @abstractmethod
    def get_performance_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for tools."""
        pass

    @abstractmethod
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis and patterns."""
        pass

    @abstractmethod
    def get_tool_popularity(self) -> List[Dict[str, Any]]:
        """Get tool popularity rankings."""
        pass


class ToolSecurityInterface(ABC):
    """Interface for tool security management."""

    @abstractmethod
    def validate_tool_safety(self, tool_definition: ToolDefinition) -> bool:
        """Validate that a tool is safe to execute."""
        pass

    @abstractmethod
    def check_execution_permissions(self, tool_call: ToolCall, context: Dict[str, Any]) -> bool:
        """Check if execution is permitted."""
        pass

    @abstractmethod
    def sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for safe execution."""
        pass

    @abstractmethod
    def audit_tool_execution(self, execution: ToolExecution, result: ToolResult) -> None:
        """Audit tool execution for security."""
        pass


class ToolCacheInterface(ABC):
    """Interface for tool result caching."""

    @abstractmethod
    async def get_cached_result(self, tool_call: ToolCall) -> Optional[ToolResult]:
        """Get cached result for a tool call."""
        pass

    @abstractmethod
    async def cache_result(self, tool_call: ToolCall, result: ToolResult) -> None:
        """Cache a tool result."""
        pass

    @abstractmethod
    async def invalidate_cache(self, tool_name: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class ToolConfigInterface(ABC):
    """Interface for tool configuration management."""

    @abstractmethod
    def load_config(self, config_path: str) -> bool:
        """Load tool configuration from file."""
        pass

    @abstractmethod
    def save_config(self, config_path: str) -> bool:
        """Save current tool configuration to file."""
        pass

    @abstractmethod
    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific tool."""
        pass

    @abstractmethod
    def update_tool_config(self, tool_name: str, config: Dict[str, Any]) -> bool:
        """Update configuration for a specific tool."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate tool configuration."""
        pass
