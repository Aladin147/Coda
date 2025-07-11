"""
Tool manager for Coda.

This module provides the main ToolManager class that orchestrates
all tool system components.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_tool import ToolError

# Import built-in tools
from .basic_tools import (
    CalculatorTool,
    GetDateTool,
    GetTimeTool,
    RandomNumberTool,
    TimestampTool,
    UUIDGeneratorTool,
)
from .executor import ToolExecutor
from .function_calling import FunctionCallingManager
from .interfaces import ToolInterface, ToolManagerInterface
from .memory_tools import (
    DeleteMemoryTool,
    GetMemoryStatsTool,
    SearchMemoryTool,
    StoreMemoryTool,
    UpdateMemoryTool,
)
from .models import ToolCall, ToolConfig, ToolDefinition, ToolResult
from .plugin_loader import PluginLoader
from .plugin_metadata import PluginMetadata
from .registry import ToolRegistry
from .system_tools import (
    ExecuteCommandTool,
    GetEnvironmentTool,
    GetProcessInfoTool,
    GetSystemInfoTool,
)

logger = logging.getLogger("coda.tools.manager")


class ToolManager(ToolManagerInterface):
    """
    Main tool manager that orchestrates all tool system components.

    Features:
    - Tool registration and management
    - Function call processing
    - Integration with memory and personality systems
    - Plugin loading
    - Analytics and monitoring
    """

    def __init__(self, config: Optional[ToolConfig] = None):
        """
        Initialize the tool manager.

        Args:
            config: Tool system configuration
        """
        self.config = config or ToolConfig()

        # Initialize components
        registry_config = getattr(self.config, "registry", None)
        executor_config = getattr(self.config, "executor", None)

        self.registry = ToolRegistry(registry_config)
        self.executor = ToolExecutor(self.registry, executor_config)
        self.function_calling = FunctionCallingManager()
        self.plugin_loader = PluginLoader()

        # Context for tool execution
        self._execution_context: Dict[str, Any] = {}

        # Plugin tracking
        self._loaded_plugins: Dict[str, PluginMetadata] = {}

        # Register built-in tools
        self._register_builtin_tools()

        logger.info("ToolManager initialized")

    async def initialize(self) -> None:
        """
        Perform async initialization of the tool manager.

        This method performs any additional async setup needed for:
        - Plugin discovery and loading
        - Tool validation
        - Integration component setup
        """
        try:
            # Load plugins if auto-discovery is enabled
            registry_config = getattr(self.config, "registry", None)
            if registry_config and getattr(registry_config, "auto_discover_plugins", False):
                plugin_dirs = getattr(registry_config, "plugin_directories", [])
                for plugin_dir in plugin_dirs:
                    try:
                        await self._load_plugins_from_directory(plugin_dir)
                    except Exception as e:
                        logger.warning(f"Failed to load plugins from {plugin_dir}: {e}")

            # Initialize tools that need async setup
            for tool_name in self.registry.get_tool_names():
                tool = self.registry.get_tool(tool_name)
                if tool and hasattr(tool, "initialize"):
                    try:
                        await tool.initialize()
                    except Exception as e:
                        logger.warning(f"Failed to initialize tool {tool_name}: {e}")

            logger.info("ToolManager async initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize ToolManager: {e}")
            raise

    async def _load_plugins_from_directory(self, plugin_dir: str) -> None:
        """Load plugins from a directory."""
        try:
            # Discover plugins
            plugin_paths = await self.plugin_loader.discover_plugins(plugin_dir)

            if not plugin_paths:
                logger.debug(f"No plugins found in {plugin_dir}")
                return

            # Load plugins with dependency resolution
            plugin_results = await self.plugin_loader.load_plugins_with_dependencies(plugin_paths)

            # Register tools from loaded plugins
            for tools, metadata in plugin_results:
                for tool in tools:
                    success = self.registry.register_tool(tool)
                    if success:
                        logger.info(f"Registered tool: {tool.get_definition().name}")
                    else:
                        logger.warning(f"Failed to register tool: {tool.get_definition().name}")

                # Track plugin metadata
                if metadata:
                    plugin_path = next(
                        (
                            path
                            for path, meta in self.plugin_loader._plugin_metadata.items()
                            if meta == metadata
                        ),
                        None,
                    )
                    if plugin_path:
                        self._loaded_plugins[plugin_path] = metadata

            logger.info(f"Loaded {len(plugin_results)} plugins from {plugin_dir}")

        except Exception as e:
            logger.error(f"Failed to load plugins from {plugin_dir}: {e}")
            raise

    async def load_plugin(self, plugin_path: str) -> bool:
        """
        Load a single plugin.

        Args:
            plugin_path: Path to plugin file or directory

        Returns:
            True if plugin loaded successfully
        """
        try:
            tools, metadata = await self.plugin_loader.load_plugin(plugin_path)

            # Register tools
            registered_count = 0
            for tool in tools:
                if self.registry.register_tool(tool):
                    registered_count += 1

            # Track plugin metadata
            if metadata:
                self._loaded_plugins[plugin_path] = metadata

            logger.info(
                f"Loaded plugin {plugin_path}: {registered_count}/{len(tools)} tools registered"
            )
            return registered_count > 0

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False

    async def unload_plugin(self, plugin_path: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_path: Path to plugin file or directory

        Returns:
            True if plugin unloaded successfully
        """
        try:
            # Get plugin tools before unloading
            loaded_plugins = self.plugin_loader.get_loaded_plugins()
            tool_names = loaded_plugins.get(plugin_path, [])

            # Unregister tools
            for tool_name in tool_names:
                self.registry.unregister_tool(tool_name)

            # Unload plugin
            success = await self.plugin_loader.unload_plugin(plugin_path)

            # Remove from tracking
            self._loaded_plugins.pop(plugin_path, None)

            logger.info(f"Unloaded plugin {plugin_path}: {len(tool_names)} tools unregistered")
            return success

        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_path}: {e}")
            return False

    async def reload_plugin(self, plugin_path: str) -> bool:
        """
        Reload a plugin.

        Args:
            plugin_path: Path to plugin file or directory

        Returns:
            True if plugin reloaded successfully
        """
        try:
            # Unload first
            await self.unload_plugin(plugin_path)

            # Load again
            return await self.load_plugin(plugin_path)

        except Exception as e:
            logger.error(f"Failed to reload plugin {plugin_path}: {e}")
            return False

    def get_loaded_plugins(self) -> Dict[str, PluginMetadata]:
        """Get information about loaded plugins."""
        return self._loaded_plugins.copy()

    async def process_function_call(self, function_call: Dict[str, Any]) -> ToolResult:
        """
        Process a function call from LLM.

        Args:
            function_call: Function call data from LLM

        Returns:
            Tool execution result
        """
        try:
            # Parse function call
            tool_call = self.function_calling.parse_function_call(function_call)

            # Validate tool call
            if not self.registry.validate_tool_call(tool_call):
                return self.function_calling.create_error_result(
                    tool_call.call_id,
                    tool_call.tool_name,
                    f"Invalid tool call: {tool_call.tool_name}",
                )

            # Add execution context
            tool_call.context = self._execution_context.copy()

            # Execute the tool
            result = await self.executor.execute_tool(tool_call)

            logger.info(f"Processed function call: {tool_call.tool_name} -> {result.success}")
            return result

        except Exception as e:
            logger.error(f"Failed to process function call: {e}")
            return ToolResult(
                call_id=str(uuid.uuid4()),
                tool_name=function_call.get("name", "unknown"),
                success=False,
                error=f"Function call processing failed: {e}",
                execution_time_ms=0,
                timestamp=datetime.now(),
            )

    async def process_function_calls(
        self, function_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """
        Process multiple function calls from LLM.

        Args:
            function_calls: List of function call data

        Returns:
            List of tool execution results
        """
        if not function_calls:
            return []

        try:
            # Parse all function calls
            tool_calls = []
            for call_data in function_calls:
                try:
                    tool_call = self.function_calling.parse_function_call(call_data)
                    tool_call.context = self._execution_context.copy()
                    tool_calls.append(tool_call)
                except Exception as e:
                    # Create error result for invalid calls
                    error_result = ToolResult(
                        call_id=str(uuid.uuid4()),
                        tool_name=call_data.get("name", "unknown"),
                        success=False,
                        error=f"Invalid function call: {e}",
                        execution_time_ms=0,
                        timestamp=datetime.now(),
                    )
                    tool_calls.append(error_result)

            # Execute all tools
            results = await self.executor.execute_tools(tool_calls)

            logger.info(f"Processed {len(function_calls)} function calls")
            return results

        except Exception as e:
            logger.error(f"Failed to process function calls: {e}")
            # Return error results for all calls
            return [
                ToolResult(
                    call_id=str(uuid.uuid4()),
                    tool_name=call.get("name", "unknown"),
                    success=False,
                    error=f"Batch processing failed: {e}",
                    execution_time_ms=0,
                    timestamp=datetime.now(),
                )
                for call in function_calls
            ]

    def get_available_tools(self, context: Optional[Dict[str, Any]] = None) -> List[ToolDefinition]:
        """
        Get available tools for current context.

        Args:
            context: Optional context for filtering tools

        Returns:
            List of available tool definitions
        """
        # Get all available tools
        tools = self.registry.get_available_tools()

        # Apply context-based filtering if needed
        if context:
            # Filter based on personality preferences
            personality_context = context.get("personality")
            if personality_context:
                tools = self._filter_tools_by_personality(tools, personality_context)

            # Filter based on user permissions
            user_context = context.get("user")
            if user_context:
                tools = self._filter_tools_by_permissions(tools, user_context)

        return tools

    def get_function_schemas(
        self, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get function schemas for LLM.

        Args:
            context: Optional context for filtering schemas

        Returns:
            List of function schemas
        """
        available_tools = self.get_available_tools(context)
        return self.function_calling.generate_function_schemas(available_tools)

    async def register_tool(self, tool: ToolInterface) -> bool:
        """
        Register a new tool.

        Args:
            tool: Tool to register

        Returns:
            True if registration successful
        """
        success = self.registry.register_tool(tool)
        if success:
            logger.info(f"Registered tool: {tool.get_definition().name}")
        return success

    async def load_plugin(self, plugin_path: str) -> bool:
        """
        Load a tool plugin.

        Args:
            plugin_path: Path to plugin file

        Returns:
            True if loading successful
        """
        # This would be implemented with the plugin loader
        # For now, return False as not implemented
        logger.warning(f"Plugin loading not yet implemented: {plugin_path}")
        return False

    def get_tool_analytics(self) -> Dict[str, Any]:
        """Get comprehensive tool analytics."""
        registry_stats = self.registry.get_registry_stats()
        execution_stats = self.executor.get_execution_stats()
        performance_metrics = self.executor.get_performance_metrics()

        return {
            "registry": registry_stats,
            "execution": execution_stats.model_dump(),
            "performance": performance_metrics,
            "function_calling": self.function_calling.get_function_calling_stats(),
            "context": {
                "has_memory_manager": "memory_manager" in self._execution_context,
                "has_personality_manager": "personality_manager" in self._execution_context,
                "context_keys": list(self._execution_context.keys()),
            },
        }

    def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for tool context."""
        self._execution_context["memory_manager"] = memory_manager
        logger.info("Memory manager set in tool context")

    def set_personality_manager(self, personality_manager: Any) -> None:
        """Set personality manager for tool context."""
        self._execution_context["personality_manager"] = personality_manager
        logger.info("Personality manager set in tool context")

    def set_context_value(self, key: str, value: Any) -> None:
        """Set a value in the execution context."""
        self._execution_context[key] = value
        logger.debug(f"Set context value: {key}")

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the execution context."""
        return self._execution_context.get(key, default)

    def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        builtin_tools = [
            # Basic tools
            GetTimeTool(),
            GetDateTool(),
            CalculatorTool(),
            RandomNumberTool(),
            UUIDGeneratorTool(),
            TimestampTool(),
            # Memory tools
            StoreMemoryTool(),
            SearchMemoryTool(),
            GetMemoryStatsTool(),
            DeleteMemoryTool(),
            UpdateMemoryTool(),
            # System tools
            GetSystemInfoTool(),
            GetEnvironmentTool(),
            ExecuteCommandTool(),
            GetProcessInfoTool(),
        ]

        registered_count = 0
        for tool in builtin_tools:
            if self.registry.register_tool(tool):
                registered_count += 1
            else:
                logger.warning(f"Failed to register builtin tool: {tool.get_definition().name}")

        logger.info(f"Registered {registered_count}/{len(builtin_tools)} builtin tools")

    def _filter_tools_by_personality(
        self, tools: List[ToolDefinition], personality_context: Dict[str, Any]
    ) -> List[ToolDefinition]:
        """Filter tools based on personality preferences."""
        # This could be enhanced to consider personality traits
        # For now, just return all tools
        return tools

    def _filter_tools_by_permissions(
        self, tools: List[ToolDefinition], user_context: Dict[str, Any]
    ) -> List[ToolDefinition]:
        """Filter tools based on user permissions."""
        # Filter out dangerous tools if user doesn't have permission
        user_permissions = user_context.get("permissions", [])

        filtered_tools = []
        for tool in tools:
            if tool.is_dangerous and "dangerous_tools" not in user_permissions:
                continue
            if tool.requires_auth and "authenticated" not in user_permissions:
                continue
            filtered_tools.append(tool)

        return filtered_tools

    def get_tool_by_name(self, tool_name: str) -> Optional[ToolInterface]:
        """Get a tool by name."""
        return self.registry.get_tool(tool_name)

    def list_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """List tools by category."""
        from .models import ToolCategory

        try:
            cat_enum = ToolCategory(category)
            return self.registry.list_tools(cat_enum)
        except ValueError:
            return []

    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by query."""
        return self.registry.search_tools(query)

    async def cleanup(self) -> None:
        """Cleanup tool manager resources."""
        logger.info("Cleaning up tool manager")

        # Cleanup all registered tools
        for tool_name in self.registry.get_tool_names():
            tool = self.registry.get_tool(tool_name)
            if tool:
                try:
                    await tool.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up tool {tool_name}: {e}")

        # Clear execution context
        self._execution_context.clear()

        logger.info("Tool manager cleanup completed")

    async def shutdown(self) -> None:
        """Shutdown tool manager (alias for cleanup)."""
        await self.cleanup()

    async def set_memory_manager(self, memory_manager) -> None:
        """Set memory manager for context enhancement."""
        self.memory_manager = memory_manager
        logger.info("Memory manager integrated with ToolManager")

    def is_healthy(self) -> bool:
        """Check if the tool manager is healthy."""
        try:
            return (
                self.config is not None and
                self.registry is not None and
                self.executor is not None and
                self.function_calling_manager is not None and
                len(self.registry.get_all_tools()) > 0
            )
        except Exception:
            return False
