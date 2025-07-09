"""
Tool manager for Coda.

This module provides the main ToolManager class that orchestrates
all tool system components.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .interfaces import ToolManagerInterface, ToolInterface
from .models import ToolCall, ToolResult, ToolConfig
from .registry import ToolRegistry
from .executor import ToolExecutor
from .function_calling import FunctionCallingManager
from .base_tool import ToolError

# Import built-in tools
from .basic_tools import (
    GetTimeTool, GetDateTool, CalculatorTool, RandomNumberTool,
    UUIDGeneratorTool, TimestampTool
)
from .memory_tools import (
    StoreMemoryTool, SearchMemoryTool, GetMemoryStatsTool,
    DeleteMemoryTool, UpdateMemoryTool
)
from .system_tools import (
    GetSystemInfoTool, GetEnvironmentTool, ExecuteCommandTool,
    GetProcessInfoTool
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
        self.registry = ToolRegistry(self.config.registry)
        self.executor = ToolExecutor(self.registry, self.config.executor)
        self.function_calling = FunctionCallingManager()
        
        # Context for tool execution
        self._execution_context: Dict[str, Any] = {}
        
        # Register built-in tools
        self._register_builtin_tools()
        
        logger.info("ToolManager initialized")
    
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
                    f"Invalid tool call: {tool_call.tool_name}"
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
                timestamp=datetime.now()
            )
    
    async def process_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[ToolResult]:
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
                        timestamp=datetime.now()
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
                    timestamp=datetime.now()
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
    
    def get_function_schemas(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
                "context_keys": list(self._execution_context.keys())
            }
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
    
    def _filter_tools_by_personality(self, tools: List[ToolDefinition], personality_context: Dict[str, Any]) -> List[ToolDefinition]:
        """Filter tools based on personality preferences."""
        # This could be enhanced to consider personality traits
        # For now, just return all tools
        return tools
    
    def _filter_tools_by_permissions(self, tools: List[ToolDefinition], user_context: Dict[str, Any]) -> List[ToolDefinition]:
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
