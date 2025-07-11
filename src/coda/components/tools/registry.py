"""
Tool registry for Coda.

This module provides the ToolRegistry class for managing tool registration,
discovery, and metadata.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .interfaces import ToolInterface, ToolRegistryInterface
from .models import (
    ToolCall,
    ToolCategory,
    ToolDefinition,
    ToolRegistryConfig,
)

logger = logging.getLogger("coda.tools.registry")


class ToolRegistry(ToolRegistryInterface):
    """
    Registry for managing tools and their metadata.

    Features:
    - Tool registration and unregistration
    - Tool discovery and search
    - Function schema generation
    - Tool validation
    - Category-based filtering
    """

    def __init__(self, config: Optional[ToolRegistryConfig] = None):
        """
        Initialize the tool registry.

        Args:
            config: Configuration for the registry
        """
        self.config = config or ToolRegistryConfig()

        # Tool storage
        self._tools: Dict[str, ToolInterface] = {}
        self._definitions: Dict[str, ToolDefinition] = {}
        self._aliases: Dict[str, str] = {}  # alias -> tool_name
        self._categories: Dict[ToolCategory, List[str]] = defaultdict(list)
        self._tags: Dict[str, List[str]] = defaultdict(list)  # tag -> tool_names

        logger.info("ToolRegistry initialized")

    def register_tool(self, tool: ToolInterface) -> bool:
        """
        Register a tool.

        Args:
            tool: Tool to register

        Returns:
            True if registration successful, False otherwise
        """
        try:
            definition = tool.get_definition()
            tool_name = definition.name

            # Check if we've reached the maximum number of tools
            if len(self._tools) >= self.config.max_tools:
                logger.error(
                    f"Cannot register tool {tool_name}: maximum tools limit reached ({self.config.max_tools})"
                )
                return False

            # Check if tool already exists
            if tool_name in self._tools:
                logger.warning(f"Tool {tool_name} is already registered, replacing")
                self.unregister_tool(tool_name)

            # Check for dangerous tools
            if definition.is_dangerous and not self.config.allow_dangerous_tools:
                logger.error(
                    f"Cannot register dangerous tool {tool_name}: dangerous tools not allowed"
                )
                return False

            # Validate tool definition
            if not self._validate_tool_definition(definition):
                logger.error(f"Cannot register tool {tool_name}: invalid definition")
                return False

            # Register the tool
            self._tools[tool_name] = tool
            self._definitions[tool_name] = definition

            # Register aliases
            for alias in definition.aliases:
                if alias in self._aliases:
                    logger.warning(
                        f"Alias {alias} already exists for tool {self._aliases[alias]}, overriding"
                    )
                self._aliases[alias] = tool_name

            # Update category index
            self._categories[definition.category].append(tool_name)

            # Update tag index
            for tag in definition.tags:
                self._tags[tag].append(tool_name)

            logger.info(f"Registered tool: {tool_name} (category: {definition.category.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to register tool: {e}")
            return False

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if tool_name not in self._tools:
                logger.warning(f"Tool {tool_name} not found for unregistration")
                return False

            definition = self._definitions[tool_name]

            # Remove from main storage
            del self._tools[tool_name]
            del self._definitions[tool_name]

            # Remove aliases
            aliases_to_remove = [
                alias for alias, name in self._aliases.items() if name == tool_name
            ]
            for alias in aliases_to_remove:
                del self._aliases[alias]

            # Remove from category index
            if tool_name in self._categories[definition.category]:
                self._categories[definition.category].remove(tool_name)

            # Remove from tag index
            for tag in definition.tags:
                if tool_name in self._tags[tag]:
                    self._tags[tag].remove(tool_name)

            logger.info(f"Unregistered tool: {tool_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister tool {tool_name}: {e}")
            return False

    def get_tool(self, tool_name: str) -> Optional[ToolInterface]:
        """
        Get a tool by name or alias.

        Args:
            tool_name: Tool name or alias

        Returns:
            Tool instance or None if not found
        """
        # Check direct name
        if tool_name in self._tools:
            return self._tools[tool_name]

        # Check aliases
        if tool_name in self._aliases:
            actual_name = self._aliases[tool_name]
            return self._tools.get(actual_name)

        return None

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """
        List all registered tools, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of tool definitions
        """
        if category is None:
            return list(self._definitions.values())

        tool_names = self._categories.get(category, [])
        return [self._definitions[name] for name in tool_names if name in self._definitions]

    def search_tools(self, query: str) -> List[ToolDefinition]:
        """
        Search tools by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching tool definitions
        """
        query_lower = query.lower()
        matches = []

        for definition in self._definitions.values():
            # Check name
            if query_lower in definition.name.lower():
                matches.append(definition)
                continue

            # Check description
            if query_lower in definition.description.lower():
                matches.append(definition)
                continue

            # Check tags
            if any(query_lower in tag.lower() for tag in definition.tags):
                matches.append(definition)
                continue

            # Check aliases
            if any(query_lower in alias.lower() for alias in definition.aliases):
                matches.append(definition)
                continue

        # Sort by relevance (exact name matches first)
        matches.sort(
            key=lambda d: (
                0
                if query_lower == d.name.lower()
                else (
                    1
                    if query_lower in d.name.lower()
                    else 2 if query_lower in d.description.lower() else 3
                )
            )
        )

        return matches

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get tool definition by name or alias.

        Args:
            tool_name: Tool name or alias

        Returns:
            Tool definition or None if not found
        """
        # Check direct name
        if tool_name in self._definitions:
            return self._definitions[tool_name]

        # Check aliases
        if tool_name in self._aliases:
            actual_name = self._aliases[tool_name]
            return self._definitions.get(actual_name)

        return None

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function calling schemas for all tools.

        Returns:
            List of function schemas
        """
        schemas = []

        for definition in self._definitions.values():
            # Skip dangerous tools if auth is required and not provided
            if definition.is_dangerous and self.config.require_auth_for_dangerous:
                continue

            # Skip unavailable tools
            tool = self._tools.get(definition.name)
            if tool and not tool.is_available():
                continue

            schemas.append(definition.to_function_schema())

        return schemas

    def validate_tool_call(self, tool_call: ToolCall) -> bool:
        """
        Validate a tool call against registered tools.

        Args:
            tool_call: Tool call to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if tool exists
            definition = self.get_tool_definition(tool_call.tool_name)
            if not definition:
                logger.warning(f"Tool call validation failed: tool {tool_call.tool_name} not found")
                return False

            # Check if tool is available
            tool = self.get_tool(tool_call.tool_name)
            if not tool or not tool.is_available():
                logger.warning(
                    f"Tool call validation failed: tool {tool_call.tool_name} not available"
                )
                return False

            # Validate parameters
            definition.validate_parameters(tool_call.parameters)

            return True

        except Exception as e:
            logger.warning(f"Tool call validation failed: {e}")
            return False

    def get_tools_by_category(self, category: ToolCategory) -> List[ToolInterface]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_tools_by_tag(self, tag: str) -> List[ToolInterface]:
        """Get all tools with a specific tag."""
        tool_names = self._tags.get(tag, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_available_tools(self) -> List[ToolDefinition]:
        """Get all currently available tools."""
        available = []

        for name, tool in self._tools.items():
            if tool.is_available():
                available.append(self._definitions[name])

        return available

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_tools = len(self._tools)
        available_tools = len(self.get_available_tools())

        category_counts = {
            category.value: len(tool_names)
            for category, tool_names in self._categories.items()
            if tool_names
        }

        return {
            "total_tools": total_tools,
            "available_tools": available_tools,
            "unavailable_tools": total_tools - available_tools,
            "total_aliases": len(self._aliases),
            "category_distribution": category_counts,
            "total_tags": len(self._tags),
            "dangerous_tools": sum(1 for d in self._definitions.values() if d.is_dangerous),
            "tools_requiring_auth": sum(1 for d in self._definitions.values() if d.requires_auth),
        }

    def _validate_tool_definition(self, definition: ToolDefinition) -> bool:
        """Validate a tool definition."""
        try:
            # Check name format
            import re

            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", definition.name):
                logger.error(f"Invalid tool name format: {definition.name}")
                return False

            # Check description
            if not definition.description or len(definition.description.strip()) < 10:
                logger.error(f"Tool description too short: {definition.name}")
                return False

            # Validate parameters
            parameter_names = set()
            for param in definition.parameters:
                if param.name in parameter_names:
                    logger.error(
                        f"Duplicate parameter name: {param.name} in tool {definition.name}"
                    )
                    return False
                parameter_names.add(param.name)

            # Check timeout
            if definition.timeout_seconds <= 0:
                logger.error(
                    f"Invalid timeout: {definition.timeout_seconds} for tool {definition.name}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Tool definition validation failed: {e}")
            return False

    def clear_registry(self) -> None:
        """Clear all registered tools."""
        logger.info("Clearing tool registry")

        self._tools.clear()
        self._definitions.clear()
        self._aliases.clear()
        self._categories.clear()
        self._tags.clear()

    def export_registry(self) -> Dict[str, Any]:
        """Export registry data for persistence."""
        return {
            "tools": {
                name: definition.model_dump() for name, definition in self._definitions.items()
            },
            "aliases": dict(self._aliases),
            "exported_at": datetime.now().isoformat(),
        }

    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def get_tool_aliases(self) -> Dict[str, str]:
        """Get mapping of aliases to tool names."""
        return dict(self._aliases)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools or tool_name in self._aliases
