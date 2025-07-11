"""
Memory tools for Coda.

This module provides tools that integrate with Coda's memory system
for storing, retrieving, and managing memories.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool, create_simple_tool_definition
from .models import ToolCategory, ToolDefinition

logger = logging.getLogger("coda.tools.memory")


class StoreMemoryTool(BaseTool):
    """Tool to store information in memory."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="store_memory",
            description="Store information in long-term memory for future reference",
            category=ToolCategory.MEMORY,
            parameters=[
                self.create_string_parameter(
                    "content",
                    "Content to store in memory",
                    required=True,
                    examples=[
                        "User prefers detailed technical explanations",
                        "User is working on a Python machine learning project",
                        "User's favorite programming language is Python",
                    ],
                ),
                self.create_string_parameter(
                    "category",
                    "Category of memory (preference, fact, context, goal, etc.)",
                    required=False,
                    default="general",
                    examples=["preference", "fact", "context", "goal", "personal"],
                ),
                self.create_array_parameter(
                    "tags",
                    "Tags to associate with this memory for easier retrieval",
                    required=False,
                    examples=[["programming", "python"], ["user_preference"], ["project", "ml"]],
                ),
                self.create_float_parameter(
                    "importance",
                    "Importance score for this memory (0.0 to 1.0)",
                    required=False,
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            examples=[
                {
                    "content": "User prefers concise explanations",
                    "category": "preference",
                    "tags": ["communication", "style"],
                    "importance": 0.8,
                },
                {
                    "content": "User is learning machine learning with Python",
                    "category": "context",
                    "tags": ["learning", "python", "ml"],
                },
            ],
            tags=["memory", "storage", "learning"],
            timeout_seconds=10.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the tool."""
        content = self.validate_string_not_empty(parameters["content"], "content")
        category = parameters.get("category", "general")
        tags = parameters.get("tags", [])
        importance = parameters.get("importance", 0.5)

        # Get memory manager from context
        memory_manager = self.get_context_value(context, "memory_manager")
        if not memory_manager:
            raise ValueError("Memory manager not available in context")

        try:
            # Store the memory
            memory_id = await memory_manager.store_memory(
                content=content,
                category=category,
                tags=tags,
                importance=importance,
                source="tool_store_memory",
            )

            self.log_info(f"Stored memory with ID: {memory_id}")
            return f"Successfully stored memory with ID: {memory_id}"

        except Exception as e:
            self.log_error(f"Failed to store memory: {e}")
            raise ValueError(f"Failed to store memory: {e}")


class SearchMemoryTool(BaseTool):
    """Tool to search and retrieve memories."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="search_memory",
            description="Search for relevant memories based on query, tags, or category",
            category=ToolCategory.MEMORY,
            parameters=[
                self.create_string_parameter(
                    "query",
                    "Search query to find relevant memories",
                    required=False,
                    examples=[
                        "user preferences",
                        "Python programming",
                        "machine learning project",
                        "communication style",
                    ],
                ),
                self.create_array_parameter(
                    "tags",
                    "Tags to filter memories by",
                    required=False,
                    examples=[["programming"], ["user_preference", "style"], ["python", "ml"]],
                ),
                self.create_string_parameter(
                    "category",
                    "Category to filter memories by",
                    required=False,
                    examples=["preference", "fact", "context", "goal"],
                ),
                self.create_integer_parameter(
                    "limit",
                    "Maximum number of memories to return",
                    required=False,
                    default=10,
                    minimum=1,
                    maximum=50,
                ),
                self.create_float_parameter(
                    "min_relevance",
                    "Minimum relevance score for returned memories (0.0 to 1.0)",
                    required=False,
                    default=0.3,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            examples=[
                {"query": "user preferences", "limit": 5},
                {"tags": ["programming", "python"], "category": "context"},
                {"query": "machine learning", "min_relevance": 0.5},
            ],
            tags=["memory", "search", "retrieval"],
            timeout_seconds=15.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool."""
        query = parameters.get("query")
        tags = parameters.get("tags", [])
        category = parameters.get("category")
        limit = parameters.get("limit", 10)
        min_relevance = parameters.get("min_relevance", 0.3)

        # Get memory manager from context
        memory_manager = self.get_context_value(context, "memory_manager")
        if not memory_manager:
            raise ValueError("Memory manager not available in context")

        try:
            # Search memories
            memories = await memory_manager.search_memories(
                query=query, tags=tags, category=category, limit=limit, min_relevance=min_relevance
            )

            # Format results
            results = []
            for memory in memories:
                results.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "category": memory.category,
                        "tags": memory.tags,
                        "importance": memory.importance,
                        "relevance_score": getattr(memory, "relevance_score", None),
                        "created_at": memory.created_at.isoformat(),
                        "last_accessed": (
                            memory.last_accessed.isoformat() if memory.last_accessed else None
                        ),
                    }
                )

            self.log_info(f"Found {len(results)} memories")

            return {
                "memories": results,
                "total_found": len(results),
                "search_query": query,
                "filters": {"tags": tags, "category": category, "min_relevance": min_relevance},
            }

        except Exception as e:
            self.log_error(f"Failed to search memories: {e}")
            raise ValueError(f"Failed to search memories: {e}")


class GetMemoryStatsTool(BaseTool):
    """Tool to get memory system statistics."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="get_memory_stats",
            description="Get statistics about the memory system",
            category=ToolCategory.MEMORY,
            parameters=[
                self.create_boolean_parameter(
                    "include_categories",
                    "Include breakdown by category",
                    required=False,
                    default=True,
                ),
                self.create_boolean_parameter(
                    "include_tags", "Include breakdown by tags", required=False, default=True
                ),
                self.create_boolean_parameter(
                    "include_recent", "Include recent memory activity", required=False, default=True
                ),
            ],
            examples=[{"include_categories": True, "include_tags": True}, {"include_recent": True}],
            tags=["memory", "statistics", "analytics"],
            timeout_seconds=10.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool."""
        include_categories = parameters.get("include_categories", True)
        include_tags = parameters.get("include_tags", True)
        include_recent = parameters.get("include_recent", True)

        # Get memory manager from context
        memory_manager = self.get_context_value(context, "memory_manager")
        if not memory_manager:
            raise ValueError("Memory manager not available in context")

        try:
            # Get basic stats
            stats = await memory_manager.get_memory_stats()

            result = {
                "total_memories": stats.get("total_memories", 0),
                "total_size_bytes": stats.get("total_size_bytes", 0),
                "average_importance": stats.get("average_importance", 0.0),
                "oldest_memory": stats.get("oldest_memory"),
                "newest_memory": stats.get("newest_memory"),
            }

            # Add category breakdown if requested
            if include_categories:
                result["categories"] = stats.get("category_distribution", {})

            # Add tag breakdown if requested
            if include_tags:
                result["top_tags"] = stats.get("top_tags", [])

            # Add recent activity if requested
            if include_recent:
                result["recent_activity"] = stats.get("recent_activity", {})

            self.log_info("Retrieved memory statistics")
            return result

        except Exception as e:
            self.log_error(f"Failed to get memory stats: {e}")
            raise ValueError(f"Failed to get memory stats: {e}")


class DeleteMemoryTool(BaseTool):
    """Tool to delete specific memories."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="delete_memory",
            description="Delete a specific memory by ID",
            category=ToolCategory.MEMORY,
            parameters=[
                self.create_string_parameter(
                    "memory_id",
                    "ID of the memory to delete",
                    required=True,
                    examples=["mem_123456", "abc-def-ghi"],
                ),
                self.create_boolean_parameter(
                    "confirm",
                    "Confirmation that you want to delete this memory",
                    required=True,
                    examples=[True],
                ),
            ],
            examples=[{"memory_id": "mem_123456", "confirm": True}],
            tags=["memory", "delete", "management"],
            timeout_seconds=10.0,
            is_dangerous=True,  # Mark as dangerous since it deletes data
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the tool."""
        memory_id = self.validate_string_not_empty(parameters["memory_id"], "memory_id")
        confirm = parameters.get("confirm", False)

        if not confirm:
            raise ValueError("Confirmation required to delete memory")

        # Get memory manager from context
        memory_manager = self.get_context_value(context, "memory_manager")
        if not memory_manager:
            raise ValueError("Memory manager not available in context")

        try:
            # Delete the memory
            success = await memory_manager.delete_memory(memory_id)

            if success:
                self.log_info(f"Deleted memory: {memory_id}")
                return f"Successfully deleted memory: {memory_id}"
            else:
                return f"Memory not found: {memory_id}"

        except Exception as e:
            self.log_error(f"Failed to delete memory {memory_id}: {e}")
            raise ValueError(f"Failed to delete memory: {e}")


class UpdateMemoryTool(BaseTool):
    """Tool to update existing memories."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="update_memory",
            description="Update an existing memory's content, tags, or importance",
            category=ToolCategory.MEMORY,
            parameters=[
                self.create_string_parameter(
                    "memory_id",
                    "ID of the memory to update",
                    required=True,
                    examples=["mem_123456", "abc-def-ghi"],
                ),
                self.create_string_parameter(
                    "content",
                    "New content for the memory",
                    required=False,
                    examples=["Updated user preference for detailed explanations"],
                ),
                self.create_array_parameter(
                    "tags",
                    "New tags for the memory",
                    required=False,
                    examples=[["programming", "python", "updated"]],
                ),
                self.create_float_parameter(
                    "importance",
                    "New importance score (0.0 to 1.0)",
                    required=False,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
            examples=[
                {"memory_id": "mem_123456", "content": "Updated content"},
                {"memory_id": "mem_123456", "importance": 0.9},
                {"memory_id": "mem_123456", "tags": ["updated", "important"]},
            ],
            tags=["memory", "update", "management"],
            timeout_seconds=10.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the tool."""
        memory_id = self.validate_string_not_empty(parameters["memory_id"], "memory_id")
        content = parameters.get("content")
        tags = parameters.get("tags")
        importance = parameters.get("importance")

        # At least one field must be provided for update
        if not any([content, tags, importance is not None]):
            raise ValueError(
                "At least one field (content, tags, or importance) must be provided for update"
            )

        # Get memory manager from context
        memory_manager = self.get_context_value(context, "memory_manager")
        if not memory_manager:
            raise ValueError("Memory manager not available in context")

        try:
            # Update the memory
            success = await memory_manager.update_memory(
                memory_id=memory_id, content=content, tags=tags, importance=importance
            )

            if success:
                self.log_info(f"Updated memory: {memory_id}")
                return f"Successfully updated memory: {memory_id}"
            else:
                return f"Memory not found: {memory_id}"

        except Exception as e:
            self.log_error(f"Failed to update memory {memory_id}: {e}")
            raise ValueError(f"Failed to update memory: {e}")
