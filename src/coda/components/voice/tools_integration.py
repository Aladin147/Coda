"""
Voice - Tools Integration

This module provides comprehensive integration between the voice processing system
and the tools manager for function calling, tool discovery, and execution within
voice conversations.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import ConversationState, VoiceMessage, VoiceResponse

try:
    from ..tools.interfaces import ToolManagerInterface
    from ..tools.models import (
        ToolCall,
        ToolCallStatus,
        ToolDefinition,
        ToolExecutionContext,
        ToolParameter,
        ToolResult,
    )
except ImportError:
    # Create mock interfaces for testing
    class ToolManagerInterface:
        async def get_available_tools(self):
            pass

        async def execute_tool(self, tool_call, context):
            pass

    class ToolDefinition:
        def __init__(self, name, description, parameters=None, category=None):
            self.name = name
            self.description = description
            self.parameters = parameters or []
            self.category = category

    class ToolParameter:
        def __init__(self, name, type, description, required=True):
            self.name = name
            self.type = type
            self.description = description
            self.required = required

    class ToolCall:
        def __init__(self, call_id, tool_name, parameters):
            self.call_id = call_id
            self.tool_name = tool_name
            self.parameters = parameters

    class ToolResult:
        def __init__(
            self, call_id, tool_name, status, result=None, error=None, execution_time_ms=0.0
        ):
            self.call_id = call_id
            self.tool_name = tool_name
            self.status = status
            self.result = result
            self.error = error
            self.execution_time_ms = execution_time_ms

    class ToolCallStatus:
        SUCCESS = "success"
        FAILED = "failed"

    class ToolExecutionContext:
        def __init__(self, user_id, conversation_id, message_id, source):
            self.user_id = user_id
            self.conversation_id = conversation_id
            self.message_id = message_id
            self.source = source


logger = logging.getLogger("coda.voice.tools_integration")


@dataclass
class VoiceToolsConfig:
    """Configuration for voice - tools integration."""

    # Tool discovery and injection
    enable_tool_injection: bool = True
    max_tools_in_context: int = 10
    tool_relevance_threshold: float = 0.6
    include_tool_descriptions: bool = True
    include_tool_examples: bool = True

    # Function calling
    enable_function_calling: bool = True
    auto_execute_tools: bool = True
    require_confirmation: bool = False  # For voice, auto - execute is usually preferred
    max_execution_time_seconds: float = 30.0

    # Voice - specific features
    enable_voice_tool_feedback: bool = True
    provide_execution_status: bool = True
    explain_tool_usage: bool = True

    # Tool learning and adaptation
    enable_tool_learning: bool = True
    learn_from_usage_patterns: bool = True
    adapt_tool_suggestions: bool = True

    # Performance optimization
    tool_cache_ttl_minutes: int = 15
    parallel_tool_operations: bool = True
    max_concurrent_tools: int = 3


class VoiceToolsIntegration:
    """
    Comprehensive integration between voice processing and tools systems.

    Features:
    - Automatic tool discovery and injection
    - Voice - triggered function calling
    - Tool execution with voice feedback
    - Tool usage learning and adaptation
    - Performance optimization with caching
    """

    def __init__(self, tool_manager: ToolManagerInterface, config: VoiceToolsConfig):
        """Initialize voice - tools integration."""
        self.tool_manager = tool_manager
        self.config = config

        # Caching for performance
        self.tool_cache: Dict[str, List[ToolDefinition]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Statistics tracking
        self.stats = {
            "tool_injections": 0,
            "function_calls": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "cache_hits": 0,
            "tools_discovered": 0,
        }

        # Tool usage tracking
        self.tool_usage_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.conversation_tools: Dict[str, List[str]] = {}

        logger.info("VoiceToolsIntegration initialized")

    async def enhance_voice_context(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        """
        Enhance voice processing context with relevant tools.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state

        Returns:
            Enhanced context with tool information
        """

        try:
            if not self.config.enable_tool_injection:
                return {}

            # Check cache first
            cache_key = f"tools_{hash(voice_message.text_content or '')}"
            if self._is_tools_cached(cache_key):
                self.stats["cache_hits"] += 1
                return self._get_cached_tools_context(cache_key)

            # Discover relevant tools
            tools_context = await self._discover_relevant_tools(voice_message)

            # Cache the result
            self._cache_tools_context(cache_key, tools_context)

            self.stats["tool_injections"] += 1

            return tools_context

        except Exception as e:
            logger.error(f"Failed to enhance voice context with tools: {e}")
            return {}

    async def process_function_calls(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None,
    ) -> Tuple[VoiceResponse, List[ToolResult]]:
        """
        Process function calls detected in voice conversation.

        Args:
            voice_message: Original voice message
            voice_response: Voice response that may contain function calls
            conversation_state: Current conversation state

        Returns:
            Tuple of (updated voice response, tool results)
        """

        try:
            if not self.config.enable_function_calling:
                return voice_response, []

            # Extract function calls from response
            function_calls = await self._extract_function_calls(voice_response)

            if not function_calls:
                return voice_response, []

            # Execute function calls
            tool_results = []
            execution_summaries = []

            for function_call in function_calls:
                try:
                    # Execute the tool
                    result = await self._execute_function_call(function_call, voice_message)
                    tool_results.append(result)

                    # Generate execution summary for voice
                    summary = await self._generate_execution_summary(function_call, result)
                    execution_summaries.append(summary)

                    self.stats["successful_executions"] += 1

                except Exception as e:
                    logger.error(f"Function call execution failed: {e}")

                    # Create error result
                    error_result = ToolResult(
                        call_id=function_call.get("call_id", "unknown"),
                        tool_name=function_call.get("name", "unknown"),
                        status=ToolCallStatus.FAILED,
                        result=None,
                        error=str(e),
                        execution_time_ms=0.0,
                    )
                    tool_results.append(error_result)

                    execution_summaries.append(
                        f"Failed to execute {function_call.get('name', 'tool')}: {e}"
                    )
                    self.stats["failed_executions"] += 1

            # Update voice response with tool execution results
            updated_response = await self._update_response_with_tool_results(
                voice_response, tool_results, execution_summaries
            )

            # Track tool usage
            await self._track_tool_usage(voice_message, function_calls, tool_results)

            return updated_response, tool_results

        except Exception as e:
            logger.error(f"Failed to process function calls: {e}")
            return voice_response, []

    async def _discover_relevant_tools(self, voice_message: VoiceMessage) -> Dict[str, Any]:
        """Discover tools relevant to the voice message."""

        text_content = voice_message.text_content or ""
        if not text_content:
            return {}

        try:
            # Get available tools
            available_tools = await self.tool_manager.get_available_tools()

            # Filter relevant tools based on message content
            relevant_tools = []

            for tool in available_tools:
                relevance_score = await self._calculate_tool_relevance(tool, text_content)

                if relevance_score >= self.config.tool_relevance_threshold:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description,
                        "relevance_score": relevance_score,
                        "parameters": (
                            [
                                {
                                    "name": param.name,
                                    "type": param.type,
                                    "description": param.description,
                                    "required": param.required,
                                }
                                for param in tool.parameters
                            ]
                            if self.config.include_tool_descriptions
                            else []
                        ),
                    }

                    # Add examples if enabled
                    if self.config.include_tool_examples and hasattr(tool, "examples"):
                        tool_info["examples"] = tool.examples[:2]  # Limit to 2 examples

                    relevant_tools.append(tool_info)

            # Sort by relevance and limit
            relevant_tools.sort(key=lambda t: t["relevance_score"], reverse=True)
            relevant_tools = relevant_tools[: self.config.max_tools_in_context]

            # Create tools context
            tools_context = {
                "available_tools": relevant_tools,
                "tool_count": len(relevant_tools),
                "discovery_query": text_content,
                "tools_summary": self._generate_tools_summary(relevant_tools),
            }

            self.stats["tools_discovered"] += len(relevant_tools)

            logger.debug(f"Discovered {len(relevant_tools)} relevant tools")

            return tools_context

        except Exception as e:
            logger.error(f"Failed to discover relevant tools: {e}")
            return {}

    async def _calculate_tool_relevance(self, tool: ToolDefinition, text_content: str) -> float:
        """Calculate relevance score between tool and text content."""

        try:
            # Simple keyword - based relevance (in real implementation, use embeddings)
            text_lower = text_content.lower()
            tool_keywords = []

            # Extract keywords from tool name and description
            tool_keywords.extend(tool.name.lower().split("_"))
            tool_keywords.extend(tool.description.lower().split())

            # Calculate keyword overlap
            matches = 0
            total_keywords = len(tool_keywords)

            for keyword in tool_keywords:
                if len(keyword) > 3 and keyword in text_lower:
                    matches += 1

            # Base relevance score
            relevance = matches / total_keywords if total_keywords > 0 else 0.0

            # Boost for exact name matches
            if tool.name.lower() in text_lower:
                relevance += 0.3

            # Boost for category matches
            if hasattr(tool, "category"):
                category_keywords = {
                    "file": ["file", "document", "save", "read", "write"],
                    "web": ["search", "browse", "url", "website", "internet"],
                    "calculation": ["calculate", "math", "compute", "number"],
                    "system": ["system", "process", "run", "execute"],
                }

                if tool.category in category_keywords:
                    for keyword in category_keywords[tool.category]:
                        if keyword in text_lower:
                            relevance += 0.1
                            break

            return min(relevance, 1.0)

        except Exception as e:
            logger.error(f"Failed to calculate tool relevance: {e}")
            return 0.0

    def _generate_tools_summary(self, tools: List[Dict[str, Any]]) -> str:
        """Generate a summary of available tools."""

        if not tools:
            return "No relevant tools found."

        # Group tools by category or type
        tool_names = [tool["name"] for tool in tools]

        if len(tools) == 1:
            return f"Available tool: {tool_names[0]}"
        elif len(tools) <= 3:
            return f"Available tools: {', '.join(tool_names)}"
        else:
            return f"Available tools: {', '.join(tool_names[:3])} and {len(tools) - 3} more"

    async def _extract_function_calls(self, voice_response: VoiceResponse) -> List[Dict[str, Any]]:
        """Extract function calls from voice response."""

        try:
            text_content = voice_response.text_content or ""

            # Simple function call detection (in real implementation, use LLM parsing)
            function_calls = []

            # Look for function call patterns
            import re

            # Pattern: function_name(param1=value1, param2=value2)
            pattern = r"(\w+)\s*\(\s*([^)]*)\s*\)"
            matches = re.findall(pattern, text_content)

            for i, (func_name, params_str) in enumerate(matches):
                # Parse parameters
                parameters = {}
                if params_str.strip():
                    # Simple parameter parsing
                    param_pairs = params_str.split(",")
                    for pair in param_pairs:
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            parameters[key.strip()] = value.strip().strip("\"'")

                function_call = {
                    "call_id": f"call_{voice_response.response_id}_{i}",
                    "name": func_name,
                    "parameters": parameters,
                }

                function_calls.append(function_call)

            self.stats["function_calls"] += len(function_calls)

            return function_calls

        except Exception as e:
            logger.error(f"Failed to extract function calls: {e}")
            return []

    async def _execute_function_call(
        self, function_call: Dict[str, Any], voice_message: VoiceMessage
    ) -> ToolResult:
        """Execute a function call."""

        start_time = time.time()

        try:
            # Create tool call object
            tool_call = ToolCall(
                call_id=function_call["call_id"],
                tool_name=function_call["name"],
                parameters=function_call["parameters"],
            )

            # Create execution context
            context = ToolExecutionContext(
                user_id="voice_user",  # This would come from conversation state
                conversation_id=voice_message.conversation_id,
                message_id=voice_message.message_id,
                source="voice_interaction",
            )

            # Execute the tool
            result = await self.tool_manager.execute_tool(tool_call, context)

            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time

            logger.debug(f"Executed tool {function_call['name']} in {execution_time:.1f}ms")

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool execution failed: {e}")

            return ToolResult(
                call_id=function_call["call_id"],
                tool_name=function_call["name"],
                status=ToolCallStatus.FAILED,
                result=None,
                error=str(e),
                execution_time_ms=execution_time,
            )

    async def _generate_execution_summary(
        self, function_call: Dict[str, Any], result: ToolResult
    ) -> str:
        """Generate a voice - friendly summary of tool execution."""

        try:
            tool_name = function_call["name"]

            if result.status == ToolCallStatus.SUCCESS:
                if self.config.provide_execution_status:
                    if result.result:
                        # Summarize result for voice
                        result_str = str(result.result)
                        if len(result_str) > 100:
                            result_str = result_str[:100] + "..."

                        return f"Successfully executed {tool_name}. Result: {result_str}"
                    else:
                        return f"Successfully executed {tool_name}."
                else:
                    return ""  # Silent execution

            elif result.status == ToolCallStatus.FAILED:
                return f"Failed to execute {tool_name}: {result.error or 'Unknown error'}"

            else:
                return f"Tool {tool_name} execution status: {result.status.value}"

        except Exception as e:
            logger.error(f"Failed to generate execution summary: {e}")
            return f"Tool execution completed with status: {result.status.value}"

    async def _update_response_with_tool_results(
        self,
        voice_response: VoiceResponse,
        tool_results: List[ToolResult],
        execution_summaries: List[str],
    ) -> VoiceResponse:
        """Update voice response with tool execution results."""

        try:
            original_text = voice_response.text_content or ""

            # Add tool execution summaries to response
            if execution_summaries and self.config.enable_voice_tool_feedback:
                tool_feedback = "\n\n" + "\n".join(execution_summaries)
                updated_text = original_text + tool_feedback
            else:
                updated_text = original_text

            # Create updated response
            updated_response = VoiceResponse(
                response_id=voice_response.response_id,
                conversation_id=voice_response.conversation_id,
                message_id=voice_response.message_id,
                text_content=updated_text,
                audio_data=voice_response.audio_data,
                processing_mode=voice_response.processing_mode,
                total_latency_ms=voice_response.total_latency_ms,
                moshi_latency_ms=voice_response.moshi_latency_ms,
                llm_latency_ms=voice_response.llm_latency_ms,
                response_relevance=voice_response.response_relevance,
            )

            return updated_response

        except Exception as e:
            logger.error(f"Failed to update response with tool results: {e}")
            return voice_response

    async def _track_tool_usage(
        self,
        voice_message: VoiceMessage,
        function_calls: List[Dict[str, Any]],
        tool_results: List[ToolResult],
    ) -> None:
        """Track tool usage patterns for learning."""

        try:
            conversation_id = voice_message.conversation_id

            # Track tools used in this conversation
            if conversation_id not in self.conversation_tools:
                self.conversation_tools[conversation_id] = []

            for function_call in function_calls:
                tool_name = function_call["name"]

                # Add to conversation tools
                if tool_name not in self.conversation_tools[conversation_id]:
                    self.conversation_tools[conversation_id].append(tool_name)

                # Track usage pattern
                if tool_name not in self.tool_usage_patterns:
                    self.tool_usage_patterns[tool_name] = []

                usage_entry = {
                    "timestamp": datetime.now(),
                    "conversation_id": conversation_id,
                    "user_message": voice_message.text_content or "",
                    "parameters": function_call["parameters"],
                    "success": any(
                        r.status == ToolCallStatus.SUCCESS
                        for r in tool_results
                        if r.tool_name == tool_name
                    ),
                }

                self.tool_usage_patterns[tool_name].append(usage_entry)

                # Keep only recent usage patterns
                if len(self.tool_usage_patterns[tool_name]) > 50:
                    self.tool_usage_patterns[tool_name] = self.tool_usage_patterns[tool_name][-50:]

        except Exception as e:
            logger.error(f"Failed to track tool usage: {e}")

    def _is_tools_cached(self, cache_key: str) -> bool:
        """Check if tools context is cached and valid."""

        if cache_key not in self.tool_cache:
            return False

        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False

        ttl = timedelta(minutes=self.config.tool_cache_ttl_minutes)
        return datetime.now() - cache_time < ttl

    def _get_cached_tools_context(self, cache_key: str) -> Dict[str, Any]:
        """Get cached tools context."""
        return self.tool_cache.get(cache_key, {})

    def _cache_tools_context(self, cache_key: str, context: Dict[str, Any]) -> None:
        """Cache tools context."""
        self.tool_cache[cache_key] = context
        self.cache_timestamps[cache_key] = datetime.now()

        # Clean old cache entries
        self._cleanup_tools_cache()

    def _cleanup_tools_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.now()
        ttl = timedelta(minutes=self.config.tool_cache_ttl_minutes)

        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items() if now - timestamp > ttl
        ]

        for key in expired_keys:
            self.tool_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

    async def get_tool_suggestions(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> List[Dict[str, Any]]:
        """Get tool suggestions based on voice message and usage patterns."""

        try:
            text_content = voice_message.text_content or ""
            conversation_id = voice_message.conversation_id

            # Get tools used in this conversation
            conversation_tools = self.conversation_tools.get(conversation_id, [])

            # Get all available tools
            available_tools = await self.tool_manager.get_available_tools()

            suggestions = []

            for tool in available_tools:
                # Calculate suggestion score
                relevance_score = await self._calculate_tool_relevance(tool, text_content)

                # Boost score for previously used tools in conversation
                if tool.name in conversation_tools:
                    relevance_score += 0.2

                # Boost score based on usage patterns
                if tool.name in self.tool_usage_patterns:
                    usage_patterns = self.tool_usage_patterns[tool.name]
                    recent_usage = len(
                        [p for p in usage_patterns if (datetime.now() - p["timestamp"]).days < 7]
                    )

                    if recent_usage > 0:
                        relevance_score += min(0.3, recent_usage * 0.1)

                if relevance_score >= self.config.tool_relevance_threshold:
                    suggestion = {
                        "tool_name": tool.name,
                        "description": tool.description,
                        "relevance_score": relevance_score,
                        "usage_count": len(self.tool_usage_patterns.get(tool.name, [])),
                        "recently_used": tool.name in conversation_tools,
                    }
                    suggestions.append(suggestion)

            # Sort by relevance score
            suggestions.sort(key=lambda s: s["relevance_score"], reverse=True)

            return suggestions[:5]  # Return top 5 suggestions

        except Exception as e:
            logger.error(f"Failed to get tool suggestions: {e}")
            return []

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get voice - tools integration statistics."""

        return {
            "voice_tools_stats": self.stats.copy(),
            "cache_size": len(self.tool_cache),
            "active_conversations": len(self.conversation_tools),
            "tracked_tools": len(self.tool_usage_patterns),
            "config": {
                "tool_injection_enabled": self.config.enable_tool_injection,
                "function_calling_enabled": self.config.enable_function_calling,
                "auto_execute_tools": self.config.auto_execute_tools,
                "max_tools_in_context": self.config.max_tools_in_context,
                "tool_relevance_threshold": self.config.tool_relevance_threshold,
            },
        }

    async def cleanup(self) -> None:
        """Clean up integration resources."""

        try:
            # Clear caches
            self.tool_cache.clear()
            self.cache_timestamps.clear()

            # Clear usage patterns
            self.tool_usage_patterns.clear()
            self.conversation_tools.clear()

            logger.info("VoiceToolsIntegration cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class VoiceToolsManager:
    """
    High - level manager for voice - tools integration.

    Provides a simplified interface for voice components to interact with tools.
    """

    def __init__(
        self, tool_manager: ToolManagerInterface, config: Optional[VoiceToolsConfig] = None
    ):
        """Initialize voice tools manager."""
        self.tool_manager = tool_manager
        self.config = config or VoiceToolsConfig()
        self.integration = VoiceToolsIntegration(tool_manager, self.config)

        logger.info("VoiceToolsManager initialized")

    async def enhance_voice_context(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        """
        Enhance voice processing context with tools.

        Args:
            voice_message: Voice message to process
            conversation_state: Current conversation state

        Returns:
            Enhanced context dictionary
        """
        return await self.integration.enhance_voice_context(voice_message, conversation_state)

    async def process_voice_with_tools(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None,
    ) -> Tuple[VoiceResponse, List[ToolResult]]:
        """
        Process voice interaction with tool execution.

        Args:
            voice_message: User voice message
            voice_response: Assistant voice response
            conversation_state: Current conversation state

        Returns:
            Tuple of (updated voice response, tool results)
        """
        return await self.integration.process_function_calls(
            voice_message, voice_response, conversation_state
        )

    async def get_tool_suggestions(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tool suggestions for voice message.

        Args:
            voice_message: Voice message to analyze
            conversation_state: Current conversation state

        Returns:
            List of tool suggestions
        """
        return await self.integration.get_tool_suggestions(voice_message, conversation_state)

    def get_tools_stats(self) -> Dict[str, Any]:
        """Get comprehensive tools statistics."""
        return self.integration.get_integration_stats()

    async def cleanup(self) -> None:
        """Clean up manager resources."""
        await self.integration.cleanup()
