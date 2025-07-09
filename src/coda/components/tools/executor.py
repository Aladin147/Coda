"""
Tool executor for Coda.

This module provides the ToolExecutor class for executing tools with
error handling, timeouts, retries, and performance monitoring.
"""

import uuid
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from collections import defaultdict, deque
from datetime import datetime, timedelta

from .interfaces import ToolExecutorInterface, ToolRegistryInterface
from .models import (
    ToolCall,
    ToolResult,
    ToolExecution,
    ToolStats,
    ToolExecutorConfig,
)
from .base_tool import ToolError, ToolTimeoutError, ToolExecutionError

logger = logging.getLogger("coda.tools.executor")


class ToolExecutor(ToolExecutorInterface):
    """
    Executes tools with comprehensive error handling and monitoring.
    
    Features:
    - Async tool execution with timeout handling
    - Retry logic for failed executions
    - Concurrent execution management
    - Performance monitoring and statistics
    - Execution history and analytics
    """
    
    def __init__(self, registry: ToolRegistryInterface, config: Optional[ToolExecutorConfig] = None):
        """
        Initialize the tool executor.
        
        Args:
            registry: Tool registry for tool lookup
            config: Configuration for the executor
        """
        self.registry = registry
        self.config = config or ToolExecutorConfig()
        
        # Execution tracking
        self._active_executions: Dict[str, ToolExecution] = {}
        self._execution_history: deque = deque(maxlen=1000)  # Keep last 1000 executions
        self._execution_stats: Dict[str, Any] = defaultdict(int)
        self._performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Concurrency control
        self._execution_semaphore = asyncio.Semaphore(self.config.max_concurrent_executions)
        
        logger.info(f"ToolExecutor initialized with max {self.config.max_concurrent_executions} concurrent executions")
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.
        
        Args:
            tool_call: Tool call to execute
            
        Returns:
            Tool execution result
        """
        # Get tool from registry
        tool = self.registry.get_tool(tool_call.tool_name)
        if not tool:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=f"Tool '{tool_call.tool_name}' not found",
                execution_time_ms=0
            )
        
        # Get tool definition
        definition = tool.get_definition()
        
        # Create execution context
        execution = ToolExecution(
            call=tool_call,
            definition=definition,
            max_attempts=definition.retry_count + 1 if self.config.enable_retries else 1
        )
        
        # Track active execution
        self._active_executions[tool_call.call_id] = execution
        
        try:
            # Execute with retry logic
            result = await self._execute_with_retries(execution, tool)
            
            # Record execution
            self._record_execution(execution, result)
            
            return result
            
        finally:
            # Remove from active executions
            self._active_executions.pop(tool_call.call_id, None)
    
    async def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute multiple tool calls concurrently.
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            List of tool execution results
        """
        if not tool_calls:
            return []
        
        # Execute all tools concurrently
        tasks = [self.execute_tool(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    call_id=tool_calls[i].call_id,
                    tool_name=tool_calls[i].tool_name,
                    success=False,
                    error=f"Execution failed: {str(result)}",
                    execution_time_ms=0
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def execute_tool_stream(self, tool_call: ToolCall) -> AsyncGenerator[ToolResult, None]:
        """
        Execute a tool with streaming results.
        
        Args:
            tool_call: Tool call to execute
            
        Yields:
            Streaming tool results
        """
        # For now, just execute normally and yield the result
        # This can be enhanced for tools that support streaming
        result = await self.execute_tool(tool_call)
        yield result
    
    def get_execution_status(self, call_id: str) -> Optional[ToolExecution]:
        """
        Get execution status for a call.
        
        Args:
            call_id: Call ID to check
            
        Returns:
            Execution status or None if not found
        """
        return self._active_executions.get(call_id)
    
    async def cancel_execution(self, call_id: str) -> bool:
        """
        Cancel a running execution.
        
        Args:
            call_id: Call ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        execution = self._active_executions.get(call_id)
        if not execution:
            return False
        
        # Mark as cancelled (actual cancellation depends on tool implementation)
        execution.status = "cancelled"
        
        # Remove from active executions
        self._active_executions.pop(call_id, None)
        
        logger.info(f"Cancelled execution: {call_id}")
        return True
    
    def get_active_executions(self) -> List[ToolExecution]:
        """Get list of currently active executions."""
        return list(self._active_executions.values())
    
    def get_execution_stats(self) -> ToolStats:
        """Get execution statistics."""
        total_executions = len(self._execution_history)
        successful_executions = sum(1 for exec_data in self._execution_history if exec_data.get('success', False))
        failed_executions = total_executions - successful_executions
        
        # Calculate average execution time
        execution_times = [exec_data.get('execution_time_ms', 0) for exec_data in self._execution_history]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        # Tool usage counts
        tool_usage = defaultdict(int)
        for exec_data in self._execution_history:
            tool_name = exec_data.get('tool_name')
            if tool_name:
                tool_usage[tool_name] += 1
        
        most_used_tool = max(tool_usage.items(), key=lambda x: x[1])[0] if tool_usage else None
        
        # Category distribution
        category_distribution = defaultdict(int)
        for tool_name in tool_usage.keys():
            definition = self.registry.get_tool_definition(tool_name)
            if definition:
                category_distribution[definition.category] += tool_usage[tool_name]
        
        return ToolStats(
            total_tools=len(self.registry.get_tool_names()),
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            average_execution_time_ms=avg_execution_time,
            most_used_tool=most_used_tool,
            tool_usage_counts=dict(tool_usage),
            category_distribution=dict(category_distribution),
            error_rate=failed_executions / total_executions * 100 if total_executions > 0 else 0.0
        )
    
    async def _execute_with_retries(self, execution: ToolExecution, tool) -> ToolResult:
        """Execute tool with retry logic."""
        last_error = None
        
        for attempt in range(execution.max_attempts):
            try:
                # Acquire semaphore for concurrency control
                async with self._execution_semaphore:
                    execution.mark_started()
                    
                    # Execute the tool
                    result = await tool.execute(
                        execution.call.parameters,
                        execution.call.context
                    )
                    
                    # Success
                    return execution.mark_completed(result)
                    
            except ToolTimeoutError as e:
                last_error = str(e)
                logger.warning(f"Tool {execution.call.tool_name} timed out (attempt {attempt + 1}/{execution.max_attempts})")
                
                if attempt < execution.max_attempts - 1:
                    # Wait before retry
                    await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff, max 10s
                
            except ToolError as e:
                last_error = str(e)
                logger.warning(f"Tool {execution.call.tool_name} failed: {e} (attempt {attempt + 1}/{execution.max_attempts})")
                
                # Don't retry if error is not recoverable
                if not e.recoverable:
                    break
                
                if attempt < execution.max_attempts - 1:
                    await asyncio.sleep(min(2 ** attempt, 10))
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error in tool {execution.call.tool_name}: {e}")
                break
        
        # All attempts failed
        return execution.mark_failed(last_error or "Tool execution failed")
    
    def _record_execution(self, execution: ToolExecution, result: ToolResult) -> None:
        """Record execution for analytics."""
        if not self.config.execution_logging:
            return
        
        # Record in history
        execution_data = {
            "call_id": execution.call.call_id,
            "tool_name": execution.call.tool_name,
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
            "attempts": execution.attempts,
            "timestamp": result.timestamp.isoformat(),
            "error": result.error
        }
        
        self._execution_history.append(execution_data)
        
        # Update stats
        self._execution_stats["total_executions"] += 1
        if result.success:
            self._execution_stats["successful_executions"] += 1
        else:
            self._execution_stats["failed_executions"] += 1
        
        # Record performance metrics
        if self.config.performance_monitoring:
            self._performance_metrics[execution.call.tool_name].append(result.execution_time_ms)
            
            # Keep only recent metrics (last 100 executions per tool)
            if len(self._performance_metrics[execution.call.tool_name]) > 100:
                self._performance_metrics[execution.call.tool_name] = \
                    self._performance_metrics[execution.call.tool_name][-100:]
    
    def get_performance_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for tools."""
        if tool_name:
            metrics = self._performance_metrics.get(tool_name, [])
            if not metrics:
                return {"tool_name": tool_name, "no_data": True}
            
            return {
                "tool_name": tool_name,
                "execution_count": len(metrics),
                "average_time_ms": sum(metrics) / len(metrics),
                "min_time_ms": min(metrics),
                "max_time_ms": max(metrics),
                "recent_executions": metrics[-10:]  # Last 10
            }
        else:
            # Return metrics for all tools
            all_metrics = {}
            for tool_name, metrics in self._performance_metrics.items():
                if metrics:
                    all_metrics[tool_name] = {
                        "execution_count": len(metrics),
                        "average_time_ms": sum(metrics) / len(metrics),
                        "min_time_ms": min(metrics),
                        "max_time_ms": max(metrics)
                    }
            
            return all_metrics
    
    def clear_history(self) -> None:
        """Clear execution history and metrics."""
        self._execution_history.clear()
        self._execution_stats.clear()
        self._performance_metrics.clear()
        logger.info("Cleared execution history and metrics")
