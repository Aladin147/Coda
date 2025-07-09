"""
Function calling orchestrator for Coda LLM system.

This module provides orchestration of function calls between LLMs and tools.
"""

import json
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .interfaces import FunctionCallingOrchestratorInterface
from .models import (
    LLMMessage,
    LLMResponse,
    FunctionCall,
    FunctionCallResult,
    MessageRole,
)

logger = logging.getLogger("coda.llm.function_calling")


class FunctionCallingOrchestrator(FunctionCallingOrchestratorInterface):
    """
    Orchestrates function calls between LLMs and tools.
    
    Features:
    - Function call parsing and validation
    - Tool integration and execution
    - Result formatting for LLMs
    - Parallel function execution
    - Error handling and recovery
    """
    
    def __init__(self):
        """Initialize the function calling orchestrator."""
        self._tool_manager: Optional[Any] = None
        self._execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_execution_time": 0.0
        }
        
        logger.info("FunctionCallingOrchestrator initialized")
    
    async def process_function_calls(
        self,
        function_calls: List[FunctionCall],
        context: Optional[Dict[str, Any]] = None
    ) -> List[FunctionCallResult]:
        """
        Process function calls and return results.
        
        Args:
            function_calls: List of function calls to process
            context: Optional execution context
            
        Returns:
            List of function call results
        """
        if not function_calls:
            return []
        
        if not self._tool_manager:
            logger.warning("No tool manager available for function calling")
            return [
                FunctionCallResult(
                    call_id=call.call_id,
                    function_name=call.name,
                    success=False,
                    error="Tool manager not available",
                    execution_time_ms=0
                )
                for call in function_calls
            ]
        
        results = []
        
        # Process each function call
        for function_call in function_calls:
            try:
                result = await self._execute_function_call(function_call, context)
                results.append(result)
                
                # Update statistics
                self._execution_stats["total_calls"] += 1
                if result.success:
                    self._execution_stats["successful_calls"] += 1
                else:
                    self._execution_stats["failed_calls"] += 1
                
            except Exception as e:
                logger.error(f"Function call execution failed: {e}")
                results.append(
                    FunctionCallResult(
                        call_id=function_call.call_id,
                        function_name=function_call.name,
                        success=False,
                        error=f"Execution failed: {str(e)}",
                        execution_time_ms=0
                    )
                )
                self._execution_stats["total_calls"] += 1
                self._execution_stats["failed_calls"] += 1
        
        logger.info(f"Processed {len(function_calls)} function calls")
        return results
    
    async def get_available_functions(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available functions for LLM.
        
        Args:
            context: Optional context for filtering functions
            
        Returns:
            List of function schemas
        """
        if not self._tool_manager:
            return []
        
        try:
            # Get function schemas from tool manager
            schemas = self._tool_manager.get_function_schemas(context)
            
            logger.debug(f"Retrieved {len(schemas)} available functions")
            return schemas
            
        except Exception as e:
            logger.error(f"Failed to get available functions: {e}")
            return []
    
    def parse_function_calls_from_response(
        self,
        response: LLMResponse
    ) -> List[FunctionCall]:
        """
        Parse function calls from LLM response.
        
        Args:
            response: LLM response containing function calls
            
        Returns:
            List of parsed function calls
        """
        function_calls = []
        
        try:
            # Parse function_calls (OpenAI format)
            for func_call in response.function_calls:
                call = FunctionCall(
                    call_id=str(uuid.uuid4()),
                    name=func_call["name"],
                    arguments=json.loads(func_call["arguments"]) if isinstance(func_call["arguments"], str) else func_call["arguments"],
                    timestamp=datetime.now()
                )
                function_calls.append(call)
            
            # Parse tool_calls (newer OpenAI format)
            for tool_call in response.tool_calls:
                if tool_call["type"] == "function":
                    function_data = tool_call["function"]
                    call = FunctionCall(
                        call_id=tool_call.get("id", str(uuid.uuid4())),
                        name=function_data["name"],
                        arguments=json.loads(function_data["arguments"]) if isinstance(function_data["arguments"], str) else function_data["arguments"],
                        timestamp=datetime.now()
                    )
                    function_calls.append(call)
            
            logger.debug(f"Parsed {len(function_calls)} function calls from response")
            return function_calls
            
        except Exception as e:
            logger.error(f"Failed to parse function calls from response: {e}")
            return []
    
    def format_function_results_for_llm(
        self,
        results: List[FunctionCallResult]
    ) -> List[LLMMessage]:
        """
        Format function results as messages for LLM.
        
        Args:
            results: List of function call results
            
        Returns:
            List of formatted messages
        """
        messages = []
        
        for result in results:
            # Format result content
            if result.success:
                content = self._format_successful_result(result)
            else:
                content = f"Error executing {result.function_name}: {result.error}"
            
            # Create function result message
            message = LLMMessage(
                role=MessageRole.FUNCTION,
                content=content,
                name=result.function_name,
                timestamp=datetime.now(),
                metadata={
                    "call_id": result.call_id,
                    "execution_time_ms": result.execution_time_ms,
                    "success": result.success
                }
            )
            
            messages.append(message)
        
        return messages
    
    async def validate_function_call(
        self,
        function_call: FunctionCall
    ) -> bool:
        """
        Validate a function call.
        
        Args:
            function_call: Function call to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self._tool_manager:
            return False
        
        try:
            # Check if tool exists
            tool = self._tool_manager.get_tool_by_name(function_call.name)
            if not tool:
                logger.warning(f"Tool not found: {function_call.name}")
                return False
            
            # Validate parameters
            definition = tool.get_definition()
            validated_params = definition.validate_parameters(function_call.arguments)
            
            return True
            
        except Exception as e:
            logger.warning(f"Function call validation failed: {e}")
            return False
    
    def set_tool_manager(self, tool_manager: Any) -> None:
        """Set tool manager for function execution."""
        self._tool_manager = tool_manager
        logger.info("Tool manager set for function calling orchestrator")
    
    async def _execute_function_call(
        self,
        function_call: FunctionCall,
        context: Optional[Dict[str, Any]] = None
    ) -> FunctionCallResult:
        """Execute a single function call."""
        start_time = datetime.now()
        
        try:
            # Validate function call
            if not await self.validate_function_call(function_call):
                return FunctionCallResult(
                    call_id=function_call.call_id,
                    function_name=function_call.name,
                    success=False,
                    error="Function call validation failed",
                    execution_time_ms=0
                )
            
            # Convert to tool manager format
            tool_call_data = {
                "name": function_call.name,
                "arguments": json.dumps(function_call.arguments)
            }
            
            # Execute via tool manager
            result = await self._tool_manager.process_function_call(tool_call_data)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return FunctionCallResult(
                call_id=function_call.call_id,
                function_name=function_call.name,
                success=result.success,
                result=result.result,
                error=result.error,
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return FunctionCallResult(
                call_id=function_call.call_id,
                function_name=function_call.name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )
    
    def _format_successful_result(self, result: FunctionCallResult) -> str:
        """Format a successful function result."""
        if result.result is None:
            return f"{result.function_name} executed successfully"
        
        if isinstance(result.result, str):
            return result.result
        elif isinstance(result.result, (dict, list)):
            try:
                return json.dumps(result.result, indent=2)
            except (TypeError, ValueError):
                return str(result.result)
        else:
            return str(result.result)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get function calling execution statistics."""
        total_calls = self._execution_stats["total_calls"]
        
        if total_calls > 0:
            success_rate = (self._execution_stats["successful_calls"] / total_calls) * 100
        else:
            success_rate = 0.0
        
        return {
            **self._execution_stats,
            "success_rate": success_rate,
            "tool_manager_available": self._tool_manager is not None
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_execution_time": 0.0
        }
    
    async def execute_function_calls_parallel(
        self,
        function_calls: List[FunctionCall],
        context: Optional[Dict[str, Any]] = None
    ) -> List[FunctionCallResult]:
        """
        Execute function calls in parallel.
        
        Args:
            function_calls: List of function calls to execute
            context: Optional execution context
            
        Returns:
            List of function call results
        """
        if not function_calls:
            return []
        
        import asyncio
        
        # Create tasks for parallel execution
        tasks = [
            self._execute_function_call(call, context)
            for call in function_calls
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    FunctionCallResult(
                        call_id=function_calls[i].call_id,
                        function_name=function_calls[i].name,
                        success=False,
                        error=f"Parallel execution failed: {str(result)}",
                        execution_time_ms=0
                    )
                )
            else:
                final_results.append(result)
        
        return final_results
    
    def create_function_call_from_dict(self, call_data: Dict[str, Any]) -> FunctionCall:
        """Create FunctionCall from dictionary data."""
        return FunctionCall(
            call_id=call_data.get("id", str(uuid.uuid4())),
            name=call_data["name"],
            arguments=call_data.get("arguments", {}),
            timestamp=datetime.now()
        )
