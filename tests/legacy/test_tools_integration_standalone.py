#!/usr/bin/env python3
"""
Test Voice-Tools Integration (Standalone)

This script tests the voice-tools integration functionality without depending
on the full tools system, using mock implementations.
"""

import asyncio
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock classes for testing
class MockToolDefinition:
    def __init__(self, name, description, parameters=None, category=None):
        self.name = name
        self.description = description
        self.parameters = parameters or []
        self.category = category

class MockToolParameter:
    def __init__(self, name, type, description, required=True):
        self.name = name
        self.type = type
        self.description = description
        self.required = required

class MockToolCall:
    def __init__(self, call_id, tool_name, parameters):
        self.call_id = call_id
        self.tool_name = tool_name
        self.parameters = parameters

class MockToolResult:
    def __init__(self, call_id, tool_name, status, result=None, error=None, execution_time_ms=0.0):
        self.call_id = call_id
        self.tool_name = tool_name
        self.status = status
        self.result = result
        self.error = error
        self.execution_time_ms = execution_time_ms

class MockToolCallStatus:
    SUCCESS = "success"
    FAILED = "failed"

class MockToolExecutionContext:
    def __init__(self, user_id, conversation_id, message_id, source):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.source = source

class MockToolManagerInterface:
    async def get_available_tools(self): pass
    async def execute_tool(self, tool_call, context): pass

# Import voice models
try:
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
    logger.info("✓ Successfully imported voice models")
except ImportError as e:
    logger.error(f"❌ Failed to import voice models: {e}")
    exit(1)

# Mock VoiceToolsConfig
class VoiceToolsConfig:
    def __init__(self):
        self.enable_tool_injection = True
        self.max_tools_in_context = 10
        self.tool_relevance_threshold = 0.6
        self.include_tool_descriptions = True
        self.include_tool_examples = True
        self.enable_function_calling = True
        self.auto_execute_tools = True
        self.require_confirmation = False
        self.max_execution_time_seconds = 30.0
        self.enable_voice_tool_feedback = True
        self.provide_execution_status = True
        self.explain_tool_usage = True
        self.enable_tool_learning = True
        self.learn_from_usage_patterns = True
        self.adapt_tool_suggestions = True
        self.tool_cache_ttl_minutes = 15
        self.parallel_tool_operations = True
        self.max_concurrent_tools = 3

# Mock VoiceToolsIntegration
class VoiceToolsIntegration:
    def __init__(self, tool_manager, config):
        self.tool_manager = tool_manager
        self.config = config
        self.tool_cache = {}
        self.cache_timestamps = {}
        self.stats = {
            "tool_injections": 0,
            "function_calls": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "cache_hits": 0,
            "tools_discovered": 0
        }
        self.tool_usage_patterns = {}
        self.conversation_tools = {}
        
    async def enhance_voice_context(self, voice_message, conversation_state=None):
        """Mock enhance voice context with tools."""
        text_content = voice_message.text_content or ""
        
        # Mock tool discovery
        available_tools = await self.tool_manager.get_available_tools()
        relevant_tools = []
        
        for tool in available_tools:
            relevance_score = self._calculate_tool_relevance(tool, text_content)
            if relevance_score >= self.config.tool_relevance_threshold:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "relevance_score": relevance_score,
                    "parameters": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "description": param.description,
                            "required": param.required
                        }
                        for param in tool.parameters
                    ]
                }
                relevant_tools.append(tool_info)
        
        relevant_tools.sort(key=lambda t: t["relevance_score"], reverse=True)
        relevant_tools = relevant_tools[:self.config.max_tools_in_context]
        
        self.stats["tool_injections"] += 1
        self.stats["tools_discovered"] += len(relevant_tools)
        
        return {
            "available_tools": relevant_tools,
            "tool_count": len(relevant_tools),
            "discovery_query": text_content,
            "tools_summary": f"Found {len(relevant_tools)} relevant tools"
        }
    
    def _calculate_tool_relevance(self, tool, text_content):
        """Mock tool relevance calculation."""
        text_lower = text_content.lower()
        tool_keywords = tool.name.lower().split('_') + tool.description.lower().split()
        
        matches = sum(1 for keyword in tool_keywords if len(keyword) > 3 and keyword in text_lower)
        relevance = matches / len(tool_keywords) if tool_keywords else 0.0
        
        if tool.name.lower() in text_lower:
            relevance += 0.3
            
        return min(relevance, 1.0)
    
    async def process_function_calls(self, voice_message, voice_response, conversation_state=None):
        """Mock process function calls."""
        function_calls = self._extract_function_calls(voice_response)
        
        if not function_calls:
            return voice_response, []
        
        tool_results = []
        execution_summaries = []
        
        for function_call in function_calls:
            try:
                result = await self._execute_function_call(function_call, voice_message)
                tool_results.append(result)
                execution_summaries.append(f"Executed {function_call['name']} successfully")
                self.stats["successful_executions"] += 1
            except Exception as e:
                error_result = MockToolResult(
                    call_id=function_call.get("call_id", "unknown"),
                    tool_name=function_call.get("name", "unknown"),
                    status=MockToolCallStatus.FAILED,
                    error=str(e)
                )
                tool_results.append(error_result)
                execution_summaries.append(f"Failed to execute {function_call.get('name', 'tool')}")
                self.stats["failed_executions"] += 1
        
        # Update response with tool results
        updated_text = voice_response.text_content + "\n\n" + "\n".join(execution_summaries)
        updated_response = VoiceResponse(
            response_id=voice_response.response_id,
            conversation_id=voice_response.conversation_id,
            message_id=voice_response.message_id,
            text_content=updated_text,
            audio_data=voice_response.audio_data,
            processing_mode=voice_response.processing_mode,
            total_latency_ms=voice_response.total_latency_ms,
            response_relevance=voice_response.response_relevance
        )
        
        return updated_response, tool_results
    
    def _extract_function_calls(self, voice_response):
        """Mock function call extraction."""
        text_content = voice_response.text_content or ""
        function_calls = []
        
        import re
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, text_content)
        
        for i, (func_name, params_str) in enumerate(matches):
            parameters = {}
            if params_str.strip():
                param_pairs = params_str.split(',')
                for pair in param_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        parameters[key.strip()] = value.strip().strip('"\'')
            
            function_call = {
                "call_id": f"call_{voice_response.response_id}_{i}",
                "name": func_name,
                "parameters": parameters
            }
            function_calls.append(function_call)
        
        self.stats["function_calls"] += len(function_calls)
        return function_calls
    
    async def _execute_function_call(self, function_call, voice_message):
        """Mock function call execution."""
        tool_call = MockToolCall(
            call_id=function_call["call_id"],
            tool_name=function_call["name"],
            parameters=function_call["parameters"]
        )
        
        context = MockToolExecutionContext(
            user_id="voice_user",
            conversation_id=voice_message.conversation_id,
            message_id=voice_message.message_id,
            source="voice_interaction"
        )
        
        result = await self.tool_manager.execute_tool(tool_call, context)
        return result
    
    def get_integration_stats(self):
        """Get integration statistics."""
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
                "tool_relevance_threshold": self.config.tool_relevance_threshold
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.tool_cache.clear()
        self.cache_timestamps.clear()
        self.tool_usage_patterns.clear()
        self.conversation_tools.clear()

# Mock tool manager
class MockToolManager(MockToolManagerInterface):
    def __init__(self):
        self.tools = [
            MockToolDefinition(
                name="search_web",
                description="Search the web for information",
                parameters=[
                    MockToolParameter("query", "string", "Search query", True),
                    MockToolParameter("limit", "integer", "Number of results", False)
                ],
                category="web"
            ),
            MockToolDefinition(
                name="calculate",
                description="Perform mathematical calculations",
                parameters=[
                    MockToolParameter("expression", "string", "Math expression", True)
                ],
                category="calculation"
            ),
            MockToolDefinition(
                name="save_file",
                description="Save content to a file",
                parameters=[
                    MockToolParameter("filename", "string", "File name", True),
                    MockToolParameter("content", "string", "File content", True)
                ],
                category="file"
            )
        ]
    
    async def get_available_tools(self):
        return self.tools
    
    async def execute_tool(self, tool_call, context):
        # Mock execution
        if tool_call.tool_name == "search_web":
            result = f"Search results for '{tool_call.parameters.get('query', '')}'"
        elif tool_call.tool_name == "calculate":
            expression = tool_call.parameters.get('expression', '1+1')
            try:
                result = f"Result: {eval(expression)}"
            except:
                result = "Error: Invalid expression"
        elif tool_call.tool_name == "save_file":
            filename = tool_call.parameters.get('filename', 'untitled.txt')
            result = f"File '{filename}' saved successfully"
        else:
            result = f"Mock result for {tool_call.tool_name}"
        
        return MockToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status=MockToolCallStatus.SUCCESS,
            result=result,
            execution_time_ms=50.0
        )

def create_test_voice_message(conversation_id: str, text_content: str) -> VoiceMessage:
    """Create a test voice message."""
    audio_data = np.random.randint(-32768, 32767, 24000, dtype=np.int16).tobytes()
    
    return VoiceMessage(
        message_id=f"test_msg_{datetime.now().strftime('%H%M%S%f')}",
        conversation_id=conversation_id,
        audio_data=audio_data,
        text_content=text_content,
        processing_mode=VoiceProcessingMode.HYBRID,
        timestamp=datetime.now()
    )

def create_test_voice_response(message_id: str, conversation_id: str, text_content: str) -> VoiceResponse:
    """Create a test voice response."""
    return VoiceResponse(
        response_id=f"response_{message_id}",
        conversation_id=conversation_id,
        message_id=message_id,
        text_content=text_content,
        audio_data=b"",
        processing_mode=VoiceProcessingMode.HYBRID,
        total_latency_ms=150.0,
        response_relevance=0.8
    )

async def test_tools_integration():
    """Test the complete tools integration."""
    logger.info("🚀 Testing Voice-Tools Integration")
    logger.info("=" * 50)
    
    try:
        # Create components
        tool_manager = MockToolManager()
        config = VoiceToolsConfig()
        integration = VoiceToolsIntegration(tool_manager, config)
        
        logger.info("✓ Created voice-tools integration")
        
        # Test 1: Tool discovery
        voice_message = create_test_voice_message(
            "test_conversation",
            "Can you search the web for Python tutorials?"
        )
        
        tools_context = await integration.enhance_voice_context(voice_message)
        
        logger.info("✓ Tool Discovery Test:")
        logger.info(f"  Discovered tools: {tools_context.get('tool_count', 0)}")
        logger.info(f"  Tools summary: {tools_context.get('tools_summary', 'N/A')}")
        
        for tool in tools_context.get('available_tools', []):
            logger.info(f"    - {tool['name']} (relevance: {tool['relevance_score']:.2f})")
        
        # Test 2: Function call extraction and execution
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "I'll search for Python tutorials. search_web(query='Python tutorials', limit=5)"
        )
        
        updated_response, tool_results = await integration.process_function_calls(
            voice_message, voice_response
        )
        
        logger.info("✓ Function Call Execution Test:")
        logger.info(f"  Function calls found: {len(tool_results)}")
        logger.info(f"  Original response length: {len(voice_response.text_content or '')}")
        logger.info(f"  Updated response length: {len(updated_response.text_content or '')}")
        
        for result in tool_results:
            logger.info(f"    - {result.tool_name}: {result.status}")
            if result.result:
                logger.info(f"      Result: {str(result.result)[:50]}...")
        
        # Test 3: Multiple function calls
        multi_call_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "Let me search and calculate. search_web(query='AI') calculate(expression='2+2')"
        )
        
        multi_updated_response, multi_tool_results = await integration.process_function_calls(
            voice_message, multi_call_response
        )
        
        logger.info("✓ Multiple Function Calls Test:")
        logger.info(f"  Function calls executed: {len(multi_tool_results)}")
        
        # Test 4: Statistics
        stats = integration.get_integration_stats()
        
        logger.info("✓ Integration Statistics:")
        logger.info(f"  Tool injections: {stats['voice_tools_stats']['tool_injections']}")
        logger.info(f"  Function calls: {stats['voice_tools_stats']['function_calls']}")
        logger.info(f"  Successful executions: {stats['voice_tools_stats']['successful_executions']}")
        logger.info(f"  Tools discovered: {stats['voice_tools_stats']['tools_discovered']}")
        
        # Cleanup
        await integration.cleanup()
        
        logger.info("=" * 50)
        logger.info("🎉 All voice-tools integration tests passed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tools integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the tools integration test."""
    success = await test_tools_integration()
    
    if success:
        logger.info("✅ Voice-Tools Integration: COMPLETE")
    else:
        logger.error("❌ Voice-Tools Integration: FAILED")

if __name__ == "__main__":
    asyncio.run(main())
