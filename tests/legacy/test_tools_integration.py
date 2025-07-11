#!/usr/bin/env python3
"""
Test Voice-Tools Integration

This script tests the integration between voice processing and tools systems.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.tools_integration import (
        VoiceToolsIntegration, VoiceToolsConfig, VoiceToolsManager
    )
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
    logger.info("✓ Successfully imported voice-tools integration components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    # Create mock classes for testing

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

    # Use mock classes
    ToolDefinition = MockToolDefinition
    ToolParameter = MockToolParameter
    ToolCall = MockToolCall
    ToolResult = MockToolResult
    ToolCallStatus = MockToolCallStatus
    ToolExecutionContext = MockToolExecutionContext

    logger.info("✓ Using mock tools components for testing")


class MockToolManager:
    """Mock tool manager for testing."""
    
    def __init__(self):
        self.tools = self._create_mock_tools()
        self.execution_results = {}
        
    def _create_mock_tools(self) -> List[ToolDefinition]:
        """Create mock tools for testing."""
        
        tools = [
            ToolDefinition(
                name="search_web",
                description="Search the web for information",
                parameters=[
                    ToolParameter(name="query", type="string", description="Search query", required=True),
                    ToolParameter(name="limit", type="integer", description="Number of results", required=False)
                ],
                category="web"
            ),
            ToolDefinition(
                name="calculate",
                description="Perform mathematical calculations",
                parameters=[
                    ToolParameter(name="expression", type="string", description="Math expression", required=True)
                ],
                category="calculation"
            ),
            ToolDefinition(
                name="save_file",
                description="Save content to a file",
                parameters=[
                    ToolParameter(name="filename", type="string", description="File name", required=True),
                    ToolParameter(name="content", type="string", description="File content", required=True)
                ],
                category="file"
            ),
            ToolDefinition(
                name="get_weather",
                description="Get weather information for a location",
                parameters=[
                    ToolParameter(name="location", type="string", description="Location name", required=True)
                ],
                category="web"
            )
        ]
        
        return tools
    
    async def get_available_tools(self) -> List[ToolDefinition]:
        """Get available tools."""
        return self.tools
    
    async def execute_tool(self, tool_call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        """Execute a tool call."""
        
        # Mock execution based on tool name
        if tool_call.tool_name == "search_web":
            result = f"Search results for '{tool_call.parameters.get('query', '')}': Found 5 relevant articles"
            
        elif tool_call.tool_name == "calculate":
            expression = tool_call.parameters.get('expression', '1+1')
            try:
                # Simple calculation
                result = f"Result: {eval(expression)}"
            except:
                result = "Error: Invalid expression"
                
        elif tool_call.tool_name == "save_file":
            filename = tool_call.parameters.get('filename', 'untitled.txt')
            result = f"File '{filename}' saved successfully"
            
        elif tool_call.tool_name == "get_weather":
            location = tool_call.parameters.get('location', 'Unknown')
            result = f"Weather in {location}: 22°C, partly cloudy"
            
        else:
            result = f"Mock result for {tool_call.tool_name}"
        
        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status=ToolCallStatus.SUCCESS,
            result=result,
            error=None,
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


async def test_voice_tools_config():
    """Test voice tools configuration."""
    logger.info("=== Testing Voice Tools Configuration ===")
    
    try:
        config = VoiceToolsConfig()
        
        logger.info("✓ Default configuration created")
        logger.info(f"  Tool injection enabled: {config.enable_tool_injection}")
        logger.info(f"  Function calling enabled: {config.enable_function_calling}")
        logger.info(f"  Auto execute tools: {config.auto_execute_tools}")
        logger.info(f"  Max tools in context: {config.max_tools_in_context}")
        logger.info(f"  Tool relevance threshold: {config.tool_relevance_threshold}")
        
        # Test custom configuration
        custom_config = VoiceToolsConfig(
            enable_tool_injection=False,
            max_tools_in_context=5,
            tool_relevance_threshold=0.8
        )
        
        logger.info("✓ Custom configuration created")
        logger.info(f"  Tool injection enabled: {custom_config.enable_tool_injection}")
        logger.info(f"  Max tools in context: {custom_config.max_tools_in_context}")
        
        logger.info("✓ Voice tools configuration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_tool_discovery():
    """Test tool discovery and relevance calculation."""
    logger.info("=== Testing Tool Discovery ===")
    
    try:
        tool_manager = MockToolManager()
        config = VoiceToolsConfig()
        integration = VoiceToolsIntegration(tool_manager, config)
        
        logger.info("✓ Voice tools integration created")
        
        # Test tool discovery for different queries
        test_queries = [
            "Can you search the web for information about Python?",
            "I need to calculate 15 * 23 + 7",
            "Please save this data to a file",
            "What's the weather like in New York?"
        ]
        
        for query in test_queries:
            voice_message = create_test_voice_message("test_conversation", query)
            tools_context = await integration.enhance_voice_context(voice_message)
            
            available_tools = tools_context.get("available_tools", [])
            
            logger.info(f"✓ Query: '{query[:50]}...'")
            logger.info(f"  Discovered tools: {len(available_tools)}")
            
            for tool in available_tools:
                logger.info(f"    - {tool['name']} (relevance: {tool['relevance_score']:.2f})")
        
        await integration.cleanup()
        logger.info("✓ Tool discovery test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_function_call_extraction():
    """Test function call extraction from voice responses."""
    logger.info("=== Testing Function Call Extraction ===")
    
    try:
        tool_manager = MockToolManager()
        config = VoiceToolsConfig()
        integration = VoiceToolsIntegration(tool_manager, config)
        
        # Test responses with function calls
        test_responses = [
            "I'll search for that information. search_web(query='Python programming', limit=5)",
            "Let me calculate that for you. calculate(expression='15 * 23 + 7')",
            "I'll save this to a file. save_file(filename='data.txt', content='Hello World')",
            "No function calls in this response."
        ]
        
        for i, response_text in enumerate(test_responses):
            voice_message = create_test_voice_message("test_conversation", f"Test query {i+1}")
            voice_response = create_test_voice_response(
                voice_message.message_id,
                voice_message.conversation_id,
                response_text
            )
            
            function_calls = await integration._extract_function_calls(voice_response)
            
            logger.info(f"✓ Response: '{response_text[:50]}...'")
            logger.info(f"  Function calls found: {len(function_calls)}")
            
            for call in function_calls:
                logger.info(f"    - {call['name']}({call['parameters']})")
        
        await integration.cleanup()
        logger.info("✓ Function call extraction test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Function call extraction test failed: {e}")
        return False


async def test_tool_execution():
    """Test tool execution functionality."""
    logger.info("=== Testing Tool Execution ===")
    
    try:
        tool_manager = MockToolManager()
        config = VoiceToolsConfig()
        integration = VoiceToolsIntegration(tool_manager, config)
        
        # Test tool execution
        voice_message = create_test_voice_message(
            "test_conversation",
            "Can you search for Python tutorials?"
        )
        
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "I'll search for Python tutorials. search_web(query='Python tutorials', limit=3)"
        )
        
        updated_response, tool_results = await integration.process_function_calls(
            voice_message, voice_response
        )
        
        logger.info("✓ Tool execution completed")
        logger.info(f"  Original response: {voice_response.text_content}")
        logger.info(f"  Updated response: {updated_response.text_content}")
        logger.info(f"  Tool results: {len(tool_results)}")
        
        for result in tool_results:
            logger.info(f"    - {result.tool_name}: {result.status.value}")
            if result.result:
                logger.info(f"      Result: {str(result.result)[:50]}...")
        
        await integration.cleanup()
        logger.info("✓ Tool execution test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool execution test failed: {e}")
        return False


async def test_tool_usage_tracking():
    """Test tool usage tracking and learning."""
    logger.info("=== Testing Tool Usage Tracking ===")
    
    try:
        tool_manager = MockToolManager()
        config = VoiceToolsConfig()
        integration = VoiceToolsIntegration(tool_manager, config)
        
        # Simulate multiple tool uses
        test_interactions = [
            ("Can you search for machine learning?", "search_web(query='machine learning')"),
            ("Calculate 25 * 4", "calculate(expression='25 * 4')"),
            ("Search for Python tutorials", "search_web(query='Python tutorials')"),
            ("What's 100 / 5?", "calculate(expression='100 / 5')")
        ]
        
        for user_query, response_text in test_interactions:
            voice_message = create_test_voice_message("test_conversation", user_query)
            voice_response = create_test_voice_response(
                voice_message.message_id,
                voice_message.conversation_id,
                response_text
            )
            
            await integration.process_function_calls(voice_message, voice_response)
        
        # Check usage tracking
        stats = integration.get_integration_stats()
        
        logger.info("✓ Tool usage tracking completed")
        logger.info(f"  Function calls: {stats['voice_tools_stats']['function_calls']}")
        logger.info(f"  Successful executions: {stats['voice_tools_stats']['successful_executions']}")
        logger.info(f"  Tracked tools: {stats['tracked_tools']}")
        logger.info(f"  Active conversations: {stats['active_conversations']}")
        
        # Test tool suggestions
        new_message = create_test_voice_message("test_conversation", "I need to do some math")
        suggestions = await integration.get_tool_suggestions(new_message)
        
        logger.info(f"✓ Tool suggestions for math query: {len(suggestions)}")
        for suggestion in suggestions:
            logger.info(f"    - {suggestion['tool_name']} (score: {suggestion['relevance_score']:.2f})")
        
        await integration.cleanup()
        logger.info("✓ Tool usage tracking test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool usage tracking test failed: {e}")
        return False


async def test_tools_caching():
    """Test tools context caching."""
    logger.info("=== Testing Tools Caching ===")
    
    try:
        tool_manager = MockToolManager()
        config = VoiceToolsConfig(tool_cache_ttl_minutes=1)
        integration = VoiceToolsIntegration(tool_manager, config)
        
        # Test caching with same query
        voice_message = create_test_voice_message(
            "test_conversation",
            "Can you search for information about AI?"
        )
        
        # First call (should cache)
        context1 = await integration.enhance_voice_context(voice_message)
        
        # Second call (should use cache)
        context2 = await integration.enhance_voice_context(voice_message)
        
        # Check cache statistics
        stats = integration.get_integration_stats()
        cache_hits = stats['voice_tools_stats']['cache_hits']
        
        logger.info(f"✓ Tools caching test:")
        logger.info(f"  Cache hits: {cache_hits}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        
        # Verify contexts are similar (from cache)
        assert context1.get('discovery_query') == context2.get('discovery_query')
        logger.info("✓ Cached context matches original")
        
        await integration.cleanup()
        logger.info("✓ Tools caching test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Caching test failed: {e}")
        return False


async def test_voice_tools_manager():
    """Test high-level voice tools manager."""
    logger.info("=== Testing Voice Tools Manager ===")
    
    try:
        tool_manager = MockToolManager()
        config = VoiceToolsConfig()
        voice_tools_manager = VoiceToolsManager(tool_manager, config)
        
        logger.info("✓ Voice tools manager created")
        
        # Test context enhancement
        voice_message = create_test_voice_message(
            "test_conversation",
            "I need to search for information about quantum computing"
        )
        
        enhanced_context = await voice_tools_manager.enhance_voice_context(voice_message)
        
        logger.info("✓ Voice context enhanced with tools")
        logger.info(f"  Available tools: {enhanced_context.get('tool_count', 0)}")
        logger.info(f"  Tools summary: {enhanced_context.get('tools_summary', 'N/A')}")
        
        # Test voice processing with tools
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "I'll search for quantum computing information. search_web(query='quantum computing')"
        )
        
        updated_response, tool_results = await voice_tools_manager.process_voice_with_tools(
            voice_message, voice_response
        )
        
        logger.info("✓ Voice processed with tools")
        logger.info(f"  Tool results: {len(tool_results)}")
        logger.info(f"  Response updated: {len(updated_response.text_content or '') > len(voice_response.text_content or '')}")
        
        # Test tool suggestions
        suggestions = await voice_tools_manager.get_tool_suggestions(voice_message)
        
        logger.info(f"✓ Tool suggestions: {len(suggestions)}")
        for suggestion in suggestions[:3]:
            logger.info(f"    - {suggestion['tool_name']} (relevance: {suggestion['relevance_score']:.2f})")
        
        # Test statistics
        stats = voice_tools_manager.get_tools_stats()
        logger.info(f"✓ Tools statistics:")
        logger.info(f"  Tool injections: {stats['voice_tools_stats']['tool_injections']}")
        logger.info(f"  Function calls: {stats['voice_tools_stats']['function_calls']}")
        
        await voice_tools_manager.cleanup()
        logger.info("✓ Voice tools manager test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tools manager test failed: {e}")
        return False


async def main():
    """Run all voice-tools integration tests."""
    logger.info("🚀 Starting Voice-Tools Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Voice Tools Configuration", test_voice_tools_config),
        ("Tool Discovery", test_tool_discovery),
        ("Function Call Extraction", test_function_call_extraction),
        ("Tool Execution", test_tool_execution),
        ("Tool Usage Tracking", test_tool_usage_tracking),
        ("Tools Caching", test_tools_caching),
        ("Voice Tools Manager", test_voice_tools_manager),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "❌ CRASHED"
    
    # Print results summary
    logger.info("=" * 50)
    logger.info("🏁 Voice-Tools Integration Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<30}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice-tools integration tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
