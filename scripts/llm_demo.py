#!/usr/bin/env python3
"""
LLM system demonstration script.

This script demonstrates the LLM system functionality by:
1. Setting up multiple LLM providers
2. Demonstrating conversation management
3. Showing function calling capabilities
4. Testing streaming responses
5. Displaying WebSocket integration
6. Showing memory and personality integration
"""

import asyncio
import logging
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coda.components.llm.models import (
    LLMConfig,
    ProviderConfig,
    LLMProvider,
    MessageRole,
    LLMMessage,
)
from coda.components.llm.manager import LLMManager
from coda.components.llm.websocket_integration import WebSocketLLMManager
from coda.components.llm.providers.openai_provider import OpenAIProvider
from coda.components.tools.manager import ToolManager
from coda.components.memory.manager import MemoryManager
from coda.components.personality.manager import PersonalityManager
from coda.interfaces.websocket.server import CodaWebSocketServer
from coda.interfaces.websocket.integration import CodaWebSocketIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_demo")


class MockLLMProvider:
    """Mock LLM provider for demonstration."""
    
    def __init__(self, config):
        self.config = config
        self._available = True
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
    
    def get_provider_name(self):
        return LLMProvider.OPENAI
    
    def get_model_name(self):
        return self.config.model
    
    async def generate_response(self, messages, stream=False, functions=None, **kwargs):
        self._request_count += 1
        
        # Simulate response based on last message
        last_message = messages[-1] if messages else None
        content = f"Mock response to: {last_message.content if last_message else 'unknown'}"
        
        if stream:
            # Return async generator for streaming
            async def mock_stream():
                for i, word in enumerate(content.split()):
                    from coda.components.llm.models import LLMStreamChunk
                    yield LLMStreamChunk(
                        chunk_id=f"chunk-{i}",
                        content=" ".join(content.split()[:i+1]),
                        delta=word + " " if i < len(content.split()) - 1 else word,
                        finish_reason="stop" if i == len(content.split()) - 1 else None
                    )
                    await asyncio.sleep(0.1)  # Simulate streaming delay
            
            return mock_stream()
        else:
            from coda.components.llm.models import LLMResponse
            return LLMResponse(
                response_id="mock-response",
                content=content,
                provider=LLMProvider.OPENAI,
                model=self.config.model,
                prompt_tokens=len(str(messages)) // 4,
                completion_tokens=len(content) // 4,
                total_tokens=(len(str(messages)) + len(content)) // 4,
                response_time_ms=100.0
            )
    
    async def generate_streaming_response(self, messages, functions=None, **kwargs):
        async for chunk in await self.generate_response(messages, stream=True, functions=functions, **kwargs):
            yield chunk
    
    async def validate_connection(self):
        return self._available
    
    def is_available(self):
        return self._available
    
    def get_request_count(self):
        return self._request_count
    
    def get_total_tokens(self):
        return self._total_tokens
    
    def get_total_cost(self):
        return self._total_cost
    
    def get_last_error(self):
        return None
    
    def supports_streaming(self):
        return True
    
    def supports_function_calling(self):
        return True
    
    async def cleanup(self):
        pass


async def demonstrate_basic_llm():
    """Demonstrate basic LLM functionality."""
    logger.info("🤖 Starting basic LLM demonstration...")
    
    # Create LLM configuration
    config = LLMConfig(
        providers={
            "mock_openai": ProviderConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
        },
        default_provider="mock_openai"
    )
    
    # Create LLM manager
    llm_manager = LLMManager(config)
    
    # Replace with mock provider for demo
    mock_provider = MockLLMProvider(config.providers["mock_openai"])
    llm_manager._providers["mock_openai"] = mock_provider
    
    # Test basic response generation
    logger.info("💬 Generating basic response...")
    response = await llm_manager.generate_response("Hello, how are you today?")
    logger.info(f"Response: {response.content}")
    logger.info(f"Tokens: {response.total_tokens}, Time: {response.response_time_ms}ms")
    
    # Test conversation continuation
    logger.info("🔄 Testing conversation continuation...")
    conversation_id = response.conversation_id
    
    response2 = await llm_manager.continue_conversation(
        conversation_id, 
        "What's the weather like?"
    )
    logger.info(f"Response 2: {response2.content}")
    
    # Get conversation
    conversation = await llm_manager.get_conversation(conversation_id)
    logger.info(f"Conversation has {len(conversation.messages)} messages")
    
    # Test streaming response
    logger.info("📡 Testing streaming response...")
    print("Streaming response: ", end="", flush=True)
    
    async for chunk in llm_manager.generate_response(
        "Tell me a short story", 
        conversation_id=conversation_id,
        stream=True
    ):
        print(chunk.delta, end="", flush=True)
    
    print()  # New line after streaming
    
    logger.info("✅ Basic LLM demonstration completed!")


async def demonstrate_provider_management():
    """Demonstrate provider management."""
    logger.info("🔧 Starting provider management demonstration...")
    
    # Create configuration with multiple providers
    config = LLMConfig(
        providers={
            "openai": ProviderConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo"
            ),
            "anthropic": ProviderConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-sonnet"
            ),
            "ollama": ProviderConfig(
                provider=LLMProvider.OLLAMA,
                model="llama2",
                api_base="http://localhost:11434"
            )
        },
        default_provider="openai"
    )
    
    llm_manager = LLMManager(config)
    
    # Replace with mock providers
    for provider_name, provider_config in config.providers.items():
        mock_provider = MockLLMProvider(provider_config)
        llm_manager._providers[provider_name] = mock_provider
    
    # List providers
    providers = await llm_manager.list_providers()
    logger.info(f"📋 Available providers: {providers}")
    
    # Check provider status
    for provider in providers:
        status = await llm_manager.get_provider_status(provider)
        logger.info(f"  {provider}: {'✅' if status['available'] else '❌'} {status.get('model', 'unknown')}")
    
    # Test different providers
    test_prompt = "What is artificial intelligence?"
    
    for provider in providers[:2]:  # Test first 2 providers
        logger.info(f"🧪 Testing provider: {provider}")
        try:
            response = await llm_manager.generate_response(
                test_prompt,
                provider=provider
            )
            logger.info(f"  Response: {response.content[:50]}...")
        except Exception as e:
            logger.error(f"  Error: {e}")
    
    # Get analytics
    analytics = await llm_manager.get_analytics()
    logger.info("📊 LLM Analytics:")
    logger.info(f"  Providers: {len(analytics['providers'])}")
    logger.info(f"  Total conversations: {analytics['conversations']['total_conversations']}")
    
    logger.info("✅ Provider management demonstration completed!")


async def demonstrate_function_calling():
    """Demonstrate function calling integration."""
    logger.info("🔧 Starting function calling demonstration...")
    
    # Create LLM manager
    config = LLMConfig(
        function_calling={"enabled": True, "auto_execute": True}
    )
    llm_manager = LLMManager(config)
    
    # Add mock provider
    mock_provider = MockLLMProvider(ProviderConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo"
    ))
    llm_manager._providers["mock"] = mock_provider
    llm_manager._default_provider = "mock"
    
    # Create and integrate tool manager
    tool_manager = ToolManager()
    await llm_manager.set_tool_manager(tool_manager)
    
    # Get available functions
    functions = await llm_manager.function_calling.get_available_functions()
    logger.info(f"🛠️ Available functions: {len(functions)}")
    
    for func in functions[:5]:  # Show first 5 functions
        logger.info(f"  - {func['name']}: {func['description']}")
    
    # Test function calling (simulated)
    logger.info("⚡ Simulating function call...")
    
    # Create a mock response with function calls
    from coda.components.llm.models import LLMResponse, FunctionCall
    
    mock_response = LLMResponse(
        response_id="test",
        content="",
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        function_calls=[{
            "name": "get_time",
            "arguments": '{"format": "24h"}'
        }]
    )
    
    # Parse and process function calls
    function_calls = llm_manager.function_calling.parse_function_calls_from_response(mock_response)
    logger.info(f"📞 Parsed {len(function_calls)} function calls")
    
    if function_calls:
        results = await llm_manager.function_calling.process_function_calls(function_calls)
        logger.info(f"✅ Executed {len(results)} function calls")
        
        for result in results:
            logger.info(f"  {result.function_name}: {'✅' if result.success else '❌'} {result.result}")
    
    logger.info("✅ Function calling demonstration completed!")


async def demonstrate_memory_integration():
    """Demonstrate memory system integration."""
    logger.info("🧠 Starting memory integration demonstration...")
    
    # Create LLM manager
    llm_manager = LLMManager()
    
    # Add mock provider
    mock_provider = MockLLMProvider(ProviderConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo"
    ))
    llm_manager._providers["mock"] = mock_provider
    llm_manager._default_provider = "mock"
    
    # Create and integrate memory manager
    memory_manager = MemoryManager()
    await llm_manager.set_memory_manager(memory_manager)
    
    # Store some memories
    logger.info("💾 Storing memories...")
    await memory_manager.store_memory(
        content="User prefers detailed technical explanations",
        category="preference",
        tags=["communication", "style"],
        importance=0.8
    )
    
    await memory_manager.store_memory(
        content="User is working on a Python machine learning project",
        category="context",
        tags=["programming", "python", "ml"],
        importance=0.7
    )
    
    # Test memory-enhanced conversation
    logger.info("🗣️ Testing memory-enhanced conversation...")
    
    response = await llm_manager.generate_response(
        "Can you help me with my Python project?",
        conversation_id="memory-test"
    )
    
    logger.info(f"Response: {response.content}")
    
    # Check if memories were used (in a real implementation, this would be reflected in the prompt)
    memories = await memory_manager.search_memories(query="Python project", limit=3)
    logger.info(f"🔍 Found {len(memories)} relevant memories")
    
    for memory in memories:
        logger.info(f"  - {memory.content} (relevance: {getattr(memory, 'relevance_score', 'N/A')})")
    
    logger.info("✅ Memory integration demonstration completed!")


async def demonstrate_websocket_integration():
    """Demonstrate WebSocket integration."""
    logger.info("🌐 Starting WebSocket LLM demonstration...")
    
    try:
        # Set up WebSocket server
        server = CodaWebSocketServer(host="localhost", port=8769)
        integration = CodaWebSocketIntegration(server)
        
        # Create WebSocket-enabled LLM manager
        config = LLMConfig(websocket_events_enabled=True, stream_responses=True)
        llm_manager = WebSocketLLMManager(config)
        
        # Add mock provider
        mock_provider = MockLLMProvider(ProviderConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo"
        ))
        llm_manager._providers["mock"] = mock_provider
        llm_manager._default_provider = "mock"
        
        await llm_manager.set_websocket_integration(integration)
        
        # Start WebSocket server
        await server.start()
        logger.info(f"🌐 WebSocket server running at ws://{server.host}:{server.port}")
        logger.info("💡 Connect with: wscat -c ws://localhost:8769")
        
        # Wait for potential clients
        logger.info("⏳ Waiting 3 seconds for WebSocket clients...")
        await asyncio.sleep(3)
        
        # Demonstrate LLM operations with WebSocket events
        logger.info("🎬 Demonstrating LLM operations with WebSocket events...")
        
        # Generate response with events
        response = await llm_manager.generate_response(
            "Tell me about artificial intelligence",
            conversation_id="websocket-demo"
        )
        await asyncio.sleep(0.5)
        
        # Generate streaming response with events
        logger.info("📡 Generating streaming response with events...")
        async for chunk in llm_manager.generate_response(
            "What are the benefits of AI?",
            conversation_id="websocket-demo",
            stream=True
        ):
            pass  # Events are automatically broadcasted
        
        await asyncio.sleep(0.5)
        
        # Broadcast analytics
        await llm_manager.broadcast_analytics_update()
        
        # Trigger system snapshot
        snapshot = await llm_manager.trigger_llm_snapshot()
        logger.info(f"📸 LLM system snapshot captured with {len(snapshot)} data points")
        
        # Show server stats
        server_stats = server.get_stats()
        logger.info(f"📊 WebSocket server stats: {server_stats}")
        
        logger.info("✅ WebSocket LLM demonstration completed!")
        logger.info("⏳ Server will stop in 3 seconds...")
        await asyncio.sleep(3)
        
        await server.stop()
        
    except Exception as e:
        logger.error(f"❌ Error in WebSocket demonstration: {e}")


async def demonstrate_conversation_management():
    """Demonstrate advanced conversation management."""
    logger.info("💬 Starting conversation management demonstration...")
    
    llm_manager = LLMManager()
    
    # Add mock provider
    mock_provider = MockLLMProvider(ProviderConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo"
    ))
    llm_manager._providers["mock"] = mock_provider
    llm_manager._default_provider = "mock"
    
    # Create multiple conversations
    conversations = []
    for i in range(3):
        conv_id = f"demo-conv-{i}"
        
        # Start conversation
        response = await llm_manager.generate_response(
            f"Hello, this is conversation {i}",
            conversation_id=conv_id
        )
        
        # Add more messages
        await llm_manager.continue_conversation(conv_id, "How are you?")
        await llm_manager.continue_conversation(conv_id, "What can you help me with?")
        
        conversations.append(conv_id)
    
    logger.info(f"📝 Created {len(conversations)} conversations")
    
    # List conversations
    conv_list = await llm_manager.conversation_manager.list_conversations()
    logger.info(f"📋 Listed {len(conv_list)} conversations")
    
    for conv in conv_list:
        stats = await llm_manager.conversation_manager.get_conversation_stats(conv.conversation_id)
        logger.info(f"  {conv.conversation_id}: {stats['message_count']} messages, {stats['total_tokens']} tokens")
    
    # Test conversation context management
    logger.info("🔍 Testing conversation context...")
    context = await llm_manager.conversation_manager.get_conversation_context(
        conversations[0], 
        max_tokens=500
    )
    logger.info(f"Retrieved context with {len(context)} messages")
    
    # Test conversation summarization
    logger.info("📄 Testing conversation summarization...")
    summary = await llm_manager.conversation_manager.summarize_conversation(conversations[0])
    logger.info(f"Summary: {summary}")
    
    logger.info("✅ Conversation management demonstration completed!")


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Coda LLM System Demonstration")
    
    try:
        # Run demonstrations
        await demonstrate_basic_llm()
        await asyncio.sleep(1)
        
        await demonstrate_provider_management()
        await asyncio.sleep(1)
        
        await demonstrate_conversation_management()
        await asyncio.sleep(1)
        
        await demonstrate_function_calling()
        await asyncio.sleep(1)
        
        await demonstrate_memory_integration()
        await asyncio.sleep(1)
        
        await demonstrate_websocket_integration()
        
        logger.info("🎉 All LLM system demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
