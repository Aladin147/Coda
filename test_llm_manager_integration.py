#!/usr/bin/env python3
"""
Test Voice-LLM Manager Integration

This script tests the integration between voice processing and the LLM manager.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, AsyncGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.llm_manager_integration import (
        VoiceLLMManagerIntegration, VoiceLLMManagerConfig, VoiceLLMManager
    )
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
    logger.info("✓ Successfully imported voice-LLM manager integration components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


# Mock LLM components
class MockLLMMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class MockLLMResponse:
    def __init__(self, content: str, conversation_id: str):
        self.id = f"llm_response_{datetime.now().strftime('%H%M%S%f')}"
        self.content = content
        self.conversation_id = conversation_id
        self.usage = type('Usage', (), {'total_tokens': 150})()

class MockLLMStreamChunk:
    def __init__(self, delta: str):
        self.delta = delta

class MockLLMConversation:
    def __init__(self, conversation_id: str):
        self.id = conversation_id
        self.messages = [
            MockLLMMessage("user", "Hello"),
            MockLLMMessage("assistant", "Hi there! How can I help you?"),
            MockLLMMessage("user", "Tell me about AI"),
            MockLLMMessage("assistant", "AI is a fascinating field...")
        ]

class MockLLMManager:
    """Mock LLM manager for testing."""
    
    def __init__(self):
        self.conversations = {}
        self.providers = ["ollama", "openai", "anthropic"]
        
    async def generate_response(self, prompt: str, conversation_id: str = None, **kwargs):
        """Mock generate response."""
        if conversation_id:
            self.conversations[conversation_id] = MockLLMConversation(conversation_id)
        
        # Mock response based on prompt
        if "hello" in prompt.lower():
            content = "Hello! I'm here to help you with any questions you have."
        elif "calculate" in prompt.lower():
            content = "I can help you with calculations. What would you like me to compute?"
        elif "weather" in prompt.lower():
            content = "I'd be happy to help with weather information. What location are you interested in?"
        else:
            content = f"I understand you're asking about: {prompt[:50]}... Let me help you with that."
        
        return MockLLMResponse(content, conversation_id or "default")
    
    async def continue_conversation(self, conversation_id: str, message: str, stream: bool = False, **kwargs):
        """Mock continue conversation."""
        if stream:
            # Return async generator for streaming
            async def streaming_generator():
                async for chunk in self._generate_streaming_response(message, conversation_id):
                    yield chunk
            return streaming_generator()
        else:
            return await self.generate_response(message, conversation_id, **kwargs)
    
    async def _generate_streaming_response(self, message: str, conversation_id: str):
        """Mock streaming response generator."""
        response_text = f"Streaming response to: {message}"
        
        # Yield chunks
        for i, word in enumerate(response_text.split()):
            chunk = MockLLMStreamChunk(word + " ")
            yield chunk
            await asyncio.sleep(0.01)  # Simulate streaming delay
    
    async def get_conversation(self, conversation_id: str):
        """Mock get conversation."""
        return self.conversations.get(conversation_id)
    
    async def list_providers(self):
        """Mock list providers."""
        return self.providers


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


async def test_voice_llm_config():
    """Test voice LLM manager configuration."""
    logger.info("=== Testing Voice LLM Manager Configuration ===")
    
    try:
        config = VoiceLLMManagerConfig()
        
        logger.info("✓ Default configuration created")
        logger.info(f"  LLM provider: {config.llm_provider}")
        logger.info(f"  LLM model: {config.llm_model}")
        logger.info(f"  Voice timeout: {config.voice_timeout_seconds}s")
        logger.info(f"  Max tokens: {config.voice_response_max_tokens}")
        logger.info(f"  Streaming enabled: {config.enable_streaming_for_voice}")
        
        # Test custom configuration
        custom_config = VoiceLLMManagerConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            voice_timeout_seconds=10.0,
            voice_response_max_tokens=500
        )
        
        logger.info("✓ Custom configuration created")
        logger.info(f"  Custom provider: {custom_config.llm_provider}")
        logger.info(f"  Custom model: {custom_config.llm_model}")
        
        logger.info("✓ Voice LLM manager configuration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_voice_response_generation():
    """Test voice response generation with LLM manager."""
    logger.info("=== Testing Voice Response Generation ===")
    
    try:
        llm_manager = MockLLMManager()
        config = VoiceLLMManagerConfig()
        integration = VoiceLLMManagerIntegration(llm_manager, config)
        
        logger.info("✓ Voice LLM integration created")
        
        # Test different types of voice messages
        test_messages = [
            "Hello, how are you today?",
            "Can you help me calculate 15 * 23?",
            "What's the weather like in San Francisco?",
            "Tell me about machine learning algorithms"
        ]
        
        for i, text_content in enumerate(test_messages):
            voice_message = create_test_voice_message(
                f"test_conversation_{i}",
                text_content
            )
            
            # Test enhanced context
            enhanced_context = {
                "memory": {
                    "relevant_memories": [{"content": "User likes technical topics"}],
                    "context_summary": "User has technical background"
                },
                "personality": {
                    "description": "Assistant is helpful and enthusiastic"
                },
                "tools": {
                    "tools_summary": "Calculator and web search available"
                }
            }
            
            voice_response = await integration.generate_voice_response(
                voice_message, enhanced_context=enhanced_context
            )
            
            logger.info(f"✓ Response {i+1}:")
            logger.info(f"  Query: {text_content}")
            logger.info(f"  Response: {voice_response.text_content[:80]}...")
            logger.info(f"  Latency: {voice_response.total_latency_ms:.1f}ms")
            logger.info(f"  Processing mode: {voice_response.processing_mode}")
        
        await integration.cleanup()
        logger.info("✓ Voice response generation test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Response generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_response():
    """Test streaming voice response generation."""
    logger.info("=== Testing Streaming Response Generation ===")
    
    try:
        llm_manager = MockLLMManager()
        config = VoiceLLMManagerConfig(enable_streaming_for_voice=True)
        integration = VoiceLLMManagerIntegration(llm_manager, config)
        
        voice_message = create_test_voice_message(
            "test_conversation_streaming",
            "Tell me a story about artificial intelligence"
        )
        
        logger.info("✓ Starting streaming response generation")
        
        chunk_count = 0
        accumulated_content = ""
        
        async for voice_chunk in integration.generate_streaming_voice_response(voice_message):
            chunk_count += 1
            accumulated_content = voice_chunk.text_content
            
            logger.info(f"  Chunk {chunk_count}: {len(voice_chunk.text_content)} chars")
        
        logger.info(f"✓ Streaming completed:")
        logger.info(f"  Total chunks: {chunk_count}")
        logger.info(f"  Final content: {accumulated_content[:80]}...")
        
        await integration.cleanup()
        logger.info("✓ Streaming response test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming response test failed: {e}")
        return False


async def test_conversation_continuity():
    """Test conversation continuity across voice sessions."""
    logger.info("=== Testing Conversation Continuity ===")
    
    try:
        llm_manager = MockLLMManager()
        config = VoiceLLMManagerConfig()
        integration = VoiceLLMManagerIntegration(llm_manager, config)
        
        conversation_id = "test_conversation_continuity"
        
        # First message
        voice_message1 = create_test_voice_message(
            conversation_id,
            "Hello, my name is Alice"
        )
        
        response1 = await integration.generate_voice_response(voice_message1)
        
        logger.info("✓ First interaction:")
        logger.info(f"  User: {voice_message1.text_content}")
        logger.info(f"  Assistant: {response1.text_content[:60]}...")
        
        # Second message (should continue conversation)
        voice_message2 = create_test_voice_message(
            conversation_id,
            "What did I just tell you my name was?"
        )
        
        response2 = await integration.generate_voice_response(voice_message2)
        
        logger.info("✓ Second interaction:")
        logger.info(f"  User: {voice_message2.text_content}")
        logger.info(f"  Assistant: {response2.text_content[:60]}...")
        
        # Check conversation history
        history = await integration.get_conversation_history(conversation_id)
        
        logger.info(f"✓ Conversation history: {len(history)} messages")
        
        await integration.cleanup()
        logger.info("✓ Conversation continuity test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation continuity test failed: {e}")
        return False


async def test_response_caching():
    """Test response caching functionality."""
    logger.info("=== Testing Response Caching ===")
    
    try:
        llm_manager = MockLLMManager()
        config = VoiceLLMManagerConfig(enable_response_caching=True)
        integration = VoiceLLMManagerIntegration(llm_manager, config)
        
        voice_message = create_test_voice_message(
            "test_conversation_caching",
            "What is artificial intelligence?"
        )
        
        # First request (should cache)
        response1 = await integration.generate_voice_response(voice_message)
        
        # Second request (should use cache)
        response2 = await integration.generate_voice_response(voice_message)
        
        # Check cache statistics
        stats = integration.get_integration_stats()
        cache_hits = stats['voice_llm_stats']['cache_hits']
        
        logger.info(f"✓ Response caching test:")
        logger.info(f"  Cache hits: {cache_hits}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        logger.info(f"  Response 1 ID: {response1.response_id}")
        logger.info(f"  Response 2 ID: {response2.response_id}")
        
        await integration.cleanup()
        logger.info("✓ Response caching test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Response caching test failed: {e}")
        return False


async def test_voice_llm_manager():
    """Test high-level voice LLM manager."""
    logger.info("=== Testing Voice LLM Manager ===")
    
    try:
        llm_manager = MockLLMManager()
        config = VoiceLLMManagerConfig()
        voice_llm_manager = VoiceLLMManager(llm_manager, config)
        
        logger.info("✓ Voice LLM manager created")
        
        # Test non-streaming response
        voice_message = create_test_voice_message(
            "test_conversation_manager",
            "Explain quantum computing in simple terms"
        )
        
        enhanced_context = {
            "personality": {"description": "Friendly and educational assistant"},
            "memory": {"context_summary": "User is learning about technology"}
        }
        
        response = await voice_llm_manager.generate_voice_response(
            voice_message,
            enhanced_context=enhanced_context,
            stream=False
        )
        
        logger.info("✓ Non-streaming response generated")
        logger.info(f"  Response: {response.text_content[:60]}...")
        logger.info(f"  Latency: {response.total_latency_ms:.1f}ms")
        
        # Test streaming response
        logger.info("✓ Testing streaming response...")
        
        chunk_count = 0
        async for chunk in voice_llm_manager.generate_voice_response(
            voice_message,
            enhanced_context=enhanced_context,
            stream=True
        ):
            chunk_count += 1
        
        logger.info(f"  Streaming chunks: {chunk_count}")
        
        # Test conversation history
        history = await voice_llm_manager.get_conversation_history("test_conversation_manager")
        logger.info(f"✓ Conversation history: {len(history)} messages")
        
        # Test statistics
        stats = voice_llm_manager.get_llm_stats()
        logger.info(f"✓ LLM statistics:")
        logger.info(f"  Voice LLM requests: {stats['voice_llm_stats']['voice_llm_requests']}")
        logger.info(f"  Average response time: {stats['voice_llm_stats']['average_response_time_ms']:.1f}ms")
        logger.info(f"  Active conversations: {stats['active_conversations']}")
        
        await voice_llm_manager.cleanup()
        logger.info("✓ Voice LLM manager test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice LLM manager test failed: {e}")
        return False


async def test_integration_statistics():
    """Test integration statistics and monitoring."""
    logger.info("=== Testing Integration Statistics ===")
    
    try:
        llm_manager = MockLLMManager()
        config = VoiceLLMManagerConfig()
        integration = VoiceLLMManagerIntegration(llm_manager, config)
        
        # Generate multiple responses to build statistics
        for i in range(5):
            voice_message = create_test_voice_message(
                f"test_conversation_stats_{i}",
                f"Test message {i+1} for statistics"
            )
            
            await integration.generate_voice_response(voice_message)
        
        # Get comprehensive statistics
        stats = integration.get_integration_stats()
        
        logger.info(f"✓ Integration statistics:")
        logger.info(f"  Voice LLM requests: {stats['voice_llm_stats']['voice_llm_requests']}")
        logger.info(f"  Average response time: {stats['voice_llm_stats']['average_response_time_ms']:.1f}ms")
        logger.info(f"  Total tokens used: {stats['voice_llm_stats']['total_tokens_used']}")
        logger.info(f"  Active conversations: {stats['active_conversations']}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        
        # Check configuration in stats
        config_stats = stats['config']
        logger.info(f"  Configuration:")
        logger.info(f"    LLM provider: {config_stats['llm_provider']}")
        logger.info(f"    LLM model: {config_stats['llm_model']}")
        logger.info(f"    Streaming enabled: {config_stats['streaming_enabled']}")
        
        await integration.cleanup()
        logger.info("✓ Integration statistics test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration statistics test failed: {e}")
        return False


async def main():
    """Run all voice-LLM manager integration tests."""
    logger.info("🚀 Starting Voice-LLM Manager Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Voice LLM Configuration", test_voice_llm_config),
        ("Voice Response Generation", test_voice_response_generation),
        ("Streaming Response", test_streaming_response),
        ("Conversation Continuity", test_conversation_continuity),
        ("Response Caching", test_response_caching),
        ("Voice LLM Manager", test_voice_llm_manager),
        ("Integration Statistics", test_integration_statistics),
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
    logger.info("🏁 Voice-LLM Manager Integration Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<30}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice-LLM manager integration tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
