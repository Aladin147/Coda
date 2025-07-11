#!/usr/bin/env python3
"""
Test Voice-Memory Integration

This script tests the integration between voice processing and memory systems.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.memory_integration import (
        VoiceMemoryIntegration, VoiceMemoryConfig, VoiceMemoryManager
    )
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
    from src.coda.components.memory.manager import MemoryManager
    from src.coda.components.memory.models import MemoryManagerConfig, ConversationRole
    logger.info("✓ Successfully imported voice-memory integration components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self):
        self.turns = []
        self.facts = []
        self.search_results = []
        
    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        self.turns.append(turn)
        return turn
    
    def get_context(self, max_tokens: int = 800):
        """Get conversation context."""
        class MockContext:
            def __init__(self, messages, total_tokens, summary):
                self.messages = messages
                self.total_tokens = total_tokens
                self.summary = summary
        
        messages = []
        for turn in self.turns[-10:]:  # Last 10 turns
            msg = type('Message', (), {
                'role': turn['role'],
                'content': turn['content'],
                'timestamp': turn['timestamp']
            })()
            messages.append(msg)
        
        return MockContext(messages, len(str(messages)), "Mock conversation summary")
    
    async def search_memories(self, query: str, limit: int = 5, min_relevance: float = 0.0):
        """Search memories."""
        # Mock search results
        results = []
        for i in range(min(limit, 3)):  # Return up to 3 mock results
            result = type('MemoryResult', (), {
                'memory': type('Memory', (), {
                    'content': f"Mock memory {i+1} related to: {query[:30]}...",
                    'type': type('MemoryType', (), {'value': 'conversation'})(),
                    'created_at': datetime.now(),
                    'importance': 0.7 + i * 0.1
                })(),
                'final_score': 0.8 - i * 0.1
            })()
            results.append(result)
        
        return results
    
    async def store_fact(self, fact: str, source: str = "user", importance: float = 0.7):
        """Store a fact."""
        fact_id = f"fact_{len(self.facts)}"
        self.facts.append({
            "id": fact_id,
            "content": fact,
            "source": source,
            "importance": importance
        })
        return fact_id
    
    async def consolidate_short_term(self):
        """Consolidate short-term memory."""
        return len(self.turns) // 2  # Mock consolidation
    
    async def get_memory_stats(self):
        """Get memory statistics."""
        return {
            "short_term": {
                "turn_count": len(self.turns),
                "total_tokens": sum(len(turn["content"]) for turn in self.turns)
            },
            "long_term": {
                "memory_count": len(self.facts)
            }
        }


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


async def test_voice_memory_config():
    """Test voice memory configuration."""
    logger.info("=== Testing Voice Memory Configuration ===")
    
    try:
        config = VoiceMemoryConfig()
        
        logger.info("✓ Default configuration created")
        logger.info(f"  Memory injection enabled: {config.enable_memory_injection}")
        logger.info(f"  Auto storage enabled: {config.enable_auto_storage}")
        logger.info(f"  Max relevant memories: {config.max_relevant_memories}")
        logger.info(f"  Relevance threshold: {config.memory_relevance_threshold}")
        logger.info(f"  Consolidation interval: {config.consolidation_interval}")
        
        # Test custom configuration
        custom_config = VoiceMemoryConfig(
            enable_memory_injection=False,
            max_relevant_memories=10,
            memory_relevance_threshold=0.8
        )
        
        logger.info("✓ Custom configuration created")
        logger.info(f"  Memory injection enabled: {custom_config.enable_memory_injection}")
        logger.info(f"  Max relevant memories: {custom_config.max_relevant_memories}")
        
        logger.info("✓ Voice memory configuration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_voice_memory_integration():
    """Test voice memory integration functionality."""
    logger.info("=== Testing Voice Memory Integration ===")
    
    try:
        # Create mock memory manager and integration
        memory_manager = MockMemoryManager()
        config = VoiceMemoryConfig()
        integration = VoiceMemoryIntegration(memory_manager, config)
        
        logger.info("✓ Voice memory integration created")
        
        # Test processing voice message
        voice_message = create_test_voice_message(
            "test_conversation",
            "Hello, I'm interested in learning about machine learning algorithms"
        )
        
        context = await integration.process_voice_message(voice_message)
        
        logger.info("✓ Voice message processed with memory integration")
        logger.info(f"  Memory context keys: {list(context.get('memory', {}).keys())}")
        logger.info(f"  Conversation context keys: {list(context.get('conversation', {}).keys())}")
        logger.info(f"  Message info keys: {list(context.get('message_info', {}).keys())}")
        
        # Check if user message was stored
        assert len(memory_manager.turns) == 1
        assert memory_manager.turns[0]["role"] == ConversationRole.USER.value
        logger.info("✓ User message stored in memory")
        
        # Test processing voice response
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "Machine learning algorithms are computational methods that learn patterns from data..."
        )
        
        await integration.process_voice_response(voice_response)
        
        # Check if assistant response was stored
        assert len(memory_manager.turns) == 2
        assert memory_manager.turns[1]["role"] == ConversationRole.ASSISTANT.value
        logger.info("✓ Assistant response stored in memory")
        
        # Test statistics
        stats = integration.get_integration_stats()
        logger.info(f"✓ Integration statistics:")
        logger.info(f"  Turns stored: {stats['voice_memory_stats']['turns_stored']}")
        logger.info(f"  Context enhancements: {stats['voice_memory_stats']['context_enhancements']}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        
        await integration.cleanup()
        logger.info("✓ Voice memory integration test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_context_retrieval():
    """Test memory context retrieval functionality."""
    logger.info("=== Testing Memory Context Retrieval ===")
    
    try:
        memory_manager = MockMemoryManager()
        config = VoiceMemoryConfig()
        integration = VoiceMemoryIntegration(memory_manager, config)
        
        # Add some conversation history
        memory_manager.add_turn("user", "I'm working on a Python project")
        memory_manager.add_turn("assistant", "That's great! What kind of Python project?")
        memory_manager.add_turn("user", "A machine learning application")
        
        # Test memory context retrieval
        voice_message = create_test_voice_message(
            "test_conversation",
            "Can you help me with neural networks?"
        )
        
        context = await integration.process_voice_message(voice_message)
        
        # Check memory context
        memory_context = context.get("memory", {})
        logger.info(f"✓ Memory context retrieved:")
        logger.info(f"  Relevant memories: {memory_context.get('memory_count', 0)}")
        logger.info(f"  Average relevance: {memory_context.get('average_relevance', 0):.2f}")
        logger.info(f"  Context summary: {memory_context.get('context_summary', 'N/A')}")
        
        # Check conversation context
        conversation_context = context.get("conversation", {})
        logger.info(f"✓ Conversation context retrieved:")
        logger.info(f"  Total messages: {conversation_context.get('total_messages', 0)}")
        logger.info(f"  Context tokens: {conversation_context.get('context_tokens', 0)}")
        
        await integration.cleanup()
        logger.info("✓ Memory context retrieval test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context retrieval test failed: {e}")
        return False


async def test_memory_consolidation():
    """Test memory consolidation functionality."""
    logger.info("=== Testing Memory Consolidation ===")
    
    try:
        memory_manager = MockMemoryManager()
        config = VoiceMemoryConfig(consolidation_interval=3)  # Consolidate every 3 messages
        integration = VoiceMemoryIntegration(memory_manager, config)
        
        # Process multiple messages to trigger consolidation
        for i in range(5):
            voice_message = create_test_voice_message(
                "test_conversation",
                f"Test message {i+1} for consolidation testing"
            )
            
            await integration.process_voice_message(voice_message)
            
            voice_response = create_test_voice_response(
                voice_message.message_id,
                voice_message.conversation_id,
                f"Response {i+1} for consolidation testing"
            )
            
            await integration.process_voice_response(voice_response)
        
        # Check statistics
        stats = integration.get_integration_stats()
        logger.info(f"✓ Consolidation test results:")
        logger.info(f"  Turns stored: {stats['voice_memory_stats']['turns_stored']}")
        logger.info(f"  Consolidations performed: {stats['voice_memory_stats']['consolidations_performed']}")
        logger.info(f"  Messages since consolidation: {stats['message_count_since_consolidation']}")
        
        await integration.cleanup()
        logger.info("✓ Memory consolidation test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Consolidation test failed: {e}")
        return False


async def test_voice_memory_manager():
    """Test high-level voice memory manager."""
    logger.info("=== Testing Voice Memory Manager ===")
    
    try:
        memory_manager = MockMemoryManager()
        config = VoiceMemoryConfig()
        voice_memory_manager = VoiceMemoryManager(memory_manager, config)
        
        logger.info("✓ Voice memory manager created")
        
        # Test voice context enhancement
        voice_message = create_test_voice_message(
            "test_conversation",
            "What are the best practices for deep learning?"
        )
        
        enhanced_context = await voice_memory_manager.enhance_voice_context(voice_message)
        
        logger.info("✓ Voice context enhanced")
        logger.info(f"  Context keys: {list(enhanced_context.keys())}")
        
        # Test storing complete voice interaction
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "Deep learning best practices include proper data preprocessing, regularization techniques..."
        )
        
        await voice_memory_manager.store_voice_interaction(voice_message, voice_response)
        
        logger.info("✓ Voice interaction stored")
        
        # Test memory statistics
        stats = voice_memory_manager.get_memory_stats()
        logger.info(f"✓ Memory statistics:")
        logger.info(f"  Turns stored: {stats['voice_memory_stats']['turns_stored']}")
        logger.info(f"  Context enhancements: {stats['voice_memory_stats']['context_enhancements']}")
        
        await voice_memory_manager.cleanup()
        logger.info("✓ Voice memory manager test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory manager test failed: {e}")
        return False


async def test_memory_caching():
    """Test memory caching functionality."""
    logger.info("=== Testing Memory Caching ===")
    
    try:
        memory_manager = MockMemoryManager()
        config = VoiceMemoryConfig(memory_cache_ttl_minutes=1)
        integration = VoiceMemoryIntegration(memory_manager, config)
        
        # Process same message twice to test caching
        voice_message = create_test_voice_message(
            "test_conversation",
            "Tell me about artificial intelligence"
        )
        
        # First processing (should cache)
        context1 = await integration.process_voice_message(voice_message)
        
        # Second processing (should use cache)
        context2 = await integration.process_voice_message(voice_message)
        
        # Check cache statistics
        stats = integration.get_integration_stats()
        cache_hits = stats['voice_memory_stats']['cache_hits']
        
        logger.info(f"✓ Memory caching test:")
        logger.info(f"  Cache hits: {cache_hits}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        
        # Verify contexts are similar (from cache)
        assert context1.get('memory', {}).get('query') == context2.get('memory', {}).get('query')
        logger.info("✓ Cached context matches original")
        
        await integration.cleanup()
        logger.info("✓ Memory caching test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Caching test failed: {e}")
        return False


async def main():
    """Run all voice-memory integration tests."""
    logger.info("🚀 Starting Voice-Memory Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Voice Memory Configuration", test_voice_memory_config),
        ("Voice Memory Integration", test_voice_memory_integration),
        ("Memory Context Retrieval", test_memory_context_retrieval),
        ("Memory Consolidation", test_memory_consolidation),
        ("Voice Memory Manager", test_voice_memory_manager),
        ("Memory Caching", test_memory_caching),
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
    logger.info("🏁 Voice-Memory Integration Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<30}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice-memory integration tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
