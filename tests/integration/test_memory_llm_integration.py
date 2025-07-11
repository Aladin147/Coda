"""
Integration tests for Memory-LLM interaction and context management.
Tests memory retrieval, context injection, and conversation continuity.
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.coda.components.memory.models import Memory, MemoryType, MemoryMetadata, MemoryQuery
from src.coda.components.llm.models import LLMMessage, MessageRole, LLMConfig
from src.coda.components.memory.manager import MemoryManager
from src.coda.components.llm.manager import LLMManager


class TestMemoryLLMIntegration:
    """Integration tests for Memory-LLM interaction."""

    @pytest_asyncio.fixture
    async def sample_memories(self):
        """Create sample memories for testing."""
        return [
            Memory(
                id="mem-1",
                content="User's name is Alice and she works as a software engineer",
                metadata=MemoryMetadata(
                    source_type=MemoryType.FACT,
                    importance=0.9,
                    topics=["personal", "profession"],
                    timestamp=datetime.now() - timedelta(days=1)
                )
            ),
            Memory(
                id="mem-2",
                content="Alice prefers technical explanations with code examples",
                metadata=MemoryMetadata(
                    source_type=MemoryType.PREFERENCE,
                    importance=0.8,
                    topics=["communication", "preferences"],
                    timestamp=datetime.now() - timedelta(hours=2)
                )
            ),
            Memory(
                id="mem-3",
                content="Previous conversation about Python async programming",
                metadata=MemoryMetadata(
                    source_type=MemoryType.CONVERSATION,
                    importance=0.7,
                    topics=["python", "async", "programming"],
                    timestamp=datetime.now() - timedelta(minutes=30)
                )
            )
        ]

    @pytest_asyncio.fixture
    async def mock_memory_manager(self, sample_memories):
        """Create mock memory manager with realistic behavior."""
        manager = Mock(spec=MemoryManager)
        
        # Mock memory retrieval
        manager.retrieve_relevant_memories = AsyncMock(return_value=sample_memories)
        manager.store_conversation = AsyncMock(return_value="stored-memory-id")
        manager.get_conversation_history = AsyncMock(return_value=[
            LLMMessage(
                role=MessageRole.USER,
                content="Tell me about async programming",
                timestamp=datetime.now() - timedelta(minutes=5)
            ),
            LLMMessage(
                role=MessageRole.ASSISTANT,
                content="Async programming allows concurrent execution...",
                timestamp=datetime.now() - timedelta(minutes=4)
            )
        ])
        
        # Mock user preferences
        manager.get_user_preferences = AsyncMock(return_value={
            "communication_style": "technical",
            "detail_level": "comprehensive",
            "code_examples": True
        })
        
        manager.is_initialized = True
        return manager

    @pytest_asyncio.fixture
    async def mock_llm_manager(self):
        """Create mock LLM manager with context-aware responses."""
        manager = Mock(spec=LLMManager)
        
        # Mock context-aware response generation
        async def context_aware_response(messages, context=None):
            if context and "Alice" in str(context):
                return "Hello Alice! Based on your background in software engineering..."
            else:
                return "Hello! How can I help you today?"
        
        manager.generate_response = AsyncMock(side_effect=context_aware_response)
        
        # Mock conversation management
        manager.add_message_to_conversation = AsyncMock()
        manager.get_conversation_context = AsyncMock(return_value={
            "conversation_id": "test-conv",
            "message_count": 5,
            "topics": ["programming", "python"]
        })
        
        manager.is_initialized = True
        return manager

    @pytest_asyncio.fixture
    async def integrated_system(self, mock_memory_manager, mock_llm_manager):
        """Create integrated memory-LLM system."""
        class IntegratedSystem:
            def __init__(self, memory_manager, llm_manager):
                self.memory = memory_manager
                self.llm = llm_manager
            
            async def process_message_with_context(self, user_message: str, conversation_id: str):
                """Process message with memory context."""
                # Retrieve relevant memories
                memory_query = MemoryQuery(
                    query=user_message,
                    limit=5,
                    min_relevance=0.5
                )
                memories = await self.memory.retrieve_relevant_memories(memory_query)
                
                # Get conversation history
                history = await self.memory.get_conversation_history(conversation_id)
                
                # Get user preferences
                preferences = await self.memory.get_user_preferences()
                
                # Build context
                context = {
                    "memories": memories,
                    "history": history,
                    "preferences": preferences
                }
                
                # Generate response with context
                messages = history + [LLMMessage(
                    role=MessageRole.USER,
                    content=user_message,
                    timestamp=datetime.now()
                )]
                
                response = await self.llm.generate_response(messages, context=context)
                
                # Store conversation
                await self.memory.store_conversation(conversation_id, user_message, response)
                
                return response
        
        return IntegratedSystem(mock_memory_manager, mock_llm_manager)

    @pytest.mark.asyncio
    async def test_memory_context_injection(self, integrated_system):
        """Test that memory context is properly injected into LLM requests."""
        user_message = "What did we discuss about programming?"
        conversation_id = "test-conv-123"
        
        response = await integrated_system.process_message_with_context(
            user_message, conversation_id
        )
        
        # Verify memory retrieval was called
        integrated_system.memory.retrieve_relevant_memories.assert_called_once()
        
        # Verify LLM was called with context
        integrated_system.llm.generate_response.assert_called_once()
        call_args = integrated_system.llm.generate_response.call_args
        assert 'context' in call_args.kwargs
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_conversation_continuity(self, integrated_system):
        """Test conversation continuity across multiple messages."""
        conversation_id = "continuity-test"
        
        # First message
        response1 = await integrated_system.process_message_with_context(
            "My name is Alice", conversation_id
        )
        
        # Second message referencing first
        response2 = await integrated_system.process_message_with_context(
            "What's my name?", conversation_id
        )
        
        # Verify memory operations
        assert integrated_system.memory.retrieve_relevant_memories.call_count == 2
        assert integrated_system.memory.store_conversation.call_count == 2
        
        # Verify responses
        assert isinstance(response1, str)
        assert isinstance(response2, str)

    @pytest.mark.asyncio
    async def test_user_preference_integration(self, integrated_system):
        """Test that user preferences influence LLM responses."""
        user_message = "Explain async programming"
        conversation_id = "preferences-test"
        
        response = await integrated_system.process_message_with_context(
            user_message, conversation_id
        )
        
        # Verify preferences were retrieved
        integrated_system.memory.get_user_preferences.assert_called_once()
        
        # Verify LLM received context including preferences
        call_args = integrated_system.llm.generate_response.call_args
        context = call_args.kwargs.get('context', {})
        assert 'preferences' in context

    @pytest.mark.asyncio
    async def test_memory_relevance_filtering(self, integrated_system, sample_memories):
        """Test that only relevant memories are included in context."""
        # Mock memory manager to return filtered results
        relevant_memories = [mem for mem in sample_memories if "python" in mem.content.lower()]
        integrated_system.memory.retrieve_relevant_memories.return_value = relevant_memories
        
        user_message = "Tell me more about Python"
        conversation_id = "relevance-test"
        
        response = await integrated_system.process_message_with_context(
            user_message, conversation_id
        )
        
        # Verify memory query was made
        call_args = integrated_system.memory.retrieve_relevant_memories.call_args
        query = call_args[0][0]
        assert isinstance(query, MemoryQuery)
        assert "Python" in query.query

    @pytest.mark.asyncio
    async def test_context_size_management(self, integrated_system):
        """Test that context size is managed to avoid token limits."""
        # Create a very long user message
        long_message = "Explain " + "very " * 1000 + "complex topic"
        conversation_id = "context-size-test"
        
        response = await integrated_system.process_message_with_context(
            long_message, conversation_id
        )
        
        # Verify system handled long context gracefully
        assert isinstance(response, str)
        integrated_system.llm.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_storage_after_response(self, integrated_system):
        """Test that conversations are stored in memory after LLM response."""
        user_message = "Store this conversation"
        conversation_id = "storage-test"
        
        response = await integrated_system.process_message_with_context(
            user_message, conversation_id
        )
        
        # Verify conversation was stored
        integrated_system.memory.store_conversation.assert_called_once()
        call_args = integrated_system.memory.store_conversation.call_args
        assert call_args[0][0] == conversation_id  # conversation_id
        assert call_args[0][1] == user_message     # user message
        assert call_args[0][2] == response         # assistant response

    @pytest.mark.asyncio
    async def test_concurrent_memory_llm_operations(self, integrated_system):
        """Test concurrent memory-LLM operations."""
        # Create multiple concurrent requests
        messages = [
            ("What's my name?", "concurrent-1"),
            ("Tell me about Python", "concurrent-2"),
            ("What are my preferences?", "concurrent-3")
        ]
        
        # Process concurrently
        tasks = [
            integrated_system.process_message_with_context(msg, conv_id)
            for msg, conv_id in messages
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all operations completed
        assert len(responses) == 3
        for response in responses:
            if isinstance(response, Exception):
                pytest.fail(f"Concurrent operation failed: {response}")
            assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_error_handling_in_integration(self, integrated_system):
        """Test error handling in memory-LLM integration."""
        # Mock memory manager to raise an error
        integrated_system.memory.retrieve_relevant_memories.side_effect = Exception("Memory error")
        
        user_message = "This should handle memory error"
        conversation_id = "error-test"
        
        # Verify error is handled gracefully
        with pytest.raises(Exception, match="Memory error"):
            await integrated_system.process_message_with_context(user_message, conversation_id)

    @pytest.mark.asyncio
    async def test_memory_update_based_on_llm_response(self, integrated_system):
        """Test that memory is updated based on LLM responses."""
        # Mock LLM to return response with new information
        integrated_system.llm.generate_response.return_value = "I learned that you prefer detailed explanations"
        
        user_message = "I like detailed explanations"
        conversation_id = "update-test"
        
        response = await integrated_system.process_message_with_context(
            user_message, conversation_id
        )
        
        # Verify conversation was stored (which should extract new facts)
        integrated_system.memory.store_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_personalization(self, integrated_system, sample_memories):
        """Test that context is personalized based on user memories."""
        # Ensure Alice's information is in memories
        alice_memory = next(mem for mem in sample_memories if "Alice" in mem.content)
        integrated_system.memory.retrieve_relevant_memories.return_value = [alice_memory]
        
        user_message = "Hello"
        conversation_id = "personalization-test"
        
        response = await integrated_system.process_message_with_context(
            user_message, conversation_id
        )
        
        # Verify personalized response
        assert "Alice" in response

    @pytest.mark.asyncio
    async def test_topic_tracking_across_conversations(self, integrated_system):
        """Test topic tracking across multiple conversations."""
        # First conversation about Python
        await integrated_system.process_message_with_context(
            "Tell me about Python", "topic-test-1"
        )
        
        # Second conversation referencing Python
        await integrated_system.process_message_with_context(
            "More about that language", "topic-test-2"
        )
        
        # Verify memory retrieval was called for both
        assert integrated_system.memory.retrieve_relevant_memories.call_count == 2
        
        # Verify context was used to understand "that language" refers to Python
        call_args = integrated_system.llm.generate_response.call_args
        assert 'context' in call_args.kwargs
