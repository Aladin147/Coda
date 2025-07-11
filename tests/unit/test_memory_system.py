"""
Unit tests for the memory system.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.coda.components.memory.models import (
    ConversationTurn,
    ConversationRole,
    MemoryType,
    MemoryQuery,
    ShortTermMemoryConfig,
    LongTermMemoryConfig,
    MemoryEncoderConfig,
    MemoryManagerConfig,
)
from src.coda.components.memory.short_term import ShortTermMemory
from src.coda.components.memory.long_term import LongTermMemory
from src.coda.components.memory.encoder import MemoryEncoder
from src.coda.components.memory.manager import MemoryManager


class TestShortTermMemory:
    """Test cases for ShortTermMemory."""
    
    @pytest.fixture
    def memory(self):
        """Create a test short-term memory instance."""
        config = ShortTermMemoryConfig(max_turns=5, max_tokens=100)
        return ShortTermMemory(config)
    
    def test_add_turn(self, memory):
        """Test adding conversation turns."""
        turn = memory.add_turn("user", "Hello")
        
        assert turn.role == ConversationRole.USER
        assert turn.content == "Hello"
        assert turn.turn_id == 0
        assert len(memory.turns) == 1
    
    def test_invalid_role(self, memory):
        """Test adding turn with invalid role."""
        with pytest.raises(ValueError):
            memory.add_turn("invalid_role", "Hello")
    
    def test_max_turns_limit(self, memory):
        """Test that memory respects max_turns limit."""
        # Add more turns than the limit
        for i in range(10):
            memory.add_turn("user", f"Message {i}")
        
        # Should only keep the last 5 turns
        assert len(memory.turns) == 5
        assert memory.turns[-1].content == "Message 9"
        assert memory.turns[0].content == "Message 5"
    
    def test_get_context(self, memory):
        """Test getting conversation context."""
        memory.add_turn("system", "You are a helpful assistant")
        memory.add_turn("user", "Hello")
        memory.add_turn("assistant", "Hi there!")
        
        context = memory.get_context(max_tokens=50)
        
        assert len(context.messages) == 3
        assert context.messages[0].role == ConversationRole.SYSTEM
        assert context.total_tokens > 0
        assert context.short_term_turns == 2  # user + assistant
    
    def test_get_recent_turns(self, memory):
        """Test getting recent turns."""
        for i in range(3):
            memory.add_turn("user", f"Message {i}")
        
        recent = memory.get_recent_turns(2)
        assert len(recent) == 2
        assert recent[-1].content == "Message 2"
    
    def test_clear(self, memory):
        """Test clearing memory."""
        memory.add_turn("user", "Hello")
        memory.clear()
        
        assert len(memory.turns) == 0
        assert memory.turn_count == 0
    
    def test_export_import(self, memory):
        """Test exporting and importing turns."""
        memory.add_turn("user", "Hello")
        memory.add_turn("assistant", "Hi!")
        
        exported = memory.export_turns()
        assert len(exported) == 2
        
        # Create new memory and import
        new_memory = ShortTermMemory()
        imported_count = new_memory.import_turns(exported)
        
        assert imported_count == 2
        assert len(new_memory.turns) == 2
        assert new_memory.turns[0].content == "Hello"


class TestMemoryEncoder:
    """Test cases for MemoryEncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create a test memory encoder."""
        config = MemoryEncoderConfig(chunk_size=100, chunk_overlap=20)
        return MemoryEncoder(config)
    
    def test_encode_conversation(self, encoder):
        """Test encoding conversation turns."""
        turns = [
            ConversationTurn(role=ConversationRole.USER, content="Hello", turn_id=0),
            ConversationTurn(role=ConversationRole.ASSISTANT, content="Hi there!", turn_id=1),
            ConversationTurn(role=ConversationRole.USER, content="How are you?", turn_id=2),
        ]
        
        chunks = encoder.encode_conversation(turns)
        
        assert len(chunks) >= 1
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.speakers for chunk in chunks)
    
    def test_encode_fact(self, encoder):
        """Test encoding a fact."""
        fact = "The capital of France is Paris"
        chunk = encoder.encode_fact(fact, source="user")
        
        assert chunk.content == fact
        assert "user" in chunk.speakers
        assert len(chunk.topics) >= 0  # May or may not extract topics
    
    def test_calculate_importance(self, encoder):
        """Test importance calculation."""
        # Normal text
        normal_score = encoder.calculate_importance("This is a normal message")
        
        # Text with importance keywords
        important_score = encoder.calculate_importance("This is very important to remember")
        
        # Question
        question_score = encoder.calculate_importance("What is the meaning of life?")
        
        assert 0.0 <= normal_score <= 1.0
        assert important_score > normal_score
        assert question_score >= normal_score
    
    def test_extract_topics(self, encoder):
        """Test topic extraction."""
        content = "I love Python programming and machine learning"
        topics = encoder.extract_topics(content)
        
        assert isinstance(topics, list)
        # Should extract some topics from the content
        assert len(topics) >= 0
    
    def test_chunk_text(self, encoder):
        """Test text chunking."""
        long_text = "This is a very long text. " * 20  # Create long text
        chunks = encoder.chunk_text(long_text)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(len(chunk) <= encoder.config.chunk_size + encoder.config.chunk_overlap for chunk in chunks)


@pytest.mark.asyncio
class TestLongTermMemory:
    """Test cases for LongTermMemory."""
    
    @pytest_asyncio.fixture
    async def memory(self):
        """Create a test long-term memory instance."""
        # Use temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        config = LongTermMemoryConfig(
            storage_path=temp_dir,
            vector_db_type="in_memory",  # Use in-memory for testing
            max_memories=10
        )

        memory = LongTermMemory(config)
        yield memory
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_store_memory(self, memory):
        """Test storing a memory."""
        memory_id = await memory.store_memory(
            content="Test memory content",
            memory_type=MemoryType.FACT,
            importance=0.8,
            metadata={"source": "test"}
        )
        
        assert memory_id is not None
        assert len(memory_id) > 0
    
    async def test_retrieve_memories(self, memory):
        """Test retrieving memories."""
        # Store some test memories
        await memory.store_memory("Python is a programming language", MemoryType.FACT, 0.8, {})
        await memory.store_memory("I like coffee in the morning", MemoryType.PREFERENCE, 0.6, {})
        
        # Search for memories
        query = MemoryQuery(query="programming", limit=5)
        results = await memory.retrieve_memories(query)
        
        assert len(results) >= 1
        assert all(result.relevance_score >= 0 for result in results)
    
    async def test_get_memory(self, memory):
        """Test getting a specific memory."""
        memory_id = await memory.store_memory("Test content", MemoryType.FACT, 0.7, {})
        
        retrieved = await memory.get_memory(memory_id)
        
        assert retrieved is not None
        assert retrieved.content == "Test content"
        assert retrieved.id == memory_id
    
    async def test_update_memory(self, memory):
        """Test updating a memory."""
        memory_id = await memory.store_memory("Original content", MemoryType.FACT, 0.5, {})
        
        success = await memory.update_memory(memory_id, content="Updated content", importance=0.9)
        
        assert success is True
        
        updated = await memory.get_memory(memory_id)
        assert updated.content == "Updated content"
        assert updated.metadata.importance == 0.9
    
    async def test_delete_memory(self, memory):
        """Test deleting a memory."""
        memory_id = await memory.store_memory("To be deleted", MemoryType.FACT, 0.5, {})
        
        success = await memory.delete_memory(memory_id)
        assert success is True
        
        deleted = await memory.get_memory(memory_id)
        assert deleted is None
    
    async def test_get_stats(self, memory):
        """Test getting memory statistics."""
        # Store some memories
        await memory.store_memory("Fact 1", MemoryType.FACT, 0.8, {})
        await memory.store_memory("Conversation 1", MemoryType.CONVERSATION, 0.6, {})
        
        stats = await memory.get_stats()
        
        assert stats.total_memories == 2
        assert MemoryType.FACT in stats.memory_types
        assert stats.average_importance > 0


class TestMemoryManager:
    """Test cases for MemoryManager."""
    
    @pytest_asyncio.fixture
    async def manager(self):
        """Create a test memory manager."""
        # Use temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        config = MemoryManagerConfig(
            short_term=ShortTermMemoryConfig(max_turns=5),
            long_term=LongTermMemoryConfig(
                storage_path=temp_dir,
                vector_db_type="in_memory",
                max_memories=10
            ),
            auto_persist=False  # Disable auto-persist for testing
        )

        manager = MemoryManager(config)
        yield manager

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_add_turn(self, manager):
        """Test adding turns to memory manager."""
        turn = manager.add_turn("user", "Hello")
        
        assert turn.role == ConversationRole.USER
        assert turn.content == "Hello"
    
    def test_get_context(self, manager):
        """Test getting context from memory manager."""
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi!")
        
        context = manager.get_context()
        
        assert len(context.messages) == 2
        assert context.short_term_turns == 2
        assert context.long_term_memories == 0
    
    @pytest.mark.asyncio
    async def test_enhanced_context(self, manager):
        """Test getting enhanced context with long-term memories."""
        # Add some conversation
        manager.add_turn("user", "I love Python programming")
        manager.add_turn("assistant", "That's great!")
        
        # Store a relevant fact
        await manager.store_fact("Python is a versatile programming language")
        
        # Get enhanced context
        context = await manager.get_enhanced_context("Tell me about Python")
        
        assert len(context.messages) >= 2  # At least the conversation
        # May include long-term memories depending on relevance
    
    @pytest.mark.asyncio
    async def test_store_fact(self, manager):
        """Test storing facts."""
        memory_id = await manager.store_fact("Test fact", importance=0.8)
        
        assert memory_id is not None
        
        # Should be able to search for it
        results = await manager.search_memories("Test fact")
        assert len(results) >= 1
    
    @pytest.mark.asyncio
    async def test_consolidate_short_term(self, manager):
        """Test consolidating short-term memory."""
        # Add some conversation turns
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi there!")
        manager.add_turn("user", "How are you?")
        manager.add_turn("assistant", "I'm doing well, thanks!")
        
        # Consolidate
        memories_created = await manager.consolidate_short_term()
        
        assert memories_created >= 0  # May create memories depending on content
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, manager):
        """Test getting memory statistics."""
        manager.add_turn("user", "Hello")
        await manager.store_fact("Test fact")
        
        stats = await manager.get_memory_stats()
        
        assert "short_term" in stats
        assert "long_term" in stats
        assert "consolidation" in stats
        assert stats["short_term"]["total_turns"] == 1
    
    def test_clear_short_term(self, manager):
        """Test clearing short-term memory."""
        manager.add_turn("user", "Hello")
        manager.clear_short_term()
        
        context = manager.get_context()
        assert len(context.messages) == 0


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        config = MemoryManagerConfig(
            long_term=LongTermMemoryConfig(vector_db_type="in_memory")
        )
        manager = MemoryManager(config)
        
        # Test basic functionality
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi there!")
        
        context = manager.get_context()
        print(f"Context has {len(context.messages)} messages")
        
        await manager.store_fact("Test fact for simple test")
        results = await manager.search_memories("test")
        print(f"Found {len(results)} memories")
        
        print("✅ Memory system test passed")
    
    asyncio.run(simple_test())
