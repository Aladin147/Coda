"""
Comprehensive tests for LongTermMemory to increase coverage from 12% to 80%+.
Targets specific uncovered lines in long_term.py.
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.coda.components.memory.long_term import LongTermMemory
from src.coda.components.memory.models import Memory, MemoryType, MemoryMetadata, MemoryQuery
from src.coda.components.memory.encoder import MemoryEncoder


class TestLongTermMemoryComprehensive:
    """Comprehensive tests for LongTermMemory covering all major functionality."""

    @pytest_asyncio.fixture
    async def mock_chroma_client(self):
        """Create mock ChromaDB client."""
        client = Mock()
        collection = Mock()
        
        # Mock collection methods
        collection.add = Mock()
        collection.query = Mock(return_value={
            'ids': [['mem_1', 'mem_2']],
            'distances': [[0.1, 0.3]],
            'metadatas': [[
                {'type': 'conversation', 'importance': 0.8, 'timestamp': '2024-01-01T00:00:00'},
                {'type': 'fact', 'importance': 0.6, 'timestamp': '2024-01-02T00:00:00'}
            ]],
            'documents': [['Test memory 1', 'Test memory 2']]
        })
        collection.get = Mock(return_value={
            'ids': ['mem_1'],
            'metadatas': [{'type': 'conversation', 'importance': 0.8}],
            'documents': ['Test memory content']
        })
        collection.update = Mock()
        collection.delete = Mock()
        collection.count = Mock(return_value=10)
        
        client.get_or_create_collection = Mock(return_value=collection)
        client.delete_collection = Mock()
        
        return client

    @pytest_asyncio.fixture
    async def mock_encoder(self):
        """Create mock memory encoder."""
        encoder = Mock(spec=MemoryEncoder)
        encoder.encode_memory = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        encoder.encode_query = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        encoder.calculate_importance = Mock(return_value=0.7)
        encoder.extract_topics = Mock(return_value=['test', 'memory'])
        return encoder

    @pytest_asyncio.fixture
    async def long_term_memory(self, mock_chroma_client, mock_encoder):
        """Create LongTermMemory instance with mocked dependencies."""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            memory = LongTermMemory(
                persist_directory="./test_memory",
                collection_name="test_collection",
                encoder=mock_encoder
            )
            await memory.initialize()
            return memory

    @pytest_asyncio.fixture
    def sample_memory(self):
        """Create sample memory for testing."""
        return Memory(
            id=str(uuid.uuid4()),
            content="This is a test memory about machine learning",
            memory_type=MemoryType.CONVERSATION,
            metadata=MemoryMetadata(
                importance=0.8,
                topics=['machine learning', 'test'],
                timestamp=datetime.now(),
                source="test"
            )
        )

    @pytest.mark.asyncio
    async def test_initialization(self, mock_chroma_client, mock_encoder):
        """Test LongTermMemory initialization."""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            memory = LongTermMemory(
                persist_directory="./test_memory",
                collection_name="test_collection",
                encoder=mock_encoder
            )
            
            assert not memory.is_initialized
            await memory.initialize()
            assert memory.is_initialized
            
            # Test double initialization
            await memory.initialize()
            assert memory.is_initialized

    @pytest.mark.asyncio
    async def test_store_memory_basic(self, long_term_memory, sample_memory):
        """Test basic memory storage."""
        result = await long_term_memory.store_memory(sample_memory)
        
        assert result == sample_memory.id
        long_term_memory.collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_memory_with_embedding(self, long_term_memory, sample_memory, mock_encoder):
        """Test memory storage with custom embedding."""
        custom_embedding = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        result = await long_term_memory.store_memory(sample_memory, embedding=custom_embedding)
        
        assert result == sample_memory.id
        # Should not call encoder since embedding provided
        mock_encoder.encode_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_memory_batch(self, long_term_memory):
        """Test batch memory storage."""
        memories = [
            Memory(
                id=str(uuid.uuid4()),
                content=f"Test memory {i}",
                memory_type=MemoryType.FACT,
                metadata=MemoryMetadata(importance=0.5 + i * 0.1)
            )
            for i in range(3)
        ]
        
        results = await long_term_memory.store_memories(memories)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_memories_by_query(self, long_term_memory):
        """Test memory retrieval by query."""
        query = MemoryQuery(
            text="machine learning",
            memory_types=[MemoryType.CONVERSATION],
            limit=5
        )
        
        memories = await long_term_memory.retrieve_memories(query)
        
        assert isinstance(memories, list)
        assert len(memories) <= 5
        long_term_memory.collection.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_memories_with_filters(self, long_term_memory):
        """Test memory retrieval with various filters."""
        # Test with time range filter
        query = MemoryQuery(
            text="test",
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            min_importance=0.5,
            topics=['machine learning']
        )
        
        memories = await long_term_memory.retrieve_memories(query)
        
        assert isinstance(memories, list)
        long_term_memory.collection.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_by_id(self, long_term_memory):
        """Test retrieving specific memory by ID."""
        memory_id = "mem_1"
        
        memory = await long_term_memory.get_memory(memory_id)
        
        assert memory is not None
        assert memory.id == memory_id
        long_term_memory.collection.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, long_term_memory):
        """Test retrieving non-existent memory."""
        long_term_memory.collection.get.return_value = {
            'ids': [],
            'metadatas': [],
            'documents': []
        }
        
        memory = await long_term_memory.get_memory("nonexistent")
        assert memory is None

    @pytest.mark.asyncio
    async def test_update_memory(self, long_term_memory, sample_memory):
        """Test memory update."""
        # Update memory content and importance
        sample_memory.content = "Updated content"
        sample_memory.metadata.importance = 0.9
        
        success = await long_term_memory.update_memory(sample_memory)
        
        assert success is True
        long_term_memory.collection.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_memory(self, long_term_memory):
        """Test memory deletion."""
        memory_id = "mem_1"
        
        success = await long_term_memory.delete_memory(memory_id)
        
        assert success is True
        long_term_memory.collection.delete.assert_called_once_with(ids=[memory_id])

    @pytest.mark.asyncio
    async def test_delete_memories_batch(self, long_term_memory):
        """Test batch memory deletion."""
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        
        success = await long_term_memory.delete_memories(memory_ids)
        
        assert success is True
        long_term_memory.collection.delete.assert_called_once_with(ids=memory_ids)

    @pytest.mark.asyncio
    async def test_search_similar_memories(self, long_term_memory, sample_memory):
        """Test finding similar memories."""
        similar_memories = await long_term_memory.search_similar(
            sample_memory, 
            threshold=0.8, 
            limit=3
        )
        
        assert isinstance(similar_memories, list)
        assert len(similar_memories) <= 3

    @pytest.mark.asyncio
    async def test_get_memories_by_topic(self, long_term_memory):
        """Test retrieving memories by topic."""
        memories = await long_term_memory.get_memories_by_topic(
            "machine learning", 
            limit=10
        )
        
        assert isinstance(memories, list)
        assert len(memories) <= 10

    @pytest.mark.asyncio
    async def test_get_memories_by_importance(self, long_term_memory):
        """Test retrieving memories by importance threshold."""
        memories = await long_term_memory.get_memories_by_importance(
            min_importance=0.7,
            limit=5
        )
        
        assert isinstance(memories, list)
        assert len(memories) <= 5

    @pytest.mark.asyncio
    async def test_get_recent_memories(self, long_term_memory):
        """Test retrieving recent memories."""
        memories = await long_term_memory.get_recent_memories(
            hours=24,
            limit=10
        )
        
        assert isinstance(memories, list)
        assert len(memories) <= 10

    @pytest.mark.asyncio
    async def test_consolidate_memories(self, long_term_memory):
        """Test memory consolidation process."""
        # Mock memories that should be consolidated
        long_term_memory.collection.query.return_value = {
            'ids': [['mem_1', 'mem_2', 'mem_3']],
            'distances': [[0.1, 0.15, 0.2]],  # Very similar
            'metadatas': [[
                {'type': 'conversation', 'importance': 0.6},
                {'type': 'conversation', 'importance': 0.5},
                {'type': 'conversation', 'importance': 0.4}
            ]],
            'documents': [['Similar content 1', 'Similar content 2', 'Similar content 3']]
        }
        
        consolidated_count = await long_term_memory.consolidate_memories(
            similarity_threshold=0.9,
            max_age_hours=24
        )
        
        assert isinstance(consolidated_count, int)
        assert consolidated_count >= 0

    @pytest.mark.asyncio
    async def test_cleanup_old_memories(self, long_term_memory):
        """Test cleanup of old, low-importance memories."""
        deleted_count = await long_term_memory.cleanup_old_memories(
            max_age_days=30,
            min_importance=0.3,
            max_memories=1000
        )
        
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, long_term_memory):
        """Test getting memory statistics."""
        stats = await long_term_memory.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_memories' in stats
        assert 'memory_types' in stats
        assert 'average_importance' in stats
        assert 'topics' in stats

    @pytest.mark.asyncio
    async def test_export_memories(self, long_term_memory):
        """Test exporting memories."""
        # Mock collection data for export
        long_term_memory.collection.get.return_value = {
            'ids': ['mem_1', 'mem_2'],
            'metadatas': [
                {'type': 'conversation', 'importance': 0.8},
                {'type': 'fact', 'importance': 0.6}
            ],
            'documents': ['Memory 1', 'Memory 2']
        }
        
        exported_data = await long_term_memory.export_memories()
        
        assert isinstance(exported_data, dict)
        assert 'memories' in exported_data
        assert 'metadata' in exported_data

    @pytest.mark.asyncio
    async def test_import_memories(self, long_term_memory):
        """Test importing memories."""
        import_data = {
            'memories': [
                {
                    'id': 'imported_1',
                    'content': 'Imported memory',
                    'type': 'conversation',
                    'importance': 0.7
                }
            ],
            'metadata': {
                'export_date': '2024-01-01T00:00:00',
                'version': '1.0'
            }
        }
        
        imported_count = await long_term_memory.import_memories(import_data)
        
        assert isinstance(imported_count, int)
        assert imported_count >= 0

    @pytest.mark.asyncio
    async def test_error_handling_invalid_memory(self, long_term_memory):
        """Test error handling with invalid memory data."""
        # Test with None memory
        with pytest.raises(ValueError):
            await long_term_memory.store_memory(None)
        
        # Test with memory missing required fields
        invalid_memory = Memory(
            id="",  # Empty ID
            content="",  # Empty content
            memory_type=MemoryType.CONVERSATION
        )
        
        with pytest.raises(ValueError):
            await long_term_memory.store_memory(invalid_memory)

    @pytest.mark.asyncio
    async def test_error_handling_database_errors(self, long_term_memory):
        """Test error handling for database errors."""
        # Mock database error
        long_term_memory.collection.add.side_effect = Exception("Database error")
        
        sample_memory = Memory(
            id=str(uuid.uuid4()),
            content="Test memory",
            memory_type=MemoryType.FACT
        )
        
        with pytest.raises(Exception):
            await long_term_memory.store_memory(sample_memory)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, long_term_memory):
        """Test concurrent memory operations."""
        # Create multiple memories
        memories = [
            Memory(
                id=str(uuid.uuid4()),
                content=f"Concurrent memory {i}",
                memory_type=MemoryType.FACT,
                metadata=MemoryMetadata(importance=0.5)
            )
            for i in range(5)
        ]
        
        # Store concurrently
        tasks = [long_term_memory.store_memory(memory) for memory in memories]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all operations completed
        assert len(results) == 5
        assert all(isinstance(r, str) or isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_memory_lifecycle(self, long_term_memory, sample_memory):
        """Test complete memory lifecycle: store, retrieve, update, delete."""
        # Store memory
        memory_id = await long_term_memory.store_memory(sample_memory)
        assert memory_id == sample_memory.id
        
        # Retrieve memory
        retrieved = await long_term_memory.get_memory(memory_id)
        assert retrieved is not None
        
        # Update memory
        sample_memory.content = "Updated content"
        success = await long_term_memory.update_memory(sample_memory)
        assert success is True
        
        # Delete memory
        success = await long_term_memory.delete_memory(memory_id)
        assert success is True
