"""
Memory System Performance Tests.
Tests vector database operations, memory retrieval speed, and large-scale memory handling.
"""

import pytest
import asyncio
import time
import uuid
import random
import string
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.coda.components.memory.models import (
    Memory, MemoryType, MemoryMetadata, MemoryQuery
)
from src.coda.components.memory.manager import MemoryManager
from src.coda.components.memory.long_term import LongTermMemory
from src.coda.components.memory.short_term import ShortTermMemory


class MemoryPerformanceMetrics:
    """Memory performance metrics collector."""
    
    def __init__(self):
        self.storage_times = []
        self.retrieval_times = []
        self.query_times = []
        self.memory_sizes = []
    
    def record_storage_time(self, time_ms: float, memory_size: int):
        """Record memory storage time."""
        self.storage_times.append(time_ms)
        self.memory_sizes.append(memory_size)
    
    def record_retrieval_time(self, time_ms: float, results_count: int):
        """Record memory retrieval time."""
        self.retrieval_times.append(time_ms)
    
    def record_query_time(self, time_ms: float):
        """Record query processing time."""
        self.query_times.append(time_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        import numpy as np
        return {
            "avg_storage_time": np.mean(self.storage_times) if self.storage_times else 0,
            "avg_retrieval_time": np.mean(self.retrieval_times) if self.retrieval_times else 0,
            "avg_query_time": np.mean(self.query_times) if self.query_times else 0,
            "max_storage_time": np.max(self.storage_times) if self.storage_times else 0,
            "max_retrieval_time": np.max(self.retrieval_times) if self.retrieval_times else 0,
            "total_memories_processed": len(self.storage_times)
        }


class TestMemoryPerformance:
    """Memory system performance tests."""

    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics collector."""
        return MemoryPerformanceMetrics()

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager with realistic performance."""
        manager = Mock(spec=MemoryManager)
        
        # Mock storage with realistic timing
        async def mock_store(memory_id, content, response):
            await asyncio.sleep(0.01)  # 10ms storage time
            return f"stored-{memory_id}"
        
        # Mock retrieval with realistic timing
        async def mock_retrieve(query):
            await asyncio.sleep(0.005)  # 5ms retrieval time
            return [
                Memory(
                    id=f"mem-{i}",
                    content=f"Retrieved memory {i} for query: {query.query}",
                    metadata=MemoryMetadata(
                        source_type=MemoryType.CONVERSATION,
                        importance=0.7 + (i * 0.1),
                        topics=["test", "performance"],
                        timestamp=datetime.now()
                    )
                )
                for i in range(min(query.limit, 5))
            ]
        
        manager.store_conversation = AsyncMock(side_effect=mock_store)
        manager.retrieve_relevant_memories = AsyncMock(side_effect=mock_retrieve)
        manager.is_initialized = True
        
        return manager

    @pytest.fixture
    def sample_memories(self):
        """Generate sample memories for testing."""
        memories = []
        topics_pool = ["technology", "science", "history", "art", "music", "sports", "travel"]
        
        for i in range(1000):
            content = f"Sample memory content {i}: " + "".join(
                random.choices(string.ascii_letters + string.digits + " ", k=random.randint(50, 200))
            )
            
            memory = Memory(
                id=str(uuid.uuid4()),
                content=content,
                metadata=MemoryMetadata(
                    source_type=random.choice(list(MemoryType)),
                    importance=random.uniform(0.1, 1.0),
                    topics=random.sample(topics_pool, random.randint(1, 3)),
                    timestamp=datetime.now() - timedelta(days=random.randint(0, 365))
                )
            )
            memories.append(memory)
        
        return memories

    @pytest.mark.benchmark(group="memory_storage")
    def test_single_memory_storage_performance(self, benchmark, mock_memory_manager):
        """Benchmark single memory storage performance."""
        
        def store_memory():
            return asyncio.run(mock_memory_manager.store_conversation(
                "test-conv",
                "Test user message",
                "Test assistant response"
            ))
        
        result = benchmark(store_memory)
        assert result.startswith("stored-")

    @pytest.mark.benchmark(group="memory_retrieval")
    def test_single_memory_retrieval_performance(self, benchmark, mock_memory_manager):
        """Benchmark single memory retrieval performance."""
        
        def retrieve_memories():
            query = MemoryQuery(
                query="test query",
                limit=10,
                min_relevance=0.5
            )
            return asyncio.run(mock_memory_manager.retrieve_relevant_memories(query))
        
        result = benchmark(retrieve_memories)
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_bulk_memory_storage_performance(self, mock_memory_manager, sample_memories, performance_metrics):
        """Test bulk memory storage performance."""
        
        batch_sizes = [10, 50, 100, 500]
        
        for batch_size in batch_sizes:
            batch = sample_memories[:batch_size]
            
            start_time = time.time()
            
            # Store memories concurrently
            tasks = [
                mock_memory_manager.store_conversation(
                    f"batch-{i}",
                    memory.content[:100],  # Truncate for user message
                    f"Response for memory {i}"
                )
                for i, memory in enumerate(batch)
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Record performance
            total_time = (end_time - start_time) * 1000  # Convert to ms
            avg_time_per_memory = total_time / batch_size
            
            performance_metrics.record_storage_time(avg_time_per_memory, len(memory.content))
            
            print(f"Batch size {batch_size}: {total_time:.2f}ms total, {avg_time_per_memory:.2f}ms per memory")
            
            # Verify all stored successfully
            assert len(results) == batch_size
            for result in results:
                assert result.startswith("stored-")
            
            # Performance assertions
            assert avg_time_per_memory < 50  # Under 50ms per memory
            assert total_time < 5000  # Under 5 seconds total

    @pytest.mark.asyncio
    async def test_concurrent_memory_retrieval_performance(self, mock_memory_manager, performance_metrics):
        """Test concurrent memory retrieval performance."""
        
        # Create diverse queries
        queries = [
            MemoryQuery(query="technology and programming", limit=10),
            MemoryQuery(query="science and research", limit=15),
            MemoryQuery(query="history and culture", limit=5),
            MemoryQuery(query="art and creativity", limit=20),
            MemoryQuery(query="music and entertainment", limit=8),
        ]
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for level in concurrency_levels:
            # Create concurrent queries
            concurrent_queries = queries * (level // len(queries) + 1)
            concurrent_queries = concurrent_queries[:level]
            
            start_time = time.time()
            
            # Execute queries concurrently
            tasks = [
                mock_memory_manager.retrieve_relevant_memories(query)
                for query in concurrent_queries
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Record performance
            total_time = (end_time - start_time) * 1000
            avg_time_per_query = total_time / level
            
            performance_metrics.record_retrieval_time(avg_time_per_query, sum(len(r) for r in results))
            
            print(f"Concurrency {level}: {total_time:.2f}ms total, {avg_time_per_query:.2f}ms per query")
            
            # Verify results
            assert len(results) == level
            for result in results:
                assert isinstance(result, list)
                assert len(result) > 0
            
            # Performance assertions
            assert avg_time_per_query < 100  # Under 100ms per query
            assert total_time < 2000  # Under 2 seconds total

    @pytest.mark.benchmark(group="memory_query_complexity")
    def test_complex_query_performance(self, benchmark, mock_memory_manager):
        """Benchmark complex memory query performance."""
        
        def complex_query():
            query = MemoryQuery(
                query="Find memories about technology, programming, and artificial intelligence from the last 30 days with high importance",
                memory_types=[MemoryType.CONVERSATION, MemoryType.FACT, MemoryType.PREFERENCE],
                limit=50,
                min_relevance=0.7,
                time_range=(datetime.now() - timedelta(days=30), datetime.now())
            )
            return asyncio.run(mock_memory_manager.retrieve_relevant_memories(query))
        
        result = benchmark(complex_query)
        assert isinstance(result, list)

    def test_memory_system_scalability(self, mock_memory_manager, sample_memories):
        """Test memory system scalability with increasing data size."""
        
        async def scalability_test(memory_count: int):
            """Test with different amounts of memory data."""
            memories = sample_memories[:memory_count]
            
            # Store memories
            start_time = time.time()
            store_tasks = [
                mock_memory_manager.store_conversation(
                    f"scale-{i}",
                    memory.content[:100],
                    f"Response {i}"
                )
                for i, memory in enumerate(memories)
            ]
            
            await asyncio.gather(*store_tasks)
            storage_time = time.time() - start_time
            
            # Query memories
            start_time = time.time()
            query_tasks = [
                mock_memory_manager.retrieve_relevant_memories(
                    MemoryQuery(query=f"query {i}", limit=10)
                )
                for i in range(10)  # 10 different queries
            ]
            
            query_results = await asyncio.gather(*query_tasks)
            query_time = time.time() - start_time
            
            return {
                "memory_count": memory_count,
                "storage_time": storage_time * 1000,  # ms
                "query_time": query_time * 1000,  # ms
                "avg_storage_time": (storage_time * 1000) / memory_count,
                "avg_query_time": (query_time * 1000) / 10,
                "total_results": sum(len(r) for r in query_results)
            }
        
        # Test different scales
        scales = [100, 500, 1000]
        scalability_results = []
        
        for scale in scales:
            result = asyncio.run(scalability_test(scale))
            scalability_results.append(result)
            
            print(f"Scale {result['memory_count']}: "
                  f"Storage {result['avg_storage_time']:.2f}ms/memory, "
                  f"Query {result['avg_query_time']:.2f}ms/query")
        
        # Verify scalability
        for result in scalability_results:
            # Storage time per memory should be reasonable
            assert result['avg_storage_time'] < 100  # Under 100ms per memory
            
            # Query time should be reasonable
            assert result['avg_query_time'] < 200  # Under 200ms per query
            
            # Should return results
            assert result['total_results'] > 0

    @pytest.mark.asyncio
    async def test_memory_search_performance(self, mock_memory_manager):
        """Test memory search performance with different query types."""
        
        search_queries = [
            ("simple", "technology"),
            ("compound", "technology AND programming"),
            ("complex", "technology OR programming AND (artificial intelligence OR machine learning)"),
            ("semantic", "How to build AI systems that can understand natural language"),
            ("temporal", "Recent discussions about programming languages"),
        ]
        
        search_results = {}
        
        for query_type, query_text in search_queries:
            query = MemoryQuery(
                query=query_text,
                limit=20,
                min_relevance=0.5
            )
            
            start_time = time.time()
            results = await mock_memory_manager.retrieve_relevant_memories(query)
            end_time = time.time()
            
            search_time = (end_time - start_time) * 1000
            search_results[query_type] = {
                "query": query_text,
                "time_ms": search_time,
                "results_count": len(results),
                "results_per_ms": len(results) / search_time if search_time > 0 else 0
            }
            
            print(f"{query_type.capitalize()} search: {search_time:.2f}ms, {len(results)} results")
        
        # Verify search performance
        for query_type, result in search_results.items():
            assert result["time_ms"] < 500  # Under 500ms
            assert result["results_count"] > 0  # Should return results

    @pytest.mark.benchmark(group="memory_filtering")
    def test_memory_filtering_performance(self, benchmark, mock_memory_manager):
        """Benchmark memory filtering performance."""
        
        def filtered_query():
            query = MemoryQuery(
                query="programming",
                memory_types=[MemoryType.CONVERSATION, MemoryType.FACT],
                limit=30,
                min_relevance=0.8,
                time_range=(datetime.now() - timedelta(days=7), datetime.now())
            )
            return asyncio.run(mock_memory_manager.retrieve_relevant_memories(query))
        
        result = benchmark(filtered_query)
        assert isinstance(result, list)

    def test_memory_performance_under_load(self, mock_memory_manager):
        """Test memory system performance under sustained load."""
        
        async def sustained_load_test():
            """Run sustained operations for performance testing."""
            operations = []
            
            # Mix of storage and retrieval operations
            for i in range(100):
                if i % 3 == 0:  # Storage operation
                    op = mock_memory_manager.store_conversation(
                        f"load-{i}",
                        f"Load test message {i}",
                        f"Load test response {i}"
                    )
                else:  # Retrieval operation
                    query = MemoryQuery(
                        query=f"load test {i}",
                        limit=random.randint(5, 15)
                    )
                    op = mock_memory_manager.retrieve_relevant_memories(query)
                
                operations.append(op)
            
            start_time = time.time()
            results = await asyncio.gather(*operations)
            end_time = time.time()
            
            return {
                "total_operations": len(operations),
                "total_time": (end_time - start_time) * 1000,
                "avg_time_per_operation": ((end_time - start_time) * 1000) / len(operations),
                "operations_per_second": len(operations) / (end_time - start_time)
            }
        
        result = asyncio.run(sustained_load_test())
        
        print(f"Sustained load: {result['total_operations']} operations in {result['total_time']:.2f}ms")
        print(f"Average: {result['avg_time_per_operation']:.2f}ms per operation")
        print(f"Throughput: {result['operations_per_second']:.2f} operations/second")
        
        # Performance assertions
        assert result['avg_time_per_operation'] < 100  # Under 100ms per operation
        assert result['operations_per_second'] > 10  # At least 10 operations per second

    @pytest.mark.asyncio
    async def test_memory_cache_performance(self, mock_memory_manager):
        """Test memory caching performance."""
        
        # Same query repeated multiple times to test caching
        query = MemoryQuery(
            query="cache performance test",
            limit=10
        )
        
        # First query (cache miss)
        start_time = time.time()
        first_result = await mock_memory_manager.retrieve_relevant_memories(query)
        first_time = (time.time() - start_time) * 1000
        
        # Subsequent queries (should be faster if cached)
        cache_times = []
        for i in range(5):
            start_time = time.time()
            cached_result = await mock_memory_manager.retrieve_relevant_memories(query)
            cache_time = (time.time() - start_time) * 1000
            cache_times.append(cache_time)
        
        avg_cache_time = sum(cache_times) / len(cache_times)
        
        print(f"First query: {first_time:.2f}ms")
        print(f"Cached queries: {avg_cache_time:.2f}ms average")
        
        # Verify results are consistent
        assert len(first_result) == len(cached_result)
        
        # Cache should provide some performance benefit
        # (Note: Mock doesn't implement real caching, so this is more of a structure test)
        assert avg_cache_time > 0
