# Memory System API

The Memory System API provides intelligent memory management with vector storage, semantic retrieval, and learning capabilities for enhanced conversation context.

## Quick Start

```python
from coda.components.memory import MemoryManager, MemoryConfig

# Initialize memory manager
config = MemoryConfig()
memory_manager = MemoryManager(config)
await memory_manager.initialize()

# Store a memory
memory_id = await memory_manager.store_memory(
    content="User prefers morning meetings",
    memory_type=MemoryType.PREFERENCE,
    metadata={"category": "scheduling"}
)

# Retrieve relevant memories
memories = await memory_manager.retrieve_memories(
    query="schedule a meeting",
    limit=5
)

for memory in memories:
    print(f"Memory: {memory.content} (relevance: {memory.relevance_score:.2f})")
```

## Core Classes

### MemoryManager

Main orchestrator for memory operations.

```python
class MemoryManager:
    def __init__(self, config: MemoryConfig) -> None
    
    async def initialize(self) -> None
    async def cleanup(self) -> None
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None
    ) -> str
    
    async def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_relevance: float = 0.5
    ) -> List[Memory]
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None
    ) -> bool
    
    async def delete_memory(self, memory_id: str) -> bool
    
    async def consolidate_memories(
        self,
        time_window_hours: int = 24
    ) -> ConsolidationResult
```

#### Methods

**`store_memory(content, memory_type, metadata, importance_score)`**
- Stores a new memory with vector embedding
- Returns: Memory ID
- Raises: `MemoryStorageError`, `ValidationError`

**`retrieve_memories(query, memory_types, limit, min_relevance)`**
- Retrieves memories using semantic similarity
- Returns: List of `Memory` objects with relevance scores
- Raises: `MemoryRetrievalError`

**`consolidate_memories(time_window_hours)`**
- Consolidates related memories to reduce redundancy
- Returns: `ConsolidationResult` with statistics
- Raises: `MemoryConsolidationError`

### Memory

Represents a stored memory with metadata and embeddings.

```python
@dataclass
class Memory:
    memory_id: str
    content: str
    memory_type: MemoryType
    timestamp: float
    importance_score: float
    access_count: int = 0
    last_accessed: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None
    decay_factor: Optional[float] = None
```

### MemoryType

```python
class MemoryType(str, Enum):
    EPISODIC = "episodic"        # Specific events and experiences
    SEMANTIC = "semantic"        # Facts and general knowledge
    PREFERENCE = "preference"    # User preferences and settings
    PROCEDURAL = "procedural"    # How-to knowledge and procedures
    EMOTIONAL = "emotional"      # Emotional associations and responses
    CONTEXTUAL = "contextual"    # Conversation context and patterns
```

## Memory Storage

### Short-term Memory

Temporary storage for immediate conversation context:

```python
# Store short-term memory
await memory_manager.store_short_term_memory(
    content="User asked about weather in Paris",
    conversation_id="conv_123",
    ttl_minutes=30
)

# Retrieve conversation context
context = await memory_manager.get_conversation_context(
    conversation_id="conv_123",
    max_memories=10
)
```

### Long-term Memory

Persistent storage with importance-based retention:

```python
# Store important long-term memory
memory_id = await memory_manager.store_memory(
    content="User's birthday is March 15th",
    memory_type=MemoryType.PREFERENCE,
    importance_score=0.9,
    metadata={
        "category": "personal_info",
        "date_type": "birthday"
    }
)

# Retrieve by importance
important_memories = await memory_manager.retrieve_memories(
    query="personal information",
    min_importance=0.8
)
```

## Memory Retrieval

### Semantic Search

```python
# Basic semantic search
memories = await memory_manager.retrieve_memories(
    query="machine learning projects",
    limit=5
)

# Filtered search
memories = await memory_manager.retrieve_memories(
    query="Python programming",
    memory_types=[MemoryType.PROCEDURAL, MemoryType.SEMANTIC],
    min_relevance=0.7
)

# Time-based search
recent_memories = await memory_manager.retrieve_recent_memories(
    hours=24,
    memory_types=[MemoryType.EPISODIC]
)
```

### Advanced Retrieval

```python
# Multi-query retrieval
memories = await memory_manager.retrieve_memories_multi_query(
    queries=["Python", "machine learning", "data science"],
    aggregation_method="weighted_average"
)

# Contextual retrieval
memories = await memory_manager.retrieve_contextual_memories(
    conversation_id="conv_123",
    include_related=True,
    max_depth=2
)
```

## Memory Learning

### Automatic Learning

```python
# Enable automatic learning from conversations
await memory_manager.enable_auto_learning(
    conversation_id="conv_123",
    learning_rate=0.1,
    importance_threshold=0.5
)

# Learn from interaction
await memory_manager.learn_from_interaction(
    user_input="I love hiking in the mountains",
    assistant_response="That sounds wonderful! Do you have a favorite trail?",
    user_feedback={"helpful": True, "relevant": True}
)
```

### Manual Learning

```python
# Extract and store key information
extracted_info = await memory_manager.extract_key_information(
    text="I work as a software engineer at TechCorp and I'm working on a new AI project",
    extract_types=["job", "company", "project"]
)

for info in extracted_info:
    await memory_manager.store_memory(
        content=info.content,
        memory_type=info.suggested_type,
        importance_score=info.importance_score
    )
```

## Configuration

### MemoryConfig

```python
@dataclass
class MemoryConfig:
    # Storage settings
    vector_store_type: str = "chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_memories: int = 10000
    
    # Retrieval settings
    default_retrieval_limit: int = 10
    min_relevance_threshold: float = 0.5
    enable_semantic_search: bool = True
    
    # Learning settings
    enable_auto_learning: bool = True
    learning_rate: float = 0.1
    importance_decay_rate: float = 0.01
    
    # Consolidation settings
    enable_auto_consolidation: bool = True
    consolidation_interval_hours: int = 24
    similarity_threshold: float = 0.85
    
    # Performance settings
    batch_size: int = 100
    cache_size: int = 1000
    enable_compression: bool = True
```

### Storage Configuration

```python
@dataclass
class VectorStoreConfig:
    store_type: str = "chroma"
    persist_directory: str = "data/memory/vectors"
    collection_name: str = "coda_memories"
    distance_metric: str = "cosine"
    
    # Chroma-specific settings
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    
    # Performance settings
    batch_size: int = 100
    max_batch_size: int = 1000
```

## Memory Analytics

### Usage Statistics

```python
# Get memory statistics
stats = await memory_manager.get_memory_stats()
print(f"Total memories: {stats.total_memories}")
print(f"Memory types: {stats.memory_type_distribution}")
print(f"Average importance: {stats.avg_importance_score:.2f}")

# Get retrieval analytics
retrieval_stats = await memory_manager.get_retrieval_stats()
print(f"Total retrievals: {retrieval_stats.total_retrievals}")
print(f"Average relevance: {retrieval_stats.avg_relevance_score:.2f}")
```

### Memory Health

```python
# Check memory health
health = await memory_manager.check_memory_health()
print(f"Health score: {health.overall_score:.2f}")
print(f"Issues: {health.issues}")

# Optimize memory storage
optimization_result = await memory_manager.optimize_storage()
print(f"Freed space: {optimization_result.freed_space_mb:.1f}MB")
print(f"Consolidated memories: {optimization_result.consolidated_count}")
```

## Integration Features

### WebSocket Integration

```python
# Real-time memory updates via WebSocket
await memory_manager.enable_websocket_integration(
    websocket_manager=websocket_manager,
    broadcast_updates=True
)

# Memory events are automatically broadcast
# - memory_stored
# - memory_retrieved
# - memory_consolidated
```

### Conversation Integration

```python
# Automatic conversation memory
await memory_manager.enable_conversation_integration(
    conversation_manager=conversation_manager,
    auto_store_messages=True,
    auto_extract_entities=True
)
```

## Error Handling

### Exception Hierarchy

```python
class MemoryError(Exception):
    """Base exception for memory system errors."""

class MemoryStorageError(MemoryError):
    """Memory storage related errors."""

class MemoryRetrievalError(MemoryError):
    """Memory retrieval related errors."""

class MemoryConsolidationError(MemoryError):
    """Memory consolidation related errors."""

class EmbeddingError(MemoryError):
    """Embedding generation related errors."""

class VectorStoreError(MemoryError):
    """Vector store related errors."""
```

### Error Handling Example

```python
try:
    memories = await memory_manager.retrieve_memories(query)
except MemoryRetrievalError as e:
    logger.error(f"Memory retrieval failed: {e}")
    # Fallback to cached results
except EmbeddingError as e:
    logger.error(f"Embedding generation failed: {e}")
    # Use keyword-based search
except VectorStoreError as e:
    logger.error(f"Vector store error: {e}")
    # Switch to backup storage
```

## Best Practices

### Memory Management

```python
# Regular cleanup of old, unimportant memories
await memory_manager.cleanup_old_memories(
    max_age_days=30,
    min_importance=0.3,
    max_access_count=1
)

# Batch operations for better performance
memory_batch = [
    {"content": "User likes coffee", "type": MemoryType.PREFERENCE},
    {"content": "Meeting scheduled for 2pm", "type": MemoryType.EPISODIC}
]
await memory_manager.store_memories_batch(memory_batch)
```

### Performance Optimization

```python
# Use memory caching for frequently accessed memories
await memory_manager.enable_caching(
    cache_size=1000,
    ttl_minutes=60
)

# Preload important memories
await memory_manager.preload_important_memories(
    min_importance=0.8,
    max_count=100
)
```
