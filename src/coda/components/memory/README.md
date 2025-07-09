# Coda Memory System

> **Sophisticated dual-layer memory architecture for intelligent conversation and knowledge retention**

The memory system provides comprehensive memory management for Coda, featuring both short-term conversation context and long-term persistent memory with vector-based semantic search.

## Features

- 🧠 **Dual-layer architecture** - Short-term + long-term memory
- 🔍 **Semantic search** with vector embeddings
- 📝 **Automatic consolidation** from short-term to long-term
- ⚡ **Real-time WebSocket events** for monitoring
- 🎯 **Importance scoring** and relevance ranking
- 🕒 **Time-based decay** for memory relevance
- 💾 **Backup and restore** functionality
- 🔧 **Configurable chunking** and encoding
- 📊 **Comprehensive analytics** and statistics

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Short-term     │───▶│  Memory          │───▶│  Long-term      │
│  Memory         │    │  Manager         │    │  Memory         │
│  (Conversation) │    │  (Orchestrator)  │    │  (Vector DB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Context        │    │  Memory          │    │  Semantic       │
│  Generation     │    │  Encoder         │    │  Search         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Usage

```python
from coda.components.memory import MemoryManager, MemoryManagerConfig

# Create memory manager
config = MemoryManagerConfig()
memory = MemoryManager(config)

# Add conversation turns
memory.add_turn("user", "Hello, I'm learning Python")
memory.add_turn("assistant", "Great! Python is an excellent language to learn.")

# Get conversation context
context = memory.get_context(max_tokens=500)
print(f"Context has {len(context.messages)} messages")

# Store facts in long-term memory
await memory.store_fact("Python is a versatile programming language", importance=0.8)

# Search memories
results = await memory.search_memories("Python programming", limit=5)
for result in results:
    print(f"Score: {result.final_score:.3f} - {result.memory.content}")
```

### Enhanced Context with Long-term Memories

```python
# Get context enhanced with relevant long-term memories
enhanced_context = await memory.get_enhanced_context(
    user_input="Tell me about Python",
    max_tokens=800,
    max_memories=3
)

print(f"Enhanced context: {enhanced_context.short_term_turns} short-term + {enhanced_context.long_term_memories} long-term")
```

### WebSocket Integration

```python
from coda.components.memory import WebSocketMemoryManager
from coda.interfaces.websocket import CodaWebSocketServer, CodaWebSocketIntegration

# Set up WebSocket server
server = CodaWebSocketServer()
integration = CodaWebSocketIntegration(server)

# Create WebSocket-enabled memory manager
memory = WebSocketMemoryManager(config)
await memory.set_websocket_integration(integration)

# All memory operations now broadcast real-time events
await memory.store_fact("This will broadcast an event")
```

## Components

### Short-term Memory

Manages conversation context within token limits:

```python
from coda.components.memory import ShortTermMemory, ShortTermMemoryConfig

config = ShortTermMemoryConfig(max_turns=20, max_tokens=800)
short_term = ShortTermMemory(config)

# Add turns
turn = short_term.add_turn("user", "Hello")

# Get context within token budget
context = short_term.get_context(max_tokens=500)

# Export/import for persistence
exported = short_term.export_turns()
short_term.import_turns(exported)
```

### Long-term Memory

Persistent memory with vector embeddings:

```python
from coda.components.memory import LongTermMemory, LongTermMemoryConfig, MemoryType

config = LongTermMemoryConfig(
    storage_path="data/memory",
    vector_db_type="chroma",  # or "sqlite", "in_memory"
    embedding_model="all-MiniLM-L6-v2",
    max_memories=1000
)
long_term = LongTermMemory(config)

# Store memory
memory_id = await long_term.store_memory(
    content="Important information",
    memory_type=MemoryType.FACT,
    importance=0.8,
    metadata={"source": "user"}
)

# Retrieve memories
from coda.components.memory import MemoryQuery
query = MemoryQuery(query="information", limit=5, min_relevance=0.3)
results = await long_term.retrieve_memories(query)
```

### Memory Encoder

Converts conversations into memory chunks:

```python
from coda.components.memory import MemoryEncoder, MemoryEncoderConfig

config = MemoryEncoderConfig(chunk_size=200, chunk_overlap=50)
encoder = MemoryEncoder(config)

# Encode conversation
chunks = encoder.encode_conversation(conversation_turns)

# Encode facts
fact_chunk = encoder.encode_fact("Python is great for AI", source="user")

# Calculate importance
importance = encoder.calculate_importance("This is very important to remember")
```

## Configuration

### Memory Manager Configuration

```python
from coda.components.memory import (
    MemoryManagerConfig,
    ShortTermMemoryConfig,
    LongTermMemoryConfig,
    MemoryEncoderConfig
)

config = MemoryManagerConfig(
    short_term=ShortTermMemoryConfig(
        max_turns=20,
        max_tokens=800,
        include_system_in_context=True
    ),
    long_term=LongTermMemoryConfig(
        storage_path="data/memory/long_term",
        vector_db_type="chroma",
        embedding_model="all-MiniLM-L6-v2",
        max_memories=1000,
        device="cpu",  # or "cuda"
        time_decay_days=30.0
    ),
    encoder=MemoryEncoderConfig(
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_length=50,
        topic_extraction_enabled=True
    ),
    auto_persist=True,
    persist_interval=5
)
```

### Vector Database Options

#### ChromaDB (Recommended)
```python
config = LongTermMemoryConfig(
    vector_db_type="chroma",
    storage_path="data/memory/chroma"
)
```

#### SQLite
```python
config = LongTermMemoryConfig(
    vector_db_type="sqlite",
    storage_path="data/memory/sqlite"
)
```

#### In-Memory (Testing)
```python
config = LongTermMemoryConfig(
    vector_db_type="in_memory"
)
```

## Memory Types

The system supports different types of memories:

- `CONVERSATION` - Consolidated conversation chunks
- `FACT` - Explicit facts and information
- `PREFERENCE` - User preferences and settings
- `SYSTEM` - System-related information
- `TOOL_RESULT` - Results from tool executions

## Advanced Features

### Memory Consolidation

```python
# Automatic consolidation (configured)
memory.set_auto_persist(enabled=True, interval=5)

# Manual consolidation
memories_created = await memory.consolidate_short_term()
```

### Memory Analytics

```python
# Get comprehensive statistics
stats = await memory.get_memory_stats()

# Get conversation summary
summary = await memory.get_conversation_summary()

# Memory timeline (WebSocket version)
timeline = await memory.get_memory_timeline(days=7)
```

### Backup and Restore

```python
# Backup all memories
success = await memory.backup_all_memories("backup.json")

# Restore from backup
restored_count = await memory.restore_memories("backup.json")

# Cleanup old memories
deleted_count = await memory.cleanup_old_memories(max_age_days=365)
```

### Memory Management

```python
# Update memory importance
await memory.update_memory_importance(memory_id, 0.9)

# Delete specific memory
await memory.delete_memory(memory_id)

# Get specific memory
memory_data = await memory.get_memory_by_id(memory_id)
```

## WebSocket Events

When using `WebSocketMemoryManager`, the following events are broadcast:

- `memory_store` - Memory storage operations
- `memory_retrieve` - Memory search operations
- `memory_consolidate` - Short-term to long-term consolidation
- `memory_stats` - Memory statistics updates
- `memory_timeline` - Memory timeline data
- `conversation_summary` - Conversation summaries

## Testing

### Run Unit Tests
```bash
pytest tests/unit/test_memory_system.py -v
```

### Run Demo
```bash
python scripts/memory_demo.py
```

## Performance Considerations

- **Embedding Model**: Choose appropriate model for your hardware
  - `all-MiniLM-L6-v2` - Fast, good quality (default)
  - `all-mpnet-base-v2` - Higher quality, slower
  - Custom models for specific domains

- **Vector Database**: 
  - ChromaDB for production use
  - SQLite for simpler deployments
  - In-memory for testing only

- **Memory Limits**:
  - Set appropriate `max_memories` for your use case
  - Configure `time_decay_days` for relevance management
  - Use `cleanup_old_memories()` periodically

## Migration from Coda Lite

Key improvements over the original implementation:

✅ **Type-safe models** with Pydantic  
✅ **Modern async/await** architecture  
✅ **Comprehensive error handling**  
✅ **WebSocket integration** for real-time events  
✅ **Configurable vector databases**  
✅ **Advanced memory analytics**  
✅ **Backup and restore** functionality  
✅ **Memory importance scoring**  
✅ **Time-based relevance decay**  
✅ **Comprehensive test coverage**  

## Next Steps

- [ ] Advanced topic modeling integration
- [ ] Memory deduplication algorithms
- [ ] Cross-session memory sharing
- [ ] Memory compression techniques
- [ ] Advanced analytics dashboard
- [ ] Memory export formats (JSON, CSV, etc.)

## Dependencies

- `sentence-transformers` - Text embeddings
- `chromadb` - Vector database (optional)
- `numpy` - Numerical operations
- `pydantic` - Data validation
- `sqlite3` - Alternative vector storage (built-in)
