"""
Long-term memory implementation for Coda.

This module provides a LongTermMemory class for persistent memory storage
using vector embeddings and semantic search capabilities.
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Vector database imports
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from .interfaces import LongTermMemoryInterface
from .models import (
    Memory,
    MemoryMetadata,
    MemoryQuery,
    MemoryResult,
    MemoryStats,
    MemoryType,
    LongTermMemoryConfig,
)

logger = logging.getLogger("coda.memory.long_term")


class LongTermMemory(LongTermMemoryInterface):
    """
    Manages long-term memory using vector embeddings for semantic search.
    
    Features:
    - Vector-based semantic search with ChromaDB or SQLite
    - Time-based relevance decay
    - Memory importance scoring
    - Metadata filtering and search
    - Automatic memory pruning
    - Backup and restore functionality
    - Async operations for better performance
    """
    
    def __init__(self, config: Optional[LongTermMemoryConfig] = None):
        """
        Initialize the long-term memory system.
        
        Args:
            config: Configuration for long-term memory
        """
        self.config = config or LongTermMemoryConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model {self.config.embedding_model} on {self.config.device}")
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model, 
            device=self.config.device
        )
        
        # Initialize vector database
        self._init_vector_db()
        
        # Initialize metadata storage
        self.metadata_path = self.storage_path / "metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"LongTermMemory initialized with {len(self.metadata.get('memories', {}))} memories")
    
    def _init_vector_db(self) -> None:
        """Initialize the vector database based on configuration."""
        if self.config.vector_db_type == "chroma" and CHROMA_AVAILABLE:
            logger.info(f"Initializing ChromaDB at {self.storage_path}")
            self.vector_db = chromadb.PersistentClient(path=str(self.storage_path))
            self.collection = self.vector_db.get_or_create_collection(
                name="memories",
                metadata={"description": "Coda's long-term memories"}
            )
        elif self.config.vector_db_type == "sqlite" and SQLITE_AVAILABLE:
            logger.info(f"Initializing SQLite vector database at {self.storage_path}")
            db_path = self.storage_path / "memories.db"
            self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._init_sqlite_db()
        else:
            # Fallback to in-memory storage
            logger.warning("Using in-memory vector storage (not persistent)")
            self.vectors: Dict[str, np.ndarray] = {}
            self.contents: Dict[str, str] = {}
            self.vector_metadata: Dict[str, Dict[str, Any]] = {}
    
    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp TEXT NOT NULL,
                importance REAL NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
        """)
        self.conn.commit()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        
        return {
            "version": "2.0.0",
            "created_at": datetime.now().isoformat(),
            "memories": {},
            "memory_count": 0,
            "last_cleanup": None,
        }
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def store_memory(self, content: str, memory_type: MemoryType, 
                          importance: float, metadata: Dict[str, Any]) -> str:
        """
        Store a memory with vector embedding.
        
        Args:
            content: The memory content
            memory_type: Type of memory
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            The generated memory ID
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create memory metadata
        memory_metadata = MemoryMetadata(
            source_type=memory_type,
            timestamp=timestamp,
            importance=max(0.0, min(1.0, importance)),
            **metadata
        )
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            metadata=memory_metadata,
            embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            created_at=timestamp,
            accessed_at=timestamp,
            access_count=0
        )
        
        # Store in vector database
        await self._store_in_vector_db(memory)
        
        # Update metadata
        self.metadata["memories"][memory_id] = {
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "timestamp": timestamp.isoformat(),
            "importance": importance,
            "memory_type": memory_type.value,
            "metadata": metadata
        }
        self.metadata["memory_count"] = len(self.metadata["memories"])
        self._save_metadata()
        
        logger.info(f"Stored memory {memory_id} ({len(content)} chars, importance={importance:.2f})")
        
        # Check if we need to prune memories
        if self.metadata["memory_count"] > self.config.max_memories:
            await self._prune_memories()
        
        return memory_id
    
    async def _generate_embedding(self, content: str) -> np.ndarray:
        """Generate embedding for content."""
        # Run embedding generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            self.embedding_model.encode, 
            content
        )
        return embedding
    
    async def _store_in_vector_db(self, memory: Memory) -> None:
        """Store memory in the vector database."""
        if self.config.vector_db_type == "chroma" and hasattr(self, 'collection'):
            # ChromaDB storage
            self.collection.add(
                ids=[memory.id],
                embeddings=[memory.embedding],
                metadatas=[memory.metadata.model_dump()],
                documents=[memory.content]
            )
        elif self.config.vector_db_type == "sqlite" and hasattr(self, 'conn'):
            # SQLite storage
            cursor = self.conn.cursor()
            embedding_bytes = np.array(memory.embedding).tobytes()
            cursor.execute(
                "INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.content,
                    embedding_bytes,
                    memory.created_at.isoformat(),
                    memory.metadata.importance,
                    json.dumps(memory.metadata.model_dump())
                )
            )
            self.conn.commit()
        else:
            # In-memory storage
            self.vectors[memory.id] = np.array(memory.embedding)
            self.contents[memory.id] = memory.content
            self.vector_metadata[memory.id] = memory.metadata.model_dump()
    
    async def retrieve_memories(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        Retrieve memories based on query parameters.
        
        Args:
            query: Memory query parameters
            
        Returns:
            List of memory results with relevance scores
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query.query)
        
        # Retrieve from vector database
        if self.config.vector_db_type == "chroma" and hasattr(self, 'collection'):
            results = await self._query_chroma(query_embedding, query)
        elif self.config.vector_db_type == "sqlite" and hasattr(self, 'conn'):
            results = await self._query_sqlite(query_embedding, query)
        else:
            results = await self._query_memory(query_embedding, query)
        
        # Apply time decay and filtering
        filtered_results = []
        for result in results:
            # Calculate time decay factor
            time_decay = self._calculate_time_decay(result.memory.created_at)
            
            # Calculate final score
            final_score = result.relevance_score * time_decay * result.memory.metadata.importance
            
            # Apply minimum relevance filter
            if final_score >= query.min_relevance:
                result.time_decay_factor = time_decay
                result.final_score = final_score
                filtered_results.append(result)
        
        # Sort by final score and limit results
        filtered_results.sort(key=lambda r: r.final_score, reverse=True)
        return filtered_results[:query.limit]

    async def _query_chroma(self, query_embedding: np.ndarray, query: MemoryQuery) -> List[MemoryResult]:
        """Query ChromaDB for similar memories."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(query.limit * 2, 50),  # Get more results for filtering
                include=["documents", "metadatas", "distances"]
            )

            memory_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                similarity = 1.0 - distance

                # Create memory object
                memory_metadata = MemoryMetadata(**metadata)
                memory = Memory(
                    id=results["ids"][0][i],
                    content=doc,
                    metadata=memory_metadata,
                    created_at=memory_metadata.timestamp,
                    accessed_at=datetime.now(),
                    access_count=0
                )

                memory_results.append(MemoryResult(
                    memory=memory,
                    relevance_score=similarity,
                    time_decay_factor=1.0,  # Will be calculated later
                    final_score=similarity
                ))

            return memory_results

        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    async def _query_sqlite(self, query_embedding: np.ndarray, query: MemoryQuery) -> List[MemoryResult]:
        """Query SQLite database for similar memories."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, content, embedding, timestamp, importance, metadata FROM memories")
            rows = cursor.fetchall()

            memory_results = []
            for row in rows:
                memory_id, content, embedding_bytes, timestamp_str, importance, metadata_json = row

                # Reconstruct embedding
                stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, stored_embedding)

                # Parse metadata
                metadata_dict = json.loads(metadata_json)
                memory_metadata = MemoryMetadata(**metadata_dict)

                # Create memory object
                memory = Memory(
                    id=memory_id,
                    content=content,
                    metadata=memory_metadata,
                    created_at=datetime.fromisoformat(timestamp_str),
                    accessed_at=datetime.now(),
                    access_count=0
                )

                memory_results.append(MemoryResult(
                    memory=memory,
                    relevance_score=similarity,
                    time_decay_factor=1.0,  # Will be calculated later
                    final_score=similarity
                ))

            return memory_results

        except Exception as e:
            logger.error(f"Error querying SQLite: {e}")
            return []

    async def _query_memory(self, query_embedding: np.ndarray, query: MemoryQuery) -> List[MemoryResult]:
        """Query in-memory storage for similar memories."""
        memory_results = []

        for memory_id, stored_embedding in self.vectors.items():
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)

            # Get content and metadata
            content = self.contents[memory_id]
            metadata_dict = self.vector_metadata[memory_id]
            memory_metadata = MemoryMetadata(**metadata_dict)

            # Create memory object
            memory = Memory(
                id=memory_id,
                content=content,
                metadata=memory_metadata,
                created_at=memory_metadata.timestamp,
                accessed_at=datetime.now(),
                access_count=0
            )

            memory_results.append(MemoryResult(
                memory=memory,
                relevance_score=similarity,
                time_decay_factor=1.0,  # Will be calculated later
                final_score=similarity
            ))

        return memory_results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """Calculate time decay factor for a memory."""
        now = datetime.now()
        age_days = (now - timestamp).total_seconds() / (24 * 3600)

        # Exponential decay: score = 2^(-age_days / decay_constant)
        decay_factor = 2 ** (-age_days / self.config.time_decay_days)
        return max(0.01, decay_factor)  # Minimum decay factor of 0.01

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        if memory_id not in self.metadata.get("memories", {}):
            return None

        try:
            if self.config.vector_db_type == "chroma" and hasattr(self, 'collection'):
                results = self.collection.get(ids=[memory_id], include=["documents", "metadatas"])
                if results["ids"]:
                    doc = results["documents"][0]
                    metadata_dict = results["metadatas"][0]
                    memory_metadata = MemoryMetadata(**metadata_dict)

                    memory = Memory(
                        id=memory_id,
                        content=doc,
                        metadata=memory_metadata,
                        created_at=memory_metadata.timestamp,
                        accessed_at=datetime.now(),
                        access_count=0
                    )
                    memory.update_access()
                    return memory

            elif self.config.vector_db_type == "sqlite" and hasattr(self, 'conn'):
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT content, timestamp, importance, metadata FROM memories WHERE id = ?",
                    (memory_id,)
                )
                row = cursor.fetchone()
                if row:
                    content, timestamp_str, importance, metadata_json = row
                    metadata_dict = json.loads(metadata_json)
                    memory_metadata = MemoryMetadata(**metadata_dict)

                    memory = Memory(
                        id=memory_id,
                        content=content,
                        metadata=memory_metadata,
                        created_at=datetime.fromisoformat(timestamp_str),
                        accessed_at=datetime.now(),
                        access_count=0
                    )
                    memory.update_access()
                    return memory

            else:
                # In-memory storage
                if memory_id in self.contents:
                    content = self.contents[memory_id]
                    metadata_dict = self.vector_metadata[memory_id]
                    memory_metadata = MemoryMetadata(**metadata_dict)

                    memory = Memory(
                        id=memory_id,
                        content=content,
                        metadata=memory_metadata,
                        created_at=memory_metadata.timestamp,
                        accessed_at=datetime.now(),
                        access_count=0
                    )
                    memory.update_access()
                    return memory

        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")

        return None

    async def update_memory(self, memory_id: str, content: Optional[str] = None,
                           importance: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory."""
        memory = await self.get_memory(memory_id)
        if not memory:
            return False

        try:
            # Update fields if provided
            if content is not None:
                memory.content = content
                # Regenerate embedding for new content
                memory.embedding = (await self._generate_embedding(content)).tolist()

            if importance is not None:
                memory.metadata.importance = max(0.0, min(1.0, importance))

            if metadata is not None:
                memory.metadata.additional.update(metadata)

            # Store updated memory
            await self._store_in_vector_db(memory)

            # Update metadata
            self.metadata["memories"][memory_id].update({
                "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                "importance": memory.metadata.importance,
                "metadata": memory.metadata.additional
            })
            self._save_metadata()

            logger.info(f"Updated memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if memory_id not in self.metadata.get("memories", {}):
            return False

        try:
            # Delete from vector database
            if self.config.vector_db_type == "chroma" and hasattr(self, 'collection'):
                self.collection.delete(ids=[memory_id])
            elif self.config.vector_db_type == "sqlite" and hasattr(self, 'conn'):
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                self.conn.commit()
            else:
                # In-memory storage
                self.vectors.pop(memory_id, None)
                self.contents.pop(memory_id, None)
                self.vector_metadata.pop(memory_id, None)

            # Remove from metadata
            self.metadata["memories"].pop(memory_id, None)
            self.metadata["memory_count"] = len(self.metadata["memories"])
            self._save_metadata()

            logger.info(f"Deleted memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False

    async def get_stats(self) -> MemoryStats:
        """Get statistics about long-term memory."""
        memories = self.metadata.get("memories", {})

        if not memories:
            return MemoryStats()

        # Calculate statistics
        memory_types = {}
        total_importance = 0.0
        total_content_length = 0
        oldest_timestamp = None
        newest_timestamp = None
        most_accessed_id = None
        max_access_count = 0
        total_access_count = 0

        for memory_id, memory_info in memories.items():
            # Memory type distribution
            memory_type = memory_info.get("memory_type", "unknown")
            memory_types[memory_type] = memory_types.get(memory_type, 0) + 1

            # Importance
            importance = memory_info.get("importance", 0.5)
            total_importance += importance

            # Content length (estimate from preview)
            content_preview = memory_info.get("content_preview", "")
            if content_preview.endswith("..."):
                estimated_length = len(content_preview) * 2  # Rough estimate
            else:
                estimated_length = len(content_preview)
            total_content_length += estimated_length

            # Timestamps
            timestamp_str = memory_info.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if oldest_timestamp is None or timestamp < oldest_timestamp:
                    oldest_timestamp = timestamp
                if newest_timestamp is None or timestamp > newest_timestamp:
                    newest_timestamp = timestamp

        return MemoryStats(
            total_memories=len(memories),
            memory_types={MemoryType(k): v for k, v in memory_types.items() if k in [t.value for t in MemoryType]},
            average_importance=total_importance / len(memories),
            oldest_memory=oldest_timestamp,
            newest_memory=newest_timestamp,
            total_content_length=total_content_length,
            average_content_length=total_content_length / len(memories),
            most_accessed_memory_id=most_accessed_id,
            total_access_count=total_access_count,
        )

    async def cleanup_old_memories(self, max_age_days: int = 365) -> int:
        """Clean up old memories and return count of deleted memories."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0

        memories_to_delete = []
        for memory_id, memory_info in self.metadata.get("memories", {}).items():
            timestamp_str = memory_info.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp < cutoff_date:
                    memories_to_delete.append(memory_id)

        # Delete old memories
        for memory_id in memories_to_delete:
            if await self.delete_memory(memory_id):
                deleted_count += 1

        # Update cleanup timestamp
        self.metadata["last_cleanup"] = datetime.now().isoformat()
        self._save_metadata()

        logger.info(f"Cleaned up {deleted_count} old memories (older than {max_age_days} days)")
        return deleted_count

    async def _prune_memories(self) -> None:
        """Prune memories when exceeding max_memories limit."""
        memories = self.metadata.get("memories", {})
        if len(memories) <= self.config.max_memories:
            return

        # Calculate scores for all memories (importance * recency)
        memory_scores = []
        for memory_id, memory_info in memories.items():
            importance = memory_info.get("importance", 0.5)
            timestamp_str = memory_info.get("timestamp")

            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now() - timestamp).total_seconds() / (24 * 3600)
                recency_score = 2 ** (-age_days / self.config.time_decay_days)
                final_score = importance * recency_score
            else:
                final_score = importance

            memory_scores.append((memory_id, final_score))

        # Sort by score and keep only the top memories
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        memories_to_keep = memory_scores[:self.config.max_memories]
        memories_to_delete = memory_scores[self.config.max_memories:]

        # Delete excess memories
        deleted_count = 0
        for memory_id, _ in memories_to_delete:
            if await self.delete_memory(memory_id):
                deleted_count += 1

        logger.info(f"Pruned {deleted_count} memories to stay within limit of {self.config.max_memories}")

    async def backup_memories(self, backup_path: str) -> bool:
        """Backup memories to a file."""
        try:
            backup_data = {
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "config": self.config.model_dump(),
                "metadata": self.metadata,
                "memories": []
            }

            # Export all memories
            for memory_id in self.metadata.get("memories", {}):
                memory = await self.get_memory(memory_id)
                if memory:
                    backup_data["memories"].append({
                        "id": memory.id,
                        "content": memory.content,
                        "metadata": memory.metadata.model_dump(),
                        "embedding": memory.embedding,
                        "created_at": memory.created_at.isoformat(),
                        "accessed_at": memory.accessed_at.isoformat(),
                        "access_count": memory.access_count,
                    })

            # Write backup file
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Backed up {len(backup_data['memories'])} memories to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    async def restore_memories(self, backup_path: str) -> int:
        """Restore memories from a backup file."""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            restored_count = 0
            for memory_data in backup_data.get("memories", []):
                try:
                    # Reconstruct memory object
                    memory_metadata = MemoryMetadata(**memory_data["metadata"])
                    memory = Memory(
                        id=memory_data["id"],
                        content=memory_data["content"],
                        metadata=memory_metadata,
                        embedding=memory_data["embedding"],
                        created_at=datetime.fromisoformat(memory_data["created_at"]),
                        accessed_at=datetime.fromisoformat(memory_data["accessed_at"]),
                        access_count=memory_data["access_count"],
                    )

                    # Store memory
                    await self._store_in_vector_db(memory)

                    # Update metadata
                    self.metadata["memories"][memory.id] = {
                        "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                        "timestamp": memory.created_at.isoformat(),
                        "importance": memory.metadata.importance,
                        "memory_type": memory.metadata.source_type.value,
                        "metadata": memory.metadata.additional
                    }

                    restored_count += 1

                except Exception as e:
                    logger.warning(f"Failed to restore memory {memory_data.get('id', 'unknown')}: {e}")
                    continue

            # Update metadata
            self.metadata["memory_count"] = len(self.metadata["memories"])
            self._save_metadata()

            logger.info(f"Restored {restored_count} memories from {backup_path}")
            return restored_count

        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return 0
