#!/usr/bin/env python3
"""
Memory system demonstration script.

This script demonstrates the memory system functionality by:
1. Creating a memory manager with both short-term and long-term memory
2. Simulating conversations and fact storage
3. Demonstrating memory retrieval and consolidation
4. Showing WebSocket integration capabilities
"""

import asyncio
import logging
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coda.components.memory.models import (
    MemoryManagerConfig,
    ShortTermMemoryConfig,
    LongTermMemoryConfig,
    MemoryEncoderConfig,
    MemoryQuery,
    MemoryType,
)
from coda.components.memory.manager import MemoryManager
from coda.components.memory.websocket_integration import WebSocketMemoryManager
from coda.interfaces.websocket.server import CodaWebSocketServer
from coda.interfaces.websocket.integration import CodaWebSocketIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memory_demo")


async def demonstrate_basic_memory():
    """Demonstrate basic memory functionality."""
    logger.info("🧠 Starting basic memory demonstration...")
    
    # Create temporary directory for this demo
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Configure memory system
        config = MemoryManagerConfig(
            short_term=ShortTermMemoryConfig(max_turns=10, max_tokens=500),
            long_term=LongTermMemoryConfig(
                storage_path=temp_dir,
                vector_db_type="in_memory",  # Use in-memory for demo
                max_memories=50,
                embedding_model="all-MiniLM-L6-v2"
            ),
            encoder=MemoryEncoderConfig(chunk_size=150, chunk_overlap=30),
            auto_persist=True,
            persist_interval=3
        )
        
        # Create memory manager
        memory = MemoryManager(config)
        
        # Simulate a conversation
        logger.info("💬 Simulating conversation...")
        memory.add_turn("system", "You are Coda, a helpful AI assistant.")
        memory.add_turn("user", "Hello! I'm working on a Python project.")
        memory.add_turn("assistant", "Hello! I'd be happy to help with your Python project. What are you working on?")
        memory.add_turn("user", "I'm building a voice assistant using machine learning.")
        memory.add_turn("assistant", "That sounds exciting! Voice assistants typically involve speech recognition, natural language processing, and text-to-speech. What specific part are you focusing on?")
        memory.add_turn("user", "I'm particularly interested in the memory system - how to make the assistant remember things.")
        memory.add_turn("assistant", "Memory systems are crucial for creating intelligent assistants. You'll want both short-term memory for conversation context and long-term memory for persistent knowledge. Vector embeddings work well for semantic search.")
        
        # Get conversation context
        context = memory.get_context(max_tokens=300)
        logger.info(f"📝 Current context has {len(context.messages)} messages ({context.total_tokens} tokens)")
        
        # Store some facts
        logger.info("📚 Storing facts in long-term memory...")
        await memory.store_fact("Python is a versatile programming language popular for AI development", importance=0.8)
        await memory.store_fact("Vector embeddings can represent text as numerical vectors for similarity search", importance=0.9)
        await memory.store_fact("The user is building a voice assistant project", importance=0.7)
        await memory.store_fact("Machine learning is used in speech recognition and natural language processing", importance=0.8)
        
        # Search memories
        logger.info("🔍 Searching memories...")
        search_queries = [
            "Python programming",
            "voice assistant",
            "machine learning",
            "memory system"
        ]
        
        for query in search_queries:
            results = await memory.search_memories(query, limit=3)
            logger.info(f"Query '{query}': Found {len(results)} relevant memories")
            for i, result in enumerate(results):
                logger.info(f"  {i+1}. Score: {result.final_score:.3f} - {result.memory.content[:60]}...")
        
        # Get enhanced context with long-term memories
        logger.info("🔗 Getting enhanced context...")
        enhanced_context = await memory.get_enhanced_context(
            "Tell me more about building voice assistants with Python",
            max_tokens=400,
            max_memories=2
        )
        logger.info(f"Enhanced context: {enhanced_context.short_term_turns} short-term + {enhanced_context.long_term_memories} long-term")
        
        # Force consolidation
        logger.info("💾 Consolidating short-term memory...")
        memories_created = await memory.consolidate_short_term()
        logger.info(f"Created {memories_created} new memories from conversation")
        
        # Get memory statistics
        stats = await memory.get_memory_stats()
        logger.info("📊 Memory Statistics:")
        logger.info(f"  Short-term: {stats['short_term']['total_turns']} turns")
        logger.info(f"  Long-term: {stats['long_term']['total_memories']} memories")
        logger.info(f"  Average importance: {stats['long_term']['average_importance']:.2f}")
        
        logger.info("✅ Basic memory demonstration completed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


async def demonstrate_websocket_integration():
    """Demonstrate WebSocket integration with memory system."""
    logger.info("🌐 Starting WebSocket memory demonstration...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Set up WebSocket server
        server = CodaWebSocketServer(host="localhost", port=8766)
        integration = CodaWebSocketIntegration(server)
        
        # Configure WebSocket memory manager
        config = MemoryManagerConfig(
            short_term=ShortTermMemoryConfig(max_turns=8),
            long_term=LongTermMemoryConfig(
                storage_path=temp_dir,
                vector_db_type="in_memory",
                max_memories=20
            ),
            auto_persist=True,
            persist_interval=2
        )
        
        # Create WebSocket-enabled memory manager
        memory = WebSocketMemoryManager(config)
        await memory.set_websocket_integration(integration)
        
        # Start WebSocket server
        await server.start()
        logger.info(f"🌐 WebSocket server running at ws://{server.host}:{server.port}")
        logger.info("💡 Connect with: wscat -c ws://localhost:8766")
        
        # Wait a moment for potential clients
        logger.info("⏳ Waiting 3 seconds for WebSocket clients...")
        await asyncio.sleep(3)
        
        # Simulate memory operations with WebSocket events
        logger.info("🎬 Simulating memory operations with WebSocket events...")
        
        # Add conversation turns
        memory.add_turn("user", "I need help with data science")
        memory.add_turn("assistant", "I'd be happy to help! What specific area of data science are you interested in?")
        memory.add_turn("user", "I'm learning about neural networks and deep learning")
        memory.add_turn("assistant", "Neural networks are fascinating! They're inspired by biological neurons and can learn complex patterns from data.")
        
        # Store facts (will broadcast events)
        await memory.store_fact("Neural networks are computational models inspired by biological neural networks", importance=0.9)
        await memory.store_fact("Deep learning uses multiple layers of neural networks", importance=0.8)
        await memory.store_fact("The user is learning about data science and neural networks", importance=0.7)
        
        # Search memories (will broadcast events)
        await memory.search_memories("neural networks")
        await memory.search_memories("data science")
        
        # Get enhanced context (will broadcast events)
        await memory.get_enhanced_context("Explain backpropagation in neural networks")
        
        # Force consolidation (will broadcast events)
        await memory.consolidate_short_term()
        
        # Broadcast memory statistics
        await memory.broadcast_memory_stats()
        
        # Broadcast conversation summary
        await memory.broadcast_conversation_summary()
        
        # Show final server stats
        server_stats = server.get_stats()
        logger.info(f"📊 WebSocket server stats: {server_stats}")
        
        logger.info("✅ WebSocket memory demonstration completed!")
        logger.info("⏳ Server will stop in 3 seconds...")
        await asyncio.sleep(3)
        
        await server.stop()
        
    except Exception as e:
        logger.error(f"❌ Error in WebSocket demonstration: {e}")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


async def demonstrate_advanced_features():
    """Demonstrate advanced memory features."""
    logger.info("🚀 Starting advanced memory features demonstration...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = MemoryManagerConfig(
            long_term=LongTermMemoryConfig(
                storage_path=temp_dir,
                vector_db_type="in_memory",
                max_memories=15,
                time_decay_days=7.0  # Faster decay for demo
            )
        )
        
        memory = MemoryManager(config)
        
        # Store memories with different importance levels
        logger.info("📚 Storing memories with different importance levels...")
        memories = [
            ("Critical system information", 1.0),
            ("Important user preference", 0.8),
            ("Useful fact", 0.6),
            ("Minor detail", 0.3),
            ("Trivial information", 0.1),
        ]
        
        memory_ids = []
        for content, importance in memories:
            memory_id = await memory.store_fact(content, importance=importance)
            memory_ids.append(memory_id)
            logger.info(f"  Stored: {content} (importance: {importance})")
        
        # Test memory retrieval with different queries
        logger.info("🔍 Testing memory retrieval...")
        test_queries = [
            ("system", 0.0),
            ("important", 0.5),
            ("information", 0.3),
        ]
        
        for query, min_relevance in test_queries:
            results = await memory.search_memories(query, limit=5, min_relevance=min_relevance)
            logger.info(f"Query '{query}' (min_relevance={min_relevance}): {len(results)} results")
            for result in results:
                logger.info(f"  Score: {result.final_score:.3f} - {result.memory.content}")
        
        # Test memory updates
        logger.info("✏️ Testing memory updates...")
        if memory_ids:
            test_id = memory_ids[0]
            success = await memory.update_memory_importance(test_id, 0.95)
            logger.info(f"Updated memory importance: {success}")
            
            updated_memory = await memory.get_memory_by_id(test_id)
            if updated_memory:
                logger.info(f"New importance: {updated_memory['metadata']['importance']}")
        
        # Test backup and restore
        logger.info("💾 Testing backup functionality...")
        backup_path = Path(temp_dir) / "memory_backup.json"
        backup_success = await memory.backup_all_memories(str(backup_path))
        logger.info(f"Backup created: {backup_success}")
        
        if backup_path.exists():
            logger.info(f"Backup file size: {backup_path.stat().st_size} bytes")
        
        # Test memory cleanup
        logger.info("🧹 Testing memory cleanup...")
        deleted_count = await memory.cleanup_old_memories(max_age_days=0)  # Delete all for demo
        logger.info(f"Deleted {deleted_count} old memories")
        
        # Final statistics
        final_stats = await memory.get_memory_stats()
        logger.info("📊 Final Statistics:")
        logger.info(f"  Total memories: {final_stats['long_term']['total_memories']}")
        logger.info(f"  Memory types: {final_stats['long_term']['memory_types']}")
        
        logger.info("✅ Advanced features demonstration completed!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Coda Memory System Demonstration")
    
    try:
        # Run demonstrations
        await demonstrate_basic_memory()
        await asyncio.sleep(1)
        
        await demonstrate_websocket_integration()
        await asyncio.sleep(1)
        
        await demonstrate_advanced_features()
        
        logger.info("🎉 All memory system demonstrations completed successfully!")
        
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
