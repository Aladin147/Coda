#!/usr/bin/env python3
"""
Memory Integration Example

This example demonstrates how to use Coda's memory system for intelligent
conversation context and learning.

Features demonstrated:
- Memory storage and retrieval
- Semantic search capabilities
- Conversation context enhancement
- Memory-based learning
- Different memory types

Usage:
    python examples/memory_integration.py [--interactive]
"""

import asyncio
import logging
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.memory import (
        MemoryManager, MemoryConfig, Memory, MemoryType
    )
except ImportError as e:
    logger.error(f"Failed to import memory components: {e}")
    logger.error("Please ensure dependencies are installed: pip install -e .")
    exit(1)


class MemoryIntegrationDemo:
    """Demonstrates memory system integration and capabilities."""
    
    def __init__(self):
        self.memory_manager: MemoryManager = None
        self.conversation_context: List[str] = []
    
    async def initialize(self) -> None:
        """Initialize the memory system."""
        logger.info("🧠 Initializing memory system...")
        
        try:
            config = MemoryConfig(
                enable_auto_learning=True,
                enable_auto_consolidation=True,
                max_memories=10000,
                default_retrieval_limit=10,
                min_relevance_threshold=0.3
            )
            
            self.memory_manager = MemoryManager(config)
            await self.memory_manager.initialize()
            
            logger.info("✅ Memory system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory system: {e}")
            raise
    
    async def demonstrate_memory_types(self) -> None:
        """Demonstrate different types of memories."""
        logger.info("\n📚 === Memory Types Demo ===")
        
        # Sample memories of different types
        sample_memories = [
            {
                "content": "User is a software engineer specializing in machine learning",
                "type": MemoryType.SEMANTIC,
                "importance": 0.9,
                "metadata": {"category": "profession", "domain": "technology"}
            },
            {
                "content": "User prefers morning meetings and dislikes late afternoon calls",
                "type": MemoryType.PREFERENCE,
                "importance": 0.8,
                "metadata": {"category": "scheduling", "time_preference": "morning"}
            },
            {
                "content": "User asked about optimizing neural network training on 2024-03-15",
                "type": MemoryType.EPISODIC,
                "importance": 0.6,
                "metadata": {"topic": "neural_networks", "date": "2024-03-15"}
            },
            {
                "content": "To debug Python code: use print statements, debugger, or logging",
                "type": MemoryType.PROCEDURAL,
                "importance": 0.7,
                "metadata": {"skill": "debugging", "language": "python"}
            },
            {
                "content": "User expressed excitement about new AI developments",
                "type": MemoryType.EMOTIONAL,
                "importance": 0.5,
                "metadata": {"emotion": "excitement", "topic": "AI"}
            }
        ]
        
        # Store memories
        logger.info("Storing sample memories...")
        stored_ids = []
        for memory_data in sample_memories:
            memory_id = await self.memory_manager.store_memory(
                content=memory_data["content"],
                memory_type=memory_data["type"],
                importance_score=memory_data["importance"],
                metadata=memory_data["metadata"]
            )
            stored_ids.append(memory_id)
            logger.info(f"  ✓ {memory_data['type'].value}: {memory_data['content'][:50]}...")
        
        return stored_ids
    
    async def demonstrate_semantic_search(self) -> None:
        """Demonstrate semantic search capabilities."""
        logger.info("\n🔍 === Semantic Search Demo ===")
        
        # Test queries with different contexts
        test_queries = [
            "schedule a meeting",
            "machine learning help",
            "Python programming",
            "user emotions",
            "debugging code",
            "AI technology"
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            
            memories = await self.memory_manager.retrieve_memories(
                query=query,
                limit=3,
                min_relevance=0.3
            )
            
            if memories:
                logger.info(f"Found {len(memories)} relevant memories:")
                for i, memory in enumerate(memories, 1):
                    logger.info(f"  {i}. [{memory.memory_type.value}] {memory.content[:60]}...")
                    logger.info(f"     Relevance: {memory.relevance_score:.3f}, Importance: {memory.importance_score:.3f}")
            else:
                logger.info("  No relevant memories found")
    
    async def demonstrate_conversation_context(self) -> None:
        """Demonstrate conversation context enhancement."""
        logger.info("\n💬 === Conversation Context Demo ===")
        
        # Simulate a conversation
        conversation_turns = [
            "Hi, I'm working on a machine learning project",
            "I need help with neural network optimization",
            "Can you schedule a meeting to discuss this?",
            "I'm feeling excited about this new AI project",
            "How do I debug Python code effectively?"
        ]
        
        for turn_num, user_input in enumerate(conversation_turns, 1):
            logger.info(f"\n--- Turn {turn_num} ---")
            logger.info(f"User: {user_input}")
            
            # Retrieve relevant context
            context_memories = await self.memory_manager.retrieve_memories(
                query=user_input,
                limit=3,
                min_relevance=0.4
            )
            
            # Build context
            context = []
            if context_memories:
                logger.info("Relevant context:")
                for memory in context_memories:
                    context.append(memory.content)
                    logger.info(f"  - {memory.content[:50]}... (relevance: {memory.relevance_score:.3f})")
            
            # Generate context-aware response
            response = await self._generate_contextual_response(user_input, context)
            logger.info(f"Assistant: {response}")
            
            # Store the interaction
            await self.memory_manager.store_memory(
                content=f"User said: {user_input}",
                memory_type=MemoryType.EPISODIC,
                importance_score=0.5,
                metadata={"turn": turn_num, "type": "user_input"}
            )
            
            await self.memory_manager.store_memory(
                content=f"Assistant responded: {response}",
                memory_type=MemoryType.EPISODIC,
                importance_score=0.4,
                metadata={"turn": turn_num, "type": "assistant_response"}
            )
    
    async def _generate_contextual_response(self, user_input: str, context: List[str]) -> str:
        """Generate a response using memory context."""
        # This is a simplified response generator
        # In a real implementation, this would use the LLM with context
        
        if "schedule" in user_input.lower() and any("morning" in c for c in context):
            return "I'd be happy to help schedule that! I remember you prefer morning meetings. How about 10 AM tomorrow?"
        elif "machine learning" in user_input.lower() and any("engineer" in c for c in context):
            return "Given your background as a software engineer in ML, I can definitely help with that. What specific aspect would you like to focus on?"
        elif "debug" in user_input.lower() and any("Python" in c for c in context):
            return "For Python debugging, you can use print statements, the built-in debugger (pdb), or logging. Which approach would you prefer?"
        elif "excited" in user_input.lower():
            return "That's wonderful! Your enthusiasm for AI projects really comes through. What aspects are you most excited about?"
        else:
            return f"I understand you're asking about '{user_input}'. Let me help you with that based on what I know about your interests."
    
    async def demonstrate_memory_learning(self) -> None:
        """Demonstrate memory-based learning."""
        logger.info("\n🎓 === Memory Learning Demo ===")
        
        # Simulate learning from user feedback
        learning_scenarios = [
            {
                "interaction": "User asked about Python debugging",
                "response": "Use print statements for debugging",
                "feedback": {"helpful": True, "too_simple": True},
                "learning": "User prefers more detailed technical explanations"
            },
            {
                "interaction": "User asked about meeting scheduling",
                "response": "I can help schedule meetings",
                "feedback": {"helpful": True, "remember_preference": True},
                "learning": "Always mention morning preference for meetings"
            },
            {
                "interaction": "User expressed excitement about AI",
                "response": "That's great!",
                "feedback": {"helpful": False, "too_generic": True},
                "learning": "User prefers specific, technical responses about AI topics"
            }
        ]
        
        for scenario in learning_scenarios:
            logger.info(f"\nLearning scenario: {scenario['interaction']}")
            logger.info(f"Original response: {scenario['response']}")
            logger.info(f"User feedback: {scenario['feedback']}")
            
            # Store learning
            await self.memory_manager.store_memory(
                content=scenario["learning"],
                memory_type=MemoryType.PREFERENCE,
                importance_score=0.8,
                metadata={
                    "source": "user_feedback",
                    "feedback": scenario["feedback"],
                    "learned_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✓ Learned: {scenario['learning']}")
    
    async def demonstrate_memory_analytics(self) -> None:
        """Demonstrate memory analytics and insights."""
        logger.info("\n📊 === Memory Analytics Demo ===")
        
        # Get memory statistics
        stats = await self.memory_manager.get_memory_stats()
        
        logger.info("Memory Statistics:")
        logger.info(f"  Total memories: {stats.total_memories}")
        logger.info(f"  Memory types: {dict(stats.memory_type_distribution)}")
        logger.info(f"  Average importance: {stats.avg_importance_score:.3f}")
        logger.info(f"  Most accessed: {stats.most_accessed_memory_id}")
        
        # Get recent memories
        recent_memories = await self.memory_manager.retrieve_recent_memories(
            hours=24,
            limit=5
        )
        
        logger.info(f"\nRecent memories ({len(recent_memories)}):")
        for memory in recent_memories:
            age_hours = (datetime.now().timestamp() - memory.timestamp) / 3600
            logger.info(f"  - {memory.content[:50]}... ({age_hours:.1f}h ago)")
    
    async def interactive_mode(self) -> None:
        """Run interactive memory exploration."""
        logger.info("\n🎮 === Interactive Memory Mode ===")
        logger.info("Commands:")
        logger.info("  search <query>  - Search memories")
        logger.info("  store <content> - Store new memory")
        logger.info("  stats          - Show statistics")
        logger.info("  recent         - Show recent memories")
        logger.info("  quit           - Exit")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ['quit', 'exit']:
                    break
                elif command.startswith('search '):
                    query = command[7:]
                    memories = await self.memory_manager.retrieve_memories(query, limit=5)
                    print(f"Found {len(memories)} memories:")
                    for i, memory in enumerate(memories, 1):
                        print(f"  {i}. {memory.content[:60]}... (relevance: {memory.relevance_score:.3f})")
                
                elif command.startswith('store '):
                    content = command[6:]
                    memory_id = await self.memory_manager.store_memory(
                        content=content,
                        memory_type=MemoryType.EPISODIC,
                        importance_score=0.5
                    )
                    print(f"✓ Stored memory: {memory_id}")
                
                elif command == 'stats':
                    stats = await self.memory_manager.get_memory_stats()
                    print(f"Total memories: {stats.total_memories}")
                    print(f"Average importance: {stats.avg_importance_score:.3f}")
                
                elif command == 'recent':
                    memories = await self.memory_manager.retrieve_recent_memories(hours=24, limit=5)
                    print(f"Recent memories ({len(memories)}):")
                    for memory in memories:
                        print(f"  - {memory.content[:50]}...")
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.memory_manager:
            await self.memory_manager.cleanup()


async def main():
    """Main function to run the memory integration demo."""
    parser = argparse.ArgumentParser(description="Memory Integration Example")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    demo = MemoryIntegrationDemo()
    
    try:
        await demo.initialize()
        
        if args.interactive:
            await demo.interactive_mode()
        else:
            # Run all demonstrations
            await demo.demonstrate_memory_types()
            await demo.demonstrate_semantic_search()
            await demo.demonstrate_conversation_context()
            await demo.demonstrate_memory_learning()
            await demo.demonstrate_memory_analytics()
        
        logger.info("\n🎉 Memory integration demo completed!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1
    finally:
        await demo.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
