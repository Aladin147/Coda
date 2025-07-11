#!/usr/bin/env python3
"""
Comprehensive Coda Demo Script

This script demonstrates the full capabilities of Coda's integrated systems:
- Voice Processing with Kyutai Moshi
- Memory System with intelligent retrieval
- Personality Engine with adaptive responses
- Tools System with function calling
- LLM Manager with multi-provider support
- WebSocket integration for real-time communication

Usage:
    python scripts/comprehensive_demo.py [--mode voice|text|websocket]
"""

import asyncio
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Coda components
try:
    from src.coda.components.voice import VoiceManager, VoiceConfig, VoiceMessage, VoiceProcessingMode
    from src.coda.components.memory import MemoryManager, MemoryConfig, MemoryType
    from src.coda.components.personality import PersonalityManager, PersonalityConfig
    from src.coda.components.tools import ToolsManager, ToolsConfig
    from src.coda.core.llm_manager import LLMManager, LLMConfig
    from src.coda.interfaces.websocket import WebSocketManager, WebSocketConfig
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    logger.error("Please ensure you're running from the project root and all dependencies are installed")
    exit(1)


class CodaDemo:
    """Comprehensive demo of Coda's integrated systems."""
    
    def __init__(self):
        self.voice_manager: Optional[VoiceManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.personality_manager: Optional[PersonalityManager] = None
        self.tools_manager: Optional[ToolsManager] = None
        self.llm_manager: Optional[LLMManager] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        
        self.conversation_id: Optional[str] = None
        self.demo_stats = {
            "start_time": time.time(),
            "interactions": 0,
            "voice_responses": 0,
            "memories_stored": 0,
            "tools_called": 0,
            "personality_adaptations": 0
        }
    
    async def initialize_all_systems(self) -> None:
        """Initialize all Coda systems with optimal configurations."""
        logger.info("🚀 Initializing Coda systems...")
        
        try:
            # Initialize Memory System
            logger.info("📚 Initializing Memory System...")
            memory_config = MemoryConfig(
                enable_auto_learning=True,
                enable_auto_consolidation=True,
                max_memories=10000
            )
            self.memory_manager = MemoryManager(memory_config)
            await self.memory_manager.initialize()
            logger.info("✅ Memory System initialized")
            
            # Initialize Personality Engine
            logger.info("🎭 Initializing Personality Engine...")
            personality_config = PersonalityConfig(
                enable_adaptation=True,
                enable_learning=True,
                adaptation_rate=0.1
            )
            self.personality_manager = PersonalityManager(personality_config)
            await self.personality_manager.initialize()
            logger.info("✅ Personality Engine initialized")
            
            # Initialize Tools System
            logger.info("🛠️ Initializing Tools System...")
            tools_config = ToolsConfig(
                enable_auto_discovery=True,
                enable_function_calling=True
            )
            self.tools_manager = ToolsManager(tools_config)
            await self.tools_manager.initialize()
            logger.info("✅ Tools System initialized")
            
            # Initialize LLM Manager
            logger.info("🧠 Initializing LLM Manager...")
            llm_config = LLMConfig(
                provider="ollama",
                model="gemma2:2b",
                temperature=0.8
            )
            self.llm_manager = LLMManager(llm_config)
            await self.llm_manager.initialize()
            logger.info("✅ LLM Manager initialized")
            
            # Initialize Voice System (if available)
            try:
                logger.info("🎙️ Initializing Voice System...")
                voice_config = VoiceConfig(
                    enable_memory_integration=True,
                    enable_personality_integration=True,
                    enable_tools_integration=True,
                    default_mode=VoiceProcessingMode.ADAPTIVE
                )
                self.voice_manager = VoiceManager(voice_config)
                await self.voice_manager.initialize()
                logger.info("✅ Voice System initialized")
            except Exception as e:
                logger.warning(f"⚠️ Voice System initialization failed: {e}")
                logger.info("Continuing with text-only mode...")
            
            # Initialize WebSocket System
            try:
                logger.info("🌐 Initializing WebSocket System...")
                websocket_config = WebSocketConfig(
                    host="localhost",
                    port=8765,
                    enable_voice_streaming=bool(self.voice_manager)
                )
                self.websocket_manager = WebSocketManager(websocket_config)
                await self.websocket_manager.initialize()
                logger.info("✅ WebSocket System initialized")
            except Exception as e:
                logger.warning(f"⚠️ WebSocket System initialization failed: {e}")
            
            logger.info("🎉 All systems initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    async def demonstrate_memory_system(self) -> None:
        """Demonstrate memory storage and retrieval capabilities."""
        logger.info("\n📚 === Memory System Demo ===")
        
        # Store various types of memories
        demo_memories = [
            {
                "content": "User prefers morning meetings and dislikes late afternoon calls",
                "type": MemoryType.PREFERENCE,
                "importance": 0.8
            },
            {
                "content": "User is working on a machine learning project using Python and TensorFlow",
                "type": MemoryType.SEMANTIC,
                "importance": 0.7
            },
            {
                "content": "User mentioned feeling excited about the new AI developments",
                "type": MemoryType.EMOTIONAL,
                "importance": 0.6
            },
            {
                "content": "User asked how to optimize neural network training",
                "type": MemoryType.EPISODIC,
                "importance": 0.5
            }
        ]
        
        logger.info("Storing demo memories...")
        for memory_data in demo_memories:
            memory_id = await self.memory_manager.store_memory(
                content=memory_data["content"],
                memory_type=memory_data["type"],
                importance_score=memory_data["importance"]
            )
            logger.info(f"  ✓ Stored: {memory_data['content'][:50]}... (ID: {memory_id[:8]})")
            self.demo_stats["memories_stored"] += 1
        
        # Demonstrate retrieval
        logger.info("\nTesting memory retrieval...")
        queries = [
            "schedule a meeting",
            "machine learning help",
            "user emotions",
            "Python programming"
        ]
        
        for query in queries:
            memories = await self.memory_manager.retrieve_memories(
                query=query,
                limit=3,
                min_relevance=0.3
            )
            logger.info(f"  Query: '{query}' -> {len(memories)} relevant memories")
            for memory in memories:
                logger.info(f"    - {memory.content[:60]}... (relevance: {memory.relevance_score:.2f})")
    
    async def demonstrate_personality_system(self) -> None:
        """Demonstrate personality adaptation and learning."""
        logger.info("\n🎭 === Personality System Demo ===")
        
        # Get current personality state
        personality_state = await self.personality_manager.get_personality_state()
        logger.info(f"Current personality traits:")
        for trait, value in personality_state.traits.items():
            logger.info(f"  - {trait}: {value:.2f}")
        
        # Simulate personality adaptation
        logger.info("\nSimulating personality adaptation...")
        adaptation_scenarios = [
            {"user_feedback": {"engagement": 0.9, "helpfulness": 0.8}, "context": "technical discussion"},
            {"user_feedback": {"engagement": 0.6, "helpfulness": 0.9}, "context": "casual conversation"},
            {"user_feedback": {"engagement": 0.8, "helpfulness": 0.7}, "context": "problem solving"}
        ]
        
        for scenario in adaptation_scenarios:
            await self.personality_manager.adapt_personality(
                user_feedback=scenario["user_feedback"],
                context=scenario["context"]
            )
            self.demo_stats["personality_adaptations"] += 1
            logger.info(f"  ✓ Adapted personality for: {scenario['context']}")
        
        # Show updated personality
        updated_state = await self.personality_manager.get_personality_state()
        logger.info(f"\nUpdated personality traits:")
        for trait, value in updated_state.traits.items():
            change = value - personality_state.traits.get(trait, 0)
            change_str = f"({change:+.2f})" if abs(change) > 0.01 else ""
            logger.info(f"  - {trait}: {value:.2f} {change_str}")
    
    async def demonstrate_tools_system(self) -> None:
        """Demonstrate tool discovery and function calling."""
        logger.info("\n🛠️ === Tools System Demo ===")
        
        # Discover available tools
        available_tools = await self.tools_manager.discover_tools()
        logger.info(f"Discovered {len(available_tools)} tools:")
        for tool in available_tools[:5]:  # Show first 5 tools
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # Simulate tool usage
        logger.info("\nSimulating tool usage...")
        tool_scenarios = [
            {"query": "What time is it?", "expected_tool": "get_current_time"},
            {"query": "Calculate 15 * 23", "expected_tool": "calculator"},
            {"query": "Search for Python tutorials", "expected_tool": "web_search"}
        ]
        
        for scenario in tool_scenarios:
            try:
                # Get tool suggestions
                suggestions = await self.tools_manager.get_tool_suggestions(
                    query=scenario["query"]
                )
                
                if suggestions:
                    tool = suggestions[0]
                    logger.info(f"  Query: '{scenario['query']}'")
                    logger.info(f"    → Suggested tool: {tool.name}")
                    
                    # Simulate tool execution
                    result = await self.tools_manager.execute_tool(
                        tool_name=tool.name,
                        parameters={}
                    )
                    logger.info(f"    → Result: {str(result)[:100]}...")
                    self.demo_stats["tools_called"] += 1
                else:
                    logger.info(f"  Query: '{scenario['query']}' → No suitable tools found")
                    
            except Exception as e:
                logger.warning(f"  Tool execution failed: {e}")
    
    async def demonstrate_integrated_conversation(self) -> None:
        """Demonstrate integrated conversation with all systems."""
        logger.info("\n💬 === Integrated Conversation Demo ===")
        
        # Start a conversation
        if self.voice_manager:
            self.conversation_id = await self.voice_manager.start_conversation()
            logger.info(f"Started voice conversation: {self.conversation_id}")
        else:
            self.conversation_id = f"demo_conv_{int(time.time())}"
            logger.info(f"Started text conversation: {self.conversation_id}")
        
        # Simulate conversation turns
        conversation_turns = [
            "Hello, I'm working on a Python project and need some help",
            "Can you help me understand machine learning algorithms?",
            "What's the best way to optimize neural network training?",
            "I prefer working in the mornings, can you remember that?",
            "Thanks for the help! You've been very helpful today."
        ]
        
        for i, user_input in enumerate(conversation_turns, 1):
            logger.info(f"\n--- Turn {i} ---")
            logger.info(f"User: {user_input}")
            
            try:
                # Process with integrated systems
                response = await self.process_integrated_input(user_input)
                logger.info(f"Assistant: {response}")
                
                self.demo_stats["interactions"] += 1
                
                # Simulate brief pause between turns
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process input: {e}")
    
    async def process_integrated_input(self, user_input: str) -> str:
        """Process user input through all integrated systems."""
        try:
            # 1. Retrieve relevant memories
            memories = await self.memory_manager.retrieve_memories(
                query=user_input,
                limit=3
            )
            memory_context = [m.content for m in memories]
            
            # 2. Get personality-adapted context
            personality_context = await self.personality_manager.get_response_context(
                user_input=user_input,
                conversation_context=memory_context
            )
            
            # 3. Get tool suggestions
            tool_suggestions = await self.tools_manager.get_tool_suggestions(
                query=user_input
            )
            
            # 4. Generate LLM response with full context
            enhanced_context = {
                "memories": memory_context,
                "personality": personality_context,
                "available_tools": [t.name for t in tool_suggestions[:3]],
                "conversation_id": self.conversation_id
            }

            # Create a mock response for demo purposes
            response_content = f"Thank you for your input about '{user_input[:30]}...'. I've considered your previous preferences and current context to provide this response."

            # In a real implementation, this would be:
            # response = await self.llm_manager.generate_response(
            #     messages=[{"role": "user", "content": user_input}],
            #     context=enhanced_context
            # )

            class MockResponse:
                def __init__(self, content):
                    self.content = content

            response = MockResponse(response_content)
            
            # 5. Store interaction in memory
            await self.memory_manager.store_memory(
                content=f"User said: {user_input}",
                memory_type=MemoryType.EPISODIC,
                importance_score=0.5
            )
            
            await self.memory_manager.store_memory(
                content=f"Assistant responded: {response.content}",
                memory_type=MemoryType.EPISODIC,
                importance_score=0.5
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Integrated processing failed: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    async def show_demo_statistics(self) -> None:
        """Display comprehensive demo statistics."""
        logger.info("\n📊 === Demo Statistics ===")
        
        duration = time.time() - self.demo_stats["start_time"]
        
        logger.info(f"Demo Duration: {duration:.1f} seconds")
        logger.info(f"Total Interactions: {self.demo_stats['interactions']}")
        logger.info(f"Memories Stored: {self.demo_stats['memories_stored']}")
        logger.info(f"Tools Called: {self.demo_stats['tools_called']}")
        logger.info(f"Personality Adaptations: {self.demo_stats['personality_adaptations']}")
        
        # System-specific statistics
        if self.memory_manager:
            memory_stats = await self.memory_manager.get_memory_stats()
            logger.info(f"Total Memories in System: {memory_stats.total_memories}")
        
        if self.personality_manager:
            personality_stats = await self.personality_manager.get_adaptation_stats()
            logger.info(f"Personality Adaptations: {personality_stats.total_adaptations}")
        
        if self.tools_manager:
            tools_stats = await self.tools_manager.get_usage_stats()
            logger.info(f"Available Tools: {tools_stats.total_tools}")
    
    async def cleanup(self) -> None:
        """Clean up all systems."""
        logger.info("\n🧹 Cleaning up systems...")
        
        cleanup_tasks = []
        
        if self.voice_manager:
            cleanup_tasks.append(self.voice_manager.cleanup())
        if self.memory_manager:
            cleanup_tasks.append(self.memory_manager.cleanup())
        if self.personality_manager:
            cleanup_tasks.append(self.personality_manager.cleanup())
        if self.tools_manager:
            cleanup_tasks.append(self.tools_manager.cleanup())
        if self.llm_manager:
            cleanup_tasks.append(self.llm_manager.cleanup())
        if self.websocket_manager:
            cleanup_tasks.append(self.websocket_manager.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("✅ Cleanup completed")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Comprehensive Coda Demo")
    parser.add_argument(
        "--mode",
        choices=["full", "memory", "personality", "tools", "conversation"],
        default="full",
        help="Demo mode to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    demo = CodaDemo()
    
    try:
        # Initialize systems
        await demo.initialize_all_systems()
        
        # Run selected demo mode
        if args.mode == "full":
            await demo.demonstrate_memory_system()
            await demo.demonstrate_personality_system()
            await demo.demonstrate_tools_system()
            await demo.demonstrate_integrated_conversation()
        elif args.mode == "memory":
            await demo.demonstrate_memory_system()
        elif args.mode == "personality":
            await demo.demonstrate_personality_system()
        elif args.mode == "tools":
            await demo.demonstrate_tools_system()
        elif args.mode == "conversation":
            await demo.demonstrate_integrated_conversation()
        
        # Show statistics
        await demo.show_demo_statistics()
        
        logger.info("\n🎉 Demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        raise
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
