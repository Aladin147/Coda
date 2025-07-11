#!/usr/bin/env python3
"""
Interactive Coda Testing Interface
Real-time testing with all primary components - NO FALLBACKS

This provides an interactive command-line interface to test:
- Ollama qwen3:30b-a3b LLM (PRIMARY, /no_think enabled)
- ChromaDB Memory System (PRIMARY)
- Real-time conversation with memory persistence

NO FALLBACKS ACTIVE - Pure primary systems only!
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("interactive_coda")

# Import core components
try:
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    from coda.components.llm.manager import LLMManager
    from coda.components.memory.manager import MemoryManager
    from coda.components.memory.models import MemoryManagerConfig
    
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    exit(1)


class InteractiveCodaTest:
    """Interactive testing interface for Coda system."""
    
    def __init__(self):
        self.llm_manager = None
        self.memory_manager = None
        self.conversation_count = 0
        
    async def initialize(self):
        """Initialize core components."""
        print("=" * 60)
        print("🚀 CODA INTERACTIVE TEST - PRIMARY COMPONENTS ONLY")
        print("=" * 60)
        print("Initializing components...")
        
        # Initialize LLM
        print("🧠 Initializing LLM (Ollama qwen3:30b-a3b, /no_think)...")
        config = LLMConfig(
            providers={
                "ollama": ProviderConfig(
                    provider=LLMProvider.OLLAMA,
                    model="qwen3:30b-a3b",
                    host="http://localhost:11434",
                    temperature=0.7,
                    max_tokens=512,
                    system_message="/no_think You are Coda, a helpful voice assistant. Respond naturally and concisely."
                )
            },
            default_provider="ollama"
        )
        
        self.llm_manager = LLMManager(config)
        print("✅ LLM Manager initialized")
        
        # Initialize Memory
        print("🧠 Initializing Memory (ChromaDB)...")
        memory_config = MemoryManagerConfig()
        self.memory_manager = MemoryManager(memory_config)
        await self.memory_manager.initialize()
        print("✅ Memory Manager initialized")
        
        print("=" * 60)
        print("🎉 CODA READY FOR INTERACTIVE TESTING!")
        print("=" * 60)
        print("💡 Features:")
        print("  • Fast LLM responses with /no_think enabled")
        print("  • Memory persistence across conversations")
        print("  • Real-time performance metrics")
        print("  • NO FALLBACKS - Primary systems only")
        print("=" * 60)
        print("Commands:")
        print("  • Type your message and press Enter")
        print("  • Type 'quit' or 'exit' to stop")
        print("  • Type 'memory' to search memory")
        print("  • Type 'stats' to see performance stats")
        print("=" * 60)
        
    async def chat_loop(self):
        """Main interactive chat loop."""
        total_response_time = 0
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'memory':
                    await self.search_memory()
                    continue
                    
                elif user_input.lower() == 'stats':
                    self.show_stats(total_response_time)
                    continue
                    
                elif not user_input:
                    continue
                
                # Generate response
                start_time = time.time()
                
                response = await self.llm_manager.generate_response(
                    prompt=f"/no_think {user_input}",
                    provider="ollama"
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                total_response_time += response_time
                self.conversation_count += 1
                
                # Store in memory
                self.memory_manager.add_turn("user", user_input)
                self.memory_manager.add_turn("assistant", response.content)
                
                # Display response
                print(f"🤖 Coda: {response.content}")
                print(f"⚡ Response time: {response_time:.1f}ms")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    async def search_memory(self):
        """Search memory for previous conversations."""
        query = input("🔍 Search memory for: ").strip()
        if not query:
            return
            
        try:
            memories = await self.memory_manager.search_memories(query, limit=5)
            
            if memories:
                print(f"\n💭 Found {len(memories)} relevant memories:")
                for i, memory in enumerate(memories, 1):
                    print(f"  {i}. {memory.content[:100]}... (score: {memory.relevance_score:.3f})")
            else:
                print("💭 No relevant memories found")
                
        except Exception as e:
            print(f"❌ Memory search error: {e}")
            
    def show_stats(self, total_response_time):
        """Show performance statistics."""
        if self.conversation_count > 0:
            avg_response_time = total_response_time / self.conversation_count
            print(f"\n📊 Performance Stats:")
            print(f"  • Conversations: {self.conversation_count}")
            print(f"  • Average response time: {avg_response_time:.1f}ms")
            print(f"  • Total response time: {total_response_time:.1f}ms")
            print(f"  • LLM: Ollama qwen3:30b-a3b (/no_think)")
            print(f"  • Memory: ChromaDB")
            print(f"  • Fallbacks: DISABLED")
        else:
            print("📊 No conversations yet")


async def main():
    """Main function."""
    tester = InteractiveCodaTest()
    
    try:
        await tester.initialize()
        await tester.chat_loop()
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
