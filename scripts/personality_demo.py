#!/usr/bin/env python3
"""
Personality system demonstration script.

This script demonstrates the personality system functionality by:
1. Creating a personality manager with all components
2. Simulating conversations with personality adaptation
3. Demonstrating behavioral learning and topic awareness
4. Showing WebSocket integration capabilities
5. Displaying personality analytics and insights
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coda.components.personality.models import (
    PersonalityConfig,
    PersonalityTraitType,
    TopicCategory,
)
from coda.components.personality.manager import PersonalityManager
from coda.components.personality.websocket_integration import WebSocketPersonalityManager
from coda.interfaces.websocket.server import CodaWebSocketServer
from coda.interfaces.websocket.integration import CodaWebSocketIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("personality_demo")


async def demonstrate_basic_personality():
    """Demonstrate basic personality functionality."""
    logger.info("🧠 Starting basic personality demonstration...")
    
    # Create personality manager
    config = PersonalityConfig()
    personality = PersonalityManager(config)
    
    # Simulate a conversation with personality adaptation
    logger.info("💬 Simulating conversation with personality adaptation...")
    
    # Technical conversation
    logger.info("📚 Technical conversation phase...")
    result1 = await personality.process_user_input("I'm learning Python programming and need help with algorithms")
    logger.info(f"Topic detected: {result1['topic_context']['current_topic']} ({result1['topic_context']['category']})")
    logger.info(f"Personality adjustments: {len(result1['personality_adjustments'])}")
    
    # Enhanced prompt for technical context
    enhanced_prompt = await personality.enhance_prompt(
        "You are a helpful programming assistant.",
        "technical programming discussion"
    )
    logger.info(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
    
    # Process assistant response
    await personality.process_assistant_response(
        "I'd be happy to help you with Python algorithms! Let's start with sorting algorithms - they're fundamental and great for learning."
    )
    
    # Creative conversation shift
    logger.info("🎨 Creative conversation phase...")
    result2 = await personality.process_user_input("Actually, I'm also interested in creative writing. Can you help me write a story?")
    logger.info(f"Topic shift to: {result2['topic_context']['current_topic']} ({result2['topic_context']['category']})")
    
    # User feedback processing
    logger.info("📝 Processing user feedback...")
    feedback_result = await personality.process_feedback("I like your creative suggestions, but please be more concise")
    logger.info(f"Feedback processed with {len(feedback_result['personality_adjustments'])} adjustments")
    
    # Personal conversation
    logger.info("💭 Personal conversation phase...")
    result3 = await personality.process_user_input("I'm feeling a bit overwhelmed with all this learning")
    logger.info(f"Topic: {result3['topic_context']['category']}")
    
    # Get personality state
    state = personality.get_personality_state()
    logger.info("📊 Current personality state:")
    for trait, info in state["parameters"].items():
        deviation = info["deviation"]
        if abs(deviation) > 0.1:
            direction = "↑" if deviation > 0 else "↓"
            logger.info(f"  {trait}: {info['value']:.2f} {direction} (Δ{deviation:+.2f})")
    
    # Session information
    session_info = state["session"]
    logger.info(f"📅 Session: {session_info['turn_count']} turns, {session_info['duration_minutes']:.1f} minutes")
    
    # Get analytics
    analytics = personality.get_analytics()
    logger.info("📈 Analytics overview:")
    logger.info(f"  Total interactions: {analytics['overview']['total_interactions']}")
    logger.info(f"  Total adjustments: {analytics['overview']['total_adjustments']}")
    logger.info(f"  Most adjusted trait: {analytics['overview']['most_adjusted_trait']}")
    
    logger.info("✅ Basic personality demonstration completed!")


async def demonstrate_behavioral_learning():
    """Demonstrate behavioral learning capabilities."""
    logger.info("🎓 Starting behavioral learning demonstration...")
    
    personality = PersonalityManager()
    
    # Simulate user preference patterns
    logger.info("📚 Simulating user preference learning...")
    
    # User prefers brief responses
    await personality.process_user_input("Please be brief")
    await personality.process_feedback("That was too long, please be more concise")
    await personality.process_user_input("Keep it short")
    
    # User likes technical detail
    await personality.process_user_input("Can you explain how neural networks work in detail?")
    await personality.process_feedback("Great explanation! I love the technical depth")
    await personality.process_user_input("Tell me more about the mathematical foundations")
    
    # User enjoys humor
    await personality.process_user_input("Make this more fun and engaging")
    await personality.process_feedback("I love your sense of humor!")
    
    # Apply learned adjustments
    logger.info("🔄 Applying learned behavioral adjustments...")
    learning_result = await personality.apply_learned_adjustments()
    logger.info(f"Applied {learning_result['adjustments_applied']} learned adjustments")
    
    # Show behavior profile
    behavior_stats = personality.behavioral_conditioner.get_learning_stats()
    logger.info("🧠 Behavioral learning stats:")
    logger.info(f"  Observations: {behavior_stats['observation_count']}")
    logger.info(f"  Learned preferences: {behavior_stats['learned_preferences']}")
    logger.info(f"  Average confidence: {behavior_stats['average_confidence']:.2f}")
    
    # Show personality changes
    final_state = personality.get_personality_state()
    logger.info("📊 Learned personality adjustments:")
    for trait, info in final_state["parameters"].items():
        if info["adjustments"] > 0:
            logger.info(f"  {trait}: {info['value']:.2f} ({info['adjustments']} adjustments)")
    
    logger.info("✅ Behavioral learning demonstration completed!")


async def demonstrate_websocket_integration():
    """Demonstrate WebSocket integration with personality system."""
    logger.info("🌐 Starting WebSocket personality demonstration...")
    
    try:
        # Set up WebSocket server
        server = CodaWebSocketServer(host="localhost", port=8767)
        integration = CodaWebSocketIntegration(server)
        
        # Create WebSocket-enabled personality manager
        config = PersonalityConfig()
        personality = WebSocketPersonalityManager(config)
        await personality.set_websocket_integration(integration)
        
        # Start WebSocket server
        await server.start()
        logger.info(f"🌐 WebSocket server running at ws://{server.host}:{server.port}")
        logger.info("💡 Connect with: wscat -c ws://localhost:8767")
        
        # Wait for potential clients
        logger.info("⏳ Waiting 3 seconds for WebSocket clients...")
        await asyncio.sleep(3)
        
        # Simulate personality operations with WebSocket events
        logger.info("🎬 Simulating personality operations with WebSocket events...")
        
        # Topic detection and personality adjustment
        await personality.process_user_input("I'm working on a machine learning project")
        await asyncio.sleep(0.5)
        
        # Behavioral feedback
        await personality.process_feedback("I prefer more technical detail in explanations")
        await asyncio.sleep(0.5)
        
        # Topic shift
        await personality.process_user_input("Let's talk about creative writing instead")
        await asyncio.sleep(0.5)
        
        # Session closure simulation
        await personality.process_user_input("Thanks, that's all for now")
        await asyncio.sleep(0.5)
        
        # Broadcast comprehensive analytics
        await personality.broadcast_personality_analytics()
        await personality.broadcast_behavior_insights()
        await personality.broadcast_lore_usage()
        
        # Trigger personality snapshot
        snapshot = await personality.trigger_personality_snapshot()
        logger.info(f"📸 Personality snapshot captured with {len(snapshot)} data points")
        
        # Show server stats
        server_stats = server.get_stats()
        logger.info(f"📊 WebSocket server stats: {server_stats}")
        
        logger.info("✅ WebSocket personality demonstration completed!")
        logger.info("⏳ Server will stop in 3 seconds...")
        await asyncio.sleep(3)
        
        await server.stop()
        
    except Exception as e:
        logger.error(f"❌ Error in WebSocket demonstration: {e}")


async def demonstrate_advanced_features():
    """Demonstrate advanced personality features."""
    logger.info("🚀 Starting advanced personality features demonstration...")
    
    personality = PersonalityManager()
    
    # Demonstrate prompt enhancement
    logger.info("✨ Demonstrating prompt enhancement...")
    
    base_prompts = [
        "You are a helpful assistant.",
        "Help the user with their programming question.",
        "Provide creative writing assistance."
    ]
    
    for i, base_prompt in enumerate(base_prompts):
        enhanced = await personality.enhance_prompt(base_prompt, f"context_{i}")
        enhancement_ratio = len(enhanced) / len(base_prompt)
        logger.info(f"  Prompt {i+1}: {len(base_prompt)} → {len(enhanced)} chars (×{enhancement_ratio:.1f})")
    
    # Demonstrate lore injection
    logger.info("📚 Demonstrating personal lore injection...")
    
    lore_stats = personality.personal_lore.get_lore_usage_stats()
    logger.info(f"  Available lore: {lore_stats['total_quirks']} quirks, {lore_stats['total_memories']} memories")
    
    # Get relevant lore for different contexts
    contexts = ["technical", "creative", "personal"]
    for context in contexts:
        relevant_lore = personality.personal_lore.get_relevant_lore(context, [context])
        logger.info(f"  {context.title()} context: {len(relevant_lore)} relevant lore elements")
    
    # Demonstrate session management
    logger.info("⏰ Demonstrating session management...")
    
    # Simulate extended conversation
    for i in range(15):
        await personality.process_user_input(f"Message {i+1}")
        await personality.process_assistant_response(f"Response {i+1}")
    
    session_analytics = personality.session_manager.get_session_analytics()
    logger.info("📊 Session analytics:")
    logger.info(f"  Interaction patterns: {session_analytics['interaction_patterns']}")
    logger.info(f"  Engagement analysis: {session_analytics['engagement_analysis']}")
    logger.info(f"  Session flow: {session_analytics['session_flow']}")
    
    # Demonstrate topic awareness
    logger.info("🎯 Demonstrating topic awareness...")
    
    topic_inputs = [
        "Let's discuss machine learning algorithms",
        "I want to write a creative story",
        "Help me with my work presentation",
        "I'm feeling stressed about my studies"
    ]
    
    for topic_input in topic_inputs:
        await personality.process_user_input(topic_input)
    
    topic_stats = personality.topic_awareness.get_topic_stats()
    logger.info(f"  Topics detected: {topic_stats['total_topics']}")
    logger.info(f"  Category distribution: {topic_stats['category_distribution']}")
    logger.info(f"  Most common category: {topic_stats['most_common_category']}")
    
    # Export and import personality state
    logger.info("💾 Demonstrating state persistence...")
    
    exported_state = personality.export_personality_state()
    logger.info(f"  Exported state size: {len(str(exported_state))} characters")
    
    # Create new personality and import state
    new_personality = PersonalityManager()
    import_success = new_personality.import_personality_state(exported_state)
    logger.info(f"  State import successful: {import_success}")
    
    logger.info("✅ Advanced features demonstration completed!")


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Coda Personality System Demonstration")
    
    try:
        # Run demonstrations
        await demonstrate_basic_personality()
        await asyncio.sleep(1)
        
        await demonstrate_behavioral_learning()
        await asyncio.sleep(1)
        
        await demonstrate_websocket_integration()
        await asyncio.sleep(1)
        
        await demonstrate_advanced_features()
        
        logger.info("🎉 All personality system demonstrations completed successfully!")
        
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
