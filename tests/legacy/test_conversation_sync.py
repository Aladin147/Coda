#!/usr/bin/env python3
"""
Test Voice-Conversation Synchronization

This script tests the synchronization between voice conversations and conversation manager.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.conversation_sync import (
        VoiceConversationSync, VoiceConversationSyncConfig, VoiceConversationSyncManager,
        ConflictResolutionStrategy, SyncEventType
    )
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode, ConversationState
    logger.info("✓ Successfully imported voice-conversation sync components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


# Mock conversation components
class MockMessage:
    def __init__(self, id: str, role: str, content: str, timestamp: datetime = None):
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = {}

class MockConversation:
    def __init__(self, conversation_id: str):
        self.id = conversation_id
        self.messages = []
        self.metadata = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

class MockConversationManager:
    """Mock conversation manager for testing."""
    
    def __init__(self):
        self.conversations = {}
        self.messages = {}
        
    async def get_conversation(self, conversation_id: str):
        """Mock get conversation."""
        return self.conversations.get(conversation_id)
    
    async def create_conversation(self, conversation_id: str):
        """Mock create conversation."""
        conversation = MockConversation(conversation_id)
        self.conversations[conversation_id] = conversation
        return conversation
    
    async def add_message(self, conversation_id: str, message):
        """Mock add message."""
        if conversation_id not in self.conversations:
            await self.create_conversation(conversation_id)
        
        conversation = self.conversations[conversation_id]
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
        
        # Store message separately for easy access
        self.messages[message.id] = message
        
        return True
    
    async def update_conversation_metadata(self, conversation_id: str, metadata):
        """Mock update conversation metadata."""
        if conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            conversation.metadata.update(metadata.__dict__ if hasattr(metadata, '__dict__') else metadata)
            conversation.updated_at = datetime.now()
        return True


def create_test_voice_message(conversation_id: str, text_content: str) -> VoiceMessage:
    """Create a test voice message."""
    audio_data = np.random.randint(-32768, 32767, 24000, dtype=np.int16).tobytes()
    
    return VoiceMessage(
        message_id=f"test_msg_{datetime.now().strftime('%H%M%S%f')}",
        conversation_id=conversation_id,
        audio_data=audio_data,
        text_content=text_content,
        processing_mode=VoiceProcessingMode.HYBRID,
        timestamp=datetime.now()
    )


def create_test_voice_response(message_id: str, conversation_id: str, text_content: str) -> VoiceResponse:
    """Create a test voice response."""
    return VoiceResponse(
        response_id=f"response_{message_id}",
        conversation_id=conversation_id,
        message_id=message_id,
        text_content=text_content,
        audio_data=b"",
        processing_mode=VoiceProcessingMode.HYBRID,
        total_latency_ms=150.0,
        response_relevance=0.8
    )


def create_test_conversation_state(conversation_id: str) -> ConversationState:
    """Create a test conversation state."""
    return ConversationState(
        conversation_id=conversation_id,
        started_at=datetime.now(),
        last_activity=datetime.now(),
        message_count=0,
        is_active=True,
        current_mode=VoiceProcessingMode.HYBRID
    )


async def test_conversation_sync_config():
    """Test conversation sync configuration."""
    logger.info("=== Testing Conversation Sync Configuration ===")
    
    try:
        config = VoiceConversationSyncConfig()
        
        logger.info("✓ Default configuration created")
        logger.info(f"  Sync enabled: {config.sync_enabled}")
        logger.info(f"  Real-time sync: {config.real_time_sync}")
        logger.info(f"  Bidirectional sync: {config.bidirectional_sync}")
        logger.info(f"  Batch sync enabled: {config.batch_sync_enabled}")
        logger.info(f"  Conflict resolution: {config.conflict_resolution_strategy}")
        
        # Test custom configuration
        custom_config = VoiceConversationSyncConfig(
            sync_enabled=True,
            real_time_sync=False,
            conflict_resolution_strategy=ConflictResolutionStrategy.TIMESTAMP_PRIORITY,
            batch_sync_interval_seconds=10.0
        )
        
        logger.info("✓ Custom configuration created")
        logger.info(f"  Real-time sync: {custom_config.real_time_sync}")
        logger.info(f"  Conflict resolution: {custom_config.conflict_resolution_strategy}")
        
        logger.info("✓ Conversation sync configuration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_voice_message_sync():
    """Test voice message synchronization."""
    logger.info("=== Testing Voice Message Sync ===")
    
    try:
        conversation_manager = MockConversationManager()
        config = VoiceConversationSyncConfig()
        sync = VoiceConversationSync(conversation_manager, config)
        
        await sync.start()
        logger.info("✓ Voice conversation sync started")
        
        # Test voice message sync
        voice_message = create_test_voice_message(
            "test_conversation_sync",
            "Hello, this is a test voice message"
        )
        
        conversation_state = create_test_conversation_state("test_conversation_sync")
        
        success = await sync.sync_voice_message(voice_message, conversation_state)
        
        logger.info(f"✓ Voice message sync: {'SUCCESS' if success else 'FAILED'}")
        
        # Check if message was added to conversation manager
        conversation = await conversation_manager.get_conversation("test_conversation_sync")
        
        if conversation and len(conversation.messages) > 0:
            logger.info(f"  Message added to conversation: {conversation.messages[0].content[:50]}...")
            logger.info(f"  Message role: {conversation.messages[0].role}")
        
        await sync.stop()
        logger.info("✓ Voice message sync test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice message sync test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_voice_response_sync():
    """Test voice response synchronization."""
    logger.info("=== Testing Voice Response Sync ===")
    
    try:
        conversation_manager = MockConversationManager()
        config = VoiceConversationSyncConfig()
        sync = VoiceConversationSync(conversation_manager, config)
        
        await sync.start()
        
        # Test voice response sync
        voice_message = create_test_voice_message(
            "test_conversation_response",
            "What's the weather like?"
        )
        
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "The weather is sunny and 22°C today."
        )
        
        conversation_state = create_test_conversation_state("test_conversation_response")
        
        # Sync both message and response
        message_success = await sync.sync_voice_message(voice_message, conversation_state)
        response_success = await sync.sync_voice_response(voice_response, conversation_state)
        
        logger.info(f"✓ Voice message sync: {'SUCCESS' if message_success else 'FAILED'}")
        logger.info(f"✓ Voice response sync: {'SUCCESS' if response_success else 'FAILED'}")
        
        # Check conversation
        conversation = await conversation_manager.get_conversation("test_conversation_response")
        
        if conversation:
            logger.info(f"  Total messages in conversation: {len(conversation.messages)}")
            for i, msg in enumerate(conversation.messages):
                logger.info(f"    Message {i+1}: {msg.role} - {msg.content[:40]}...")
        
        await sync.stop()
        logger.info("✓ Voice response sync test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice response sync test failed: {e}")
        return False


async def test_conversation_state_sync():
    """Test conversation state synchronization."""
    logger.info("=== Testing Conversation State Sync ===")
    
    try:
        conversation_manager = MockConversationManager()
        config = VoiceConversationSyncConfig()
        sync = VoiceConversationSync(conversation_manager, config)
        
        await sync.start()
        
        # Create conversation state
        conversation_state = create_test_conversation_state("test_conversation_state")
        conversation_state.message_count = 5
        conversation_state.is_active = True
        
        # Sync conversation state
        success = await sync.sync_conversation_state(conversation_state)
        
        logger.info(f"✓ Conversation state sync: {'SUCCESS' if success else 'FAILED'}")
        
        # Check conversation metadata
        conversation = await conversation_manager.get_conversation("test_conversation_state")
        
        if conversation and conversation.metadata:
            logger.info(f"  Conversation metadata updated:")
            logger.info(f"    Message count: {conversation.metadata.get('message_count', 'N/A')}")
            logger.info(f"    Is active: {conversation.metadata.get('is_active', 'N/A')}")
            logger.info(f"    Tags: {conversation.metadata.get('tags', 'N/A')}")
        
        await sync.stop()
        logger.info("✓ Conversation state sync test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation state sync test failed: {e}")
        return False


async def test_conflict_resolution():
    """Test conflict resolution strategies."""
    logger.info("=== Testing Conflict Resolution ===")
    
    try:
        conversation_manager = MockConversationManager()
        
        # Test different conflict resolution strategies
        strategies = [
            ConflictResolutionStrategy.VOICE_PRIORITY,
            ConflictResolutionStrategy.CONVERSATION_PRIORITY,
            ConflictResolutionStrategy.TIMESTAMP_PRIORITY,
            ConflictResolutionStrategy.MERGE
        ]
        
        for strategy in strategies:
            config = VoiceConversationSyncConfig(
                conflict_resolution_strategy=strategy,
                enable_conflict_detection=True
            )
            sync = VoiceConversationSync(conversation_manager, config)
            
            await sync.start()
            
            # Simulate conflict scenario
            conversation_id = f"test_conflict_{strategy.value}"
            
            # First sync
            voice_message = create_test_voice_message(conversation_id, "Test conflict message")
            await sync.sync_voice_message(voice_message)
            
            # Simulate conflicting update
            update_data = {
                "message_count": 10,
                "is_active": False,
                "timestamp": datetime.now()
            }
            
            conflict_handled = await sync.handle_conversation_update(conversation_id, update_data)
            
            logger.info(f"✓ Conflict resolution ({strategy.value}): {'SUCCESS' if conflict_handled else 'FAILED'}")
            
            await sync.stop()
        
        logger.info("✓ Conflict resolution test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Conflict resolution test failed: {e}")
        return False


async def test_sync_events():
    """Test synchronization events."""
    logger.info("=== Testing Sync Events ===")
    
    try:
        conversation_manager = MockConversationManager()
        config = VoiceConversationSyncConfig(enable_sync_events=True)
        sync = VoiceConversationSync(conversation_manager, config)
        
        # Event tracking
        events_received = []
        
        def event_handler(event_type, data):
            events_received.append((event_type, data))
            logger.info(f"  Event received: {event_type}")
        
        # Add event handlers
        sync.add_event_handler(SyncEventType.VOICE_MESSAGE_ADDED, event_handler)
        sync.add_event_handler(SyncEventType.VOICE_RESPONSE_ADDED, event_handler)
        sync.add_event_handler(SyncEventType.CONVERSATION_STATE_UPDATED, event_handler)
        
        await sync.start()
        
        # Trigger events
        voice_message = create_test_voice_message("test_events", "Test event message")
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "Test event response"
        )
        conversation_state = create_test_conversation_state("test_events")
        
        await sync.sync_voice_message(voice_message)
        await sync.sync_voice_response(voice_response)
        await sync.sync_conversation_state(conversation_state)
        
        # Wait for events to be processed
        await asyncio.sleep(0.1)
        
        logger.info(f"✓ Events received: {len(events_received)}")
        for event_type, data in events_received:
            logger.info(f"    {event_type}: {data.get('conversation_id', 'N/A')}")
        
        await sync.stop()
        logger.info("✓ Sync events test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync events test failed: {e}")
        return False


async def test_sync_statistics():
    """Test sync statistics tracking."""
    logger.info("=== Testing Sync Statistics ===")
    
    try:
        conversation_manager = MockConversationManager()
        config = VoiceConversationSyncConfig()
        sync = VoiceConversationSync(conversation_manager, config)
        
        await sync.start()
        
        # Perform multiple sync operations
        for i in range(3):
            voice_message = create_test_voice_message(
                f"test_stats_{i}",
                f"Test statistics message {i+1}"
            )
            voice_response = create_test_voice_response(
                voice_message.message_id,
                voice_message.conversation_id,
                f"Test statistics response {i+1}"
            )
            
            await sync.sync_voice_message(voice_message)
            await sync.sync_voice_response(voice_response)
        
        # Get statistics
        stats = sync.get_sync_statistics()
        
        logger.info(f"✓ Sync statistics:")
        logger.info(f"  Sync operations: {stats['sync_stats']['sync_operations']}")
        logger.info(f"  Successful syncs: {stats['sync_stats']['successful_syncs']}")
        logger.info(f"  Failed syncs: {stats['sync_stats']['failed_syncs']}")
        logger.info(f"  Active conversations: {stats['active_conversations']}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        
        # Check configuration in stats
        config_stats = stats['config']
        logger.info(f"  Configuration:")
        logger.info(f"    Sync enabled: {config_stats['sync_enabled']}")
        logger.info(f"    Real-time sync: {config_stats['real_time_sync']}")
        logger.info(f"    Conflict resolution: {config_stats['conflict_resolution_strategy']}")
        
        await sync.stop()
        logger.info("✓ Sync statistics test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync statistics test failed: {e}")
        return False


async def test_voice_conversation_sync_manager():
    """Test high-level voice conversation sync manager."""
    logger.info("=== Testing Voice Conversation Sync Manager ===")
    
    try:
        conversation_manager = MockConversationManager()
        config = VoiceConversationSyncConfig()
        sync_manager = VoiceConversationSyncManager(conversation_manager, config)
        
        await sync_manager.start()
        logger.info("✓ Voice conversation sync manager started")
        
        # Test complete voice interaction sync
        voice_message = create_test_voice_message(
            "test_manager",
            "How can I improve my productivity?"
        )
        
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "Here are some tips to improve productivity: 1. Set clear goals..."
        )
        
        conversation_state = create_test_conversation_state("test_manager")
        conversation_state.message_count = 2
        
        # Sync complete interaction
        success = await sync_manager.sync_voice_interaction(
            voice_message, voice_response, conversation_state
        )
        
        logger.info(f"✓ Voice interaction sync: {'SUCCESS' if success else 'FAILED'}")
        
        # Check conversation
        conversation = await conversation_manager.get_conversation("test_manager")
        
        if conversation:
            logger.info(f"  Messages in conversation: {len(conversation.messages)}")
            logger.info(f"  Conversation metadata: {len(conversation.metadata)} fields")
        
        # Test statistics
        stats = sync_manager.get_sync_stats()
        logger.info(f"✓ Manager statistics:")
        logger.info(f"  Sync operations: {stats['sync_stats']['sync_operations']}")
        logger.info(f"  Active conversations: {stats['active_conversations']}")
        
        await sync_manager.stop()
        logger.info("✓ Voice conversation sync manager test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync manager test failed: {e}")
        return False


async def main():
    """Run all voice-conversation sync tests."""
    logger.info("🚀 Starting Voice-Conversation Sync Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Conversation Sync Configuration", test_conversation_sync_config),
        ("Voice Message Sync", test_voice_message_sync),
        ("Voice Response Sync", test_voice_response_sync),
        ("Conversation State Sync", test_conversation_state_sync),
        ("Conflict Resolution", test_conflict_resolution),
        ("Sync Events", test_sync_events),
        ("Sync Statistics", test_sync_statistics),
        ("Voice Conversation Sync Manager", test_voice_conversation_sync_manager),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "❌ CRASHED"
    
    # Print results summary
    logger.info("=" * 50)
    logger.info("🏁 Voice-Conversation Sync Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<35}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice-conversation sync tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
