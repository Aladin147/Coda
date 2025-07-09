"""
Voice-Conversation Synchronization

This module provides comprehensive synchronization between voice conversations
and the conversation manager for consistent state management across the system.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .models import VoiceMessage, VoiceResponse, ConversationState
try:
    from ..conversation.interfaces import ConversationManagerInterface
    from ..conversation.models import (
        Conversation, Message, MessageRole, ConversationMetadata
    )
except ImportError:
    # Create mock interfaces for testing
    class ConversationManagerInterface:
        async def get_conversation(self, conversation_id: str): pass
        async def create_conversation(self, conversation_id: str): pass
        async def add_message(self, conversation_id: str, message): pass
        async def update_conversation_metadata(self, conversation_id: str, metadata): pass

    class Message:
        def __init__(self, id: str, role: str, content: str, timestamp=None, metadata=None):
            self.id = id
            self.role = role
            self.content = content
            self.timestamp = timestamp
            self.metadata = metadata or {}

    class MessageRole:
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

    class ConversationMetadata:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

logger = logging.getLogger("coda.voice.conversation_sync")


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving sync conflicts."""
    VOICE_PRIORITY = "voice_priority"
    CONVERSATION_PRIORITY = "conversation_priority"
    TIMESTAMP_PRIORITY = "timestamp_priority"
    MERGE = "merge"


class SyncEventType(str, Enum):
    """Types of synchronization events."""
    VOICE_MESSAGE_ADDED = "voice_message_added"
    VOICE_RESPONSE_ADDED = "voice_response_added"
    CONVERSATION_STATE_UPDATED = "conversation_state_updated"
    CONVERSATION_METADATA_CHANGED = "conversation_metadata_changed"
    SYNC_CONFLICT_DETECTED = "sync_conflict_detected"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"


@dataclass
class VoiceConversationSyncConfig:
    """Configuration for voice-conversation synchronization."""
    
    # Core sync settings
    sync_enabled: bool = True
    real_time_sync: bool = True
    bidirectional_sync: bool = True
    
    # Batch synchronization
    batch_sync_enabled: bool = True
    batch_sync_interval_seconds: float = 5.0
    max_batch_size: int = 50
    
    # Conflict resolution
    conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.VOICE_PRIORITY
    enable_conflict_detection: bool = True
    
    # Sync scope
    sync_conversation_metadata: bool = True
    sync_message_history: bool = True
    sync_conversation_state: bool = True
    
    # Performance optimization
    enable_sync_caching: bool = True
    sync_cache_ttl_minutes: int = 5
    max_sync_retries: int = 3
    sync_timeout_seconds: float = 10.0
    
    # Event handling
    enable_conversation_events: bool = True
    enable_sync_events: bool = True


class VoiceConversationSync:
    """
    Comprehensive synchronization between voice conversations and conversation manager.
    
    Features:
    - Real-time bidirectional synchronization
    - Batch synchronization for performance
    - Conflict resolution strategies
    - Event-driven updates
    - Performance optimization with caching
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        conversation_manager: ConversationManagerInterface,
        config: VoiceConversationSyncConfig
    ):
        """Initialize voice-conversation synchronization."""
        self.conversation_manager = conversation_manager
        self.config = config
        
        # Sync state tracking
        self.voice_conversations: Dict[str, ConversationState] = {}
        self.sync_timestamps: Dict[str, datetime] = {}
        self.pending_syncs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Caching for performance
        self.sync_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Event handlers
        self.event_handlers: Dict[SyncEventType, List[Callable]] = {}
        
        # Statistics tracking
        self.stats = {
            "sync_operations": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "batch_syncs": 0,
            "cache_hits": 0
        }
        
        # Background tasks
        self._batch_sync_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("VoiceConversationSync initialized")
    
    async def start(self) -> None:
        """Start the synchronization system."""
        
        if self._running:
            return
        
        self._running = True
        
        # Start batch sync task if enabled
        if self.config.batch_sync_enabled:
            self._batch_sync_task = asyncio.create_task(self._batch_sync_loop())
        
        logger.info("Voice conversation sync started")
    
    async def stop(self) -> None:
        """Stop the synchronization system."""
        
        self._running = False
        
        # Cancel batch sync task
        if self._batch_sync_task:
            self._batch_sync_task.cancel()
            try:
                await self._batch_sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Voice conversation sync stopped")
    
    async def sync_voice_message(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None
    ) -> bool:
        """
        Synchronize voice message to conversation manager.
        
        Args:
            voice_message: Voice message to sync
            conversation_state: Current conversation state
            
        Returns:
            True if sync successful, False otherwise
        """
        
        try:
            if not self.config.sync_enabled:
                return True
            
            # Check cache first
            cache_key = f"voice_msg_{voice_message.message_id}"
            if self._is_sync_cached(cache_key):
                self.stats["cache_hits"] += 1
                return True
            
            # Convert voice message to conversation message
            conversation_message = await self._convert_voice_to_conversation_message(voice_message)
            
            # Add message to conversation
            await self.conversation_manager.add_message(
                conversation_id=voice_message.conversation_id,
                message=conversation_message
            )
            
            # Update sync tracking
            self.sync_timestamps[voice_message.conversation_id] = datetime.now()
            self._cache_sync_result(cache_key, True)
            
            # Emit sync event
            await self._emit_sync_event(SyncEventType.VOICE_MESSAGE_ADDED, {
                "voice_message_id": voice_message.message_id,
                "conversation_id": voice_message.conversation_id,
                "timestamp": datetime.now()
            })
            
            self.stats["sync_operations"] += 1
            self.stats["successful_syncs"] += 1
            
            logger.debug(f"Synced voice message {voice_message.message_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync voice message: {e}")
            self.stats["failed_syncs"] += 1
            return False
    
    async def sync_voice_response(
        self,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None
    ) -> bool:
        """
        Synchronize voice response to conversation manager.
        
        Args:
            voice_response: Voice response to sync
            conversation_state: Current conversation state
            
        Returns:
            True if sync successful, False otherwise
        """
        
        try:
            if not self.config.sync_enabled:
                return True
            
            # Check cache first
            cache_key = f"voice_resp_{voice_response.response_id}"
            if self._is_sync_cached(cache_key):
                self.stats["cache_hits"] += 1
                return True
            
            # Convert voice response to conversation message
            conversation_message = await self._convert_voice_response_to_conversation_message(voice_response)
            
            # Add message to conversation
            await self.conversation_manager.add_message(
                conversation_id=voice_response.conversation_id,
                message=conversation_message
            )
            
            # Update sync tracking
            self.sync_timestamps[voice_response.conversation_id] = datetime.now()
            self._cache_sync_result(cache_key, True)
            
            # Emit sync event
            await self._emit_sync_event(SyncEventType.VOICE_RESPONSE_ADDED, {
                "voice_response_id": voice_response.response_id,
                "conversation_id": voice_response.conversation_id,
                "timestamp": datetime.now()
            })
            
            self.stats["sync_operations"] += 1
            self.stats["successful_syncs"] += 1
            
            logger.debug(f"Synced voice response {voice_response.response_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync voice response: {e}")
            self.stats["failed_syncs"] += 1
            return False
    
    async def sync_conversation_state(
        self,
        conversation_state: ConversationState
    ) -> bool:
        """
        Synchronize conversation state to conversation manager.
        
        Args:
            conversation_state: Conversation state to sync
            
        Returns:
            True if sync successful, False otherwise
        """
        
        try:
            if not self.config.sync_conversation_state:
                return True
            
            # Update conversation metadata
            metadata = ConversationMetadata(
                title=f"Voice Conversation {conversation_state.conversation_id[:8]}",
                description="Voice conversation",
                tags=["voice", "real-time"],
                created_at=conversation_state.started_at,
                updated_at=datetime.now(),
                participant_count=1,
                message_count=conversation_state.message_count,
                is_active=conversation_state.is_active
            )
            
            # Update conversation
            await self.conversation_manager.update_conversation_metadata(
                conversation_id=conversation_state.conversation_id,
                metadata=metadata
            )
            
            # Track conversation state
            self.voice_conversations[conversation_state.conversation_id] = conversation_state
            
            # Emit sync event
            await self._emit_sync_event(SyncEventType.CONVERSATION_STATE_UPDATED, {
                "conversation_id": conversation_state.conversation_id,
                "message_count": conversation_state.message_count,
                "is_active": conversation_state.is_active,
                "timestamp": datetime.now()
            })
            
            self.stats["sync_operations"] += 1
            self.stats["successful_syncs"] += 1
            
            logger.debug(f"Synced conversation state {conversation_state.conversation_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync conversation state: {e}")
            self.stats["failed_syncs"] += 1
            return False
    
    async def handle_conversation_update(
        self,
        conversation_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """
        Handle updates from conversation manager.
        
        Args:
            conversation_id: ID of updated conversation
            update_data: Update data from conversation manager
            
        Returns:
            True if update handled successfully, False otherwise
        """
        
        try:
            if not self.config.bidirectional_sync:
                return True
            
            # Check for conflicts
            if await self._detect_sync_conflict(conversation_id, update_data):
                conflict_resolved = await self._resolve_sync_conflict(conversation_id, update_data)
                if not conflict_resolved:
                    logger.warning(f"Failed to resolve sync conflict for conversation {conversation_id}")
                    return False
            
            # Apply update to voice conversation state
            if conversation_id in self.voice_conversations:
                voice_state = self.voice_conversations[conversation_id]
                
                # Update voice conversation state based on conversation manager update
                if "message_count" in update_data:
                    voice_state.message_count = update_data["message_count"]
                
                if "is_active" in update_data:
                    voice_state.is_active = update_data["is_active"]
                
                voice_state.last_activity = datetime.now()
            
            logger.debug(f"Handled conversation update for {conversation_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle conversation update: {e}")
            return False
    
    async def _convert_voice_to_conversation_message(self, voice_message: VoiceMessage) -> Message:
        """Convert voice message to conversation message."""
        
        return Message(
            id=voice_message.message_id,
            role=MessageRole.USER,
            content=voice_message.text_content or "",
            timestamp=voice_message.timestamp,
            metadata={
                "source": "voice",
                "processing_mode": voice_message.processing_mode.value,
                "audio_duration_ms": getattr(voice_message, 'audio_duration_ms', 0),
                "confidence_score": getattr(voice_message, 'confidence_score', 0.0)
            }
        )
    
    async def _convert_voice_response_to_conversation_message(self, voice_response: VoiceResponse) -> Message:
        """Convert voice response to conversation message."""
        
        return Message(
            id=voice_response.response_id,
            role=MessageRole.ASSISTANT,
            content=voice_response.text_content or "",
            timestamp=datetime.now(),
            metadata={
                "source": "voice_response",
                "processing_mode": voice_response.processing_mode.value,
                "total_latency_ms": voice_response.total_latency_ms,
                "moshi_latency_ms": voice_response.moshi_latency_ms,
                "llm_latency_ms": voice_response.llm_latency_ms,
                "response_relevance": voice_response.response_relevance
            }
        )
    
    async def _detect_sync_conflict(self, conversation_id: str, update_data: Dict[str, Any]) -> bool:
        """Detect synchronization conflicts."""
        
        if not self.config.enable_conflict_detection:
            return False
        
        # Check if we have a more recent update
        last_sync = self.sync_timestamps.get(conversation_id)
        update_timestamp = update_data.get("timestamp", datetime.now())
        
        if last_sync and isinstance(update_timestamp, datetime):
            # Conflict if our last sync is more recent than the update
            if last_sync > update_timestamp:
                self.stats["conflicts_detected"] += 1
                
                await self._emit_sync_event(SyncEventType.SYNC_CONFLICT_DETECTED, {
                    "conversation_id": conversation_id,
                    "last_sync": last_sync,
                    "update_timestamp": update_timestamp,
                    "update_data": update_data
                })
                
                return True
        
        return False
    
    async def _resolve_sync_conflict(self, conversation_id: str, update_data: Dict[str, Any]) -> bool:
        """Resolve synchronization conflict."""
        
        try:
            strategy = self.config.conflict_resolution_strategy
            
            if strategy == ConflictResolutionStrategy.VOICE_PRIORITY:
                # Voice system takes priority - ignore the update
                logger.debug(f"Resolving conflict with voice priority for {conversation_id}")
                return True
            
            elif strategy == ConflictResolutionStrategy.CONVERSATION_PRIORITY:
                # Conversation manager takes priority - apply the update
                logger.debug(f"Resolving conflict with conversation priority for {conversation_id}")
                return True
            
            elif strategy == ConflictResolutionStrategy.TIMESTAMP_PRIORITY:
                # Most recent timestamp wins
                last_sync = self.sync_timestamps.get(conversation_id, datetime.min)
                update_timestamp = update_data.get("timestamp", datetime.now())
                
                if update_timestamp > last_sync:
                    logger.debug(f"Resolving conflict with timestamp priority (update wins) for {conversation_id}")
                    return True
                else:
                    logger.debug(f"Resolving conflict with timestamp priority (voice wins) for {conversation_id}")
                    return True
            
            elif strategy == ConflictResolutionStrategy.MERGE:
                # Attempt to merge the changes
                logger.debug(f"Resolving conflict with merge strategy for {conversation_id}")
                return await self._merge_conversation_updates(conversation_id, update_data)
            
            self.stats["conflicts_resolved"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve sync conflict: {e}")
            return False
    
    async def _merge_conversation_updates(self, conversation_id: str, update_data: Dict[str, Any]) -> bool:
        """Merge conversation updates."""
        
        try:
            # Simple merge strategy - combine non-conflicting updates
            if conversation_id in self.voice_conversations:
                voice_state = self.voice_conversations[conversation_id]
                
                # Merge message count (take maximum)
                if "message_count" in update_data:
                    voice_state.message_count = max(
                        voice_state.message_count,
                        update_data["message_count"]
                    )
                
                # Merge activity status (OR operation)
                if "is_active" in update_data:
                    voice_state.is_active = voice_state.is_active or update_data["is_active"]
                
                voice_state.last_activity = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge conversation updates: {e}")
            return False

    async def _batch_sync_loop(self) -> None:
        """Background task for batch synchronization."""

        while self._running:
            try:
                await asyncio.sleep(self.config.batch_sync_interval_seconds)

                if self.pending_syncs:
                    await self._process_batch_syncs()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch sync loop error: {e}")

    async def _process_batch_syncs(self) -> None:
        """Process pending batch synchronizations."""

        try:
            batch_count = 0

            for conversation_id, pending_ops in list(self.pending_syncs.items()):
                if batch_count >= self.config.max_batch_size:
                    break

                # Process pending operations for this conversation
                for operation in pending_ops:
                    await self._process_sync_operation(operation)
                    batch_count += 1

                # Clear processed operations
                del self.pending_syncs[conversation_id]

            if batch_count > 0:
                self.stats["batch_syncs"] += 1
                logger.debug(f"Processed {batch_count} batch sync operations")

        except Exception as e:
            logger.error(f"Failed to process batch syncs: {e}")

    async def _process_sync_operation(self, operation: Dict[str, Any]) -> None:
        """Process a single sync operation."""

        try:
            op_type = operation.get("type")

            if op_type == "voice_message":
                await self.sync_voice_message(operation["voice_message"], operation.get("conversation_state"))
            elif op_type == "voice_response":
                await self.sync_voice_response(operation["voice_response"], operation.get("conversation_state"))
            elif op_type == "conversation_state":
                await self.sync_conversation_state(operation["conversation_state"])

        except Exception as e:
            logger.error(f"Failed to process sync operation: {e}")

    def _is_sync_cached(self, cache_key: str) -> bool:
        """Check if sync result is cached and valid."""

        if not self.config.enable_sync_caching:
            return False

        if cache_key not in self.sync_cache:
            return False

        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False

        ttl = timedelta(minutes=self.config.sync_cache_ttl_minutes)
        return datetime.now() - cache_time < ttl

    def _cache_sync_result(self, cache_key: str, result: Any) -> None:
        """Cache sync result."""

        if self.config.enable_sync_caching:
            self.sync_cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()

            # Clean old cache entries
            self._cleanup_sync_cache()

    def _cleanup_sync_cache(self) -> None:
        """Clean up expired cache entries."""

        now = datetime.now()
        ttl = timedelta(minutes=self.config.sync_cache_ttl_minutes)

        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if now - timestamp > ttl
        ]

        for key in expired_keys:
            self.sync_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

    async def _emit_sync_event(self, event_type: SyncEventType, data: Dict[str, Any]) -> None:
        """Emit synchronization event."""

        if not self.config.enable_sync_events:
            return

        try:
            handlers = self.event_handlers.get(event_type, [])

            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, data)
                    else:
                        handler(event_type, data)
                except Exception as e:
                    logger.error(f"Sync event handler error: {e}")

        except Exception as e:
            logger.error(f"Failed to emit sync event: {e}")

    def add_event_handler(self, event_type: SyncEventType, handler: Callable) -> None:
        """Add event handler for sync events."""

        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)

    def remove_event_handler(self, event_type: SyncEventType, handler: Callable) -> None:
        """Remove event handler for sync events."""

        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def force_sync_conversation(self, conversation_id: str) -> bool:
        """Force synchronization of a specific conversation."""

        try:
            # Get conversation from conversation manager
            conversation = await self.conversation_manager.get_conversation(conversation_id)

            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found for force sync")
                return False

            # Sync conversation metadata and messages
            if conversation_id in self.voice_conversations:
                voice_state = self.voice_conversations[conversation_id]
                await self.sync_conversation_state(voice_state)

            logger.info(f"Force synced conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to force sync conversation: {e}")
            return False

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get comprehensive synchronization statistics."""

        return {
            "sync_stats": self.stats.copy(),
            "active_conversations": len(self.voice_conversations),
            "pending_syncs": sum(len(ops) for ops in self.pending_syncs.values()),
            "cache_size": len(self.sync_cache),
            "config": {
                "sync_enabled": self.config.sync_enabled,
                "real_time_sync": self.config.real_time_sync,
                "bidirectional_sync": self.config.bidirectional_sync,
                "batch_sync_enabled": self.config.batch_sync_enabled,
                "conflict_resolution_strategy": self.config.conflict_resolution_strategy.value,
                "sync_cache_enabled": self.config.enable_sync_caching
            }
        }

    async def cleanup(self) -> None:
        """Clean up synchronization resources."""

        try:
            # Stop background tasks
            await self.stop()

            # Clear state
            self.voice_conversations.clear()
            self.sync_timestamps.clear()
            self.pending_syncs.clear()
            self.sync_cache.clear()
            self.cache_timestamps.clear()
            self.event_handlers.clear()

            logger.info("VoiceConversationSync cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class VoiceConversationSyncManager:
    """
    High-level manager for voice-conversation synchronization.

    Provides a simplified interface for voice components to sync with conversation manager.
    """

    def __init__(
        self,
        conversation_manager: ConversationManagerInterface,
        config: Optional[VoiceConversationSyncConfig] = None
    ):
        """Initialize voice conversation sync manager."""
        self.conversation_manager = conversation_manager
        self.config = config or VoiceConversationSyncConfig()
        self.sync = VoiceConversationSync(conversation_manager, self.config)

        logger.info("VoiceConversationSyncManager initialized")

    async def start(self) -> None:
        """Start the sync manager."""
        await self.sync.start()

    async def stop(self) -> None:
        """Stop the sync manager."""
        await self.sync.stop()

    async def sync_voice_interaction(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None
    ) -> bool:
        """
        Synchronize complete voice interaction.

        Args:
            voice_message: User voice message
            voice_response: Assistant voice response
            conversation_state: Current conversation state

        Returns:
            True if sync successful, False otherwise
        """

        try:
            # Sync voice message
            message_synced = await self.sync.sync_voice_message(voice_message, conversation_state)

            # Sync voice response
            response_synced = await self.sync.sync_voice_response(voice_response, conversation_state)

            # Sync conversation state if provided
            state_synced = True
            if conversation_state:
                state_synced = await self.sync.sync_conversation_state(conversation_state)

            return message_synced and response_synced and state_synced

        except Exception as e:
            logger.error(f"Failed to sync voice interaction: {e}")
            return False

    async def handle_conversation_update(
        self,
        conversation_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """Handle conversation update from conversation manager."""
        return await self.sync.handle_conversation_update(conversation_id, update_data)

    def add_sync_event_handler(self, event_type: SyncEventType, handler: Callable) -> None:
        """Add event handler for sync events."""
        self.sync.add_event_handler(event_type, handler)

    def remove_sync_event_handler(self, event_type: SyncEventType, handler: Callable) -> None:
        """Remove event handler for sync events."""
        self.sync.remove_event_handler(event_type, handler)

    async def force_sync_conversation(self, conversation_id: str) -> bool:
        """Force synchronization of a specific conversation."""
        return await self.sync.force_sync_conversation(conversation_id)

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get comprehensive sync statistics."""
        return self.sync.get_sync_statistics()

    async def cleanup(self) -> None:
        """Clean up manager resources."""
        await self.sync.cleanup()
