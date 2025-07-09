"""
Conversation state management for Kyutai Moshi integration.

This module provides comprehensive conversation state tracking and management
for Moshi voice processing, including session persistence, analytics, and
real-time state synchronization.
"""

import asyncio
import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading

from .models import ConversationState, VoiceProcessingMode
from .inner_monologue import ExtractedText

logger = logging.getLogger(__name__)


class ConversationPhase(str, Enum):
    """Conversation phases."""
    INITIALIZING = "initializing"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING = "waiting"
    ENDING = "ending"
    ENDED = "ended"


class ConversationQuality(str, Enum):
    """Conversation quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class ConversationMetrics:
    """Detailed conversation metrics."""
    total_exchanges: int = 0
    total_audio_duration_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    average_response_latency_ms: float = 0.0
    average_confidence_score: float = 0.0
    interruption_count: int = 0
    silence_periods: int = 0
    quality_score: float = 0.0
    quality_level: ConversationQuality = ConversationQuality.UNKNOWN


@dataclass
class ConversationEvent:
    """Conversation event for tracking."""
    event_id: str
    conversation_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    phase: ConversationPhase
    user_id: Optional[str] = None


@dataclass
class ConversationSnapshot:
    """Snapshot of conversation state at a point in time."""
    snapshot_id: str
    conversation_id: str
    timestamp: datetime
    phase: ConversationPhase
    metrics: ConversationMetrics
    recent_events: List[ConversationEvent]
    moshi_state: Dict[str, Any]
    inner_monologue_buffer: List[ExtractedText]


class ConversationStateManager:
    """Manages conversation state for Moshi integration."""
    
    def __init__(self, max_conversations: int = 100):
        """Initialize conversation state manager."""
        self.max_conversations = max_conversations
        self.conversations: Dict[str, ConversationState] = {}
        self.conversation_metrics: Dict[str, ConversationMetrics] = {}
        self.conversation_events: Dict[str, List[ConversationEvent]] = {}
        self.conversation_snapshots: Dict[str, List[ConversationSnapshot]] = {}
        
        # Thread safety
        self.state_lock = threading.RLock()
        
        # Configuration
        self.max_events_per_conversation = 1000
        self.max_snapshots_per_conversation = 50
        self.snapshot_interval_seconds = 30
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.snapshot_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("ConversationStateManager initialized")
    
    async def start(self) -> None:
        """Start the conversation state manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.snapshot_task = asyncio.create_task(self._snapshot_loop())
        
        logger.info("ConversationStateManager started")
    
    async def stop(self) -> None:
        """Stop the conversation state manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.snapshot_task:
            self.snapshot_task.cancel()
            try:
                await self.snapshot_task
            except asyncio.CancelledError:
                pass
        
        logger.info("ConversationStateManager stopped")
    
    def create_conversation(
        self, 
        conversation_id: str, 
        user_id: Optional[str] = None,
        mode: VoiceProcessingMode = VoiceProcessingMode.MOSHI_ONLY
    ) -> ConversationState:
        """Create a new conversation state."""
        with self.state_lock:
            if conversation_id in self.conversations:
                raise ValueError(f"Conversation {conversation_id} already exists")
            
            # Create conversation state
            conversation_state = ConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                is_active=True,
                is_speaking=False,
                is_listening=True,
                started_at=datetime.now(),
                last_activity=datetime.now(),
                current_mode=mode,
                moshi_active=True
            )
            
            # Initialize metrics
            metrics = ConversationMetrics()
            
            # Store state
            self.conversations[conversation_id] = conversation_state
            self.conversation_metrics[conversation_id] = metrics
            self.conversation_events[conversation_id] = []
            self.conversation_snapshots[conversation_id] = []
            
            # Record creation event
            self._record_event(
                conversation_id=conversation_id,
                event_type="conversation_created",
                data={"user_id": user_id, "mode": mode.value},
                phase=ConversationPhase.INITIALIZING
            )
            
            logger.info(f"Created conversation state: {conversation_id}")
            return conversation_state
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state."""
        with self.state_lock:
            return self.conversations.get(conversation_id)
    
    def update_conversation_phase(self, conversation_id: str, phase: ConversationPhase) -> None:
        """Update conversation phase."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return
            
            conversation = self.conversations[conversation_id]
            conversation.last_activity = datetime.now()
            
            # Record phase change event
            self._record_event(
                conversation_id=conversation_id,
                event_type="phase_changed",
                data={"new_phase": phase.value},
                phase=phase
            )
            
            logger.debug(f"Conversation {conversation_id} phase changed to {phase}")
    
    def record_audio_exchange(
        self, 
        conversation_id: str, 
        audio_duration_ms: float,
        processing_latency_ms: float,
        confidence_score: Optional[float] = None
    ) -> None:
        """Record an audio exchange."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return
            
            conversation = self.conversations[conversation_id]
            metrics = self.conversation_metrics[conversation_id]
            
            # Update conversation state
            conversation.message_count += 1
            conversation.total_audio_duration_ms += audio_duration_ms
            conversation.last_activity = datetime.now()
            
            # Update metrics
            metrics.total_exchanges += 1
            metrics.total_audio_duration_ms += audio_duration_ms
            metrics.total_processing_time_ms += processing_latency_ms
            
            # Update average latency
            if metrics.total_exchanges == 1:
                metrics.average_response_latency_ms = processing_latency_ms
            else:
                metrics.average_response_latency_ms = (
                    (metrics.average_response_latency_ms * (metrics.total_exchanges - 1) + processing_latency_ms)
                    / metrics.total_exchanges
                )
            
            # Update average confidence
            if confidence_score is not None:
                if metrics.total_exchanges == 1:
                    metrics.average_confidence_score = confidence_score
                else:
                    metrics.average_confidence_score = (
                        (metrics.average_confidence_score * (metrics.total_exchanges - 1) + confidence_score)
                        / metrics.total_exchanges
                    )
            
            # Update conversation average latency
            conversation.average_latency_ms = metrics.average_response_latency_ms
            
            # Record exchange event
            self._record_event(
                conversation_id=conversation_id,
                event_type="audio_exchange",
                data={
                    "audio_duration_ms": audio_duration_ms,
                    "processing_latency_ms": processing_latency_ms,
                    "confidence_score": confidence_score
                },
                phase=ConversationPhase.PROCESSING
            )
    
    def record_interruption(self, conversation_id: str) -> None:
        """Record a conversation interruption."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                return
            
            metrics = self.conversation_metrics[conversation_id]
            metrics.interruption_count += 1
            
            self._record_event(
                conversation_id=conversation_id,
                event_type="interruption",
                data={"interruption_count": metrics.interruption_count},
                phase=ConversationPhase.PROCESSING
            )
    
    def record_silence_period(self, conversation_id: str, duration_ms: float) -> None:
        """Record a silence period."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                return
            
            metrics = self.conversation_metrics[conversation_id]
            metrics.silence_periods += 1
            
            self._record_event(
                conversation_id=conversation_id,
                event_type="silence_period",
                data={"duration_ms": duration_ms, "silence_count": metrics.silence_periods},
                phase=ConversationPhase.WAITING
            )
    
    def update_moshi_state(self, conversation_id: str, moshi_state: Dict[str, Any]) -> None:
        """Update Moshi-specific state."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                return
            
            conversation = self.conversations[conversation_id]
            conversation.last_activity = datetime.now()
            
            # Store Moshi state in memory context
            if conversation.memory_context is None:
                conversation.memory_context = {}
            
            conversation.memory_context['moshi_state'] = moshi_state
            
            self._record_event(
                conversation_id=conversation_id,
                event_type="moshi_state_updated",
                data=moshi_state,
                phase=ConversationPhase.PROCESSING
            )
    
    def add_inner_monologue_text(self, conversation_id: str, extracted_text: ExtractedText) -> None:
        """Add extracted text from inner monologue."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                return
            
            # Store in conversation memory context
            conversation = self.conversations[conversation_id]
            if conversation.memory_context is None:
                conversation.memory_context = {}
            
            if 'inner_monologue' not in conversation.memory_context:
                conversation.memory_context['inner_monologue'] = []
            
            # Keep only recent inner monologue entries
            inner_monologue = conversation.memory_context['inner_monologue']
            inner_monologue.append({
                'text': extracted_text.text,
                'confidence': extracted_text.confidence,
                'timestamp': extracted_text.timestamp.isoformat(),
                'audio_duration_ms': extracted_text.audio_duration_ms
            })
            
            # Keep only last 50 entries
            if len(inner_monologue) > 50:
                inner_monologue.pop(0)
            
            self._record_event(
                conversation_id=conversation_id,
                event_type="inner_monologue_text",
                data={
                    'text': extracted_text.text,
                    'confidence': extracted_text.confidence
                },
                phase=ConversationPhase.PROCESSING
            )
    
    def end_conversation(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """End a conversation and return final metrics."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return None
            
            conversation = self.conversations[conversation_id]
            metrics = self.conversation_metrics[conversation_id]
            
            # Update conversation state
            conversation.is_active = False
            conversation.last_activity = datetime.now()
            
            # Calculate final metrics
            total_duration = (conversation.last_activity - conversation.started_at).total_seconds() * 1000
            conversation.total_duration_ms = int(total_duration)
            
            # Calculate quality score
            metrics.quality_score = self._calculate_quality_score(metrics)
            metrics.quality_level = self._determine_quality_level(metrics.quality_score)
            
            # Record end event
            self._record_event(
                conversation_id=conversation_id,
                event_type="conversation_ended",
                data={
                    "total_duration_ms": conversation.total_duration_ms,
                    "quality_score": metrics.quality_score,
                    "quality_level": metrics.quality_level.value
                },
                phase=ConversationPhase.ENDED
            )
            
            # Create final snapshot
            self._create_snapshot(conversation_id)
            
            logger.info(f"Ended conversation {conversation_id}, quality: {metrics.quality_level}")
            return metrics
    
    def get_conversation_metrics(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """Get conversation metrics."""
        with self.state_lock:
            return self.conversation_metrics.get(conversation_id)
    
    def get_conversation_events(
        self, 
        conversation_id: str, 
        event_types: Optional[Set[str]] = None,
        limit: Optional[int] = None
    ) -> List[ConversationEvent]:
        """Get conversation events."""
        with self.state_lock:
            if conversation_id not in self.conversation_events:
                return []
            
            events = self.conversation_events[conversation_id]
            
            # Filter by event types if specified
            if event_types:
                events = [event for event in events if event.event_type in event_types]
            
            # Apply limit
            if limit:
                events = events[-limit:]
            
            return events.copy()
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        with self.state_lock:
            return [
                conv_id for conv_id, conv in self.conversations.items()
                if conv.is_active
            ]
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive conversation summary."""
        with self.state_lock:
            if conversation_id not in self.conversations:
                return None
            
            conversation = self.conversations[conversation_id]
            metrics = self.conversation_metrics[conversation_id]
            recent_events = self.conversation_events[conversation_id][-10:]  # Last 10 events
            
            return {
                'conversation': conversation.model_dump() if hasattr(conversation, 'model_dump') else asdict(conversation),
                'metrics': asdict(metrics),
                'recent_events': [asdict(event) for event in recent_events],
                'event_count': len(self.conversation_events[conversation_id]),
                'snapshot_count': len(self.conversation_snapshots.get(conversation_id, []))
            }
    
    def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """Cleanup old conversations."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self.state_lock:
            conversations_to_remove = []
            
            for conv_id, conversation in self.conversations.items():
                if not conversation.is_active and conversation.last_activity < cutoff_time:
                    conversations_to_remove.append(conv_id)
            
            for conv_id in conversations_to_remove:
                self._remove_conversation_data(conv_id)
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old conversations")
        
        return cleaned_count
    
    # Private helper methods
    
    def _record_event(
        self, 
        conversation_id: str, 
        event_type: str, 
        data: Dict[str, Any],
        phase: ConversationPhase
    ) -> None:
        """Record a conversation event."""
        if conversation_id not in self.conversation_events:
            return
        
        event = ConversationEvent(
            event_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            phase=phase,
            user_id=self.conversations[conversation_id].user_id
        )
        
        events = self.conversation_events[conversation_id]
        events.append(event)
        
        # Maintain event limit
        if len(events) > self.max_events_per_conversation:
            events.pop(0)
    
    def _create_snapshot(self, conversation_id: str) -> None:
        """Create a conversation snapshot."""
        if conversation_id not in self.conversations:
            return
        
        conversation = self.conversations[conversation_id]
        metrics = self.conversation_metrics[conversation_id]
        recent_events = self.conversation_events[conversation_id][-20:]  # Last 20 events
        
        # Get Moshi state from memory context
        moshi_state = {}
        if conversation.memory_context and 'moshi_state' in conversation.memory_context:
            moshi_state = conversation.memory_context['moshi_state']
        
        # Get inner monologue buffer
        inner_monologue_buffer = []
        if conversation.memory_context and 'inner_monologue' in conversation.memory_context:
            inner_monologue_buffer = conversation.memory_context['inner_monologue']
        
        snapshot = ConversationSnapshot(
            snapshot_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            phase=ConversationPhase.PROCESSING,  # Current phase would be tracked separately
            metrics=metrics,
            recent_events=recent_events,
            moshi_state=moshi_state,
            inner_monologue_buffer=inner_monologue_buffer
        )
        
        snapshots = self.conversation_snapshots[conversation_id]
        snapshots.append(snapshot)
        
        # Maintain snapshot limit
        if len(snapshots) > self.max_snapshots_per_conversation:
            snapshots.pop(0)
    
    def _calculate_quality_score(self, metrics: ConversationMetrics) -> float:
        """Calculate conversation quality score (0.0 to 1.0)."""
        score = 0.0
        
        # Base score from confidence
        if metrics.average_confidence_score > 0:
            score += metrics.average_confidence_score * 0.4
        
        # Latency score (lower is better)
        if metrics.average_response_latency_ms > 0:
            latency_score = max(0, 1.0 - (metrics.average_response_latency_ms / 1000.0))  # 1s = 0 score
            score += latency_score * 0.3
        
        # Exchange count score
        if metrics.total_exchanges > 0:
            exchange_score = min(1.0, metrics.total_exchanges / 10.0)  # 10 exchanges = full score
            score += exchange_score * 0.2
        
        # Interruption penalty
        if metrics.total_exchanges > 0:
            interruption_rate = metrics.interruption_count / metrics.total_exchanges
            interruption_penalty = min(0.1, interruption_rate * 0.1)
            score -= interruption_penalty
        
        return max(0.0, min(1.0, score))
    
    def _determine_quality_level(self, score: float) -> ConversationQuality:
        """Determine quality level from score."""
        if score >= 0.8:
            return ConversationQuality.EXCELLENT
        elif score >= 0.6:
            return ConversationQuality.GOOD
        elif score >= 0.4:
            return ConversationQuality.FAIR
        elif score >= 0.2:
            return ConversationQuality.POOR
        else:
            return ConversationQuality.UNKNOWN
    
    def _remove_conversation_data(self, conversation_id: str) -> None:
        """Remove all data for a conversation."""
        self.conversations.pop(conversation_id, None)
        self.conversation_metrics.pop(conversation_id, None)
        self.conversation_events.pop(conversation_id, None)
        self.conversation_snapshots.pop(conversation_id, None)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.is_running:
            try:
                # Cleanup old conversations every hour
                self.cleanup_old_conversations(max_age_hours=24)
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _snapshot_loop(self) -> None:
        """Background snapshot creation loop."""
        while self.is_running:
            try:
                # Create snapshots for active conversations
                with self.state_lock:
                    active_conversations = self.get_active_conversations()
                
                for conv_id in active_conversations:
                    self._create_snapshot(conv_id)
                
                # Sleep for snapshot interval
                await asyncio.sleep(self.snapshot_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retrying
