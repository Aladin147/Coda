"""
Voice manager implementation for Coda 2.0 voice system.

This module provides the main voice manager that orchestrates all voice processing
components and integrates with other Coda systems.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from .interfaces import VoiceManagerInterface
from .models import (
    VoiceConfig,
    VoiceMessage,
    VoiceResponse,
    VoiceStreamChunk,
    ConversationState,
    VoiceAnalytics,
    VoiceProcessingMode
)
from .audio_processor import AudioProcessor
from .pipeline import AudioPipeline, PipelineManager
from .vram_manager import DynamicVRAMManager, initialize_vram_manager
from .config import load_voice_config
from .utils import PerformanceMonitor, LatencyTracker

logger = logging.getLogger(__name__)


class VoiceManager(VoiceManagerInterface):
    """Main voice manager for orchestrating voice processing."""
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        """Initialize voice manager."""
        self.config = config or load_voice_config("development")
        self.conversations: Dict[str, ConversationState] = {}
        self.pipeline_manager = PipelineManager()
        self.vram_manager: Optional[DynamicVRAMManager] = None
        self.performance_monitor = PerformanceMonitor()
        
        # Integration managers (will be set later)
        self.memory_manager = None
        self.personality_manager = None
        self.tool_manager = None
        
        # Analytics
        self.analytics = VoiceAnalytics()
        self.latency_tracker = LatencyTracker("voice_processing")
        
        logger.info("VoiceManager initialized")
    
    async def initialize(self, config: VoiceConfig) -> None:
        """Initialize the voice manager."""
        try:
            self.config = config
            
            # Initialize VRAM manager
            self.vram_manager = initialize_vram_manager(config)
            await self.vram_manager.initialize()
            
            # Set default pipeline configuration
            self.pipeline_manager.set_default_config(config.audio)
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            logger.info("VoiceManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VoiceManager: {e}")
            raise
    
    async def start_conversation(
        self, 
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """Start a new voice conversation."""
        try:
            # Generate conversation ID if not provided
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
            
            # Check if conversation already exists
            if conversation_id in self.conversations:
                logger.warning(f"Conversation {conversation_id} already exists")
                return conversation_id
            
            # Create conversation state
            conversation_state = ConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                is_active=True,
                current_mode=self.config.mode,
                started_at=datetime.now()
            )
            
            self.conversations[conversation_id] = conversation_state
            
            # Create audio pipeline for this conversation
            pipeline = await self.pipeline_manager.create_pipeline(
                pipeline_id=conversation_id,
                config=self.config.audio
            )
            
            await self.pipeline_manager.start_pipeline(conversation_id)
            
            # Update analytics
            self.analytics.total_conversations += 1
            
            logger.info(f"Started voice conversation: {conversation_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise
    
    async def process_voice_input(
        self, 
        conversation_id: str, 
        audio_data: bytes
    ) -> VoiceResponse:
        """Process voice input and generate response."""
        try:
            # Check if conversation exists
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            conversation = self.conversations[conversation_id]
            
            # Create voice message
            message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=conversation.user_id,
                audio_data=audio_data,
                processing_mode=conversation.current_mode,
                timestamp=datetime.now()
            )
            
            # Start latency tracking
            self.latency_tracker.start()
            
            # Get pipeline for this conversation
            pipeline = self.pipeline_manager.get_pipeline(conversation_id)
            if not pipeline:
                raise RuntimeError(f"No pipeline found for conversation {conversation_id}")
            
            # Process audio through pipeline
            processed_chunk = await pipeline.process_input(audio_data)
            
            # For now, create a simple response (will be enhanced in later phases)
            response_text = "Voice processing response placeholder"
            response_audio = audio_data  # Echo for now
            
            # Stop latency tracking
            latency_ms = self.latency_tracker.stop()
            
            # Create response
            response = VoiceResponse(
                response_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                message_id=message.message_id,
                audio_data=response_audio,
                text_content=response_text,
                processing_mode=conversation.current_mode,
                total_latency_ms=latency_ms,
                timestamp=datetime.now()
            )
            
            # Update conversation state
            conversation.message_count += 1
            conversation.last_activity = datetime.now()
            conversation.total_audio_duration_ms += processed_chunk.duration_ms
            
            # Update analytics
            self.analytics.total_messages += 1
            self.analytics.total_audio_duration_ms += processed_chunk.duration_ms
            
            # Update average latency
            if self.analytics.total_messages > 0:
                self.analytics.average_latency_ms = (
                    (self.analytics.average_latency_ms * (self.analytics.total_messages - 1) + latency_ms) 
                    / self.analytics.total_messages
                )
            
            logger.debug(f"Processed voice input for {conversation_id}, latency: {latency_ms:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process voice input: {e}")
            raise
    
    async def process_voice_stream(
        self, 
        conversation_id: str, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[VoiceStreamChunk, None]:
        """Process streaming voice input."""
        try:
            # Check if conversation exists
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            # Get pipeline for this conversation
            pipeline = self.pipeline_manager.get_pipeline(conversation_id)
            if not pipeline:
                raise RuntimeError(f"No pipeline found for conversation {conversation_id}")
            
            sequence_number = 0
            
            # Process audio stream
            async for audio_chunk in audio_stream:
                try:
                    # Process chunk through pipeline
                    processed_chunk = await pipeline.process_input(audio_chunk)
                    
                    # Create stream chunk
                    stream_chunk = VoiceStreamChunk(
                        chunk_id=f"{conversation_id}_{sequence_number}",
                        conversation_id=conversation_id,
                        sequence_number=sequence_number,
                        audio_data=processed_chunk.data,
                        timestamp=datetime.now(),
                        is_final=False,
                        chunk_type="audio"
                    )
                    
                    sequence_number += 1
                    yield stream_chunk
                    
                except Exception as e:
                    logger.error(f"Failed to process stream chunk: {e}")
                    continue
            
            # Send final chunk
            final_chunk = VoiceStreamChunk(
                chunk_id=f"{conversation_id}_{sequence_number}",
                conversation_id=conversation_id,
                sequence_number=sequence_number,
                is_final=True,
                chunk_type="control",
                timestamp=datetime.now()
            )
            
            yield final_chunk
            
        except Exception as e:
            logger.error(f"Failed to process voice stream: {e}")
            raise
    
    async def end_conversation(self, conversation_id: str) -> None:
        """End a voice conversation."""
        try:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return
            
            # Stop and remove pipeline
            await self.pipeline_manager.stop_pipeline(conversation_id)
            await self.pipeline_manager.remove_pipeline(conversation_id)
            
            # Update conversation state
            conversation = self.conversations[conversation_id]
            conversation.is_active = False
            
            # Remove from active conversations
            del self.conversations[conversation_id]
            
            logger.info(f"Ended voice conversation: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            raise
    
    async def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        return [
            conv_id for conv_id, conv in self.conversations.items()
            if conv.is_active
        ]
    
    async def get_conversation_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state."""
        return self.conversations.get(conversation_id)
    
    async def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for integration."""
        self.memory_manager = memory_manager
        logger.info("Memory manager integration set")
    
    async def set_personality_manager(self, personality_manager: Any) -> None:
        """Set personality manager for integration."""
        self.personality_manager = personality_manager
        logger.info("Personality manager integration set")
    
    async def set_tool_manager(self, tool_manager: Any) -> None:
        """Set tool manager for integration."""
        self.tool_manager = tool_manager
        logger.info("Tool manager integration set")
    
    async def get_analytics(self) -> VoiceAnalytics:
        """Get voice processing analytics."""
        # Update current metrics
        if self.vram_manager:
            vram_summary = self.vram_manager.get_allocation_summary()
            self.analytics.vram_usage = {
                comp: f"{alloc['allocated_mb']:.0f}MB"
                for comp, alloc in vram_summary['components'].items()
            }
        
        # Update performance metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        self.analytics.cpu_usage = current_metrics.cpu_usage
        
        return self.analytics
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get voice system status."""
        status = {
            'active_conversations': len(self.conversations),
            'total_conversations': self.analytics.total_conversations,
            'total_messages': self.analytics.total_messages,
            'average_latency_ms': self.analytics.average_latency_ms,
            'processing_mode': self.config.mode.value,
            'pipelines': self.pipeline_manager.get_all_stats()
        }
        
        if self.vram_manager:
            status['vram'] = self.vram_manager.get_allocation_summary()
        
        return status
    
    async def cleanup(self) -> None:
        """Cleanup voice manager resources."""
        try:
            # End all conversations
            for conversation_id in list(self.conversations.keys()):
                await self.end_conversation(conversation_id)
            
            # Cleanup pipeline manager
            await self.pipeline_manager.cleanup_all()
            
            # Cleanup VRAM manager
            if self.vram_manager:
                await self.vram_manager.cleanup()
            
            logger.info("VoiceManager cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup VoiceManager: {e}")


# Placeholder classes for missing components (will be implemented in later phases)

class MoshiVoiceProcessor:
    """Placeholder for Moshi voice processor."""
    pass


class HybridVoiceProcessor:
    """Placeholder for hybrid voice processor."""
    pass


class WebSocketVoiceManager:
    """Placeholder for WebSocket voice manager."""
    pass
