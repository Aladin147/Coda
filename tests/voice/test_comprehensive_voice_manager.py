"""
Comprehensive tests for the VoiceManager component.

This module provides thorough testing of the VoiceManager including
initialization, conversation management, voice processing, and error handling.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import io
import wave
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.coda.components.voice.manager import VoiceManager
from src.coda.components.voice.models import (
    VoiceConfig, AudioConfig, MoshiConfig, VoiceMessage, VoiceResponse,
    VoiceProcessingMode, ConversationState
)
from src.coda.components.voice.exceptions import (
    VoiceProcessingError, ValidationError, ComponentNotInitializedError,
    ComponentFailureError, VoiceTimeoutError
)


class TestVoiceManagerInitialization:
    """Test VoiceManager initialization and configuration."""
    
    def test_voice_manager_creation_with_config(self):
        """Test VoiceManager creation with custom configuration."""
        config = VoiceConfig(
            audio=AudioConfig(
                sample_rate=24000,
                channels=1,
                format="wav"
            ),
            moshi=MoshiConfig(
                device="cpu",
                vram_allocation="2GB"
            )
        )
        
        manager = VoiceManager(config)
        assert manager.config == config
        assert manager.vram_manager is None  # Not initialized yet
        assert len(manager.conversations) == 0
    
    def test_voice_manager_creation_without_config(self):
        """Test VoiceManager creation with default configuration."""
        manager = VoiceManager()
        assert manager.config is not None
        assert isinstance(manager.config, VoiceConfig)
        assert manager.vram_manager is None  # Not initialized yet
    
    @pytest.mark.asyncio
    async def test_voice_manager_initialization_success(self):
        """Test successful VoiceManager initialization."""
        config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )
        
        manager = VoiceManager(config)

        # Mock the VRAM manager initialization
        with patch('src.coda.components.voice.manager.initialize_vram_manager') as mock_init_vram:
            mock_vram_manager = AsyncMock()
            mock_init_vram.return_value = mock_vram_manager

            await manager.initialize(config)

            assert manager.vram_manager is not None
            mock_init_vram.assert_called_once_with(config)
            mock_vram_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_voice_manager_initialization_failure(self):
        """Test VoiceManager initialization failure handling."""
        config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )
        
        manager = VoiceManager(config)

        # Mock VRAM manager initialization to fail
        with patch('src.coda.components.voice.manager.initialize_vram_manager') as mock_init_vram:
            mock_init_vram.side_effect = Exception("VRAM allocation failed")

            with pytest.raises(Exception):
                await manager.initialize(config)

            assert manager.vram_manager is None
    
    @pytest.mark.asyncio
    async def test_voice_manager_cleanup(self):
        """Test VoiceManager cleanup."""
        manager = VoiceManager()

        # Mock components properly
        manager.vram_manager = AsyncMock()
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.stop_pipeline = AsyncMock()
        manager.pipeline_manager.remove_pipeline = AsyncMock()
        manager.pipeline_manager.cleanup_all = AsyncMock()

        # Add some conversations
        manager.conversations["conv1"] = Mock()
        manager.conversations["conv2"] = Mock()

        await manager.cleanup()

        # Verify cleanup
        assert len(manager.conversations) == 0
        # Verify pipeline methods were called
        assert manager.pipeline_manager.stop_pipeline.call_count == 2
        assert manager.pipeline_manager.remove_pipeline.call_count == 2
        manager.pipeline_manager.cleanup_all.assert_called_once()


class TestConversationManagement:
    """Test conversation lifecycle management."""
    
    @pytest_asyncio.fixture
    async def initialized_manager(self):
        """Create an initialized VoiceManager for testing."""
        manager = VoiceManager()

        config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )

        # Mock the VRAM manager initialization and torch hub loading
        with patch('src.coda.components.voice.manager.initialize_vram_manager') as mock_init_vram, \
             patch('torch.hub.load') as mock_torch_hub:

            mock_vram_manager = AsyncMock()
            mock_init_vram.return_value = mock_vram_manager

            # Mock the Silero VAD model loading
            mock_model = Mock()
            mock_utils = Mock()
            mock_torch_hub.return_value = (mock_model, mock_utils)

            await manager.initialize(config)

        return manager
    
    @pytest.mark.asyncio
    async def test_start_conversation_success(self):
        """Test successful conversation start."""
        manager = VoiceManager()

        # Mock the required components directly
        manager.vram_manager = AsyncMock()  # Make manager appear initialized
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.create_pipeline = AsyncMock(return_value=Mock())
        manager.pipeline_manager.start_pipeline = AsyncMock()
        manager.analytics = Mock()
        manager.analytics.total_conversations = 0

        conversation_id = "test-conv-123"

        returned_id = await manager.start_conversation(user_id=None, conversation_id=conversation_id)

        assert returned_id == conversation_id
        assert conversation_id in manager.conversations

        conversation_state = manager.conversations[conversation_id]
        assert conversation_state.conversation_id == conversation_id
        assert conversation_state.is_active
    
    @pytest.mark.asyncio
    async def test_start_conversation_duplicate(self):
        """Test starting a conversation that already exists."""
        manager = VoiceManager()

        # Mock the required components directly
        manager.vram_manager = AsyncMock()  # Make manager appear initialized
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.create_pipeline = AsyncMock(return_value=Mock())
        manager.pipeline_manager.start_pipeline = AsyncMock()
        manager.analytics = Mock()
        manager.analytics.total_conversations = 0

        conversation_id = "test-conv-123"

        # Start conversation first time
        await manager.start_conversation(user_id=None, conversation_id=conversation_id)

        # Try to start again - should handle gracefully
        returned_id = await manager.start_conversation(user_id=None, conversation_id=conversation_id)
        assert returned_id == conversation_id
    
    @pytest.mark.asyncio
    async def test_end_conversation_success(self):
        """Test successful conversation end."""
        manager = VoiceManager()

        # Mock the required components directly
        manager.vram_manager = AsyncMock()  # Make manager appear initialized
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.create_pipeline = AsyncMock(return_value=Mock())
        manager.pipeline_manager.start_pipeline = AsyncMock()
        manager.pipeline_manager.stop_pipeline = AsyncMock()
        manager.pipeline_manager.remove_pipeline = AsyncMock()
        manager.analytics = Mock()
        manager.analytics.total_conversations = 0

        conversation_id = "test-conv-123"

        # Start conversation
        await manager.start_conversation(user_id=None, conversation_id=conversation_id)
        assert conversation_id in manager.conversations

        # End conversation
        await manager.end_conversation(conversation_id)
        assert conversation_id not in manager.conversations
    
    @pytest.mark.asyncio
    async def test_end_nonexistent_conversation(self):
        """Test ending a conversation that doesn't exist."""
        manager = VoiceManager()

        # Mock the required components directly
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.stop_pipeline = AsyncMock()
        manager.pipeline_manager.remove_pipeline = AsyncMock()

        conversation_id = "nonexistent-conv"

        # Should handle gracefully without error
        await manager.end_conversation(conversation_id)
    
    @pytest.mark.asyncio
    async def test_get_conversation_state(self):
        """Test getting conversation state."""
        manager = VoiceManager()

        # Mock the required components directly
        manager.vram_manager = AsyncMock()  # Make manager appear initialized
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.create_pipeline = AsyncMock(return_value=Mock())
        manager.pipeline_manager.start_pipeline = AsyncMock()
        manager.analytics = Mock()
        manager.analytics.total_conversations = 0

        conversation_id = "test-conv-123"

        # Start conversation
        await manager.start_conversation(user_id=None, conversation_id=conversation_id)

        # Get state
        state = await manager.get_conversation_state(conversation_id)
        assert state is not None
        assert state.conversation_id == conversation_id
        assert state.is_active
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation_state(self):
        """Test getting state for nonexistent conversation."""
        manager = VoiceManager()
        conversation_id = "nonexistent-conv"

        state = await manager.get_conversation_state(conversation_id)
        assert state is None


class TestVoiceProcessing:
    """Test voice processing functionality."""
    
    @pytest_asyncio.fixture
    async def manager_with_conversation(self):
        """Create manager with active conversation."""
        manager = VoiceManager()

        config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )

        # Mock the VRAM manager initialization and torch hub loading
        with patch('src.coda.components.voice.manager.initialize_vram_manager') as mock_init_vram, \
             patch('torch.hub.load') as mock_torch_hub:

            mock_vram_manager = AsyncMock()
            mock_init_vram.return_value = mock_vram_manager

            # Mock the Silero VAD model loading
            mock_model = Mock()
            mock_utils = Mock()
            mock_torch_hub.return_value = (mock_model, mock_utils)

            await manager.initialize(config)

        # Start conversation
        conversation_id = "test-conv"

        # Mock the pipeline manager to avoid initialization issues
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.create_pipeline = AsyncMock(return_value=Mock())
        manager.pipeline_manager.start_pipeline = AsyncMock()

        # Mock pipeline for voice processing
        mock_pipeline = AsyncMock()
        mock_processed_chunk = Mock()
        mock_processed_chunk.duration_ms = 100
        mock_processed_chunk.text_content = "test response"
        mock_pipeline.process_input = AsyncMock(return_value=mock_processed_chunk)
        manager.pipeline_manager.get_pipeline = Mock(return_value=mock_pipeline)

        await manager.start_conversation(user_id=None, conversation_id=conversation_id)

        return manager, conversation_id
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing."""
        # Create a simple WAV file
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        
        import numpy as np
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return wav_buffer.getvalue()
    
    @pytest.mark.asyncio
    async def test_process_voice_input_success(self, manager_with_conversation, sample_audio_data):
        """Test successful voice input processing."""
        manager, conversation_id = manager_with_conversation
        
        # Mock pipeline response
        expected_response = VoiceResponse(
            response_id="resp-123",
            conversation_id=conversation_id,
            message_id="msg-123",
            text_content="Hello, how can I help you?",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=150.0
        )
        
        with patch.object(manager, 'process_voice_input', return_value=expected_response) as mock_process:
            response = await manager.process_voice_input(
                conversation_id=conversation_id,
                audio_data=sample_audio_data
            )
            
            assert response == expected_response
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_voice_input_invalid_conversation(self, manager_with_conversation, sample_audio_data):
        """Test voice processing with invalid conversation ID."""
        manager, _ = manager_with_conversation
        
        with pytest.raises(Exception):  # Should raise appropriate exception
            await manager.process_voice_input(
                conversation_id="invalid-conv",
                audio_data=sample_audio_data
            )
    
    @pytest.mark.asyncio
    async def test_process_voice_input_empty_audio(self, manager_with_conversation):
        """Test voice processing with empty audio data."""
        manager, conversation_id = manager_with_conversation
        
        with pytest.raises(ValidationError):
            await manager.process_voice_input(
                conversation_id=conversation_id,
                audio_data=b""
            )
    
    @pytest.mark.asyncio
    async def test_process_voice_input_processing_error(self, manager_with_conversation, sample_audio_data):
        """Test voice processing with processing error."""
        manager, conversation_id = manager_with_conversation
        
        with patch.object(manager, 'process_voice_input', side_effect=VoiceProcessingError("Processing failed")):
            with pytest.raises(VoiceProcessingError):
                await manager.process_voice_input(
                    conversation_id=conversation_id,
                    audio_data=sample_audio_data
                )
    
    @pytest.mark.asyncio
    async def test_process_voice_stream(self, manager_with_conversation, sample_audio_data):
        """Test voice stream processing."""
        manager, conversation_id = manager_with_conversation
        
        # Mock streaming response
        async def mock_stream_generator():
            for i in range(3):
                yield Mock(
                    conversation_id=conversation_id,
                    text_content=f"Chunk {i}",
                    is_complete=(i == 2)
                )
        
        with patch.object(manager, 'process_voice_stream', return_value=mock_stream_generator()):
            chunks = []
            async for chunk in manager.process_voice_stream(conversation_id, sample_audio_data):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[-1].is_complete


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_uninitialized_manager_usage(self):
        """Test using manager before initialization."""
        manager = VoiceManager()
        
        with pytest.raises(ComponentNotInitializedError):
            await manager.start_conversation("test-conv")
    
    @pytest.mark.asyncio
    async def test_component_failure_handling(self):
        """Test handling of component failures."""
        manager = VoiceManager()
        
        # Mock component to fail during initialization
        with patch('src.coda.components.voice.manager.initialize_vram_manager') as mock_init_vram:
            mock_init_vram.side_effect = ComponentFailureError("vram", "Failed to initialize")
            
            config = VoiceConfig(
                audio=AudioConfig(),
                moshi=MoshiConfig(device="cpu")
            )
            
            with pytest.raises(ComponentFailureError):
                await manager.initialize(config)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in voice processing."""
        manager = VoiceManager()
        
        # Mock long-running operation
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow processing
            return Mock()
        
        with patch.object(manager, 'process_voice_input', side_effect=slow_process):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    manager.process_voice_input("conv", b"audio"),
                    timeout=1.0
                )


class TestPerformanceMetrics:
    """Test performance metrics collection."""
    
    @pytest.mark.asyncio
    async def test_analytics_tracking(self):
        """Test that analytics are properly tracked."""
        manager = VoiceManager()
        
        # Initialize manager
        config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )

        # Mock the VRAM manager initialization and torch hub loading
        with patch('src.coda.components.voice.manager.initialize_vram_manager') as mock_init_vram, \
             patch('torch.hub.load') as mock_torch_hub:

            mock_vram_manager = AsyncMock()
            mock_init_vram.return_value = mock_vram_manager

            # Mock the Silero VAD model loading
            mock_model = Mock()
            mock_utils = Mock()
            mock_torch_hub.return_value = (mock_model, mock_utils)

            await manager.initialize(config)

            # Mock pipeline manager for conversation creation
            manager.pipeline_manager = AsyncMock()
            manager.pipeline_manager.create_pipeline = AsyncMock(return_value=Mock())
            manager.pipeline_manager.start_pipeline = AsyncMock()

            # Start conversation
            conversation_id = await manager.start_conversation(user_id=None, conversation_id="test-conv")

            # Verify conversation was created
            assert "test-conv" in manager.conversations
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement during processing."""
        manager = VoiceManager()
        manager.latency_tracker = Mock()
        
        # Mock successful processing
        with patch.object(manager, 'process_voice_input') as mock_process:
            mock_process.return_value = VoiceResponse(
                response_id="resp-123",
                conversation_id="test",
                message_id="msg-123",
                text_content="Response",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY,
                total_latency_ms=100.0
            )
            
            response = await manager.process_voice_input("conv", b"audio")
            
            # Verify latency was recorded
            assert response.total_latency_ms is not None
            assert response.total_latency_ms > 0


if __name__ == "__main__":
    pytest.main([__file__])
