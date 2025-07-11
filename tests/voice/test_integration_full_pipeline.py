"""
Comprehensive integration tests for the voice processing system.

These tests verify end-to-end functionality across all components,
including audio processing, model integration, and system coordination.
"""

import pytest
import asyncio
import io
import wave
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.coda.components.voice.manager import VoiceManager
from src.coda.components.voice.models import (
    VoiceConfig, AudioConfig, MoshiConfig, VoiceMessage, VoiceResponse,
    VoiceProcessingMode, ConversationState
)
from src.coda.components.voice.exceptions import (
    VoiceProcessingError, ValidationError, ComponentNotInitializedError
)


class TestVoiceSystemIntegration:
    """Integration tests for the complete voice processing system."""
    
    @pytest.fixture
    async def voice_manager(self):
        """Create and initialize a voice manager for testing."""
        config = VoiceConfig(
            audio=AudioConfig(
                sample_rate=16000,
                channels=1,
                format="wav"
            ),
            moshi=MoshiConfig(
                device="cpu",
                vram_allocation="1GB"
            )
        )
        
        manager = VoiceManager(config)
        
        # Mock external dependencies
        manager.vram_manager = Mock()
        manager.vram_manager.register_component.return_value = True
        manager.vram_manager.allocate.return_value = True
        
        # Mock integration managers
        manager.memory_manager = AsyncMock()
        manager.personality_manager = AsyncMock()
        manager.tool_manager = AsyncMock()
        
        await manager.initialize(config)
        
        yield manager
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample WAV audio data for testing."""
        # Create a 1-second WAV file with sine wave
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0  # A4 note
        
        import numpy as np
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return wav_buffer.getvalue()
    
    @pytest.fixture
    def voice_message(self, sample_audio_data):
        """Create a sample voice message for testing."""
        return VoiceMessage(
            conversation_id="test-conversation-123",
            audio_data=sample_audio_data,
            text_content="Hello, how are you today?",
            speaker="user"
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_voice_processing(self, voice_manager, voice_message):
        """Test complete end-to-end voice processing pipeline."""
        # Mock the processing components
        with patch('src.coda.components.voice.manager.AudioProcessor') as mock_audio_processor, \
             patch('src.coda.components.voice.manager.PipelineManager') as mock_pipeline:
            
            # Setup mocks
            mock_audio_instance = AsyncMock()
            mock_audio_processor.return_value = mock_audio_instance
            mock_audio_instance.process_input_audio.return_value = voice_message.audio_data
            
            mock_pipeline_instance = AsyncMock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Mock response
            expected_response = VoiceResponse(
                conversation_id=voice_message.conversation_id,
                text_content="I'm doing well, thank you for asking!",
                audio_data=b"mock_response_audio",
                processing_time=150.0,
                confidence_score=0.92
            )
            
            mock_pipeline_instance.process.return_value = expected_response
            
            # Start conversation
            conversation_state = await voice_manager.start_conversation(
                voice_message.conversation_id
            )
            
            assert conversation_state.conversation_id == voice_message.conversation_id
            assert conversation_state.is_active
            
            # Process voice input
            response = await voice_manager.process_voice_input(
                conversation_id=voice_message.conversation_id,
                audio_data=voice_message.audio_data
            )
            
            # Verify response
            assert isinstance(response, VoiceResponse)
            assert response.conversation_id == voice_message.conversation_id
            assert response.text_content is not None
            assert len(response.text_content) > 0
            
            # Verify audio processing was called
            mock_audio_instance.process_input_audio.assert_called_once()
            
            # End conversation
            await voice_manager.end_conversation(voice_message.conversation_id)
    
    @pytest.mark.asyncio
    async def test_multiple_processing_modes(self, voice_manager, voice_message):
        """Test processing with different modes."""
        modes_to_test = [
            VoiceProcessingMode.MOSHI_ONLY,
            VoiceProcessingMode.LLM_ENHANCED,
            VoiceProcessingMode.HYBRID,
            VoiceProcessingMode.ADAPTIVE
        ]
        
        # Start conversation
        await voice_manager.start_conversation(voice_message.conversation_id)
        
        for mode in modes_to_test:
            with patch.object(voice_manager, '_process_with_mode') as mock_process:
                mock_response = VoiceResponse(
                    conversation_id=voice_message.conversation_id,
                    text_content=f"Response from {mode.value} mode",
                    processing_mode=mode
                )
                mock_process.return_value = mock_response
                
                # Process with specific mode
                voice_message.processing_mode = mode
                response = await voice_manager.process_voice_input(
                    conversation_id=voice_message.conversation_id,
                    audio_data=voice_message.audio_data
                )
                
                assert response.processing_mode == mode
                mock_process.assert_called_once()
        
        # Cleanup
        await voice_manager.end_conversation(voice_message.conversation_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_conversations(self, voice_manager, sample_audio_data):
        """Test handling multiple concurrent conversations."""
        num_conversations = 5
        conversation_ids = [f"conv_{i}" for i in range(num_conversations)]
        
        # Start multiple conversations
        conversation_tasks = []
        for conv_id in conversation_ids:
            task = voice_manager.start_conversation(conv_id)
            conversation_tasks.append(task)
        
        conversations = await asyncio.gather(*conversation_tasks)
        
        # Verify all conversations started
        assert len(conversations) == num_conversations
        for i, conv in enumerate(conversations):
            assert conv.conversation_id == conversation_ids[i]
            assert conv.is_active
        
        # Process voice input in parallel
        with patch.object(voice_manager, '_process_with_mode') as mock_process:
            mock_process.return_value = VoiceResponse(
                conversation_id="mock",
                text_content="Concurrent response"
            )
            
            processing_tasks = []
            for conv_id in conversation_ids:
                task = voice_manager.process_voice_input(
                    conversation_id=conv_id,
                    audio_data=sample_audio_data
                )
                processing_tasks.append(task)
            
            responses = await asyncio.gather(*processing_tasks)
            
            # Verify all responses
            assert len(responses) == num_conversations
            assert mock_process.call_count == num_conversations
        
        # End all conversations
        end_tasks = []
        for conv_id in conversation_ids:
            task = voice_manager.end_conversation(conv_id)
            end_tasks.append(task)
        
        await asyncio.gather(*end_tasks)
    
    @pytest.mark.asyncio
    async def test_integration_with_memory_system(self, voice_manager, voice_message):
        """Test integration with memory management system."""
        # Mock memory manager responses
        memory_context = {
            "recent_topics": ["weather", "travel"],
            "user_preferences": {"language": "en", "formality": "casual"},
            "conversation_history": [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "It's sunny today."}
            ]
        }
        
        voice_manager.memory_manager.get_conversation_context.return_value = memory_context
        voice_manager.memory_manager.store_interaction.return_value = True
        
        # Start conversation
        await voice_manager.start_conversation(voice_message.conversation_id)
        
        with patch.object(voice_manager, '_process_with_mode') as mock_process:
            mock_response = VoiceResponse(
                conversation_id=voice_message.conversation_id,
                text_content="Response with memory context"
            )
            mock_process.return_value = mock_response
            
            # Process voice input
            response = await voice_manager.process_voice_input(
                conversation_id=voice_message.conversation_id,
                audio_data=voice_message.audio_data
            )
            
            # Verify memory integration was called
            voice_manager.memory_manager.get_conversation_context.assert_called_once_with(
                voice_message.conversation_id
            )
            
            # Verify interaction was stored
            voice_manager.memory_manager.store_interaction.assert_called_once()
            
            assert response.text_content == "Response with memory context"
        
        # Cleanup
        await voice_manager.end_conversation(voice_message.conversation_id)
    
    @pytest.mark.asyncio
    async def test_integration_with_personality_system(self, voice_manager, voice_message):
        """Test integration with personality management system."""
        # Mock personality manager responses
        personality_traits = {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.3
        }
        
        personality_context = {
            "traits": personality_traits,
            "communication_style": "friendly_and_helpful",
            "response_length": "moderate",
            "formality_level": "casual"
        }
        
        voice_manager.personality_manager.get_personality_context.return_value = personality_context
        voice_manager.personality_manager.adapt_response.return_value = "Adapted response with personality"
        
        # Start conversation
        await voice_manager.start_conversation(voice_message.conversation_id)
        
        with patch.object(voice_manager, '_process_with_mode') as mock_process:
            mock_response = VoiceResponse(
                conversation_id=voice_message.conversation_id,
                text_content="Base response"
            )
            mock_process.return_value = mock_response
            
            # Process voice input
            response = await voice_manager.process_voice_input(
                conversation_id=voice_message.conversation_id,
                audio_data=voice_message.audio_data
            )
            
            # Verify personality integration was called
            voice_manager.personality_manager.get_personality_context.assert_called_once()
            voice_manager.personality_manager.adapt_response.assert_called_once()
        
        # Cleanup
        await voice_manager.end_conversation(voice_message.conversation_id)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, voice_manager, voice_message):
        """Test error handling and recovery mechanisms."""
        # Start conversation
        await voice_manager.start_conversation(voice_message.conversation_id)
        
        # Test processing error recovery
        with patch.object(voice_manager, '_process_with_mode') as mock_process:
            # First call fails, second succeeds
            mock_process.side_effect = [
                VoiceProcessingError("Processing failed"),
                VoiceResponse(
                    conversation_id=voice_message.conversation_id,
                    text_content="Recovery response"
                )
            ]
            
            # Should handle error and potentially retry
            with pytest.raises(VoiceProcessingError):
                await voice_manager.process_voice_input(
                    conversation_id=voice_message.conversation_id,
                    audio_data=voice_message.audio_data
                )
        
        # Test invalid conversation ID
        with pytest.raises(Exception):  # Should raise appropriate exception
            await voice_manager.process_voice_input(
                conversation_id="nonexistent-conversation",
                audio_data=voice_message.audio_data
            )
        
        # Test invalid audio data
        with pytest.raises(ValidationError):
            await voice_manager.process_voice_input(
                conversation_id=voice_message.conversation_id,
                audio_data=b"invalid_audio_data"
            )
        
        # Cleanup
        await voice_manager.end_conversation(voice_message.conversation_id)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, voice_manager, voice_message):
        """Test performance monitoring and analytics integration."""
        # Start conversation
        await voice_manager.start_conversation(voice_message.conversation_id)
        
        with patch.object(voice_manager, '_process_with_mode') as mock_process:
            mock_response = VoiceResponse(
                conversation_id=voice_message.conversation_id,
                text_content="Performance test response",
                processing_time=250.0
            )
            mock_process.return_value = mock_response
            
            # Process voice input
            start_time = time.time()
            response = await voice_manager.process_voice_input(
                conversation_id=voice_message.conversation_id,
                audio_data=voice_message.audio_data
            )
            end_time = time.time()
            
            # Verify performance tracking
            assert response.processing_time is not None
            assert response.processing_time > 0
            
            # Verify analytics were updated
            assert voice_manager.analytics.total_requests > 0
            
            # Verify latency tracking
            assert voice_manager.latency_tracker.get_average_latency() >= 0
        
        # Cleanup
        await voice_manager.end_conversation(voice_message.conversation_id)


if __name__ == "__main__":
    pytest.main([__file__])
