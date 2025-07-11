"""
Integration tests for the complete voice processing pipeline.
Tests voice→memory→LLM→response flow end-to-end.
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.coda.components.voice.models import (
    VoiceMessage, VoiceResponse, VoiceStreamChunk, 
    VoiceProcessingMode, AudioConfig
)
from src.coda.components.memory.models import Memory, MemoryType, MemoryMetadata
from src.coda.components.llm.models import LLMMessage, MessageRole
from src.coda.core.assistant import CodaAssistant


class TestVoicePipelineIntegration:
    """Integration tests for the complete voice processing pipeline."""

    @pytest_asyncio.fixture
    async def mock_audio_data(self):
        """Create mock audio data for testing."""
        return b"fake_audio_data_representing_hello_world"

    @pytest_asyncio.fixture
    async def mock_voice_manager(self):
        """Create mock voice manager with realistic responses."""
        manager = Mock()
        
        # Mock voice processing
        manager.process_voice_input = AsyncMock(return_value=VoiceResponse(
            response_id=str(uuid.uuid4()),
            conversation_id="test-conv-123",
            message_id="test-msg-456",
            text_content="Hello! How can I help you today?",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=150.0
        ))
        
        # Mock streaming
        async def mock_stream_generator():
            chunks = [
                "Hello! ",
                "How can ",
                "I help you ",
                "today?"
            ]
            for i, text in enumerate(chunks):
                yield VoiceStreamChunk(
                    conversation_id="test-conv-123",
                    chunk_index=i,
                    text_content=text,
                    audio_data=b"fake_audio_chunk",
                    timestamp=time.time(),
                    is_complete=(i == len(chunks) - 1),
                    chunk_type="audio"
                )
                await asyncio.sleep(0.01)  # Simulate processing delay
        
        manager.stream_response = AsyncMock(return_value=mock_stream_generator())
        manager.is_initialized = True
        
        return manager

    @pytest_asyncio.fixture
    async def mock_memory_manager(self):
        """Create mock memory manager with realistic behavior."""
        manager = Mock()
        
        # Mock memory storage
        manager.store_conversation = AsyncMock(return_value="memory-id-789")
        
        # Mock memory retrieval
        relevant_memories = [
            Memory(
                id="mem-1",
                content="User previously asked about weather",
                metadata=MemoryMetadata(
                    source_type=MemoryType.CONVERSATION,
                    importance=0.7,
                    topics=["weather", "questions"],
                    timestamp=datetime.now()
                )
            ),
            Memory(
                id="mem-2", 
                content="User prefers detailed explanations",
                metadata=MemoryMetadata(
                    source_type=MemoryType.FACT,
                    importance=0.8,
                    topics=["preferences", "communication"],
                    timestamp=datetime.now()
                )
            )
        ]
        
        manager.retrieve_relevant_memories = AsyncMock(return_value=relevant_memories)
        manager.is_initialized = True
        
        return manager

    @pytest_asyncio.fixture
    async def mock_llm_manager(self):
        """Create mock LLM manager with realistic responses."""
        manager = Mock()
        
        # Mock LLM response generation
        manager.generate_response = AsyncMock(return_value="Hello! How can I help you today?")
        
        # Mock streaming response
        async def mock_llm_stream():
            tokens = ["Hello", "!", " How", " can", " I", " help", " you", " today", "?"]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)
        
        manager.stream_response = AsyncMock(return_value=mock_llm_stream())
        manager.is_initialized = True
        
        return manager

    @pytest_asyncio.fixture
    async def integrated_assistant(self, mock_voice_manager, mock_memory_manager, mock_llm_manager):
        """Create integrated assistant with mocked components."""
        with patch('src.coda.core.assistant.VoiceManager', return_value=mock_voice_manager):
            with patch('src.coda.core.assistant.MemoryManager', return_value=mock_memory_manager):
                with patch('src.coda.core.assistant.LLMManager', return_value=mock_llm_manager):
                    assistant = CodaAssistant()
                    await assistant.initialize()
                    return assistant

    @pytest.mark.asyncio
    async def test_complete_voice_pipeline(self, integrated_assistant, mock_audio_data):
        """Test the complete voice processing pipeline end-to-end."""
        # Create voice message
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="test-conv-123",
            text_content="What's the weather like today?",
            audio_data=mock_audio_data,
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Process through complete pipeline
        response = await integrated_assistant.process_voice_message(voice_message)
        
        # Verify response
        assert isinstance(response, VoiceResponse)
        assert response.conversation_id == "test-conv-123"
        assert response.text_content == "Hello! How can I help you today?"
        assert response.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_memory_integration_in_pipeline(self, integrated_assistant, mock_memory_manager):
        """Test that memory is properly integrated into the voice pipeline."""
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="test-conv-123",
            text_content="Remember my preferences",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Process message
        response = await integrated_assistant.process_voice_message(voice_message)
        
        # Verify memory operations were called
        mock_memory_manager.retrieve_relevant_memories.assert_called()
        mock_memory_manager.store_conversation.assert_called()
        
        # Verify response
        assert isinstance(response, VoiceResponse)

    @pytest.mark.asyncio
    async def test_streaming_pipeline_integration(self, integrated_assistant):
        """Test streaming response pipeline integration."""
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="test-conv-123",
            text_content="Tell me a story",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Test streaming response
        chunks = []
        async for chunk in integrated_assistant.stream_voice_response(voice_message):
            chunks.append(chunk)
            assert isinstance(chunk, VoiceStreamChunk)
            assert chunk.conversation_id == "test-conv-123"
        
        # Verify we received multiple chunks
        assert len(chunks) > 1
        
        # Verify final chunk is marked complete
        assert chunks[-1].is_complete is True

    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, integrated_assistant, mock_voice_manager):
        """Test error handling throughout the pipeline."""
        # Mock voice manager to raise an error
        mock_voice_manager.process_voice_input.side_effect = Exception("Voice processing failed")
        
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="test-conv-123",
            text_content="This should fail",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Verify error is handled gracefully
        with pytest.raises(Exception, match="Voice processing failed"):
            await integrated_assistant.process_voice_message(voice_message)

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_processing(self, integrated_assistant):
        """Test concurrent processing of multiple voice messages."""
        # Create multiple voice messages
        messages = [
            VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"test-conv-{i}",
                text_content=f"Message {i}",
                audio_data=b"fake_audio",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            for i in range(3)
        ]
        
        # Process concurrently
        tasks = [
            integrated_assistant.process_voice_message(msg)
            for msg in messages
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all responses
        assert len(responses) == 3
        for response in responses:
            if isinstance(response, Exception):
                pytest.fail(f"Concurrent processing failed: {response}")
            assert isinstance(response, VoiceResponse)

    @pytest.mark.asyncio
    async def test_pipeline_performance_metrics(self, integrated_assistant):
        """Test that performance metrics are collected throughout the pipeline."""
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="test-conv-123",
            text_content="Performance test message",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        start_time = time.time()
        response = await integrated_assistant.process_voice_message(voice_message)
        end_time = time.time()
        
        # Verify timing metrics
        assert response.total_latency_ms > 0
        assert response.total_latency_ms < (end_time - start_time) * 1000 + 100  # Allow some margin

    @pytest.mark.asyncio
    async def test_conversation_context_preservation(self, integrated_assistant, mock_memory_manager):
        """Test that conversation context is preserved across multiple messages."""
        conversation_id = "test-conv-context"
        
        # Send first message
        msg1 = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            text_content="My name is Alice",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        response1 = await integrated_assistant.process_voice_message(msg1)
        assert isinstance(response1, VoiceResponse)
        
        # Send second message referencing first
        msg2 = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            text_content="What's my name?",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        response2 = await integrated_assistant.process_voice_message(msg2)
        assert isinstance(response2, VoiceResponse)
        
        # Verify memory was accessed for context
        assert mock_memory_manager.retrieve_relevant_memories.call_count >= 2

    @pytest.mark.asyncio
    async def test_pipeline_cleanup_on_failure(self, integrated_assistant, mock_llm_manager):
        """Test that resources are properly cleaned up when pipeline fails."""
        # Mock LLM to fail
        mock_llm_manager.generate_response.side_effect = Exception("LLM failed")
        
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="test-conv-123",
            text_content="This will fail at LLM stage",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Verify cleanup happens even on failure
        with pytest.raises(Exception):
            await integrated_assistant.process_voice_message(voice_message)
        
        # Verify assistant is still in valid state
        assert integrated_assistant.is_initialized

    @pytest.mark.asyncio
    async def test_pipeline_with_different_processing_modes(self, integrated_assistant):
        """Test pipeline with different voice processing modes."""
        modes = [
            VoiceProcessingMode.MOSHI_ONLY,
            VoiceProcessingMode.LLM_ONLY,
            VoiceProcessingMode.HYBRID
        ]
        
        for mode in modes:
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id="test-conv-modes",
                text_content=f"Test message for {mode}",
                audio_data=b"fake_audio",
                processing_mode=mode
            )
            
            response = await integrated_assistant.process_voice_message(voice_message)
            assert isinstance(response, VoiceResponse)
            assert response.processing_mode == mode
