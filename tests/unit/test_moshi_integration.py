"""
Tests for Moshi Integration.
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from src.coda.components.voice.moshi_integration import (
    MoshiClient, MoshiStreamingManager, MoshiVoiceProcessor
)
from src.coda.components.voice.models import MoshiConfig, VoiceConfig, VoiceMessage


class TestMoshiClient:
    """Test Moshi client functionality."""
    
    @pytest.fixture
    def moshi_config(self):
        """Create test Moshi configuration."""
        return MoshiConfig(
            model_path="test/model",
            device="cpu",
            optimization="fp32",
            target_latency_ms=200,
            vram_allocation="2GB",
            enable_streaming=True,
            inner_monologue_enabled=True
        )
    
    @pytest.fixture
    def moshi_client(self, moshi_config):
        """Create test Moshi client."""
        return MoshiClient(moshi_config)
    
    def test_initialization(self, moshi_client, moshi_config):
        """Test Moshi client initialization."""
        assert moshi_client.config == moshi_config
        assert moshi_client.device.type == "cpu"
        assert not moshi_client.is_initialized
        assert moshi_client.model is None
        assert moshi_client.tokenizer is None
    
    @pytest.mark.asyncio
    async def test_initialize_fallback(self, moshi_client):
        """Test initialization with fallback when Moshi library not available."""
        # Mock the absence of Moshi library
        with patch('src.coda.components.voice.moshi_integration.MOSHI_AVAILABLE', False):
            await moshi_client.initialize()
            
            assert moshi_client.is_initialized
            assert moshi_client.model == "fallback_model"
    
    @pytest.mark.asyncio
    async def test_initialize_with_moshi(self, moshi_client):
        """Test initialization with real Moshi library."""
        # Mock Moshi library components
        mock_lm_model = Mock()
        mock_compression_model = Mock()
        mock_tokenizer = Mock()
        
        with patch('src.coda.components.voice.moshi_integration.MOSHI_AVAILABLE', True), \
             patch('moshi.models.get_moshi_lm', return_value=mock_lm_model), \
             patch('moshi.models.get_mimi', return_value=mock_compression_model):
            
            mock_lm_model.tokenizer = mock_tokenizer
            mock_lm_model.to.return_value = mock_lm_model
            mock_lm_model.eval.return_value = mock_lm_model
            
            await moshi_client.initialize()
            
            assert moshi_client.is_initialized
            assert moshi_client.lm_model == mock_lm_model
            assert moshi_client.compression_model == mock_compression_model
            assert moshi_client.tokenizer == mock_tokenizer
    
    @pytest.mark.asyncio
    async def test_process_audio(self, moshi_client):
        """Test audio processing."""
        await moshi_client.initialize()
        
        # Create test audio data
        audio_data = np.random.rand(1600).astype(np.float32).tobytes()
        
        processed_audio = await moshi_client.process_audio(audio_data)
        
        assert isinstance(processed_audio, bytes)
        assert len(processed_audio) > 0
    
    @pytest.mark.asyncio
    async def test_extract_text(self, moshi_client):
        """Test text extraction from audio."""
        await moshi_client.initialize()
        
        # Create test audio data
        audio_data = np.random.rand(1600).astype(np.float32).tobytes()
        
        extracted_text = await moshi_client.extract_text(audio_data)
        
        assert isinstance(extracted_text, str)
        # Should return empty string in fallback mode
        assert extracted_text == ""
    
    @pytest.mark.asyncio
    async def test_inject_text(self, moshi_client):
        """Test text injection for speech synthesis."""
        await moshi_client.initialize()
        
        test_text = "Hello, this is a test."
        
        synthesized_audio = await moshi_client.inject_text(test_text)
        
        assert isinstance(synthesized_audio, bytes)
        assert len(synthesized_audio) > 0
    
    def test_audio_tensor_processing(self, moshi_client):
        """Test audio tensor processing methods."""
        # Test tensor conversion methods
        test_audio = torch.randn(1600)
        
        # Convert to bytes and back
        audio_bytes = moshi_client._tensor_to_bytes(test_audio)
        assert isinstance(audio_bytes, bytes)
        
        recovered_tensor = moshi_client._bytes_to_tensor(audio_bytes)
        assert isinstance(recovered_tensor, torch.Tensor)
        assert recovered_tensor.shape == test_audio.shape
    
    @pytest.mark.asyncio
    async def test_process_audio_tensor_fallback(self, moshi_client):
        """Test audio tensor processing with fallback."""
        await moshi_client.initialize()
        
        test_tensor = torch.randn(1, 1600)
        
        processed_tensor = await moshi_client._process_audio_tensor(test_tensor)
        
        assert isinstance(processed_tensor, torch.Tensor)
        assert processed_tensor.shape[0] == 1600  # Should remove batch dim
    
    @pytest.mark.asyncio
    async def test_extract_text_from_audio_tensor(self, moshi_client):
        """Test text extraction from audio tensor."""
        await moshi_client.initialize()
        
        test_tensor = torch.randn(1, 1600)
        
        extracted_text = await moshi_client._extract_text_from_audio(test_tensor)
        
        assert isinstance(extracted_text, str)
    
    @pytest.mark.asyncio
    async def test_synthesize_from_text(self, moshi_client):
        """Test text-to-speech synthesis."""
        await moshi_client.initialize()
        
        test_text = "Hello world"
        
        synthesized_tensor = await moshi_client._synthesize_from_text(test_text)
        
        assert isinstance(synthesized_tensor, torch.Tensor)
        assert synthesized_tensor.numel() > 0
    
    def test_audio_enhancement(self, moshi_client):
        """Test audio enhancement fallback."""
        test_audio = torch.randn(1, 1600) * 2  # Loud audio
        
        enhanced_audio = moshi_client._apply_audio_enhancement(test_audio)
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == test_audio.shape
        # Should be normalized
        assert enhanced_audio.abs().max() <= 1.0
    
    def test_synthetic_speech_generation(self, moshi_client):
        """Test synthetic speech generation."""
        token_ids = torch.tensor([[1, 2, 3, 4, 5]])
        audio_length = 1600
        
        synthetic_audio = moshi_client._generate_synthetic_speech(token_ids, audio_length)
        
        assert isinstance(synthetic_audio, torch.Tensor)
        assert synthetic_audio.shape == (1, audio_length)
        assert synthetic_audio.abs().max() <= 1.0


class TestMoshiStreamingManager:
    """Test Moshi streaming manager."""
    
    @pytest.fixture
    def moshi_config(self):
        """Create test configuration."""
        return MoshiConfig(
            model_path="test/model",
            device="cpu",
            optimization="fp32"
        )
    
    @pytest.fixture
    async def moshi_client(self, moshi_config):
        """Create initialized Moshi client."""
        client = MoshiClient(moshi_config)
        await client.initialize()
        return client
    
    @pytest.fixture
    def streaming_manager(self, moshi_client):
        """Create streaming manager."""
        return MoshiStreamingManager(moshi_client, buffer_size=5)
    
    def test_initialization(self, streaming_manager, moshi_client):
        """Test streaming manager initialization."""
        assert streaming_manager.moshi_client == moshi_client
        assert streaming_manager.buffer_size == 5
        assert not streaming_manager.is_streaming
        assert streaming_manager.input_buffer.maxsize == 5
        assert streaming_manager.output_buffer.maxsize == 5
    
    @pytest.mark.asyncio
    async def test_start_stop_streaming(self, streaming_manager):
        """Test starting and stopping streaming."""
        # Start streaming
        await streaming_manager.start_streaming()
        assert streaming_manager.is_streaming
        assert streaming_manager.processing_thread is not None
        
        # Stop streaming
        await streaming_manager.stop_streaming()
        assert not streaming_manager.is_streaming
    
    @pytest.mark.asyncio
    async def test_stream_input(self, streaming_manager):
        """Test input streaming."""
        await streaming_manager.start_streaming()
        
        # Create test audio stream
        async def test_audio_stream():
            for i in range(3):
                yield np.random.rand(800).astype(np.float32).tobytes()
        
        # Stream input
        input_task = asyncio.create_task(
            streaming_manager.stream_input(test_audio_stream())
        )
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Stop streaming
        await streaming_manager.stop_streaming()
        
        # Wait for task completion
        try:
            await asyncio.wait_for(input_task, timeout=1.0)
        except asyncio.TimeoutError:
            input_task.cancel()
    
    @pytest.mark.asyncio
    async def test_stream_output(self, streaming_manager):
        """Test output streaming."""
        await streaming_manager.start_streaming()
        
        # Add some test output to buffer
        test_audio = np.random.rand(800).astype(np.float32).tobytes()
        streaming_manager.output_buffer.put_nowait(test_audio)
        streaming_manager.output_buffer.put_nowait(test_audio)
        
        # Collect output
        output_chunks = []
        async for chunk in streaming_manager.stream_output():
            output_chunks.append(chunk)
            if len(output_chunks) >= 2:
                break
        
        assert len(output_chunks) == 2
        assert all(isinstance(chunk, bytes) for chunk in output_chunks)
        
        await streaming_manager.stop_streaming()
    
    def test_get_metrics(self, streaming_manager):
        """Test metrics collection."""
        metrics = streaming_manager.get_metrics()
        
        assert 'is_streaming' in metrics
        assert 'buffer_sizes' in metrics
        assert 'total_chunks' in metrics
        assert 'dropped_chunks' in metrics
        assert metrics['is_streaming'] == False


class TestMoshiVoiceProcessor:
    """Test Moshi voice processor."""
    
    @pytest.fixture
    def voice_config(self):
        """Create test voice configuration."""
        return VoiceConfig.development()
    
    @pytest.fixture
    def voice_processor(self, voice_config):
        """Create voice processor."""
        return MoshiVoiceProcessor()
    
    @pytest.mark.asyncio
    async def test_initialization(self, voice_processor, voice_config):
        """Test voice processor initialization."""
        await voice_processor.initialize(voice_config)
        
        assert voice_processor.config == voice_config
        assert voice_processor.moshi_client is not None
        assert voice_processor.streaming_manager is not None
        assert voice_processor.conversation_manager is not None
    
    @pytest.mark.asyncio
    async def test_process_voice_message(self, voice_processor, voice_config):
        """Test voice message processing."""
        await voice_processor.initialize(voice_config)
        
        # Create test voice message
        test_audio = np.random.rand(1600).astype(np.float32).tobytes()
        message = VoiceMessage(
            conversation_id="test_conv",
            audio_data=test_audio,
            timestamp=asyncio.get_event_loop().time()
        )
        
        response = await voice_processor.process_voice_message(message)
        
        assert response is not None
        assert response.conversation_id == "test_conv"
        assert isinstance(response.audio_data, bytes)
    
    @pytest.mark.asyncio
    async def test_process_voice_stream(self, voice_processor, voice_config):
        """Test voice stream processing."""
        await voice_processor.initialize(voice_config)
        
        # Create test audio stream
        async def test_audio_stream():
            for i in range(3):
                yield np.random.rand(800).astype(np.float32).tobytes()
        
        conversation_id = "test_stream_conv"
        
        # Process stream
        output_chunks = []
        async for chunk in voice_processor.process_voice_stream(conversation_id, test_audio_stream()):
            output_chunks.append(chunk)
            if len(output_chunks) >= 2:  # Limit for test
                break
        
        assert len(output_chunks) >= 1
        assert all(chunk.conversation_id == conversation_id for chunk in output_chunks)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, voice_processor, voice_config):
        """Test voice processor shutdown."""
        await voice_processor.initialize(voice_config)
        
        # Shutdown should not raise errors
        await voice_processor.shutdown()
        
        # Should be able to shutdown multiple times
        await voice_processor.shutdown()
