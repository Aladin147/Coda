"""
Tests for Moshi integration components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.coda.components.voice.moshi_integration import MoshiClient, MoshiVoiceProcessor
from src.coda.components.voice.models import MoshiConfig, VoiceConfig, AudioConfig
from src.coda.components.voice.exceptions import (
    ModelLoadingError, VRAMAllocationError, ComponentNotInitializedError
)


class TestMoshiClient:
    """Test MoshiClient functionality."""
    
    @pytest.fixture
    def moshi_config(self):
        """Create test Moshi configuration."""
        return MoshiConfig(
            model_path=None,
            device="cpu",
            optimization="fp32",
            vram_allocation="2GB",
            inner_monologue_enabled=True,
            conversation_timeout=30.0
        )
    
    @pytest.fixture
    def voice_config(self, moshi_config):
        """Create test voice configuration."""
        return VoiceConfig(
            audio=AudioConfig(
                sample_rate=24000,
                channels=1,
                format="wav"
            ),
            moshi=moshi_config
        )
    
    def test_moshi_client_init(self, voice_config):
        """Test MoshiClient initialization."""
        client = MoshiClient(voice_config)
        
        assert client.config == voice_config.moshi
        assert client.device == "cpu"
        assert not client.is_initialized
        assert not client.is_conversation_active
        assert client.current_conversation_id is None
    
    @pytest.mark.asyncio
    async def test_moshi_client_init_without_moshi_installed(self, voice_config):
        """Test MoshiClient initialization when Moshi is not installed."""
        client = MoshiClient(voice_config)
        
        with patch('src.coda.components.voice.moshi_integration.import', side_effect=ImportError("No module named 'moshi'")):
            with pytest.raises(ModelLoadingError) as exc_info:
                await client.initialize()
            
            assert "not properly installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_moshi_client_vram_allocation_failure(self, voice_config):
        """Test MoshiClient VRAM allocation failure."""
        client = MoshiClient(voice_config)
        
        # Mock VRAM manager that fails allocation
        mock_vram_manager = Mock()
        mock_vram_manager.register_component.return_value = False
        client.vram_manager = mock_vram_manager
        
        with pytest.raises(VRAMAllocationError) as exc_info:
            await client.initialize()
        
        assert "Failed to register VRAM allocation" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_start_conversation_not_initialized(self, voice_config):
        """Test starting conversation when client is not initialized."""
        client = MoshiClient(voice_config)
        
        with pytest.raises(ComponentNotInitializedError):
            await client.start_conversation("test-conversation")
    
    @pytest.mark.asyncio
    async def test_start_conversation_invalid_id(self, voice_config):
        """Test starting conversation with invalid ID."""
        client = MoshiClient(voice_config)
        client.is_initialized = True
        
        with pytest.raises(ValidationError):
            await client.start_conversation("")
    
    @pytest.mark.asyncio
    async def test_conversation_lifecycle(self, voice_config):
        """Test complete conversation lifecycle."""
        client = MoshiClient(voice_config)
        client.is_initialized = True
        
        # Mock the model and other components
        client.model = Mock()
        client.inner_monologue = Mock()
        
        conversation_id = "test-conversation-123"
        
        # Start conversation
        await client.start_conversation(conversation_id)
        assert client.is_conversation_active
        assert client.current_conversation_id == conversation_id
        
        # End conversation
        await client.end_conversation(conversation_id)
        assert not client.is_conversation_active
        assert client.current_conversation_id is None
    
    @pytest.mark.asyncio
    async def test_process_audio_not_initialized(self, voice_config):
        """Test processing audio when not initialized."""
        client = MoshiClient(voice_config)
        
        with pytest.raises(ComponentNotInitializedError):
            async for chunk in client.process_audio(b"audio_data", "test-conversation"):
                pass
    
    @pytest.mark.asyncio
    async def test_process_audio_no_conversation(self, voice_config):
        """Test processing audio without active conversation."""
        client = MoshiClient(voice_config)
        client.is_initialized = True
        
        with pytest.raises(ConversationError):
            async for chunk in client.process_audio(b"audio_data", "test-conversation"):
                pass
    
    @pytest.mark.asyncio
    async def test_cleanup_resources(self, voice_config):
        """Test resource cleanup."""
        client = MoshiClient(voice_config)
        
        # Mock resources
        mock_vram_manager = Mock()
        client.vram_manager = mock_vram_manager
        client.vram_allocated = True
        
        await client.cleanup()
        
        # Verify VRAM was deallocated
        mock_vram_manager.deallocate.assert_called_once_with("moshi_client")
        assert not client.vram_allocated


class TestMoshiVoiceProcessor:
    """Test MoshiVoiceProcessor functionality."""
    
    @pytest.fixture
    def voice_config(self):
        """Create test voice configuration."""
        return VoiceConfig(
            audio=AudioConfig(
                sample_rate=24000,
                channels=1,
                format="wav"
            ),
            moshi=MoshiConfig(
                model_path=None,
                device="cpu",
                optimization="fp32",
                vram_allocation="2GB"
            )
        )
    
    def test_moshi_voice_processor_init(self, voice_config):
        """Test MoshiVoiceProcessor initialization."""
        integration = MoshiVoiceProcessor()

        # Initially, config should be None
        assert integration.config is None
        assert integration.moshi_client is None
        assert integration.streaming_manager is None
    
    @pytest.mark.asyncio
    async def test_moshi_voice_processor_initialize(self, voice_config):
        """Test MoshiVoiceProcessor initialization."""
        integration = MoshiVoiceProcessor()

        # Mock the Moshi client initialization
        with patch('src.coda.components.voice.moshi_integration.MoshiClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            await integration.initialize(voice_config)

            assert integration.config == voice_config
            mock_client.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_voice_message(self, voice_config):
        """Test processing voice message."""
        integration = MoshiVoiceProcessor()
        await integration.initialize(voice_config)
        
        # Mock the client
        integration.client = AsyncMock()
        integration.client.is_initialized = True
        integration.client.process_audio = AsyncMock()
        
        # Mock voice message
        voice_message = Mock()
        voice_message.audio_data = b"test_audio"
        voice_message.conversation_id = "test-conversation"
        
        # Mock response chunks
        mock_chunks = [
            Mock(audio_data=b"response1", text_content="Hello"),
            Mock(audio_data=b"response2", text_content="World")
        ]
        integration.client.process_audio.return_value = mock_chunks
        
        response = await integration.process_voice_message(voice_message)
        
        assert response is not None
        integration.client.process_audio.assert_called_once_with(
            b"test_audio", "test-conversation"
        )
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, voice_config):
        """Test starting conversation."""
        integration = MoshiVoiceProcessor()
        await integration.initialize(voice_config)
        
        # Mock the client
        integration.client = AsyncMock()
        integration.client.start_conversation = AsyncMock()
        
        await integration.start_conversation("test-conversation")
        
        integration.client.start_conversation.assert_called_once_with("test-conversation")
    
    @pytest.mark.asyncio
    async def test_end_conversation(self, voice_config):
        """Test ending conversation."""
        integration = MoshiVoiceProcessor()
        await integration.initialize(voice_config)
        
        # Mock the client
        integration.client = AsyncMock()
        integration.client.end_conversation = AsyncMock()
        
        await integration.end_conversation("test-conversation")
        
        integration.client.end_conversation.assert_called_once_with("test-conversation")
    
    @pytest.mark.asyncio
    async def test_cleanup(self, voice_config):
        """Test cleanup."""
        integration = MoshiVoiceProcessor()
        await integration.initialize(voice_config)
        
        # Mock the client
        integration.client = AsyncMock()
        integration.client.cleanup = AsyncMock()
        
        await integration.cleanup()
        
        integration.client.cleanup.assert_called_once()


class TestMoshiErrorHandling:
    """Test error handling in Moshi components."""
    
    @pytest.mark.asyncio
    async def test_model_loading_error_propagation(self):
        """Test that model loading errors are properly propagated."""
        config = VoiceConfig(
            audio=AudioConfig(sample_rate=24000, channels=1, format="wav"),
            moshi=MoshiConfig(device="cpu")
        )
        
        client = MoshiClient(config)
        
        with patch('src.coda.components.voice.moshi_integration.moshi') as mock_moshi:
            mock_moshi.models.get_moshi_lm.side_effect = Exception("Model not found")
            
            with pytest.raises(ModelLoadingError) as exc_info:
                await client.initialize()
            
            assert "Failed to load Moshi language model" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_vram_cleanup_on_failure(self):
        """Test VRAM cleanup when initialization fails."""
        config = VoiceConfig(
            audio=AudioConfig(sample_rate=24000, channels=1, format="wav"),
            moshi=MoshiConfig(device="cpu")
        )
        
        client = MoshiClient(config)
        
        # Mock VRAM manager
        mock_vram_manager = Mock()
        mock_vram_manager.register_component.return_value = True
        mock_vram_manager.allocate.return_value = True
        client.vram_manager = mock_vram_manager
        client.vram_allocated = True
        
        with patch('src.coda.components.voice.moshi_integration.moshi') as mock_moshi:
            mock_moshi.models.get_moshi_lm.side_effect = Exception("Model loading failed")
            
            with pytest.raises(ModelLoadingError):
                await client.initialize()
            
            # Verify VRAM was cleaned up
            mock_vram_manager.deallocate.assert_called_with("moshi_client")


if __name__ == "__main__":
    pytest.main([__file__])
