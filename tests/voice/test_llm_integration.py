"""
Tests for LLM integration components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.coda.components.voice.llm_integration import VoiceLLMProcessor, VoiceLLMConfig
from src.coda.components.voice.models import VoiceMessage, VoiceResponse, ConversationState
from src.coda.components.voice.exceptions import LLMIntegrationError, VoiceTimeoutError, NetworkError


class TestVoiceLLMConfig:
    """Test VoiceLLMConfig functionality."""
    
    def test_voice_llm_config_defaults(self):
        """Test VoiceLLMConfig with default values."""
        config = VoiceLLMConfig()
        
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama3.1:8b"
        assert config.llm_base_url == "http://localhost:11434"
        assert config.enable_streaming is True
        assert config.llm_timeout_seconds == 30.0
        assert config.max_context_length == 4096
        assert config.temperature == 0.7
    
    def test_voice_llm_config_custom(self):
        """Test VoiceLLMConfig with custom values."""
        config = VoiceLLMConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            llm_base_url="https://api.openai.com/v1",
            enable_streaming=False,
            llm_timeout_seconds=60.0,
            max_context_length=8192,
            temperature=0.5
        )
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.llm_base_url == "https://api.openai.com/v1"
        assert config.enable_streaming is False
        assert config.llm_timeout_seconds == 60.0
        assert config.max_context_length == 8192
        assert config.temperature == 0.5


class TestVoiceLLMProcessor:
    """Test VoiceLLMProcessor functionality."""
    
    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return VoiceLLMConfig(
            llm_provider="ollama",
            llm_model="llama3.1:8b",
            llm_timeout_seconds=10.0
        )
    
    @pytest.fixture
    def voice_message(self):
        """Create test voice message."""
        return VoiceMessage(
            conversation_id="test-conversation",
            text_content="Hello, how are you?",
            audio_data=b"audio_data",
            timestamp=1234567890.0,
            speaker="user"
        )
    
    @pytest.fixture
    def conversation_state(self):
        """Create test conversation state."""
        return ConversationState(
            conversation_id="test-conversation",
            is_active=True,
            participant_count=2,
            last_activity=1234567890.0
        )
    
    def test_voice_llm_processor_init(self, llm_config):
        """Test VoiceLLMProcessor initialization."""
        integration = VoiceLLMProcessor(llm_config)
        
        assert integration.config == llm_config
        assert integration.llm_manager is not None
        assert integration.latency_tracker is not None
    
    @pytest.mark.asyncio
    async def test_process_voice_message_success(self, llm_config, voice_message, conversation_state):
        """Test successful voice message processing."""
        integration = VoiceLLMProcessor(llm_config)
        
        # Mock LLM manager
        mock_llm_manager = AsyncMock()
        mock_response = Mock()
        mock_response.content = "I'm doing well, thank you!"
        mock_response.metadata = {"model": "llama3.1:8b", "tokens": 10}
        mock_llm_manager.generate_response.return_value = mock_response
        integration.llm_manager = mock_llm_manager
        
        response = await integration.process_voice_message(
            voice_message, conversation_state
        )
        
        assert isinstance(response, VoiceResponse)
        assert response.text_content == "I'm doing well, thank you!"
        assert response.conversation_id == "test-conversation"
        mock_llm_manager.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_voice_message_empty_text(self, llm_config, conversation_state):
        """Test processing voice message with empty text content."""
        integration = VoiceLLMProcessor(llm_config)
        
        voice_message = VoiceMessage(
            conversation_id="test-conversation",
            text_content="",  # Empty text
            audio_data=b"audio_data",
            timestamp=1234567890.0,
            speaker="user"
        )
        
        response = await integration.process_voice_message(
            voice_message, conversation_state
        )
        
        # Should return fallback response
        assert isinstance(response, VoiceResponse)
        assert "I didn't catch that" in response.text_content or response.text_content == ""
    
    @pytest.mark.asyncio
    async def test_process_voice_message_with_context(self, llm_config, voice_message, conversation_state):
        """Test processing voice message with additional context."""
        integration = VoiceLLMProcessor(llm_config)
        
        # Mock LLM manager
        mock_llm_manager = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Based on your personality, I think you'd enjoy this!"
        mock_llm_manager.generate_response.return_value = mock_response
        integration.llm_manager = mock_llm_manager
        
        context = {
            "memory": {"recent_topics": ["music", "movies"]},
            "personality": {"traits": {"openness": 0.8, "extraversion": 0.6}},
            "tools": {"available_functions": ["search", "weather"]}
        }
        
        response = await integration.process_voice_message(
            voice_message, conversation_state, context
        )
        
        assert isinstance(response, VoiceResponse)
        assert response.text_content == "Based on your personality, I think you'd enjoy this!"
        
        # Verify context was passed to LLM
        call_args = mock_llm_manager.generate_response.call_args
        assert call_args is not None
    
    @pytest.mark.asyncio
    async def test_process_voice_message_streaming(self, llm_config, voice_message, conversation_state):
        """Test streaming voice message processing."""
        llm_config.enable_streaming = True
        integration = VoiceLLMProcessor(llm_config)
        
        # Mock streaming LLM manager
        mock_llm_manager = AsyncMock()
        
        async def mock_stream():
            yield Mock(content="Hello", is_complete=False)
            yield Mock(content=" there", is_complete=False)
            yield Mock(content="!", is_complete=True)
        
        mock_llm_manager.generate_response_stream.return_value = mock_stream()
        integration.llm_manager = mock_llm_manager
        
        response = await integration.process_voice_message(
            voice_message, conversation_state
        )
        
        assert isinstance(response, VoiceResponse)
        # Should contain accumulated streaming content
        assert "Hello there!" in response.text_content
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, llm_config, voice_message, conversation_state):
        """Test timeout handling in LLM processing."""
        llm_config.llm_timeout_seconds = 0.1  # Very short timeout
        integration = VoiceLLMProcessor(llm_config)
        
        # Mock LLM manager that takes too long
        mock_llm_manager = AsyncMock()
        
        async def slow_response():
            await asyncio.sleep(1.0)  # Longer than timeout
            return Mock(content="Too slow")
        
        mock_llm_manager.generate_response.side_effect = slow_response
        integration.llm_manager = mock_llm_manager
        
        with pytest.raises(VoiceTimeoutError):
            await integration.process_voice_message(voice_message, conversation_state)
    
    @pytest.mark.asyncio
    async def test_retry_on_network_error(self, llm_config, voice_message, conversation_state):
        """Test retry mechanism on network errors."""
        integration = VoiceLLMProcessor(llm_config)
        
        # Mock LLM manager that fails first time, succeeds second time
        mock_llm_manager = AsyncMock()
        call_count = 0
        
        async def flaky_response():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NetworkError("Connection failed")
            return Mock(content="Success after retry")
        
        mock_llm_manager.generate_response.side_effect = flaky_response
        integration.llm_manager = mock_llm_manager
        
        response = await integration.process_voice_message(
            voice_message, conversation_state
        )
        
        assert response.text_content == "Success after retry"
        assert call_count == 2  # Should have retried once
    
    @pytest.mark.asyncio
    async def test_build_conversation_context(self, llm_config):
        """Test building conversation context for LLM."""
        integration = VoiceLLMProcessor(llm_config)
        
        conversation_id = "test-conversation"
        input_text = "What's the weather like?"
        context = {
            "memory": {
                "recent_conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            },
            "personality": {
                "traits": {"friendliness": 0.9}
            }
        }
        
        messages = await integration._build_conversation_context(
            conversation_id, input_text, context
        )
        
        assert isinstance(messages, list)
        assert len(messages) > 0
        
        # Should include system message, context, and user input
        assert any(msg.role == "system" for msg in messages)
        assert any(msg.role == "user" for msg in messages)
        assert any(input_text in msg.content for msg in messages)
    
    @pytest.mark.asyncio
    async def test_create_fallback_response(self, llm_config, voice_message):
        """Test creating fallback response."""
        integration = VoiceLLMProcessor(llm_config)
        
        fallback_response = integration._create_fallback_response(voice_message)
        
        assert isinstance(fallback_response, VoiceResponse)
        assert fallback_response.conversation_id == voice_message.conversation_id
        assert fallback_response.text_content is not None
        assert len(fallback_response.text_content) > 0


class TestLLMIntegrationErrorHandling:
    """Test error handling in LLM integration."""
    
    @pytest.mark.asyncio
    async def test_llm_manager_initialization_error(self):
        """Test error handling when LLM manager fails to initialize."""
        config = VoiceLLMConfig(llm_provider="invalid_provider")
        
        with patch('src.coda.components.voice.llm_integration.LLMManager') as mock_llm_manager:
            mock_llm_manager.side_effect = Exception("Invalid provider")
            
            with pytest.raises(LLMIntegrationError) as exc_info:
                VoiceLLMProcessor(config)
            
            assert "Failed to initialize LLM integration" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_llm_response_error_handling(self, llm_config, voice_message, conversation_state):
        """Test error handling when LLM response fails."""
        integration = VoiceLLMProcessor(llm_config)
        
        # Mock LLM manager that raises an error
        mock_llm_manager = AsyncMock()
        mock_llm_manager.generate_response.side_effect = Exception("LLM service unavailable")
        integration.llm_manager = mock_llm_manager
        
        # Should return fallback response instead of raising error
        response = await integration.process_voice_message(
            voice_message, conversation_state
        )
        
        assert isinstance(response, VoiceResponse)
        # Should be a fallback response
        assert response.text_content is not None
    
    @pytest.mark.asyncio
    async def test_context_building_error_handling(self, llm_config):
        """Test error handling when context building fails."""
        integration = VoiceLLMProcessor(llm_config)
        
        # Mock context that causes an error
        invalid_context = {
            "memory": "invalid_memory_format",  # Should be dict, not string
            "personality": None
        }
        
        # Should handle gracefully and return basic context
        messages = await integration._build_conversation_context(
            "test-conversation", "Hello", invalid_context
        )
        
        assert isinstance(messages, list)
        assert len(messages) > 0


if __name__ == "__main__":
    pytest.main([__file__])
