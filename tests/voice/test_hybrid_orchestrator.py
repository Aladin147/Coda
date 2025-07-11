"""
Tests for hybrid orchestrator components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.coda.components.voice.hybrid_orchestrator import HybridOrchestrator, HybridConfig
from src.coda.components.voice.models import (
    VoiceMessage, VoiceResponse, ConversationState, VoiceProcessingMode
)
from src.coda.components.voice.exceptions import ComponentFailureError, VoiceTimeoutError


class TestHybridConfig:
    """Test HybridConfig functionality."""
    
    def test_hybrid_config_defaults(self):
        """Test HybridConfig with default values."""
        config = HybridConfig()
        
        assert config.enable_parallel_processing is True
        assert config.moshi_timeout_seconds == 5.0
        assert config.llm_timeout_seconds == 10.0
        assert config.total_timeout_seconds == 15.0
        assert config.fallback_to_moshi_only is True
        assert config.quality_threshold == 0.7
    
    def test_hybrid_config_custom(self):
        """Test HybridConfig with custom values."""
        config = HybridConfig(
            enable_parallel_processing=False,
            moshi_timeout_seconds=3.0,
            llm_timeout_seconds=8.0,
            total_timeout_seconds=12.0,
            fallback_to_moshi_only=False,
            quality_threshold=0.8
        )
        
        assert config.enable_parallel_processing is False
        assert config.moshi_timeout_seconds == 3.0
        assert config.llm_timeout_seconds == 8.0
        assert config.total_timeout_seconds == 12.0
        assert config.fallback_to_moshi_only is False
        assert config.quality_threshold == 0.8


class TestHybridOrchestrator:
    """Test HybridOrchestrator functionality."""
    
    @pytest.fixture
    def hybrid_config(self):
        """Create test hybrid configuration."""
        return HybridConfig(
            moshi_timeout_seconds=2.0,
            llm_timeout_seconds=3.0,
            total_timeout_seconds=5.0
        )
    
    @pytest.fixture
    def voice_message(self):
        """Create test voice message."""
        return VoiceMessage(
            conversation_id="test-conversation",
            text_content="What's the weather like today?",
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
    
    def test_hybrid_orchestrator_init(self, hybrid_config):
        """Test HybridOrchestrator initialization."""
        # Mock dependencies
        mock_moshi = Mock()
        mock_llm = Mock()
        mock_fallback = Mock()
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        assert orchestrator.config == hybrid_config
        assert orchestrator.moshi_integration == mock_moshi
        assert orchestrator.llm_integration == mock_llm
        assert orchestrator.fallback_manager == mock_fallback
        assert orchestrator.latency_tracker is not None
    
    @pytest.mark.asyncio
    async def test_process_parallel_success(self, hybrid_config, voice_message, conversation_state):
        """Test successful parallel processing."""
        # Mock components
        mock_moshi = AsyncMock()
        mock_llm = AsyncMock()
        mock_fallback = AsyncMock()
        
        # Mock responses
        moshi_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="It's sunny today",
            audio_data=b"moshi_audio",
            timestamp=1234567890.0,
            processing_time=1.5,
            confidence_score=0.8
        )
        
        llm_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="Based on current weather data, it's a beautiful sunny day with 22°C",
            audio_data=None,
            timestamp=1234567890.0,
            processing_time=2.5,
            confidence_score=0.9
        )
        
        mock_moshi.process_voice_message.return_value = moshi_response
        mock_llm.process_voice_message.return_value = llm_response
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        result = await orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        assert isinstance(result, VoiceResponse)
        # Should prefer LLM response due to higher confidence
        assert result.text_content == llm_response.text_content
        assert result.confidence_score == llm_response.confidence_score
        
        # Both components should have been called
        mock_moshi.process_voice_message.assert_called_once()
        mock_llm.process_voice_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_moshi_only_mode(self, hybrid_config, voice_message, conversation_state):
        """Test Moshi-only processing mode."""
        mock_moshi = AsyncMock()
        mock_llm = AsyncMock()
        mock_fallback = AsyncMock()
        
        moshi_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="Quick Moshi response",
            audio_data=b"moshi_audio",
            timestamp=1234567890.0
        )
        
        mock_moshi.process_voice_message.return_value = moshi_response
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        result = await orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        assert result == moshi_response
        
        # Only Moshi should have been called
        mock_moshi.process_voice_message.assert_called_once()
        mock_llm.process_voice_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_llm_enhanced_mode(self, hybrid_config, voice_message, conversation_state):
        """Test LLM-enhanced processing mode."""
        mock_moshi = AsyncMock()
        mock_llm = AsyncMock()
        mock_fallback = AsyncMock()
        
        llm_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="Enhanced LLM response with reasoning",
            audio_data=None,
            timestamp=1234567890.0
        )
        
        mock_llm.process_voice_message.return_value = llm_response
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        result = await orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.LLM_ENHANCED
        )
        
        assert result == llm_response
        
        # Only LLM should have been called
        mock_llm.process_voice_message.assert_called_once()
        mock_moshi.process_voice_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_adaptive_mode_selection(self, hybrid_config, voice_message, conversation_state):
        """Test adaptive mode selection based on message characteristics."""
        mock_moshi = AsyncMock()
        mock_llm = AsyncMock()
        mock_fallback = AsyncMock()
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        # Mock the mode selection logic
        with patch.object(orchestrator, '_select_optimal_mode') as mock_select:
            mock_select.return_value = VoiceProcessingMode.MOSHI_ONLY
            
            moshi_response = VoiceResponse(
                conversation_id="test-conversation",
                text_content="Quick response",
                audio_data=b"audio",
                timestamp=1234567890.0
            )
            mock_moshi.process_voice_message.return_value = moshi_response
            
            result = await orchestrator.process_voice_message(
                voice_message, conversation_state, mode=VoiceProcessingMode.ADAPTIVE
            )
            
            assert result == moshi_response
            mock_select.assert_called_once_with(voice_message, conversation_state)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, hybrid_config, voice_message, conversation_state):
        """Test timeout handling in hybrid processing."""
        mock_moshi = AsyncMock()
        mock_llm = AsyncMock()
        mock_fallback = AsyncMock()
        
        # Mock slow responses
        async def slow_moshi_response(*args, **kwargs):
            await asyncio.sleep(10.0)  # Longer than timeout
            return VoiceResponse(conversation_id="test", text_content="slow")
        
        async def slow_llm_response(*args, **kwargs):
            await asyncio.sleep(10.0)  # Longer than timeout
            return VoiceResponse(conversation_id="test", text_content="slow")
        
        mock_moshi.process_voice_message.side_effect = slow_moshi_response
        mock_llm.process_voice_message.side_effect = slow_llm_response
        
        # Mock fallback response
        fallback_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="Fallback response",
            audio_data=None,
            timestamp=1234567890.0
        )
        mock_fallback.get_fallback_response.return_value = fallback_response
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        result = await orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        # Should return fallback response due to timeout
        assert result == fallback_response
        mock_fallback.get_fallback_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_component_failure_fallback(self, hybrid_config, voice_message, conversation_state):
        """Test fallback when components fail."""
        mock_moshi = AsyncMock()
        mock_llm = AsyncMock()
        mock_fallback = AsyncMock()
        
        # Mock component failures
        mock_moshi.process_voice_message.side_effect = ComponentFailureError(
            "moshi", "Model crashed"
        )
        mock_llm.process_voice_message.side_effect = ComponentFailureError(
            "llm", "Service unavailable"
        )
        
        # Mock fallback response
        fallback_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="I'm having technical difficulties. Please try again.",
            audio_data=None,
            timestamp=1234567890.0
        )
        mock_fallback.get_fallback_response.return_value = fallback_response
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        result = await orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        assert result == fallback_response
        mock_fallback.get_fallback_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_response_quality_assessment(self, hybrid_config, voice_message, conversation_state):
        """Test response quality assessment and selection."""
        mock_moshi = AsyncMock()
        mock_llm = AsyncMock()
        mock_fallback = AsyncMock()
        
        # Mock responses with different quality scores
        low_quality_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="Hmm",
            audio_data=b"audio",
            timestamp=1234567890.0,
            confidence_score=0.3  # Low quality
        )
        
        high_quality_response = VoiceResponse(
            conversation_id="test-conversation",
            text_content="That's an excellent question about weather patterns...",
            audio_data=None,
            timestamp=1234567890.0,
            confidence_score=0.95  # High quality
        )
        
        mock_moshi.process_voice_message.return_value = low_quality_response
        mock_llm.process_voice_message.return_value = high_quality_response
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        result = await orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        # Should select the higher quality response
        assert result == high_quality_response
        assert result.confidence_score == 0.95
    
    def test_select_optimal_mode_simple_query(self, hybrid_config):
        """Test mode selection for simple queries."""
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=Mock(),
            llm_integration=Mock(),
            fallback_manager=Mock()
        )
        
        # Simple greeting
        simple_message = VoiceMessage(
            conversation_id="test",
            text_content="Hello",
            audio_data=b"audio",
            timestamp=1234567890.0,
            speaker="user"
        )
        
        conversation_state = ConversationState(
            conversation_id="test",
            is_active=True,
            participant_count=2,
            last_activity=1234567890.0
        )
        
        mode = orchestrator._select_optimal_mode(simple_message, conversation_state)
        
        # Simple queries should prefer Moshi for speed
        assert mode == VoiceProcessingMode.MOSHI_ONLY
    
    def test_select_optimal_mode_complex_query(self, hybrid_config):
        """Test mode selection for complex queries."""
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=Mock(),
            llm_integration=Mock(),
            fallback_manager=Mock()
        )
        
        # Complex analytical question
        complex_message = VoiceMessage(
            conversation_id="test",
            text_content="Can you analyze the economic implications of climate change on agriculture?",
            audio_data=b"audio",
            timestamp=1234567890.0,
            speaker="user"
        )
        
        conversation_state = ConversationState(
            conversation_id="test",
            is_active=True,
            participant_count=2,
            last_activity=1234567890.0
        )
        
        mode = orchestrator._select_optimal_mode(complex_message, conversation_state)
        
        # Complex queries should prefer LLM for reasoning
        assert mode in [VoiceProcessingMode.LLM_ENHANCED, VoiceProcessingMode.HYBRID]


if __name__ == "__main__":
    pytest.main([__file__])
