"""
Integration tests for hybrid voice processing system.

Tests the coordination between Moshi and LLM processing,
including parallel processing, response selection, and fallback mechanisms.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.coda.components.voice.hybrid_orchestrator import (
    HybridOrchestrator, HybridConfig, ProcessingStrategy, ResponseSelection
)
from src.coda.components.voice.models import (
    VoiceMessage, VoiceResponse, VoiceProcessingMode, ConversationState
)
from src.coda.components.voice.exceptions import (
    ComponentFailureError, VoiceTimeoutError, VoiceProcessingError
)


class TestHybridProcessingIntegration:
    """Integration tests for hybrid voice processing."""
    
    @pytest.fixture
    def hybrid_config(self):
        """Create hybrid processing configuration."""
        return HybridConfig(
            default_strategy=ProcessingStrategy.HYBRID_PARALLEL,
            response_selection=ResponseSelection.BALANCED,
            moshi_timeout_ms=1000.0,
            llm_timeout_ms=2000.0,
            hybrid_timeout_ms=2500.0,
            enable_parallel_processing=True,
            enable_fallback=True
        )
    
    @pytest.fixture
    def voice_message(self):
        """Create test voice message."""
        return VoiceMessage(
            conversation_id="hybrid-test-conv",
            audio_data=b"mock_audio_data",
            text_content="What's the weather like today?",
            speaker="user"
        )
    
    @pytest.fixture
    def conversation_state(self):
        """Create test conversation state."""
        return ConversationState(
            conversation_id="hybrid-test-conv",
            is_active=True,
            participant_count=2,
            last_activity=time.time()
        )
    
    @pytest.fixture
    async def hybrid_orchestrator(self, hybrid_config):
        """Create hybrid orchestrator with mocked components."""
        # Mock Moshi integration
        mock_moshi = AsyncMock()
        mock_moshi.process_voice_message = AsyncMock()
        
        # Mock LLM integration
        mock_llm = AsyncMock()
        mock_llm.process_voice_message = AsyncMock()
        
        # Mock fallback manager
        mock_fallback = AsyncMock()
        mock_fallback.get_fallback_response = AsyncMock()
        
        orchestrator = HybridOrchestrator(
            config=hybrid_config,
            moshi_integration=mock_moshi,
            llm_integration=mock_llm,
            fallback_manager=mock_fallback
        )
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_parallel_processing_success(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test successful parallel processing with both Moshi and LLM."""
        # Setup mock responses
        moshi_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="It's sunny today!",
            audio_data=b"moshi_audio",
            processing_time=800.0,
            confidence_score=0.85
        )
        
        llm_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="Based on current weather data, it's a beautiful sunny day with 22°C and light winds.",
            processing_time=1500.0,
            confidence_score=0.95
        )
        
        hybrid_orchestrator.moshi_integration.process_voice_message.return_value = moshi_response
        hybrid_orchestrator.llm_integration.process_voice_message.return_value = llm_response
        
        # Process with hybrid mode
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        # Should select LLM response due to higher confidence
        assert result.text_content == llm_response.text_content
        assert result.confidence_score == llm_response.confidence_score
        
        # Both components should have been called
        hybrid_orchestrator.moshi_integration.process_voice_message.assert_called_once()
        hybrid_orchestrator.llm_integration.process_voice_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_moshi_only_processing(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test Moshi-only processing mode."""
        moshi_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="Quick Moshi response",
            audio_data=b"moshi_audio",
            processing_time=200.0,
            confidence_score=0.8
        )
        
        hybrid_orchestrator.moshi_integration.process_voice_message.return_value = moshi_response
        
        # Process with Moshi-only mode
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        assert result == moshi_response
        
        # Only Moshi should have been called
        hybrid_orchestrator.moshi_integration.process_voice_message.assert_called_once()
        hybrid_orchestrator.llm_integration.process_voice_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_llm_enhanced_processing(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test LLM-enhanced processing mode."""
        llm_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="Detailed LLM analysis with reasoning",
            processing_time=1200.0,
            confidence_score=0.92
        )
        
        hybrid_orchestrator.llm_integration.process_voice_message.return_value = llm_response
        
        # Process with LLM-enhanced mode
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.LLM_ENHANCED
        )
        
        assert result == llm_response
        
        # Only LLM should have been called
        hybrid_orchestrator.llm_integration.process_voice_message.assert_called_once()
        hybrid_orchestrator.moshi_integration.process_voice_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_adaptive_mode_selection(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test adaptive mode selection based on message characteristics."""
        # Mock the mode selection logic
        with patch.object(hybrid_orchestrator, '_select_optimal_mode') as mock_select:
            mock_select.return_value = VoiceProcessingMode.MOSHI_ONLY
            
            moshi_response = VoiceResponse(
                conversation_id=voice_message.conversation_id,
                text_content="Adaptive selected Moshi",
                processing_time=150.0
            )
            hybrid_orchestrator.moshi_integration.process_voice_message.return_value = moshi_response
            
            # Process with adaptive mode
            result = await hybrid_orchestrator.process_voice_message(
                voice_message, conversation_state, mode=VoiceProcessingMode.ADAPTIVE
            )
            
            assert result == moshi_response
            mock_select.assert_called_once_with(voice_message, conversation_state)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test timeout handling in hybrid processing."""
        # Mock slow responses
        async def slow_moshi_response(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return VoiceResponse(conversation_id="test", text_content="slow")
        
        async def slow_llm_response(*args, **kwargs):
            await asyncio.sleep(3.0)  # Longer than timeout
            return VoiceResponse(conversation_id="test", text_content="slow")
        
        hybrid_orchestrator.moshi_integration.process_voice_message.side_effect = slow_moshi_response
        hybrid_orchestrator.llm_integration.process_voice_message.side_effect = slow_llm_response
        
        # Mock fallback response
        fallback_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="I'm having trouble processing that. Please try again.",
            processing_time=50.0
        )
        hybrid_orchestrator.fallback_manager.get_fallback_response.return_value = fallback_response
        
        # Process should timeout and use fallback
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        assert result == fallback_response
        hybrid_orchestrator.fallback_manager.get_fallback_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_component_failure_fallback(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test fallback when components fail."""
        # Mock component failures
        hybrid_orchestrator.moshi_integration.process_voice_message.side_effect = ComponentFailureError(
            "moshi", "Model crashed"
        )
        hybrid_orchestrator.llm_integration.process_voice_message.side_effect = ComponentFailureError(
            "llm", "Service unavailable"
        )
        
        # Mock fallback response
        fallback_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="I'm experiencing technical difficulties. Please try again later.",
            processing_time=25.0
        )
        hybrid_orchestrator.fallback_manager.get_fallback_response.return_value = fallback_response
        
        # Process should use fallback
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        assert result == fallback_response
        hybrid_orchestrator.fallback_manager.get_fallback_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_response_quality_assessment(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test response quality assessment and selection."""
        # Mock responses with different quality scores
        low_quality_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="Hmm",
            processing_time=100.0,
            confidence_score=0.3
        )
        
        high_quality_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="That's an excellent question about weather patterns. Let me provide you with detailed information...",
            processing_time=800.0,
            confidence_score=0.95
        )
        
        hybrid_orchestrator.moshi_integration.process_voice_message.return_value = low_quality_response
        hybrid_orchestrator.llm_integration.process_voice_message.return_value = high_quality_response
        
        # Process with hybrid mode
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.HYBRID
        )
        
        # Should select the higher quality response
        assert result == high_quality_response
        assert result.confidence_score == 0.95
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test performance monitoring in hybrid processing."""
        moshi_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="Moshi response",
            processing_time=200.0
        )
        
        hybrid_orchestrator.moshi_integration.process_voice_message.return_value = moshi_response
        
        # Process and check performance tracking
        start_time = time.time()
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.MOSHI_ONLY
        )
        end_time = time.time()
        
        # Verify performance metrics
        assert result.processing_time is not None
        assert result.processing_time > 0
        
        # Check latency tracker
        if hasattr(hybrid_orchestrator, 'latency_tracker'):
            assert hybrid_orchestrator.latency_tracker.get_average_latency() >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_requests(self, hybrid_orchestrator, conversation_state):
        """Test handling multiple concurrent processing requests."""
        num_requests = 5
        voice_messages = [
            VoiceMessage(
                conversation_id=f"conv_{i}",
                audio_data=b"mock_audio",
                text_content=f"Message {i}",
                speaker="user"
            )
            for i in range(num_requests)
        ]
        
        # Mock responses
        def create_mock_response(i):
            return VoiceResponse(
                conversation_id=f"conv_{i}",
                text_content=f"Response {i}",
                processing_time=100.0 + i * 10
            )
        
        hybrid_orchestrator.moshi_integration.process_voice_message.side_effect = [
            create_mock_response(i) for i in range(num_requests)
        ]
        
        # Process all requests concurrently
        tasks = []
        for i, msg in enumerate(voice_messages):
            conv_state = ConversationState(
                conversation_id=f"conv_{i}",
                is_active=True,
                participant_count=2,
                last_activity=time.time()
            )
            task = hybrid_orchestrator.process_voice_message(
                msg, conv_state, mode=VoiceProcessingMode.MOSHI_ONLY
            )
            tasks.append(task)
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses
        assert len(responses) == num_requests
        for i, response in enumerate(responses):
            assert response.conversation_id == f"conv_{i}"
            assert response.text_content == f"Response {i}"
    
    @pytest.mark.asyncio
    async def test_context_integration(self, hybrid_orchestrator, voice_message, conversation_state):
        """Test integration with context management."""
        # Mock context data
        context = {
            "memory": {
                "recent_topics": ["weather", "travel"],
                "user_preferences": {"units": "metric"}
            },
            "personality": {
                "traits": {"friendliness": 0.9, "verbosity": 0.7}
            },
            "tools": {
                "available_functions": ["get_weather", "search_web"]
            }
        }
        
        # Mock response with context integration
        enhanced_response = VoiceResponse(
            conversation_id=voice_message.conversation_id,
            text_content="Based on your preference for metric units and recent interest in weather, the temperature is 22°C with sunny skies.",
            processing_time=1200.0,
            confidence_score=0.93
        )
        
        hybrid_orchestrator.llm_integration.process_voice_message.return_value = enhanced_response
        
        # Process with context
        result = await hybrid_orchestrator.process_voice_message(
            voice_message, conversation_state, mode=VoiceProcessingMode.LLM_ENHANCED, context=context
        )
        
        assert result == enhanced_response
        
        # Verify context was passed to LLM integration
        call_args = hybrid_orchestrator.llm_integration.process_voice_message.call_args
        assert call_args is not None
        # Check if context was included in the call
        if len(call_args[0]) > 2:  # If context is passed as third argument
            passed_context = call_args[0][2]
            assert "memory" in passed_context or "personality" in passed_context


if __name__ == "__main__":
    pytest.main([__file__])
