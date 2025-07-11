"""
End-to-end system integration tests.
Tests complete system functionality from voice input to final response.
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
import json
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.coda.core.assistant import CodaAssistant
from src.coda.core.config import CodaConfig
from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
from src.coda.components.memory.models import Memory, MemoryType
from src.coda.interfaces.websocket.server import CodaWebSocketServer


class TestSystemIntegration:
    """End-to-end system integration tests."""

    @pytest_asyncio.fixture
    async def system_config(self):
        """Create system configuration for testing."""
        return CodaConfig(
            voice_enabled=True,
            memory_enabled=True,
            llm_enabled=True,
            websocket_enabled=True,
            debug_mode=True
        )

    @pytest_asyncio.fixture
    async def mock_components(self):
        """Create all mocked components for system testing."""
        components = {}
        
        # Mock Voice Manager
        voice_manager = Mock()
        voice_manager.process_voice_input = AsyncMock(return_value=VoiceResponse(
            response_id=str(uuid.uuid4()),
            conversation_id="system-test",
            message_id="msg-123",
            text_content="System integration response",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=200.0
        ))
        voice_manager.is_initialized = True
        components['voice'] = voice_manager
        
        # Mock Memory Manager
        memory_manager = Mock()
        memory_manager.store_conversation = AsyncMock(return_value="memory-stored")
        memory_manager.retrieve_relevant_memories = AsyncMock(return_value=[])
        memory_manager.is_initialized = True
        components['memory'] = memory_manager
        
        # Mock LLM Manager
        llm_manager = Mock()
        llm_manager.generate_response = AsyncMock(return_value="LLM generated response")
        llm_manager.is_initialized = True
        components['llm'] = llm_manager
        
        # Mock Personality Engine
        personality_engine = Mock()
        personality_engine.apply_personality = AsyncMock(return_value="Personalized response")
        personality_engine.is_initialized = True
        components['personality'] = personality_engine
        
        # Mock Tools Manager
        tools_manager = Mock()
        tools_manager.execute_tool = AsyncMock(return_value={"result": "tool executed"})
        tools_manager.is_initialized = True
        components['tools'] = tools_manager
        
        return components

    @pytest_asyncio.fixture
    async def integrated_system(self, system_config, mock_components):
        """Create fully integrated system with mocked components."""
        with patch.multiple(
            'src.coda.core.assistant',
            VoiceManager=lambda *args, **kwargs: mock_components['voice'],
            MemoryManager=lambda *args, **kwargs: mock_components['memory'],
            LLMManager=lambda *args, **kwargs: mock_components['llm'],
            PersonalityEngine=lambda *args, **kwargs: mock_components['personality'],
            ToolsManager=lambda *args, **kwargs: mock_components['tools']
        ):
            assistant = CodaAssistant(config=system_config)
            await assistant.initialize()
            return assistant, mock_components

    @pytest.mark.asyncio
    async def test_complete_system_initialization(self, integrated_system):
        """Test complete system initialization."""
        assistant, components = integrated_system
        
        # Verify system is initialized
        assert assistant.is_initialized
        
        # Verify all components are initialized
        for component_name, component in components.items():
            assert component.is_initialized, f"{component_name} not initialized"

    @pytest.mark.asyncio
    async def test_end_to_end_voice_processing(self, integrated_system):
        """Test end-to-end voice processing pipeline."""
        assistant, components = integrated_system
        
        # Create voice message
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="e2e-test",
            text_content="Hello, test the complete system",
            audio_data=b"fake_audio_data",
            processing_mode=VoiceProcessingMode.HYBRID
        )
        
        # Process through complete system
        response = await assistant.process_voice_message(voice_message)
        
        # Verify response
        assert isinstance(response, VoiceResponse)
        assert response.conversation_id == "e2e-test"
        assert response.total_latency_ms > 0
        
        # Verify components were called
        components['voice'].process_voice_input.assert_called_once()
        components['memory'].store_conversation.assert_called()

    @pytest.mark.asyncio
    async def test_system_with_websocket_integration(self, integrated_system):
        """Test system integration with WebSocket server."""
        assistant, components = integrated_system
        
        # Create WebSocket server
        with patch('websockets.serve') as mock_serve:
            mock_server = Mock()
            mock_server.close = AsyncMock()
            mock_server.wait_closed = AsyncMock()
            mock_serve.return_value = mock_server
            
            # Create and start WebSocket server
            ws_server = CodaWebSocketServer(
                host="localhost",
                port=8765,
                assistant=assistant
            )
            
            await ws_server.start()
            assert ws_server.is_running
            
            # Simulate WebSocket message processing
            mock_websocket = Mock()
            mock_websocket.send = AsyncMock()
            mock_websocket.closed = False
            
            # Test message handling
            message = {
                "type": "voice_input",
                "conversation_id": "ws-integration",
                "message_id": str(uuid.uuid4()),
                "text_content": "WebSocket integration test"
            }
            
            # Process message through WebSocket handler
            await ws_server.voice_handler._handle_message(
                mock_websocket, 
                json.dumps(message)
            )
            
            # Verify response was sent
            mock_websocket.send.assert_called()
            
            await ws_server.stop()

    @pytest.mark.asyncio
    async def test_system_error_recovery(self, integrated_system):
        """Test system error recovery and resilience."""
        assistant, components = integrated_system
        
        # Mock voice component to fail
        components['voice'].process_voice_input.side_effect = Exception("Voice processing failed")
        
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="error-recovery",
            text_content="This should trigger error recovery",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Verify system handles error gracefully
        with pytest.raises(Exception, match="Voice processing failed"):
            await assistant.process_voice_message(voice_message)
        
        # Verify system is still operational
        assert assistant.is_initialized

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, integrated_system):
        """Test system performance under concurrent load."""
        assistant, components = integrated_system
        
        # Create multiple concurrent requests
        messages = [
            VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"load-test-{i}",
                text_content=f"Load test message {i}",
                audio_data=b"fake_audio",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            for i in range(10)
        ]
        
        # Process concurrently
        start_time = time.time()
        tasks = [assistant.process_voice_message(msg) for msg in messages]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify all requests completed
        assert len(responses) == 10
        successful_responses = [r for r in responses if isinstance(r, VoiceResponse)]
        assert len(successful_responses) > 0
        
        # Verify reasonable performance
        total_time = end_time - start_time
        assert total_time < 10.0  # Should complete within 10 seconds

    @pytest.mark.asyncio
    async def test_system_memory_persistence(self, integrated_system):
        """Test system memory persistence across conversations."""
        assistant, components = integrated_system
        
        # First conversation
        msg1 = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="persistence-test",
            text_content="Remember that I like Python programming",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.HYBRID
        )
        
        response1 = await assistant.process_voice_message(msg1)
        assert isinstance(response1, VoiceResponse)
        
        # Second conversation referencing first
        msg2 = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="persistence-test",
            text_content="What programming language do I like?",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.HYBRID
        )
        
        response2 = await assistant.process_voice_message(msg2)
        assert isinstance(response2, VoiceResponse)
        
        # Verify memory operations
        assert components['memory'].store_conversation.call_count == 2
        assert components['memory'].retrieve_relevant_memories.call_count >= 1

    @pytest.mark.asyncio
    async def test_system_component_interaction(self, integrated_system):
        """Test interaction between all system components."""
        assistant, components = integrated_system
        
        # Configure components to interact
        components['memory'].retrieve_relevant_memories.return_value = [
            Memory(
                id="interaction-test",
                content="User prefers technical explanations",
                metadata=Mock(source_type=MemoryType.PREFERENCE, importance=0.8)
            )
        ]
        
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="component-interaction",
            text_content="Explain machine learning",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.HYBRID
        )
        
        response = await assistant.process_voice_message(voice_message)
        
        # Verify all components were involved
        components['voice'].process_voice_input.assert_called()
        components['memory'].retrieve_relevant_memories.assert_called()
        components['memory'].store_conversation.assert_called()
        
        assert isinstance(response, VoiceResponse)

    @pytest.mark.asyncio
    async def test_system_configuration_changes(self, integrated_system):
        """Test system behavior with configuration changes."""
        assistant, components = integrated_system
        
        # Test with different processing modes
        modes = [
            VoiceProcessingMode.MOSHI_ONLY,
            VoiceProcessingMode.LLM_ONLY,
            VoiceProcessingMode.HYBRID
        ]
        
        for mode in modes:
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id="config-test",
                text_content=f"Test with {mode} mode",
                audio_data=b"fake_audio",
                processing_mode=mode
            )
            
            response = await assistant.process_voice_message(voice_message)
            assert isinstance(response, VoiceResponse)
            assert response.processing_mode == mode

    @pytest.mark.asyncio
    async def test_system_graceful_shutdown(self, integrated_system):
        """Test system graceful shutdown."""
        assistant, components = integrated_system
        
        # Verify system is running
        assert assistant.is_initialized
        
        # Shutdown system
        await assistant.shutdown()
        
        # Verify system is properly shut down
        assert not assistant.is_initialized

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, integrated_system):
        """Test system health monitoring and metrics."""
        assistant, components = integrated_system
        
        # Process some messages to generate metrics
        for i in range(3):
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id="health-test",
                text_content=f"Health check message {i}",
                audio_data=b"fake_audio",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            
            await assistant.process_voice_message(voice_message)
        
        # Get system health metrics
        health = await assistant.get_health_status()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'components' in health
        assert 'metrics' in health

    @pytest.mark.asyncio
    async def test_system_real_time_streaming(self, integrated_system):
        """Test system real-time streaming capabilities."""
        assistant, components = integrated_system
        
        # Mock streaming response
        async def mock_stream():
            for i in range(5):
                yield f"Stream chunk {i}"
                await asyncio.sleep(0.01)
        
        components['voice'].stream_response = AsyncMock(return_value=mock_stream())
        
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="streaming-test",
            text_content="Stream a response",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Test streaming
        chunks = []
        async for chunk in assistant.stream_voice_response(voice_message):
            chunks.append(chunk)
        
        # Verify streaming worked
        assert len(chunks) > 0
        components['voice'].stream_response.assert_called()

    @pytest.mark.asyncio
    async def test_system_data_flow_integrity(self, integrated_system):
        """Test data flow integrity throughout the system."""
        assistant, components = integrated_system
        
        # Create message with specific data
        test_data = {
            "conversation_id": "data-flow-test",
            "user_input": "Test data flow integrity",
            "timestamp": datetime.now().isoformat()
        }
        
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id=test_data["conversation_id"],
            text_content=test_data["user_input"],
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.HYBRID
        )
        
        response = await assistant.process_voice_message(voice_message)
        
        # Verify data integrity
        assert response.conversation_id == test_data["conversation_id"]
        assert response.message_id == voice_message.message_id
        
        # Verify data was passed correctly to components
        voice_call_args = components['voice'].process_voice_input.call_args[0][0]
        assert voice_call_args.conversation_id == test_data["conversation_id"]
