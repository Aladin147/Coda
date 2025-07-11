#!/usr/bin/env python3
"""
Integration test for end-to-end conversation flow.

This test verifies that all components work together correctly
to process a complete conversation from user input to assistant response.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.coda.core.assistant import CodaAssistant
from src.coda.core.config import (
    CodaConfig, LLMConfig, MemoryConfig, ToolsConfig, PersonalityConfig,
    ShortTermMemoryConfig, LongTermMemoryConfig
)
from src.coda.components.llm.models import LLMProvider, ProviderConfig
from src.coda.components.tools.models import ToolConfig
from src.coda.components.personality.models import PersonalityConfig as PersonalityManagerConfig


@pytest.mark.asyncio
class TestConversationFlow:
    """Test end-to-end conversation flow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        return CodaConfig(
            llm=LLMConfig(
                providers={
                    "test": ProviderConfig(
                        provider=LLMProvider.OPENAI,
                        model="gpt-3.5-turbo",
                        api_key="test-key"
                    )
                },
                default_provider="test"
            ),
            memory=MemoryConfig(
                short_term=ShortTermMemoryConfig(max_turns=10),
                long_term=LongTermMemoryConfig(
                    storage_path=str(Path(temp_dir) / "memory"),
                    vector_db_type="in_memory",
                    max_memories=100
                )
            ),
            tools=ToolsConfig(
                manager=ToolConfig(
                    registry={"auto_discover_plugins": False},
                    executor={"default_timeout_seconds": 10.0}
                )
            ),
            personality=PersonalityConfig(
                manager=PersonalityManagerConfig(
                    enabled=False,  # Disable for testing
                    personality_file=None
                )
            )
        )
    
    @pytest_asyncio.fixture
    async def assistant(self, test_config):
        """Create test assistant with mocked LLM provider."""
        from src.coda.components.llm.models import LLMResponse, LLMMessage, MessageRole, LLMProvider

        assistant = CodaAssistant(test_config)

        # Mock the LLM provider to avoid real API calls
        mock_provider = AsyncMock()
        mock_provider.generate_response = AsyncMock()
        mock_provider.get_provider_name.return_value = LLMProvider.OPENAI

        # Set up mock response
        mock_response = LLMResponse(
            response_id="test-response-123",
            content="Hello! I'm Coda, your AI assistant. How can I help you today?",
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25
        )
        mock_provider.generate_response.return_value = mock_response
        
        # Initialize the assistant
        await assistant.initialize()

        # Replace the provider in the LLM manager
        if assistant.llm_manager:
            assistant.llm_manager._providers["test"] = mock_provider
            assistant.llm_manager._default_provider = "test"

        yield assistant

        # Cleanup
        await assistant.shutdown()
    
    async def test_basic_conversation_flow(self, assistant):
        """Test basic conversation flow from user input to response."""
        # Start a new conversation
        conversation_id = await assistant.start_conversation()
        assert conversation_id is not None
        
        # Send a message and get response
        response = await assistant.process_message(
            "Hello, who are you?",
            conversation_id=conversation_id
        )
        
        # Verify response
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert "Coda" in response.content
        
        # Verify conversation was stored in memory
        if assistant.memory_manager:
            context = assistant.memory_manager.get_context(max_tokens=800)
            assert len(context.messages) >= 2  # User message + assistant response
            
            # Check message roles
            user_msg = context.messages[-2]  # Second to last should be user
            assistant_msg = context.messages[-1]  # Last should be assistant
            
            assert user_msg.role.value == "user"
            assert assistant_msg.role.value == "assistant"
            assert "Hello, who are you?" in user_msg.content
    
    async def test_conversation_continuity(self, assistant):
        """Test that conversation context is maintained across multiple exchanges."""
        conversation_id = await assistant.start_conversation()
        
        # First exchange
        response1 = await assistant.process_message(
            "My name is Alice",
            conversation_id=conversation_id
        )
        assert response1 is not None
        
        # Second exchange - should remember the name
        response2 = await assistant.process_message(
            "What's my name?",
            conversation_id=conversation_id
        )
        assert response2 is not None
        
        # Verify memory contains both exchanges
        if assistant.memory_manager:
            context = await assistant.memory_manager.get_context(conversation_id)
            assert len(context.messages) >= 4  # 2 user + 2 assistant messages
    
    async def test_conversation_isolation(self, assistant):
        """Test that different conversations are isolated from each other."""
        # Start two separate conversations
        conv1_id = await assistant.start_conversation()
        conv2_id = await assistant.start_conversation()
        
        assert conv1_id != conv2_id
        
        # Send different messages to each conversation
        await assistant.process_message(
            "I like pizza",
            conversation_id=conv1_id
        )
        
        await assistant.process_message(
            "I like sushi",
            conversation_id=conv2_id
        )
        
        # Verify conversations are separate
        if assistant.memory_manager:
            context1 = await assistant.memory_manager.get_context(conv1_id)
            context2 = await assistant.memory_manager.get_context(conv2_id)
            
            # Each conversation should have its own messages
            conv1_content = " ".join([msg.content for msg in context1.messages])
            conv2_content = " ".join([msg.content for msg in context2.messages])
            
            assert "pizza" in conv1_content
            assert "pizza" not in conv2_content
            assert "sushi" in conv2_content
            assert "sushi" not in conv1_content
    
    async def test_health_check_integration(self, assistant):
        """Test that health check works with all components."""
        health = await assistant.health_check()
        
        assert health["status"] == "healthy"
        assert "components" in health
        
        # Check that all major components are reported
        components = health["components"]
        assert "event_bus" in components
        assert "memory_manager" in components
        assert "llm_manager" in components
        
        # All components should be healthy
        for component, status in components.items():
            assert status["status"] == "healthy"
    
    async def test_error_handling(self, assistant):
        """Test error handling in conversation flow."""
        # Test with invalid conversation ID
        try:
            await assistant.process_message(
                "Hello",
                conversation_id="invalid-id"
            )
            # Should not raise an exception, but handle gracefully
        except Exception as e:
            # If an exception is raised, it should be a known type
            assert isinstance(e, (ValueError, KeyError))
    
    async def test_conversation_metadata(self, assistant):
        """Test that conversation metadata is properly tracked."""
        conversation_id = await assistant.start_conversation()
        
        # Process a message
        await assistant.process_message(
            "Test message",
            conversation_id=conversation_id
        )
        
        # Check if we can get conversation info
        conversations = await assistant.list_conversations()
        assert len(conversations) >= 1
        
        # Find our conversation
        our_conv = None
        for conv in conversations:
            if conv.get("id") == conversation_id:
                our_conv = conv
                break
        
        if our_conv:
            assert "created_at" in our_conv
            assert "message_count" in our_conv or "messages" in our_conv
