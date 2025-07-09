"""
Unit tests for the LLM system.
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.coda.components.llm.models import (
    LLMMessage,
    LLMConversation,
    LLMResponse,
    LLMStreamChunk,
    LLMProvider,
    ProviderConfig,
    LLMConfig,
    MessageRole,
    FunctionCall,
    FunctionCallResult,
)
from src.coda.components.llm.base_provider import BaseLLMProvider, LLMError
from src.coda.components.llm.conversation_manager import ConversationManager
from src.coda.components.llm.prompt_enhancer import PromptEnhancer
from src.coda.components.llm.function_calling_orchestrator import FunctionCallingOrchestrator
from src.coda.components.llm.manager import LLMManager
from src.coda.components.llm.providers.openai_provider import OpenAIProvider


class TestLLMMessage:
    """Test cases for LLMMessage."""
    
    def test_message_creation(self):
        """Test creating an LLM message."""
        message = LLMMessage(
            role=MessageRole.USER,
            content="Hello, world!",
            timestamp=datetime.now()
        )
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.timestamp is not None
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        message = LLMMessage(
            role=MessageRole.ASSISTANT,
            content="Hello there!",
            name="assistant"
        )
        
        message_dict = message.to_dict()
        
        assert message_dict["role"] == "assistant"
        assert message_dict["content"] == "Hello there!"
        assert message_dict["name"] == "assistant"
    
    def test_message_with_function_call(self):
        """Test message with function call."""
        function_call = {
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
        
        message = LLMMessage(
            role=MessageRole.ASSISTANT,
            content="",
            function_call=function_call
        )
        
        message_dict = message.to_dict()
        assert message_dict["function_call"] == function_call


class TestLLMConversation:
    """Test cases for LLMConversation."""
    
    def test_conversation_creation(self):
        """Test creating a conversation."""
        conversation = LLMConversation(
            conversation_id="test-conv-123"
        )
        
        assert conversation.conversation_id == "test-conv-123"
        assert len(conversation.messages) == 0
        assert conversation.total_tokens == 0
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        conversation = LLMConversation(conversation_id="test")
        
        message = LLMMessage(
            role=MessageRole.USER,
            content="Hello"
        )
        
        conversation.add_message(message)
        
        assert len(conversation.messages) == 1
        assert conversation.messages[0] == message
    
    def test_get_messages_for_llm(self):
        """Test getting messages formatted for LLM."""
        conversation = LLMConversation(conversation_id="test")
        
        # Add system message
        system_msg = LLMMessage(role=MessageRole.SYSTEM, content="You are helpful")
        conversation.add_message(system_msg)
        
        # Add user message
        user_msg = LLMMessage(role=MessageRole.USER, content="Hello")
        conversation.add_message(user_msg)
        
        # Get messages including system
        messages = conversation.get_messages_for_llm(include_system=True)
        assert len(messages) == 2
        
        # Get messages excluding system
        messages = conversation.get_messages_for_llm(include_system=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
    
    def test_update_token_usage(self):
        """Test updating token usage."""
        conversation = LLMConversation(conversation_id="test")
        
        conversation.update_token_usage(100, 50, 0.01)
        
        assert conversation.prompt_tokens == 100
        assert conversation.completion_tokens == 50
        assert conversation.total_tokens == 150
        assert conversation.total_cost == 0.01


class TestLLMResponse:
    """Test cases for LLMResponse."""
    
    def test_response_creation(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            response_id="resp-123",
            content="Hello there!",
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        
        assert response.response_id == "resp-123"
        assert response.content == "Hello there!"
        assert response.provider == LLMProvider.OPENAI
        assert response.total_tokens == 15
    
    def test_has_function_calls(self):
        """Test checking for function calls."""
        # Response without function calls
        response = LLMResponse(
            response_id="resp-1",
            content="Hello",
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo"
        )
        assert not response.has_function_calls()
        
        # Response with function calls
        response.function_calls = [{"name": "test", "arguments": "{}"}]
        assert response.has_function_calls()


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._responses = []
        self._should_fail = False
    
    def get_provider_name(self) -> LLMProvider:
        return LLMProvider.OPENAI
    
    def set_responses(self, responses):
        """Set mock responses."""
        self._responses = responses
    
    def set_should_fail(self, should_fail: bool):
        """Set whether requests should fail."""
        self._should_fail = should_fail
    
    async def _make_request(self, messages, stream=False, functions=None, **kwargs):
        if self._should_fail:
            raise LLMError("Mock error")
        
        if self._responses:
            return self._responses.pop(0)
        
        return {
            "choices": [{
                "message": {
                    "content": "Mock response",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
    
    def _parse_response(self, response):
        return LLMResponse(
            response_id=self._create_response_id(),
            content=response["choices"][0]["message"]["content"],
            provider=self.get_provider_name(),
            model=self.config.model,
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"]
        )
    
    async def _parse_streaming_response(self, response):
        yield LLMStreamChunk(
            chunk_id=self._create_chunk_id(),
            content="Mock streaming response",
            delta="Mock streaming response"
        )


class TestBaseLLMProvider:
    """Test cases for BaseLLMProvider."""
    
    @pytest.fixture
    def provider_config(self):
        return ProviderConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
    
    @pytest.fixture
    def mock_provider(self, provider_config):
        return MockLLMProvider(provider_config)
    
    def test_provider_initialization(self, mock_provider):
        """Test provider initialization."""
        assert mock_provider.get_provider_name() == LLMProvider.OPENAI
        assert mock_provider.get_model_name() == "gpt-3.5-turbo"
        assert mock_provider.is_available() is True
    
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_provider):
        """Test generating a response."""
        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello")
        ]
        
        response = await mock_provider.generate_response(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Mock response"
        assert response.total_tokens == 15
    
    @pytest.mark.asyncio
    async def test_generate_response_failure(self, mock_provider):
        """Test response generation failure."""
        mock_provider.set_should_fail(True)
        
        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello")
        ]
        
        with pytest.raises(LLMError):
            await mock_provider.generate_response(messages)
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_provider):
        """Test streaming response generation."""
        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello")
        ]
        
        chunks = []
        async for chunk in mock_provider.generate_streaming_response(messages):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert isinstance(chunks[0], LLMStreamChunk)
    
    def test_token_estimation(self, mock_provider):
        """Test token estimation."""
        text = "Hello, world!"
        tokens = mock_provider.estimate_tokens(text)
        
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_provider_stats(self, mock_provider):
        """Test provider statistics."""
        assert mock_provider.get_request_count() == 0
        assert mock_provider.get_total_tokens() == 0
        assert mock_provider.get_total_cost() == 0.0


@pytest.mark.asyncio
class TestConversationManager:
    """Test cases for ConversationManager."""
    
    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()
    
    async def test_create_conversation(self, conversation_manager):
        """Test creating a conversation."""
        conversation = await conversation_manager.create_conversation()
        
        assert conversation.conversation_id is not None
        assert len(conversation.messages) == 0
    
    async def test_get_conversation(self, conversation_manager):
        """Test getting a conversation."""
        # Create conversation
        conversation = await conversation_manager.create_conversation("test-123")
        
        # Retrieve conversation
        retrieved = await conversation_manager.get_conversation("test-123")
        
        assert retrieved is not None
        assert retrieved.conversation_id == "test-123"
    
    async def test_add_message(self, conversation_manager):
        """Test adding a message to conversation."""
        conversation = await conversation_manager.create_conversation("test-123")
        
        message = LLMMessage(
            role=MessageRole.USER,
            content="Hello"
        )
        
        await conversation_manager.add_message("test-123", message)
        
        updated_conversation = await conversation_manager.get_conversation("test-123")
        assert len(updated_conversation.messages) == 1
        assert updated_conversation.messages[0].content == "Hello"
    
    async def test_get_conversation_context(self, conversation_manager):
        """Test getting conversation context."""
        conversation = await conversation_manager.create_conversation("test-123")
        
        # Add multiple messages
        for i in range(5):
            message = LLMMessage(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"Message {i}"
            )
            await conversation_manager.add_message("test-123", message)
        
        # Get context with token limit
        context = await conversation_manager.get_conversation_context("test-123", max_tokens=100)
        
        assert len(context) <= 5
        assert all(isinstance(msg, LLMMessage) for msg in context)
    
    async def test_list_conversations(self, conversation_manager):
        """Test listing conversations."""
        # Create multiple conversations
        for i in range(3):
            await conversation_manager.create_conversation(f"test-{i}")
        
        conversations = await conversation_manager.list_conversations()
        
        assert len(conversations) == 3
    
    async def test_delete_conversation(self, conversation_manager):
        """Test deleting a conversation."""
        conversation = await conversation_manager.create_conversation("test-delete")
        
        # Delete conversation
        success = await conversation_manager.delete_conversation("test-delete")
        assert success is True
        
        # Verify deletion
        retrieved = await conversation_manager.get_conversation("test-delete")
        assert retrieved is None


class TestPromptEnhancer:
    """Test cases for PromptEnhancer."""
    
    @pytest.fixture
    def prompt_enhancer(self):
        return PromptEnhancer()
    
    @pytest.mark.asyncio
    async def test_enhance_system_prompt(self, prompt_enhancer):
        """Test enhancing system prompt."""
        base_prompt = "You are a helpful assistant."
        
        enhanced = await prompt_enhancer.enhance_system_prompt(base_prompt)
        
        assert len(enhanced) > len(base_prompt)
        assert "helpful assistant" in enhanced
    
    @pytest.mark.asyncio
    async def test_enhance_user_prompt(self, prompt_enhancer):
        """Test enhancing user prompt."""
        prompt = "What's the weather like?"
        
        enhanced = await prompt_enhancer.enhance_user_prompt(prompt)
        
        # Without memory manager, should return original prompt
        assert enhanced == prompt
    
    def test_format_conversation_history(self, prompt_enhancer):
        """Test formatting conversation history."""
        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
            LLMMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            LLMMessage(role=MessageRole.USER, content="How are you?")
        ]
        
        formatted = prompt_enhancer.format_conversation_history(messages)
        
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted
        assert "User: How are you?" in formatted


class TestFunctionCallingOrchestrator:
    """Test cases for FunctionCallingOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        return FunctionCallingOrchestrator()
    
    @pytest.mark.asyncio
    async def test_get_available_functions(self, orchestrator):
        """Test getting available functions."""
        # Without tool manager, should return empty list
        functions = await orchestrator.get_available_functions()
        assert functions == []
    
    def test_parse_function_calls_from_response(self, orchestrator):
        """Test parsing function calls from response."""
        response = LLMResponse(
            response_id="test",
            content="",
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            function_calls=[{
                "name": "get_weather",
                "arguments": '{"location": "New York"}'
            }]
        )
        
        function_calls = orchestrator.parse_function_calls_from_response(response)
        
        assert len(function_calls) == 1
        assert function_calls[0].name == "get_weather"
        assert function_calls[0].arguments == {"location": "New York"}
    
    def test_format_function_results_for_llm(self, orchestrator):
        """Test formatting function results for LLM."""
        results = [
            FunctionCallResult(
                call_id="call-1",
                function_name="get_weather",
                success=True,
                result="Sunny, 75°F",
                execution_time_ms=100
            )
        ]
        
        messages = orchestrator.format_function_results_for_llm(results)
        
        assert len(messages) == 1
        assert messages[0].role == MessageRole.FUNCTION
        assert messages[0].name == "get_weather"
        assert "Sunny, 75°F" in messages[0].content


@pytest.mark.asyncio
class TestLLMManager:
    """Test cases for LLMManager."""
    
    @pytest.fixture
    def llm_config(self):
        return LLMConfig(
            providers={
                "mock": ProviderConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-3.5-turbo"
                )
            },
            default_provider="mock"
        )
    
    @pytest.fixture
    def llm_manager(self, llm_config):
        manager = LLMManager(llm_config)
        
        # Replace provider with mock
        mock_provider = MockLLMProvider(llm_config.providers["mock"])
        manager._providers["mock"] = mock_provider
        
        return manager
    
    async def test_generate_response(self, llm_manager):
        """Test generating a response."""
        response = await llm_manager.generate_response("Hello, world!")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Mock response"
    
    async def test_continue_conversation(self, llm_manager):
        """Test continuing a conversation."""
        # Start conversation
        response1 = await llm_manager.generate_response("Hello", conversation_id="test-conv")
        
        # Continue conversation
        response2 = await llm_manager.continue_conversation("test-conv", "How are you?")
        
        assert response1.conversation_id == "test-conv"
        assert response2.conversation_id == "test-conv"
    
    async def test_list_providers(self, llm_manager):
        """Test listing providers."""
        providers = await llm_manager.list_providers()
        
        assert "mock" in providers
    
    async def test_get_provider_status(self, llm_manager):
        """Test getting provider status."""
        status = await llm_manager.get_provider_status("mock")
        
        assert "available" in status
        assert "provider" in status
        assert status["provider"] == "mock"
    
    async def test_get_analytics(self, llm_manager):
        """Test getting analytics."""
        analytics = await llm_manager.get_analytics()
        
        assert "providers" in analytics
        assert "conversations" in analytics
        assert "function_calling" in analytics
        assert "config" in analytics


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        # Test conversation manager
        conv_manager = ConversationManager()
        
        conversation = await conv_manager.create_conversation("test-123")
        print(f"Created conversation: {conversation.conversation_id}")
        
        message = LLMMessage(role=MessageRole.USER, content="Hello, world!")
        await conv_manager.add_message("test-123", message)
        
        retrieved = await conv_manager.get_conversation("test-123")
        print(f"Retrieved conversation with {len(retrieved.messages)} messages")
        
        # Test prompt enhancer
        enhancer = PromptEnhancer()
        enhanced = await enhancer.enhance_system_prompt("You are helpful.")
        print(f"Enhanced prompt length: {len(enhanced)}")
        
        # Test function calling orchestrator
        orchestrator = FunctionCallingOrchestrator()
        functions = await orchestrator.get_available_functions()
        print(f"Available functions: {len(functions)}")
        
        print("✅ LLM system test passed")
    
    asyncio.run(simple_test())
