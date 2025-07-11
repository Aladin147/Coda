"""
WebSocket integration for the LLM system.

This module provides WebSocket event broadcasting for real-time
LLM monitoring and streaming responses.
"""

import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from .interfaces import WebSocketLLMInterface
from .manager import LLMManager
from .models import (
    FunctionCall,
    LLMConfig,
    LLMConversation,
    LLMEvent,
    LLMMessage,
    LLMResponse,
    LLMResponseType,
    LLMStreamChunk,
)

logger = logging.getLogger("coda.llm.websocket")


class WebSocketLLMManager(LLMManager, WebSocketLLMInterface):
    """
    LLM manager with WebSocket integration for real-time events.

    Features:
    - Real-time response streaming
    - Function call broadcasting
    - Provider status updates
    - Conversation event streaming
    - Performance metrics broadcasting
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the WebSocket LLM manager.

        Args:
            config: LLM system configuration
        """
        super().__init__(config)

        self._websocket_integration: Optional[Any] = None
        self._event_queue: List[LLMEvent] = []

        logger.info("WebSocketLLMManager initialized")

    async def set_websocket_integration(self, integration: Any) -> None:
        """
        Set WebSocket integration for event broadcasting.

        Args:
            integration: WebSocket integration instance
        """
        self._websocket_integration = integration

        # Broadcast initial state
        await self._broadcast_initial_state()

        logger.info("WebSocket integration set for LLM manager")

    async def broadcast_llm_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast an LLM event via WebSocket.

        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.config.websocket_events_enabled or not self._websocket_integration:
            return

        event = LLMEvent(event_type=event_type, data=data, timestamp=datetime.now())

        try:
            await self._websocket_integration.broadcast_event("llm_event", event.model_dump())
            logger.debug(f"Broadcasted LLM event: {event_type}")
        except Exception as e:
            logger.warning(f"Failed to broadcast LLM event: {e}")
            # Queue event for retry
            self._event_queue.append(event)

    async def broadcast_response_start(
        self, conversation_id: str, provider: str, model: str
    ) -> None:
        """
        Broadcast response generation start.

        Args:
            conversation_id: Conversation ID
            provider: Provider name
            model: Model name
        """
        await self.broadcast_llm_event(
            "response_start",
            {
                "conversation_id": conversation_id,
                "provider": provider,
                "model": model,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def broadcast_response_chunk(self, conversation_id: str, chunk: LLMStreamChunk) -> None:
        """
        Broadcast response chunk.

        Args:
            conversation_id: Conversation ID
            chunk: Stream chunk
        """
        await self.broadcast_llm_event(
            "response_chunk",
            {
                "conversation_id": conversation_id,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "delta": chunk.delta,
                "finish_reason": chunk.finish_reason,
                "timestamp": chunk.timestamp.isoformat(),
            },
        )

    async def broadcast_response_complete(
        self, conversation_id: str, response: LLMResponse
    ) -> None:
        """
        Broadcast response completion.

        Args:
            conversation_id: Conversation ID
            response: Complete response
        """
        await self.broadcast_llm_event(
            "response_complete",
            {
                "conversation_id": conversation_id,
                "response_id": response.response_id,
                "provider": response.provider.value,
                "model": response.model,
                "content_length": len(response.content),
                "total_tokens": response.total_tokens,
                "execution_time_ms": response.response_time_ms,
                "has_function_calls": response.has_function_calls(),
                "timestamp": response.timestamp.isoformat(),
            },
        )

    async def broadcast_function_calls(
        self, conversation_id: str, function_calls: List[FunctionCall]
    ) -> None:
        """
        Broadcast function calls.

        Args:
            conversation_id: Conversation ID
            function_calls: List of function calls
        """
        await self.broadcast_llm_event(
            "function_calls",
            {
                "conversation_id": conversation_id,
                "function_calls": [
                    {
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "timestamp": call.timestamp.isoformat(),
                    }
                    for call in function_calls
                ],
                "count": len(function_calls),
            },
        )

    async def generate_response(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        provider: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponseType:
        """
        Generate response with WebSocket events.

        Args:
            prompt: User prompt
            conversation_id: Optional conversation ID
            provider: Optional provider name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            LLM response or streaming response
        """
        # Broadcast request start
        await self.broadcast_llm_event(
            "request_start",
            {
                "conversation_id": conversation_id,
                "provider": provider or self._default_provider,
                "stream": stream,
                "prompt_length": len(prompt),
                "timestamp": datetime.now().isoformat(),
            },
        )

        try:
            # Generate response using parent method
            if stream:
                return await self._generate_streaming_response_with_events(
                    prompt, conversation_id, provider, **kwargs
                )
            else:
                response = await super().generate_response(
                    prompt, conversation_id, provider, stream, **kwargs
                )

                # Broadcast completion
                await self.broadcast_response_complete(
                    response.conversation_id or conversation_id or "unknown", response
                )

                return response

        except Exception as e:
            # Broadcast error
            await self.broadcast_llm_event(
                "request_error",
                {
                    "conversation_id": conversation_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            raise

    async def _generate_streaming_response_with_events(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate streaming response with WebSocket events."""
        provider_name = provider or self._default_provider

        # Broadcast stream start
        await self.broadcast_response_start(
            conversation_id or "unknown",
            provider_name or "unknown",
            "unknown",  # Model name would be determined by provider
        )

        try:
            async for chunk in super().generate_response(
                prompt, conversation_id, provider, stream=True, **kwargs
            ):
                # Broadcast each chunk
                await self.broadcast_response_chunk(conversation_id or "unknown", chunk)

                yield chunk

        except Exception as e:
            # Broadcast stream error
            await self.broadcast_llm_event(
                "stream_error",
                {
                    "conversation_id": conversation_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            raise

    async def broadcast_conversation_created(self, conversation: LLMConversation) -> None:
        """Broadcast conversation creation."""
        await self.broadcast_llm_event(
            "conversation_created",
            {
                "conversation_id": conversation.conversation_id,
                "created_at": conversation.created_at.isoformat(),
                "message_count": len(conversation.messages),
            },
        )

    async def broadcast_conversation_updated(self, conversation: LLMConversation) -> None:
        """Broadcast conversation update."""
        await self.broadcast_llm_event(
            "conversation_updated",
            {
                "conversation_id": conversation.conversation_id,
                "updated_at": conversation.updated_at.isoformat(),
                "message_count": len(conversation.messages),
                "total_tokens": conversation.total_tokens,
                "total_cost": conversation.total_cost,
            },
        )

    async def broadcast_provider_status_change(self, provider: str, status: Dict[str, Any]) -> None:
        """Broadcast provider status change."""
        await self.broadcast_llm_event(
            "provider_status_change",
            {"provider": provider, "status": status, "timestamp": datetime.now().isoformat()},
        )

    async def broadcast_analytics_update(self) -> None:
        """Broadcast analytics update."""
        analytics = await self.get_analytics()

        await self.broadcast_llm_event(
            "analytics_update", {"analytics": analytics, "timestamp": datetime.now().isoformat()}
        )

    async def _broadcast_initial_state(self) -> None:
        """Broadcast initial LLM system state."""
        # Broadcast provider status
        providers = await self.list_providers()
        for provider in providers:
            status = await self.get_provider_status(provider)
            await self.broadcast_provider_status_change(provider, status)

        # Broadcast analytics
        await self.broadcast_analytics_update()

        # Broadcast system configuration
        await self.broadcast_llm_event(
            "system_config",
            {
                "default_provider": self._default_provider,
                "function_calling_enabled": self.config.function_calling.enabled,
                "memory_integration_enabled": self.config.memory_integration_enabled,
                "personality_integration_enabled": self.config.personality_integration_enabled,
                "stream_responses": self.config.stream_responses,
                "providers": list(self._providers.keys()),
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info("Broadcasted initial LLM system state")

    async def _retry_queued_events(self) -> None:
        """Retry broadcasting queued events."""
        if not self._event_queue or not self._websocket_integration:
            return

        retry_queue = self._event_queue.copy()
        self._event_queue.clear()

        for event in retry_queue:
            try:
                await self._websocket_integration.broadcast_event("llm_event", event.model_dump())
                logger.debug(f"Retried broadcasting event: {event.event_type}")
            except Exception as e:
                logger.warning(f"Failed to retry event broadcast: {e}")
                # Re-queue if still failing
                self._event_queue.append(event)

    def get_websocket_stats(self) -> Dict[str, Any]:
        """Get WebSocket integration statistics."""
        return {
            "websocket_enabled": self.config.websocket_events_enabled,
            "integration_connected": self._websocket_integration is not None,
            "queued_events": len(self._event_queue),
            "stream_responses": self.config.stream_responses,
        }

    async def trigger_llm_snapshot(self) -> Dict[str, Any]:
        """Trigger a comprehensive LLM system snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "providers": {},
            "conversations": self.conversation_manager.get_manager_stats(),
            "function_calling": self.function_calling.get_execution_stats(),
            "websocket": self.get_websocket_stats(),
            "analytics": await self.get_analytics(),
        }

        # Add provider details
        for provider_name in await self.list_providers():
            snapshot["providers"][provider_name] = await self.get_provider_status(provider_name)

        # Broadcast the snapshot
        await self.broadcast_llm_event("llm_system_snapshot", snapshot)

        return snapshot

    async def cleanup(self) -> None:
        """Cleanup WebSocket LLM manager."""
        # Broadcast shutdown event
        if self._websocket_integration:
            await self.broadcast_llm_event(
                "llm_system_shutdown", {"timestamp": datetime.now().isoformat()}
            )

        # Clear event queue
        self._event_queue.clear()

        # Call parent cleanup
        await super().cleanup()

        logger.info("WebSocket LLM manager cleanup completed")
