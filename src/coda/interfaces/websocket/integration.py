"""
WebSocket integration for Coda components.

This module provides integration between Coda's core components and the WebSocket server,
allowing real-time monitoring and event broadcasting.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List

from .server import CodaWebSocketServer
from .events import EventType

logger = logging.getLogger("coda.websocket.integration")


class CodaWebSocketIntegration:
    """
    Integration between Coda's core components and the WebSocket server.

    This class provides methods to connect Coda's STT, LLM, TTS, memory,
    and tool components to the WebSocket server, allowing clients to receive
    real-time events about Coda's operation.
    """

    def __init__(self, server: CodaWebSocketServer):
        """
        Initialize the WebSocket integration.

        Args:
            server: The WebSocket server to use
        """
        self.server = server
        self.session_id = str(uuid.uuid4())
        self.conversation_turn_count = 0
        self.start_time = time.time()

        logger.info(f"WebSocket integration initialized with session {self.session_id}")

    def new_session(self) -> str:
        """Start a new session and return the session ID."""
        self.session_id = str(uuid.uuid4())
        self.conversation_turn_count = 0
        logger.info(f"New session started: {self.session_id}")
        return self.session_id

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id

    # System events

    async def system_startup(self, version: str, config: Dict[str, Any]) -> None:
        """Signal system startup."""
        await self.server.broadcast_system_info({
            "event": "startup",
            "version": version,
            "config": config,
            "session_id": self.session_id,
        })
        logger.info("System startup event sent")

    async def system_shutdown(self) -> None:
        """Signal system shutdown."""
        uptime = time.time() - self.start_time
        await self.server.broadcast_system_info({
            "event": "shutdown",
            "uptime_seconds": uptime,
            "session_id": self.session_id,
        })
        logger.info("System shutdown event sent")

    async def system_error(self, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Signal a system error."""
        await self.server.broadcast_system_error(level, message, details)
        logger.debug(f"System error event sent: {level} - {message}")

    # STT events

    async def stt_start(self, mode: str = "push_to_talk") -> None:
        """Signal the start of speech-to-text processing."""
        await self.server.broadcast_event(
            EventType.STT_START,
            {"mode": mode},
            session_id=self.session_id
        )
        logger.debug(f"STT started in {mode} mode")

    async def stt_interim_result(self, text: str, confidence: float = 0.0) -> None:
        """Send an interim STT result."""
        await self.server.broadcast_event(
            EventType.STT_INTERIM,
            {"text": text, "confidence": confidence},
            session_id=self.session_id
        )
        logger.debug(f"STT interim result: {text[:30]}...")

    async def stt_final_result(self, text: str, confidence: float, duration_ms: float, 
                              language: Optional[str] = None) -> None:
        """Send the final STT result."""
        await self.server.broadcast_event(
            EventType.STT_RESULT,
            {
                "text": text,
                "confidence": confidence,
                "duration_ms": duration_ms,
                "language": language,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.debug(f"STT final result: {text[:50]}...")

    async def stt_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Signal an STT error."""
        await self.server.broadcast_event(
            EventType.STT_ERROR,
            {"error": error, "details": details},
            high_priority=True,
            session_id=self.session_id
        )
        logger.warning(f"STT error: {error}")

    # LLM events

    async def llm_start(self, prompt: str, model: str, temperature: float = 0.7) -> None:
        """Signal the start of LLM processing."""
        await self.server.broadcast_event(
            EventType.LLM_START,
            {
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "model": model,
                "temperature": temperature,
            },
            session_id=self.session_id
        )
        logger.debug(f"LLM processing started with model {model}")

    async def llm_token(self, token: str, cumulative_text: str) -> None:
        """Send an LLM token generation event."""
        await self.server.broadcast_event(
            EventType.LLM_TOKEN,
            {"token": token, "cumulative_text": cumulative_text},
            session_id=self.session_id
        )

    async def llm_result(self, text: str, duration_ms: float, token_count: int, 
                        tokens_per_second: float) -> None:
        """Send the final LLM result."""
        await self.server.broadcast_event(
            EventType.LLM_RESULT,
            {
                "text": text,
                "duration_ms": duration_ms,
                "token_count": token_count,
                "tokens_per_second": tokens_per_second,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.debug(f"LLM result: {text[:50]}... ({token_count} tokens, {tokens_per_second:.1f} tok/s)")

    async def llm_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Signal an LLM error."""
        await self.server.broadcast_event(
            EventType.LLM_ERROR,
            {"error": error, "details": details},
            high_priority=True,
            session_id=self.session_id
        )
        logger.warning(f"LLM error: {error}")

    # TTS events

    async def tts_start(self, text: str, voice_id: str, engine: str) -> None:
        """Signal the start of TTS processing."""
        await self.server.broadcast_event(
            EventType.TTS_START,
            {"text": text, "voice_id": voice_id, "engine": engine},
            session_id=self.session_id
        )
        logger.debug(f"TTS started with {engine} engine")

    async def tts_progress(self, progress_percent: float, estimated_duration_ms: Optional[float] = None) -> None:
        """Send TTS progress update."""
        await self.server.broadcast_event(
            EventType.TTS_PROGRESS,
            {
                "progress_percent": progress_percent,
                "estimated_duration_ms": estimated_duration_ms,
            },
            session_id=self.session_id
        )

    async def tts_result(self, duration_ms: float, audio_duration_ms: float, success: bool = True) -> None:
        """Send TTS result."""
        await self.server.broadcast_event(
            EventType.TTS_RESULT,
            {
                "duration_ms": duration_ms,
                "audio_duration_ms": audio_duration_ms,
                "success": success,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.debug(f"TTS completed: {duration_ms:.0f}ms processing, {audio_duration_ms:.0f}ms audio")

    async def tts_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Signal a TTS error."""
        await self.server.broadcast_event(
            EventType.TTS_ERROR,
            {"error": error, "details": details},
            high_priority=True,
            session_id=self.session_id
        )
        logger.warning(f"TTS error: {error}")

    async def tts_status(self, status: str) -> None:
        """Send TTS status update."""
        await self.server.broadcast_event(
            EventType.TTS_STATUS,
            {"status": status},
            session_id=self.session_id
        )

    # Memory events

    async def memory_store(self, content: str, memory_type: str, importance: float, memory_id: str) -> None:
        """Signal that a memory has been stored."""
        content_preview = content[:100] + "..." if len(content) > 100 else content
        
        await self.server.broadcast_event(
            EventType.MEMORY_STORE,
            {
                "content_preview": content_preview,
                "memory_type": memory_type,
                "importance": importance,
                "memory_id": memory_id,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.debug(f"Memory stored: {content_preview[:30]}...")

    async def memory_retrieve(self, query: str, results_count: int, relevance_scores: List[float]) -> None:
        """Signal memory retrieval."""
        await self.server.broadcast_event(
            EventType.MEMORY_RETRIEVE,
            {
                "query": query,
                "results_count": results_count,
                "relevance_scores": relevance_scores,
            },
            session_id=self.session_id
        )
        logger.debug(f"Memory retrieved: {results_count} results for '{query[:30]}...'")

    # Tool events

    async def tool_call(self, tool_name: str, parameters: Dict[str, Any], call_id: str) -> None:
        """Signal a tool call."""
        await self.server.broadcast_event(
            EventType.TOOL_CALL,
            {
                "tool_name": tool_name,
                "parameters": parameters,
                "call_id": call_id,
            },
            session_id=self.session_id
        )
        logger.debug(f"Tool called: {tool_name} ({call_id})")

    async def tool_result(self, tool_name: str, call_id: str, result: Any, duration_ms: float) -> None:
        """Signal a tool result."""
        await self.server.broadcast_event(
            EventType.TOOL_RESULT,
            {
                "tool_name": tool_name,
                "call_id": call_id,
                "result": result,
                "duration_ms": duration_ms,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.debug(f"Tool result: {tool_name} ({call_id}) - {duration_ms:.0f}ms")

    async def tool_error(self, tool_name: str, call_id: str, error: str, 
                        details: Optional[Dict[str, Any]] = None) -> None:
        """Signal a tool error."""
        await self.server.broadcast_event(
            EventType.TOOL_ERROR,
            {
                "tool_name": tool_name,
                "call_id": call_id,
                "error": error,
                "details": details,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.warning(f"Tool error: {tool_name} ({call_id}) - {error}")

    # Conversation events

    async def conversation_start(self, conversation_id: str) -> None:
        """Signal the start of a conversation."""
        await self.server.broadcast_event(
            EventType.CONVERSATION_START,
            {"conversation_id": conversation_id},
            high_priority=True,
            session_id=self.session_id
        )
        logger.info(f"Conversation started: {conversation_id}")

    async def conversation_turn(self, conversation_id: str, turn_number: int, 
                              user_input: str, assistant_response: str) -> None:
        """Signal a conversation turn."""
        self.conversation_turn_count += 1
        
        await self.server.broadcast_event(
            EventType.CONVERSATION_TURN,
            {
                "conversation_id": conversation_id,
                "turn_number": turn_number,
                "user_input": user_input,
                "assistant_response": assistant_response,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.debug(f"Conversation turn {turn_number}: {user_input[:30]}...")

    async def conversation_end(self, conversation_id: str, total_turns: int, duration_seconds: float) -> None:
        """Signal the end of a conversation."""
        await self.server.broadcast_event(
            EventType.CONVERSATION_END,
            {
                "conversation_id": conversation_id,
                "total_turns": total_turns,
                "duration_seconds": duration_seconds,
            },
            high_priority=True,
            session_id=self.session_id
        )
        logger.info(f"Conversation ended: {conversation_id} ({total_turns} turns, {duration_seconds:.1f}s)")

    # Performance events

    async def latency_trace(self, component: str, operation: str, duration_ms: float, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send a latency trace event."""
        await self.server.broadcast_event(
            EventType.LATENCY_TRACE,
            {
                "component": component,
                "operation": operation,
                "duration_ms": duration_ms,
                "metadata": metadata,
            },
            session_id=self.session_id
        )

    async def component_timing(self, component: str, timings: Dict[str, float]) -> None:
        """Send component timing information."""
        await self.server.broadcast_event(
            EventType.COMPONENT_TIMING,
            {"component": component, "timings": timings},
            session_id=self.session_id
        )
