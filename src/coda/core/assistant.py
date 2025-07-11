"""
Main Coda Assistant orchestrator.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..components.llm.manager import LLMManager
from ..components.memory.manager import MemoryManager
from ..components.personality.manager import PersonalityManager
from ..components.tools.manager import ToolManager
from ..components.voice.manager import VoiceManager
from ..interfaces.dashboard.server import CodaDashboardServer
from ..interfaces.websocket.server import CodaWebSocketServer
from .component_health import ComponentHealthManager, ComponentStatus, ErrorCategory
from .component_recovery import ComponentRecoveryManager, get_recovery_manager
from .config import CodaConfig
from .config_adapters import ConfigAdapter
from .error_management import (
    ErrorManager,
    get_error_manager,
    handle_error,
)
from .event_coordinator import EventCoordinator
from .events import EventBus, EventType, emit_event, get_event_bus
from .integration import ComponentIntegrationLayer, ComponentType
from .performance_monitor import CodaPerformanceMonitor, get_performance_monitor
from .performance_optimizer import (
    OptimizationLevel,
    PerformanceOptimizer,
    PerformanceTarget,
)
from .session_manager import SessionManager
from .user_error_interface import UserErrorInterface, get_user_error_interface
from .websocket_integration import ComponentWebSocketIntegration


class CodaAssistant:
    """Main Coda Assistant orchestrator."""

    def __init__(self, config: CodaConfig):
        self.config = config
        self.logger = logging.getLogger("coda.assistant")

        # Core systems
        self.event_bus: EventBus = get_event_bus()
        self.event_coordinator = EventCoordinator()
        self.health_manager = ComponentHealthManager()
        self.integration_layer = ComponentIntegrationLayer()
        self.websocket_integration: Optional[ComponentWebSocketIntegration] = None
        self.dashboard_server: Optional[CodaDashboardServer] = None
        self.performance_monitor: Optional[CodaPerformanceMonitor] = None

        # Error handling and recovery systems
        self.error_manager: ErrorManager = get_error_manager()
        self.recovery_manager: ComponentRecoveryManager = get_recovery_manager()
        self.user_error_interface: UserErrorInterface = get_user_error_interface()

        # Performance optimization system
        self.performance_optimizer: Optional[PerformanceOptimizer] = None

        self.running = False

        # Component managers
        self.memory_manager: Optional[MemoryManager] = None
        self.personality_manager: Optional[PersonalityManager] = None
        self.llm_manager: Optional[LLMManager] = None
        self.tools_manager: Optional[ToolManager] = None
        self.voice_manager: Optional[VoiceManager] = None
        self.websocket_server: Optional[CodaWebSocketServer] = None

        # Enhanced session management
        self.conversation_active = False
        self.current_session_id: Optional[str] = None
        self.session_manager = SessionManager()  # Dedicated session manager

        # Voice conversation mapping (session_id -> voice_conversation_id)
        self.voice_conversations: Dict[str, str] = {}

    def _register_components(self):
        """Register all components for health tracking."""
        # Memory is foundational for other components
        self.health_manager.register_component("memory", dependencies=[], is_critical=True)

        # LLM is critical for core functionality
        self.health_manager.register_component("llm", dependencies=[], is_critical=True)

        # Personality depends on memory for context
        self.health_manager.register_component(
            "personality", dependencies=["memory"], is_critical=False
        )

        # Tools can work independently but benefit from memory
        self.health_manager.register_component("tools", dependencies=[], is_critical=False)

        # Voice is optional but enhances user experience
        self.health_manager.register_component("voice", dependencies=[], is_critical=False)

    async def _register_and_initialize_components(self):
        """Register and initialize components with the integration layer."""

        # Initialize memory manager
        if self.config.memory:
            await self._init_memory_manager()
            self.integration_layer.register_component(ComponentType.MEMORY, self.memory_manager)

        # Initialize LLM manager (always enabled as it's critical)
        await self._init_llm_manager()
        self.integration_layer.register_component(ComponentType.LLM, self.llm_manager)

        # Initialize personality manager
        if (
            self.config.personality
            and hasattr(self.config.personality, "enabled")
            and self.config.personality.enabled
        ):
            await self._init_personality_manager()
            self.integration_layer.register_component(
                ComponentType.PERSONALITY, self.personality_manager
            )

        # Initialize tools manager
        if self.config.tools.enabled:
            await self._init_tools_manager()
            self.integration_layer.register_component(ComponentType.TOOLS, self.tools_manager)

        # Initialize voice manager
        if self.config.voice:
            await self._init_voice_manager()
            self.integration_layer.register_component(ComponentType.VOICE, self.voice_manager)

    async def initialize(self):
        """Initialize all components with enhanced error handling."""
        self.logger.info("Initializing Coda Assistant...")

        # Setup error handling and recovery systems
        await self._setup_error_handling()

        # Setup performance optimization
        await self._setup_performance_optimization()

        # Register components for health tracking
        self._register_components()

        try:
            # Initialize session manager
            await self.session_manager.initialize()

            # Start event bus
            await self.event_bus.start()
            await emit_event(EventType.SYSTEM_STARTUP, {"component": "event_bus"})

            # Initialize event coordinator
            await self.event_coordinator.initialize()
            await emit_event(EventType.SYSTEM_STARTUP, {"component": "event_coordinator"})

            # Register event bus with integration layer
            self.integration_layer.register_component(ComponentType.EVENT_BUS, self.event_bus)

            # Initialize and register components
            await self._register_and_initialize_components()

            # Initialize WebSocket integration
            self.websocket_integration = ComponentWebSocketIntegration(self.integration_layer)

            # Initialize all components through integration layer
            initialization_results = await self.integration_layer.initialize_all_components()

            # Log initialization results
            failed_components = [
                comp.value for comp, success in initialization_results.items() if not success
            ]
            if failed_components:
                self.logger.warning(f"Some components failed to initialize: {failed_components}")
            else:
                self.logger.info("All components initialized successfully")

            # Start WebSocket integration if WebSocket server is available
            if self.websocket_server:
                self.websocket_integration.set_websocket_server(self.websocket_server)
                await self.websocket_integration.start()

                # Connect WebSocket server to event coordinator
                self.event_coordinator.add_websocket_client(self.websocket_server)

            # Initialize and start dashboard server
            try:
                self.dashboard_server = CodaDashboardServer(
                    host=(
                        self.config.dashboard.host
                        if hasattr(self.config, "dashboard")
                        else "localhost"
                    ),
                    port=self.config.dashboard.port if hasattr(self.config, "dashboard") else 8080,
                )
                self.dashboard_server.set_integration_layer(self.integration_layer)
                if self.websocket_integration:
                    self.dashboard_server.set_websocket_integration(self.websocket_integration)

                await self.dashboard_server.start()
                self.logger.info(
                    f"Dashboard available at: {self.dashboard_server.get_dashboard_url()}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to start dashboard server: {e}")
                self.dashboard_server = None

            # Initialize WebSocket server
            if self.config.websocket.enabled:
                self.logger.info("Initializing WebSocket server...")
                self.websocket_server = CodaWebSocketServer(
                    host=self.config.websocket.host, port=self.config.websocket.port
                )
                await self.websocket_server.start()
                await emit_event(EventType.SYSTEM_STARTUP, {"component": "websocket"})

            # Initialize and start performance monitoring
            self.performance_monitor = get_performance_monitor()

            # Add performance alert callback for WebSocket broadcasting
            if self.websocket_integration:

                def alert_callback(alert):
                    asyncio.create_task(
                        self.websocket_integration._broadcast_component_event(
                            "performance_alert",
                            {
                                "alert_id": alert.alert_id,
                                "component": alert.component,
                                "metric_name": alert.metric_name,
                                "current_value": alert.current_value,
                                "threshold_value": alert.threshold_value,
                                "severity": alert.severity,
                                "message": alert.message,
                                "timestamp": alert.timestamp,
                            },
                        )
                    )

                self.performance_monitor.add_alert_callback(alert_callback)

            self.performance_monitor.start_monitoring()
            self.logger.info("Performance monitoring started")

            self.running = True

            # Log initialization summary
            health_summary = self.health_manager.get_initialization_summary()
            self.logger.info("Coda Assistant initialization completed")
            self.logger.info(f"System Health Summary:\n{health_summary}")
            await emit_event(
                EventType.SYSTEM_STARTUP, {"component": "assistant", "status": "ready"}
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Coda Assistant: {e}")
            await emit_event(EventType.SYSTEM_ERROR, {"error": str(e), "component": "assistant"})
            raise

    async def shutdown(self):
        """Shutdown all components."""
        self.logger.info("Shutting down Coda Assistant...")

        try:
            self.running = False

            # Stop WebSocket server
            if self.websocket_server:
                await self.websocket_server.stop()
                await emit_event(EventType.SYSTEM_SHUTDOWN, {"component": "websocket"})

            # Stop performance monitoring
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()

            # Stop performance optimization
            if self.performance_optimizer:
                await self.performance_optimizer.shutdown()

            # Stop dashboard server
            if self.dashboard_server:
                await self.dashboard_server.stop()

            # Stop WebSocket integration
            if self.websocket_integration:
                await self.websocket_integration.stop()

            # Shutdown all components through integration layer
            shutdown_results = await self.integration_layer.shutdown_all_components()

            # Log shutdown results
            failed_shutdowns = [
                comp.value for comp, success in shutdown_results.items() if not success
            ]
            if failed_shutdowns:
                self.logger.warning(
                    f"Some components failed to shutdown cleanly: {failed_shutdowns}"
                )
            else:
                self.logger.info("All components shutdown successfully")

            # Stop event bus last
            await emit_event(EventType.SYSTEM_SHUTDOWN, {"component": "assistant"})
            await self.event_bus.stop()

            self.logger.info("Coda Assistant shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    async def start_conversation(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session with enhanced session management."""
        if not self.running:
            raise RuntimeError("Assistant not initialized")

        # Create session through session manager
        session_id = await self.session_manager.create_session(session_id)

        # Set as current session
        self.current_session_id = session_id
        self.conversation_active = True

        # Initialize session-specific memory context if memory manager available
        if self.memory_manager and hasattr(self.memory_manager, "create_session"):
            await self.memory_manager.create_session(session_id)

        # Get session data for event
        session_data = self.session_manager.get_session(session_id)

        await emit_event(
            EventType.CONVERSATION_START,
            {
                "session_id": session_id,
                "timestamp": session_data["created_at"],
                "session_data": session_data,
            },
        )

        self.logger.info(f"Started conversation session: {session_id}")
        return session_id

    async def end_conversation(self, session_id: Optional[str] = None):
        """End a conversation session with enhanced session management."""
        target_session = session_id or self.current_session_id

        if target_session:
            # End session through session manager
            success = await self.session_manager.end_session(target_session)

            if success:
                session_data = self.session_manager.get_session(target_session)

                await emit_event(
                    EventType.CONVERSATION_END,
                    {
                        "session_id": target_session,
                        "timestamp": session_data.get("ended_at"),
                        "session_data": session_data,
                    },
                )

                self.logger.info(f"Ended conversation session: {target_session}")

                # Clean up voice conversation if it exists
                await self._cleanup_voice_conversation(target_session)

                # If ending current session, reset state
                if target_session == self.current_session_id:
                    self.conversation_active = False
                    self.current_session_id = None

    async def switch_session(self, session_id: str) -> bool:
        """Switch to an existing conversation session."""
        session_data = self.session_manager.get_session(session_id)

        if session_data and session_data.get("status") == "active":
            # End current session if active
            if self.conversation_active and self.current_session_id:
                await self.end_conversation(self.current_session_id)

            # Switch to new session
            self.current_session_id = session_id
            self.conversation_active = True
            await self.session_manager.update_session_activity(session_id)

            await emit_event(
                EventType.CONVERSATION_START,
                {
                    "session_id": session_id,
                    "timestamp": session_data["last_activity"],
                    "action": "session_switch",
                },
            )

            self.logger.info(f"Switched to conversation session: {session_id}")
            return True
        return False

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        return self.session_manager.get_session(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all conversation sessions."""
        return self.session_manager.list_sessions()

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all currently active sessions."""
        return self.session_manager.list_sessions(active_only=True)

    async def process_input(
        self, input_text: str, input_type: str = "text", metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process user input and generate response."""
        import time

        start_time = time.time()

        if not self.running:
            raise RuntimeError("Assistant not initialized")

        if not self.conversation_active:
            await self.start_conversation()

        # Update session activity
        if self.current_session_id:
            await self.session_manager.update_session_activity(self.current_session_id)

        try:
            # Emit conversation turn event
            await emit_event(
                EventType.CONVERSATION_TURN,
                {
                    "session_id": self.current_session_id,
                    "input": input_text,
                    "input_type": input_type,
                },
            )

            # Process through personality system if available
            enhanced_input = input_text
            if self.personality_manager and hasattr(self.personality_manager, "enhance_input"):
                enhanced_input = await self.personality_manager.enhance_input(input_text)

            # Generate LLM response
            if not self.llm_manager:
                raise RuntimeError("LLM manager not initialized")

            response = await self.llm_manager.generate_response(
                enhanced_input, conversation_id=self.current_session_id
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Store in memory if available with multi-modal context
            if self.memory_manager:
                # Add user turn with modality marker
                self.memory_manager.add_turn("user", f"[TEXT] {input_text}")
                # Add assistant turn
                response_content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                self.memory_manager.add_turn("assistant", response_content)

            # Also store in session manager with modality metadata
            if self.current_session_id:
                await self.session_manager.add_message_to_session(
                    self.current_session_id,
                    "user",
                    input_text,
                    metadata={
                        **(metadata or {}),
                        "modality": "text",
                        "input_type": "text",
                        "supports_multimodal": True,
                    },
                )
                response_content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                await self.session_manager.add_message_to_session(
                    self.current_session_id,
                    "assistant",
                    response_content,
                    metadata={
                        "modality": "text",
                        "response_type": "text",
                        "processing_time_ms": processing_time * 1000,
                    },
                )

            # Update personality if available
            if self.personality_manager and hasattr(
                self.personality_manager, "update_from_interaction"
            ):
                response_content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                await self.personality_manager.update_from_interaction(input_text, response_content)

            return {
                "response": response,
                "session_id": self.current_session_id,
                "status": "success",
            }

        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            await emit_event(
                EventType.SYSTEM_ERROR,
                {"error": str(e), "component": "assistant", "input": input_text},
            )

            return {"error": str(e), "session_id": self.current_session_id, "status": "error"}

    async def process_text_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a text message through the complete pipeline with enhanced session management.

        Pipeline: Input → Session Management → LLM Processing → Memory Storage → Response

        Args:
            message: Text message to process
            session_id: Optional session ID (creates new if not exists)
            metadata: Optional metadata for the message

        Returns:
            Processing result with response, session info, and metadata
        """
        start_time = time.time()
        processing_metadata = metadata or {}

        try:
            # Emit processing start event
            await emit_event(
                EventType.LLM_GENERATION_START,
                {
                    "message": message,
                    "session_id": session_id,
                    "input_type": "text",
                    "component": "assistant",
                },
            )

            # Session management
            if session_id and session_id != self.current_session_id:
                if not await self.switch_session(session_id):
                    # Create new session if it doesn't exist
                    await self.start_conversation(session_id)
            elif not self.conversation_active:
                # Start new session if none active
                await self.start_conversation(session_id)

            # Process through the standard pipeline
            result = await self.process_input(message, input_type="text")

            # Calculate processing time
            processing_time = time.time() - start_time

            # Enhance result with additional metadata
            enhanced_result = {
                **result,
                "input_type": "text",
                "processing_time_ms": processing_time * 1000,
                "pipeline": "text_message_pipeline",
                "metadata": processing_metadata,
                "timestamp": datetime.now().isoformat(),
            }

            # Emit processing complete event
            await emit_event(
                EventType.LLM_GENERATION_COMPLETE,
                {
                    "session_id": self.current_session_id,
                    "processing_time_ms": processing_time * 1000,
                    "status": enhanced_result.get("status", "success"),
                    "component": "assistant",
                },
            )

            return enhanced_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_result = {
                "error": str(e),
                "session_id": self.current_session_id,
                "status": "error",
                "input_type": "text",
                "processing_time_ms": processing_time * 1000,
                "pipeline": "text_message_pipeline",
                "metadata": processing_metadata,
                "timestamp": datetime.now().isoformat(),
            }

            # Emit error event
            await emit_event(
                EventType.SYSTEM_ERROR,
                {
                    "error": str(e),
                    "component": "assistant",
                    "pipeline": "text_message_pipeline",
                    "session_id": self.current_session_id,
                    "input": message,
                },
            )

            self.logger.error(f"Text message processing failed: {e}")
            return error_result

    async def process_voice_input(
        self,
        audio_data: bytes,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process voice input through the complete pipeline with Moshi integration.

        Pipeline: Audio Input → Moshi STT → LLM Processing → Memory Storage → Moshi TTS → Audio Output

        Args:
            audio_data: Raw audio data in bytes (WAV format preferred)
            session_id: Optional session ID (creates new if not exists)
            metadata: Optional metadata for the voice message

        Returns:
            Processing result with text response, audio response, session info, and metadata
        """
        start_time = time.time()
        processing_metadata = metadata or {}

        try:
            # Emit voice processing start event
            await emit_event(
                EventType.LLM_GENERATION_START,
                {
                    "session_id": session_id or self.current_session_id,
                    "input_type": "voice",
                    "component": "assistant",
                },
            )

            # Switch to specified session if provided
            if session_id and session_id != self.current_session_id:
                if not await self.switch_session(session_id):
                    await self.start_conversation(session_id)

            # Check if voice manager is available
            if not self.voice_manager:
                return {
                    "error": "Voice processing not available - voice manager not initialized",
                    "session_id": self.current_session_id,
                    "status": "error",
                    "input_type": "voice",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "pipeline": "voice_processing_pipeline",
                    "metadata": processing_metadata,
                    "timestamp": datetime.now().isoformat(),
                }

            # Get or create voice conversation for this session
            voice_conversation_id = await self._get_or_create_voice_conversation(
                self.current_session_id
            )

            # Process voice input through VoiceManager
            voice_response = await self.voice_manager.process_voice_input(
                conversation_id=voice_conversation_id, audio_data=audio_data
            )

            # Extract text from voice response for LLM processing
            extracted_text = (
                voice_response.text_content if hasattr(voice_response, "text_content") else ""
            )

            # If we have extracted text, process it through the LLM pipeline for enhanced reasoning
            llm_response = None
            if extracted_text and self.llm_manager:
                try:
                    # Process extracted text through LLM for enhanced response
                    llm_result = await self.llm_manager.generate_response(
                        prompt=extracted_text,
                        session_id=self.current_session_id,
                        context=await self._get_conversation_context(),
                    )
                    llm_response = (
                        llm_result.content if hasattr(llm_result, "content") else str(llm_result)
                    )

                    # Store the interaction in memory with multi-modal context
                    if self.memory_manager:
                        await self.memory_manager.store_interaction(
                            session_id=self.current_session_id,
                            user_input=extracted_text,
                            assistant_response=llm_response,
                            metadata={
                                **processing_metadata,
                                "input_type": "voice",
                                "modality": "voice",
                                "has_audio_input": True,
                                "has_audio_output": hasattr(voice_response, "audio_data")
                                and voice_response.audio_data,
                                "voice_conversation_id": voice_conversation_id,
                                "supports_multimodal": True,
                            },
                        )

                        # Also add to short-term memory for immediate context
                        if hasattr(self.memory_manager, "add_turn"):
                            self.memory_manager.add_turn("user", f"[VOICE] {extracted_text}")
                            self.memory_manager.add_turn("assistant", llm_response)

                    # Store in session manager with voice modality metadata
                    if self.current_session_id and extracted_text:
                        await self.session_manager.add_message_to_session(
                            self.current_session_id,
                            "user",
                            extracted_text,
                            metadata={
                                **processing_metadata,
                                "modality": "voice",
                                "input_type": "voice",
                                "has_audio_input": True,
                                "voice_conversation_id": voice_conversation_id,
                                "supports_multimodal": True,
                            },
                        )

                        final_response = (
                            llm_response or voice_response.text_content
                            if hasattr(voice_response, "text_content")
                            else "Voice processed"
                        )
                        await self.session_manager.add_message_to_session(
                            self.current_session_id,
                            "assistant",
                            final_response,
                            metadata={
                                "modality": "voice",
                                "response_type": "voice",
                                "has_audio_output": hasattr(voice_response, "audio_data")
                                and voice_response.audio_data,
                                "voice_processing": {
                                    "moshi_latency_ms": getattr(
                                        voice_response, "moshi_latency_ms", 0
                                    ),
                                    "total_latency_ms": getattr(
                                        voice_response, "total_latency_ms", 0
                                    ),
                                    "processing_mode": str(
                                        getattr(voice_response, "processing_mode", "unknown")
                                    ),
                                },
                            },
                        )

                except Exception as e:
                    self.logger.warning(
                        f"LLM processing failed for voice input, using Moshi response: {e}"
                    )
                    llm_response = (
                        voice_response.text_content
                        if hasattr(voice_response, "text_content")
                        else "Voice processed successfully"
                    )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare enhanced result
            enhanced_result = {
                "response": {
                    "content": (
                        llm_response or voice_response.text_content
                        if hasattr(voice_response, "text_content")
                        else "Voice processed"
                    ),
                    "audio_data": (
                        voice_response.audio_data if hasattr(voice_response, "audio_data") else None
                    ),
                    "extracted_text": extracted_text,
                },
                "session_id": self.current_session_id,
                "status": "success",
                "input_type": "voice",
                "processing_time_ms": processing_time * 1000,
                "pipeline": "voice_processing_pipeline",
                "voice_processing": {
                    "moshi_latency_ms": getattr(voice_response, "moshi_latency_ms", 0),
                    "total_latency_ms": getattr(voice_response, "total_latency_ms", 0),
                    "processing_mode": str(getattr(voice_response, "processing_mode", "unknown")),
                },
                "metadata": processing_metadata,
                "timestamp": datetime.now().isoformat(),
            }

            # Emit processing complete event
            await emit_event(
                EventType.LLM_GENERATION_COMPLETE,
                {
                    "session_id": self.current_session_id,
                    "processing_time_ms": processing_time * 1000,
                    "status": "success",
                    "input_type": "voice",
                    "component": "assistant",
                },
            )

            return enhanced_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_result = {
                "error": str(e),
                "session_id": self.current_session_id,
                "status": "error",
                "input_type": "voice",
                "processing_time_ms": processing_time * 1000,
                "pipeline": "voice_processing_pipeline",
                "metadata": processing_metadata,
                "timestamp": datetime.now().isoformat(),
            }

            # Emit error event
            await emit_event(
                EventType.SYSTEM_ERROR,
                {
                    "error": str(e),
                    "component": "assistant",
                    "pipeline": "voice_processing_pipeline",
                    "session_id": self.current_session_id,
                    "input_type": "voice",
                },
            )

            self.logger.error(f"Voice input processing failed: {e}")
            return error_result

    async def process_voice_stream(
        self,
        audio_stream,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Process streaming voice input for real-time conversation.

        Args:
            audio_stream: Async generator yielding audio chunks
            session_id: Optional session ID (creates new if not exists)
            metadata: Optional metadata for the voice stream

        Yields:
            Streaming voice responses with text and audio data
        """
        processing_metadata = metadata or {}

        try:
            # Switch to specified session if provided
            if session_id and session_id != self.current_session_id:
                if not await self.switch_session(session_id):
                    await self.start_conversation(session_id)

            # Check if voice manager is available
            if not self.voice_manager:
                yield {
                    "error": "Voice streaming not available - voice manager not initialized",
                    "session_id": self.current_session_id,
                    "status": "error",
                    "input_type": "voice_stream",
                }
                return

            # Get or create voice conversation for this session
            voice_conversation_id = await self._get_or_create_voice_conversation(
                self.current_session_id
            )

            # Process streaming audio through VoiceManager
            async for voice_chunk in self.voice_manager.process_voice_stream(
                conversation_id=voice_conversation_id, audio_stream=audio_stream
            ):
                # Enhance chunk with session info
                enhanced_chunk = {
                    "response": {
                        "content": getattr(voice_chunk, "text_content", ""),
                        "audio_data": getattr(voice_chunk, "audio_data", None),
                        "text_delta": getattr(voice_chunk, "text_delta", ""),
                        "is_final": getattr(voice_chunk, "is_final", False),
                    },
                    "session_id": self.current_session_id,
                    "status": "streaming",
                    "input_type": "voice_stream",
                    "metadata": processing_metadata,
                    "timestamp": datetime.now().isoformat(),
                }

                yield enhanced_chunk

        except Exception as e:
            error_chunk = {
                "error": str(e),
                "session_id": self.current_session_id,
                "status": "error",
                "input_type": "voice_stream",
                "metadata": processing_metadata,
                "timestamp": datetime.now().isoformat(),
            }

            await emit_event(
                EventType.SYSTEM_ERROR,
                {
                    "error": str(e),
                    "component": "assistant",
                    "pipeline": "voice_streaming_pipeline",
                    "session_id": self.current_session_id,
                },
            )

            self.logger.error(f"Voice stream processing failed: {e}")
            yield error_chunk

    async def _get_or_create_voice_conversation(self, session_id: str) -> str:
        """Get existing voice conversation ID or create a new one for the session."""
        if session_id in self.voice_conversations:
            return self.voice_conversations[session_id]

        if not self.voice_manager:
            raise RuntimeError("Voice manager not available")

        # Start a new voice conversation
        voice_conversation_id = await self.voice_manager.start_conversation(
            user_id=None,  # Could be enhanced to include user info
            conversation_id=session_id,  # Use session ID as conversation ID for consistency
        )

        # Map session to voice conversation
        self.voice_conversations[session_id] = voice_conversation_id

        self.logger.debug(
            f"Created voice conversation {voice_conversation_id} for session {session_id}"
        )
        return voice_conversation_id

    async def _cleanup_voice_conversation(self, session_id: str):
        """Clean up voice conversation when session ends."""
        if session_id in self.voice_conversations:
            voice_conversation_id = self.voice_conversations[session_id]

            if self.voice_manager:
                try:
                    await self.voice_manager.end_conversation(voice_conversation_id)
                    self.logger.debug(
                        f"Ended voice conversation {voice_conversation_id} for session {session_id}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to end voice conversation {voice_conversation_id}: {e}"
                    )

            # Remove from mapping
            del self.voice_conversations[session_id]

    async def _get_conversation_context(self) -> Dict[str, Any]:
        """Get enhanced conversation context for LLM processing with multi-modal support."""
        context = {}

        if self.current_session_id and self.memory_manager:
            try:
                # Get recent multi-modal conversation history
                history = await self.get_multimodal_session_history(
                    self.current_session_id, limit=5
                )
                context["recent_messages"] = history

                # Get session modality statistics
                modality_stats = await self.get_session_modality_stats(self.current_session_id)
                context["session_stats"] = modality_stats

                # Get relevant memories
                if history:
                    last_message = history[-1].get("user_input", "") if history else ""
                    if last_message:
                        memories = await self.memory_manager.search_memories(
                            query=last_message, limit=3
                        )
                        context["relevant_memories"] = memories

                # Add multi-modal context information
                context["multimodal_context"] = {
                    "supports_voice": self.voice_manager is not None,
                    "supports_text": True,
                    "current_session_id": self.current_session_id,
                    "voice_conversation_id": self.voice_conversations.get(self.current_session_id),
                    "is_multimodal_session": modality_stats.get("is_multimodal", False),
                    "primary_modality": modality_stats.get("primary_modality", "text"),
                    "modality_switches": modality_stats.get("modality_switches", 0),
                }

            except Exception as e:
                self.logger.warning(f"Failed to get conversation context: {e}")

        return context

    async def get_multimodal_session_history(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get unified conversation history including both text and voice interactions.

        Args:
            session_id: Session ID to get history for
            limit: Maximum number of messages to return

        Returns:
            List of messages with modality information
        """
        try:
            # Get session history from session manager
            history = await self.get_session_history(session_id, limit)

            # Enhance with modality information
            enhanced_history = []
            for message in history:
                enhanced_message = {
                    **message,
                    "modality": message.get("metadata", {}).get("modality", "text"),
                    "input_type": message.get("metadata", {}).get("input_type", "text"),
                    "supports_multimodal": True,
                    "session_id": session_id,
                }

                # Add voice-specific information if available
                if enhanced_message["modality"] == "voice":
                    enhanced_message["voice_metadata"] = {
                        "has_audio_input": message.get("metadata", {}).get(
                            "has_audio_input", False
                        ),
                        "has_audio_output": message.get("metadata", {}).get(
                            "has_audio_output", False
                        ),
                        "voice_conversation_id": message.get("metadata", {}).get(
                            "voice_conversation_id"
                        ),
                        "voice_processing": message.get("metadata", {}).get("voice_processing", {}),
                    }

                enhanced_history.append(enhanced_message)

            return enhanced_history

        except Exception as e:
            self.logger.error(f"Failed to get multimodal session history: {e}")
            return []

    async def get_session_modality_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about modality usage in a session.

        Args:
            session_id: Session ID to analyze

        Returns:
            Statistics about text vs voice usage
        """
        try:
            history = await self.get_multimodal_session_history(session_id)

            text_messages = [msg for msg in history if msg.get("modality") == "text"]
            voice_messages = [msg for msg in history if msg.get("modality") == "voice"]

            total_messages = len(history)

            stats = {
                "session_id": session_id,
                "total_messages": total_messages,
                "text_messages": len(text_messages),
                "voice_messages": len(voice_messages),
                "text_percentage": (
                    (len(text_messages) / total_messages * 100) if total_messages > 0 else 0
                ),
                "voice_percentage": (
                    (len(voice_messages) / total_messages * 100) if total_messages > 0 else 0
                ),
                "is_multimodal": len(text_messages) > 0 and len(voice_messages) > 0,
                "primary_modality": "voice" if len(voice_messages) > len(text_messages) else "text",
                "modality_switches": self._count_modality_switches(history),
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get session modality stats: {e}")
            return {}

    def _count_modality_switches(self, history: List[Dict[str, Any]]) -> int:
        """Count the number of times the user switched between text and voice."""
        if len(history) < 2:
            return 0

        switches = 0
        last_modality = None

        for message in history:
            if message.get("role") == "user":  # Only count user messages
                current_modality = message.get("modality", "text")
                if last_modality and last_modality != current_modality:
                    switches += 1
                last_modality = current_modality

        return switches

    async def get_session_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a specific session."""
        # First try session manager (most reliable)
        history = self.session_manager.get_session_history(session_id, limit)
        if history:
            return history

        # Fallback to memory manager if available
        if self.memory_manager and hasattr(self.memory_manager, "get_session_history"):
            return await self.memory_manager.get_session_history(session_id, limit)

        return []

    def get_event_history(
        self, limit: Optional[int] = None, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get event history from event coordinator."""
        return self.event_coordinator.get_event_history(limit, event_type)

    def get_event_stats(self) -> Dict[str, Any]:
        """Get event statistics from event coordinator."""
        return self.event_coordinator.get_event_stats()

    async def emit_gui_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a GUI-specific event through event coordinator."""
        await self.event_coordinator.emit_gui_event(event_type, data)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all processing pipelines."""
        return {
            "text_pipeline": {
                "available": bool(self.llm_manager),
                "components": {
                    "llm": bool(self.llm_manager),
                    "memory": bool(self.memory_manager),
                    "personality": bool(self.personality_manager),
                    "session_manager": True,
                },
            },
            "voice_pipeline": {
                "available": bool(self.voice_manager),
                "components": {
                    "voice": bool(self.voice_manager),
                    "llm": bool(self.llm_manager),
                    "memory": bool(self.memory_manager),
                },
            },
            "session_management": {
                "active_session": self.current_session_id,
                "conversation_active": self.conversation_active,
                "total_sessions": len(self.session_manager.sessions),
                "active_sessions": len(self.session_manager.list_sessions(active_only=True)),
            },
            "event_system": {
                "event_bus_running": (
                    self.event_bus._running if hasattr(self.event_bus, "_running") else False
                ),
                "event_coordinator": True,
                "total_events": self.event_coordinator.get_event_stats().get("total_events", 0),
            },
        }

    async def run(self):
        """Main run loop for the assistant."""
        try:
            # Initialize all components
            await self.initialize()

            # Start a conversation session
            session_id = await self.start_conversation()
            self.logger.info(f"Assistant ready with session: {session_id}")

            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
            raise
        finally:
            await self.shutdown()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        components = {}

        # Check event bus
        components["event_bus"] = {
            "status": "healthy" if (self.event_bus and self.event_bus._running) else "unhealthy"
        }

        # Check component health
        if self.memory_manager:
            try:
                if hasattr(self.memory_manager, "health_check"):
                    components["memory_manager"] = await self.memory_manager.health_check()
                else:
                    components["memory_manager"] = {"status": "healthy"}
            except Exception:
                components["memory_manager"] = {"status": "unhealthy"}

        if self.personality_manager:
            try:
                if hasattr(self.personality_manager, "health_check"):
                    components["personality_manager"] = (
                        await self.personality_manager.health_check()
                    )
                else:
                    components["personality_manager"] = {"status": "healthy"}
            except Exception:
                components["personality_manager"] = {"status": "unhealthy"}

        if self.llm_manager:
            try:
                if hasattr(self.llm_manager, "health_check"):
                    components["llm_manager"] = await self.llm_manager.health_check()
                else:
                    components["llm_manager"] = {"status": "healthy"}
            except Exception:
                components["llm_manager"] = {"status": "unhealthy"}

        if self.tools_manager:
            try:
                if hasattr(self.tools_manager, "health_check"):
                    components["tools_manager"] = await self.tools_manager.health_check()
                else:
                    components["tools_manager"] = {"status": "healthy"}
            except Exception:
                components["tools_manager"] = {"status": "unhealthy"}

        if self.voice_manager:
            try:
                if hasattr(self.voice_manager, "health_check"):
                    components["voice_manager"] = await self.voice_manager.health_check()
                else:
                    components["voice_manager"] = {"status": "healthy"}
            except Exception:
                components["voice_manager"] = {"status": "unhealthy"}

        # Overall status
        overall_status = (
            "healthy"
            if self.running and all(comp.get("status") == "healthy" for comp in components.values())
            else "unhealthy"
        )

        return {"status": overall_status, "components": components}

    async def process_message(self, message: str, conversation_id: str = None) -> Any:
        """Process a message and return the response."""
        if conversation_id and conversation_id != self.current_session_id:
            # Switch to the specified conversation
            self.current_session_id = conversation_id
            self.conversation_active = True

        result = await self.process_input(message)

        # Return the LLM response directly for compatibility
        if "response" in result:
            return result["response"]
        else:
            # Return error as a mock response
            from ..components.llm.models import LLMProvider, LLMResponse

            return LLMResponse(
                response_id="error-response",
                content=f"Error: {result.get('error', 'Unknown error')}",
                provider=LLMProvider.OPENAI,  # Dummy provider
                model="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

    async def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        conversations = []

        if self.memory_manager and hasattr(self.memory_manager, "list_conversations"):
            conversations = await self.memory_manager.list_conversations()
        elif self.current_session_id:
            # Return current session if no memory manager
            conversations = [
                {"id": self.current_session_id, "created_at": "unknown", "message_count": 0}
            ]

        return conversations

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all components."""
        return self.health_manager.get_system_health()

    def get_health_summary(self) -> str:
        """Get human-readable health summary."""
        return self.health_manager.get_initialization_summary()

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status of all components."""
        return self.integration_layer.get_integration_status()

    def get_component(self, component_type: str) -> Optional[Any]:
        """Get a component instance by type."""
        try:
            comp_type = ComponentType(component_type.lower())
            return self.integration_layer.get_component(comp_type)
        except ValueError:
            return None

    def get_websocket_integration_status(self) -> Dict[str, Any]:
        """Get WebSocket integration status."""
        if not self.websocket_integration:
            return {"status": "not_initialized", "message": "WebSocket integration not initialized"}

        return self.websocket_integration.get_integration_metrics()

    async def get_component_event_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent component events."""
        if not self.websocket_integration:
            return []

        return await self.websocket_integration.get_event_history(limit)

    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard server status."""
        if not self.dashboard_server:
            return {"status": "not_initialized", "message": "Dashboard server not initialized"}

        return self.dashboard_server.get_status()

    def get_dashboard_url(self) -> Optional[str]:
        """Get dashboard URL if available."""
        if not self.dashboard_server:
            return None

        return self.dashboard_server.get_dashboard_url()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_monitor:
            return {
                "status": "not_initialized",
                "message": "Performance monitoring not initialized",
            }

        return self.performance_monitor.get_performance_summary()

    def get_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current system metrics."""
        if not self.performance_monitor:
            return None

        metrics = self.performance_monitor.get_system_metrics()
        return metrics.__dict__ if metrics else None

    def export_performance_data(
        self, filepath: str, time_range_hours: Optional[float] = None
    ) -> bool:
        """Export performance data to file."""
        if not self.performance_monitor:
            return False

        try:
            from pathlib import Path

            self.performance_monitor.export_metrics(
                Path(filepath), include_history=True, time_range_hours=time_range_hours
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to export performance data: {e}")
            return False

    async def _initialize_component(
        self,
        component_name: str,
        init_func: Callable[[], Awaitable[Any]],
        enabled: bool = True,
        max_retries: int = 2,
    ) -> bool:
        """
        Initialize a component with enhanced error handling and retry logic.

        Args:
            component_name: Name of the component
            init_func: Async function to initialize the component
            enabled: Whether the component is enabled in config
            max_retries: Maximum number of retry attempts

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not enabled:
            self.health_manager.set_status(component_name, ComponentStatus.DISABLED)
            self.logger.info(f"Component {component_name} is disabled in configuration")
            return False

        self.health_manager.set_status(component_name, ComponentStatus.INITIALIZING)

        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Initializing {component_name}... (attempt {attempt + 1})")
                await init_func()

                self.health_manager.set_status(component_name, ComponentStatus.HEALTHY)
                await emit_event(EventType.SYSTEM_STARTUP, {"component": component_name})
                self.logger.info(f"Component {component_name} initialized successfully")
                return True

            except Exception as e:
                error_category = self._categorize_initialization_error(e)
                is_recoverable = self._is_error_recoverable(e, error_category)

                self.health_manager.record_error(component_name, e, error_category, is_recoverable)

                if attempt < max_retries and is_recoverable:
                    delay = 2.0 * (2**attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Component {component_name} initialization failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"Component {component_name} initialization failed permanently: {e}"
                    )
                    self.health_manager.set_status(component_name, ComponentStatus.FAILED)
                    return False

        return False

    def _categorize_initialization_error(self, error: Exception) -> ErrorCategory:
        """Categorize an initialization error."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        # Configuration errors
        if any(
            term in error_msg for term in ["config", "setting", "parameter", "invalid", "missing"]
        ):
            return ErrorCategory.CONFIGURATION

        # Dependency errors
        if any(
            term in error_msg
            for term in ["import", "module", "dependency", "not found", "no module"]
        ):
            return ErrorCategory.DEPENDENCY

        # Resource errors
        if any(term in error_msg for term in ["memory", "disk", "resource", "allocation", "vram"]):
            return ErrorCategory.RESOURCE

        # Network errors
        if any(
            term in error_msg for term in ["network", "connection", "timeout", "unreachable", "api"]
        ):
            return ErrorCategory.NETWORK

        # Permission errors
        if any(term in error_msg for term in ["permission", "access", "denied", "unauthorized"]):
            return ErrorCategory.PERMISSION

        return ErrorCategory.UNKNOWN

    def _is_error_recoverable(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable and worth retrying."""
        # Configuration errors are usually not recoverable without user intervention
        if category == ErrorCategory.CONFIGURATION:
            return False

        # Dependency errors are not recoverable at runtime
        if category == ErrorCategory.DEPENDENCY:
            return False

        # Permission errors are not recoverable at runtime
        if category == ErrorCategory.PERMISSION:
            return False

        # Resource and network errors might be transient
        if category in [ErrorCategory.RESOURCE, ErrorCategory.NETWORK]:
            return True

        # Unknown errors - be conservative and try once
        return True

    async def _init_memory_manager(self):
        """Initialize memory manager with proper config conversion."""
        from ..components.memory.models import LongTermMemoryConfig as MemLongTermConfig
        from ..components.memory.models import (
            MemoryManagerConfig,
        )
        from ..components.memory.models import (
            ShortTermMemoryConfig as MemShortTermConfig,
        )

        memory_config = MemoryManagerConfig(
            short_term=MemShortTermConfig(**self.config.memory.short_term.model_dump()),
            long_term=MemLongTermConfig(**self.config.memory.long_term.model_dump()),
        )

        self.memory_manager = MemoryManager(memory_config)
        if hasattr(self.memory_manager, "initialize"):
            await self.memory_manager.initialize()

    async def _init_llm_manager(self):
        """Initialize LLM manager with config adapter."""
        try:
            llm_config = ConfigAdapter.adapt_config_for_component(self.config, "llm")
            self.llm_manager = LLMManager(llm_config)
        except Exception:
            # Fall back to direct config usage
            self.llm_manager = LLMManager(self.config.llm)

        if hasattr(self.llm_manager, "initialize"):
            await self.llm_manager.initialize()

    async def _init_personality_manager(self):
        """Initialize personality manager with config adapter."""
        try:
            personality_config = ConfigAdapter.adapt_config_for_component(
                self.config, "personality"
            )
            self.personality_manager = PersonalityManager(personality_config)
        except Exception:
            # Fall back to direct config usage
            self.personality_manager = PersonalityManager(self.config.personality)

        if hasattr(self.personality_manager, "initialize"):
            await self.personality_manager.initialize()

    async def _init_tools_manager(self):
        """Initialize tools manager with config adapter."""
        try:
            tools_config = ConfigAdapter.adapt_config_for_component(self.config, "tools")
            self.tools_manager = ToolManager(tools_config)
        except Exception:
            # Fall back to direct config usage
            self.tools_manager = ToolManager(self.config.tools)

        if hasattr(self.tools_manager, "initialize"):
            await self.tools_manager.initialize()

    async def _init_voice_manager(self):
        """Initialize voice manager with enhanced integration."""
        try:
            voice_config = ConfigAdapter.adapt_config_for_component(self.config, "voice")
            self.voice_manager = VoiceManager(voice_config)
            if hasattr(self.voice_manager, "initialize"):
                await self.voice_manager.initialize(voice_config)
        except Exception:
            # Fall back to direct config usage
            self.voice_manager = VoiceManager(self.config.voice)
            if hasattr(self.voice_manager, "initialize"):
                await self.voice_manager.initialize(self.config.voice)

        # Set up integrations with other components after initialization
        try:
            if self.memory_manager:
                self.voice_manager.memory_manager = self.memory_manager
                self.logger.debug("Voice manager connected to memory manager")

            if self.personality_manager:
                self.voice_manager.personality_manager = self.personality_manager
                self.logger.debug("Voice manager connected to personality manager")

            if self.tools_manager:
                self.voice_manager.tool_manager = self.tools_manager
                self.logger.debug("Voice manager connected to tools manager")

            # Connect to WebSocket integration if available
            if hasattr(self, "websocket_integration") and self.websocket_integration:
                self.websocket_integration.set_voice_manager(self.voice_manager)
                self.logger.debug("Voice manager connected to WebSocket integration")

            self.logger.info("Voice manager initialized with component integrations")
        except Exception as e:
            await handle_error(e, "voice", "initialization")
            self.logger.error(f"Failed to initialize voice manager: {e}")

    async def _setup_error_handling(self):
        """Setup error handling and recovery systems."""
        self.logger.info("Setting up error handling and recovery systems...")

        # Register component recovery handlers
        self._register_recovery_handlers()

        # Start component health monitoring
        await self.recovery_manager.start_monitoring()

        self.logger.info("Error handling and recovery systems initialized")

    async def _setup_performance_optimization(self):
        """Setup performance optimization system."""
        self.logger.info("Setting up performance optimization system...")

        # Determine optimization level based on configuration
        optimization_level = OptimizationLevel.BALANCED
        if hasattr(self.config, "performance") and hasattr(
            self.config.performance, "optimization_level"
        ):
            optimization_level = OptimizationLevel(self.config.performance.optimization_level)

        # Create performance targets
        targets = PerformanceTarget(
            max_response_time_ms=2000.0,  # 2 second max response time
            max_memory_usage_percent=85.0,
            max_cpu_usage_percent=75.0,
            max_gpu_memory_percent=90.0,
            min_cache_hit_rate=0.7,
            target_throughput_rps=50.0,
        )

        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer(
            optimization_level=optimization_level, targets=targets
        )

        await self.performance_optimizer.initialize()

        # Register components for optimization
        self._register_components_for_optimization()

        self.logger.info("Performance optimization system initialized")

    def _register_components_for_optimization(self):
        """Register components with the performance optimizer."""
        if not self.performance_optimizer:
            return

        # Register core components
        self.performance_optimizer.register_component(
            "llm",
            "llm",
            {"cache_responses": True, "connection_pooling": True, "gpu_optimization": True},
        )

        self.performance_optimizer.register_component(
            "memory", "memory", {"cache_embeddings": True, "optimize_storage": True}
        )

        self.performance_optimizer.register_component(
            "voice",
            "voice",
            {"buffer_pooling": True, "gpu_optimization": True, "audio_caching": True},
        )

        self.performance_optimizer.register_component(
            "websocket", "websocket", {"connection_pooling": True, "message_compression": True}
        )

        self.performance_optimizer.register_component(
            "dashboard", "dashboard", {"static_caching": True, "response_compression": True}
        )

        self.logger.info("Components registered for performance optimization")

    def _register_recovery_handlers(self):
        """Register recovery handlers for all components."""

        # LLM component recovery
        async def restart_llm():
            try:
                if self.llm_manager:
                    await self.llm_manager.shutdown()
                await self._init_llm_manager()
                return True
            except Exception as e:
                self.logger.error(f"Failed to restart LLM manager: {e}")
                return False

        async def llm_fallback():
            # Implement LLM fallback mode (e.g., use a simpler model)
            self.logger.info("Enabling LLM fallback mode")
            return True

        # Memory component recovery
        async def restart_memory():
            try:
                if self.memory_manager:
                    await self.memory_manager.shutdown()
                await self._init_memory_manager()
                return True
            except Exception as e:
                self.logger.error(f"Failed to restart memory manager: {e}")
                return False

        async def memory_fallback():
            # Implement memory fallback mode (e.g., session-only memory)
            self.logger.info("Enabling memory fallback mode")
            return True

        # Voice component recovery
        async def restart_voice():
            try:
                if self.voice_manager:
                    await self.voice_manager.shutdown()
                if self.config.voice:
                    await self._init_voice_manager()
                return True
            except Exception as e:
                self.logger.error(f"Failed to restart voice manager: {e}")
                return False

        async def voice_fallback():
            # Disable voice processing, continue with text only
            self.logger.info("Disabling voice processing - text-only mode")
            if self.voice_manager:
                await self.voice_manager.shutdown()
                self.voice_manager = None
            return True

        # Tools component recovery
        async def restart_tools():
            try:
                if self.tools_manager:
                    await self.tools_manager.shutdown()
                await self._init_tools_manager()
                return True
            except Exception as e:
                self.logger.error(f"Failed to restart tools manager: {e}")
                return False

        async def tools_fallback():
            # Disable tools, continue with basic functionality
            self.logger.info("Disabling tools - basic mode")
            if self.tools_manager:
                await self.tools_manager.shutdown()
                self.tools_manager = None
            return True

        # Personality component recovery
        async def restart_personality():
            try:
                if self.personality_manager:
                    await self.personality_manager.shutdown()
                await self._init_personality_manager()
                return True
            except Exception as e:
                self.logger.error(f"Failed to restart personality manager: {e}")
                return False

        # Register components with recovery manager
        self.recovery_manager.register_component("llm", is_critical=True)
        self.recovery_manager.register_component("memory", is_critical=True)
        self.recovery_manager.register_component("voice", is_critical=False)
        self.recovery_manager.register_component("tools", is_critical=False)
        self.recovery_manager.register_component("personality", is_critical=False)

        # Register restart handlers
        self.recovery_manager.register_restart_handler("llm", restart_llm)
        self.recovery_manager.register_restart_handler("memory", restart_memory)
        self.recovery_manager.register_restart_handler("voice", restart_voice)
        self.recovery_manager.register_restart_handler("tools", restart_tools)
        self.recovery_manager.register_restart_handler("personality", restart_personality)

        # Register fallback handlers
        self.recovery_manager.register_fallback_handler("llm", llm_fallback)
        self.recovery_manager.register_fallback_handler("memory", memory_fallback)
        self.recovery_manager.register_fallback_handler("voice", voice_fallback)
        self.recovery_manager.register_fallback_handler("tools", tools_fallback)

        # Register health check handlers
        self.recovery_manager.register_health_check_handler("llm", self._check_llm_health)
        self.recovery_manager.register_health_check_handler("memory", self._check_memory_health)
        self.recovery_manager.register_health_check_handler("voice", self._check_voice_health)
        self.recovery_manager.register_health_check_handler("tools", self._check_tools_health)
        self.recovery_manager.register_health_check_handler(
            "personality", self._check_personality_health
        )

    def _check_llm_health(self) -> bool:
        """Check LLM manager health."""
        return (
            self.llm_manager is not None
            and hasattr(self.llm_manager, "is_healthy")
            and self.llm_manager.is_healthy()
        )

    def _check_memory_health(self) -> bool:
        """Check memory manager health."""
        return (
            self.memory_manager is not None
            and hasattr(self.memory_manager, "is_healthy")
            and self.memory_manager.is_healthy()
        )

    def _check_voice_health(self) -> bool:
        """Check voice manager health."""
        return (
            self.voice_manager is not None
            and hasattr(self.voice_manager, "is_healthy")
            and self.voice_manager.is_healthy()
        )

    def _check_tools_health(self) -> bool:
        """Check tools manager health."""
        return (
            self.tools_manager is not None
            and hasattr(self.tools_manager, "is_healthy")
            and self.tools_manager.is_healthy()
        )

    def _check_personality_health(self) -> bool:
        """Check personality manager health."""
        return (
            self.personality_manager is not None
            and hasattr(self.personality_manager, "is_healthy")
            and self.personality_manager.is_healthy()
        )

    async def get_error_status(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery status."""
        performance_report = {}
        if self.performance_optimizer:
            performance_report = self.performance_optimizer.get_performance_report()

        return {
            "error_statistics": self.error_manager.get_error_statistics(),
            "system_health": self.recovery_manager.get_system_health_summary(),
            "user_status": self.user_error_interface.get_system_status_message().__dict__,
            "performance": performance_report,
        }
