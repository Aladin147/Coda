"""
Component Integration Layer for Coda.

This module provides a comprehensive integration layer that manages dependencies,
communication, and coordination between all Coda components with proper
dependency injection and lifecycle management.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger("coda.core.integration")

T = TypeVar("T")


class ComponentType(Enum):
    """Types of components in the Coda system."""

    MEMORY = "memory"
    LLM = "llm"
    PERSONALITY = "personality"
    TOOLS = "tools"
    VOICE = "voice"
    WEBSOCKET = "websocket"
    EVENT_BUS = "event_bus"
    ASSISTANT = "assistant"


class ComponentState(Enum):
    """Component lifecycle states."""

    NOT_REGISTERED = "not_registered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentDependency:
    """Represents a dependency between components."""

    dependent: ComponentType
    dependency: ComponentType
    required: bool = True
    description: str = ""


@dataclass
class ComponentMetadata:
    """Metadata about a registered component."""

    component_type: ComponentType
    instance: Any
    state: ComponentState = ComponentState.REGISTERED
    dependencies: List[ComponentType] = field(default_factory=list)
    dependents: List[ComponentType] = field(default_factory=list)
    initialization_order: int = 0
    error_count: int = 0
    last_error: Optional[Exception] = None


class ComponentInterface(ABC):
    """Base interface for all Coda components."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component."""

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""


class IntegrationEvent:
    """Event for component integration notifications."""

    def __init__(self, event_type: str, component_type: ComponentType, data: Dict[str, Any] = None):
        self.event_type = event_type
        self.component_type = component_type
        self.data = data or {}
        self.timestamp = time.time()


class ComponentIntegrationLayer:
    """
    Central integration layer that manages all component dependencies,
    lifecycle, and communication.
    """

    def __init__(self):
        self.components: Dict[ComponentType, ComponentMetadata] = {}
        self.dependencies: List[ComponentDependency] = []
        self.event_handlers: Dict[str, List[Callable[[IntegrationEvent], Awaitable[None]]]] = {}
        self.integration_lock = asyncio.Lock()

        # Define component dependencies
        self._define_dependencies()

        logger.info("ComponentIntegrationLayer initialized")

    def _define_dependencies(self):
        """Define the dependency graph between components."""
        self.dependencies = [
            # Event bus is foundational
            ComponentDependency(ComponentType.MEMORY, ComponentType.EVENT_BUS, required=True),
            ComponentDependency(ComponentType.LLM, ComponentType.EVENT_BUS, required=True),
            ComponentDependency(ComponentType.PERSONALITY, ComponentType.EVENT_BUS, required=True),
            ComponentDependency(ComponentType.TOOLS, ComponentType.EVENT_BUS, required=True),
            ComponentDependency(ComponentType.VOICE, ComponentType.EVENT_BUS, required=True),
            # Memory is used by other components
            ComponentDependency(ComponentType.LLM, ComponentType.MEMORY, required=False),
            ComponentDependency(ComponentType.PERSONALITY, ComponentType.MEMORY, required=False),
            ComponentDependency(ComponentType.VOICE, ComponentType.MEMORY, required=False),
            # LLM is used by other components
            ComponentDependency(ComponentType.PERSONALITY, ComponentType.LLM, required=False),
            ComponentDependency(ComponentType.VOICE, ComponentType.LLM, required=False),
            ComponentDependency(ComponentType.TOOLS, ComponentType.LLM, required=False),
            # Tools can be used by other components
            ComponentDependency(ComponentType.LLM, ComponentType.TOOLS, required=False),
            ComponentDependency(ComponentType.VOICE, ComponentType.TOOLS, required=False),
            # WebSocket depends on other components for events
            ComponentDependency(ComponentType.WEBSOCKET, ComponentType.EVENT_BUS, required=True),
            ComponentDependency(ComponentType.WEBSOCKET, ComponentType.MEMORY, required=False),
            ComponentDependency(ComponentType.WEBSOCKET, ComponentType.LLM, required=False),
            ComponentDependency(ComponentType.WEBSOCKET, ComponentType.PERSONALITY, required=False),
            ComponentDependency(ComponentType.WEBSOCKET, ComponentType.TOOLS, required=False),
            ComponentDependency(ComponentType.WEBSOCKET, ComponentType.VOICE, required=False),
        ]

    def register_component(
        self, component_type: ComponentType, instance: Any, dependencies: List[ComponentType] = None
    ) -> None:
        """
        Register a component with the integration layer.

        Args:
            component_type: Type of the component
            instance: Component instance
            dependencies: Optional override of dependencies
        """
        if component_type in self.components:
            logger.warning(f"Component {component_type.value} already registered, replacing")

        # Determine dependencies
        component_deps = dependencies or self._get_component_dependencies(component_type)

        # Create metadata
        metadata = ComponentMetadata(
            component_type=component_type,
            instance=instance,
            dependencies=component_deps,
            initialization_order=self._calculate_initialization_order(component_type),
        )

        self.components[component_type] = metadata

        # Update dependents
        self._update_dependents()

        logger.info(f"Registered component: {component_type.value}")

    def _get_component_dependencies(self, component_type: ComponentType) -> List[ComponentType]:
        """Get dependencies for a component type."""
        return [
            dep.dependency
            for dep in self.dependencies
            if dep.dependent == component_type and dep.required
        ]

    def _calculate_initialization_order(self, component_type: ComponentType) -> int:
        """Calculate initialization order based on dependencies."""
        order_map = {
            ComponentType.EVENT_BUS: 0,
            ComponentType.MEMORY: 1,
            ComponentType.LLM: 2,
            ComponentType.TOOLS: 2,
            ComponentType.PERSONALITY: 3,
            ComponentType.VOICE: 4,
            ComponentType.WEBSOCKET: 5,
        }
        return order_map.get(component_type, 10)

    def _update_dependents(self):
        """Update dependent lists for all components."""
        for metadata in self.components.values():
            metadata.dependents.clear()

        for dep in self.dependencies:
            if dep.dependency in self.components and dep.dependent in self.components:
                self.components[dep.dependency].dependents.append(dep.dependent)

    def get_component(self, component_type: ComponentType) -> Optional[Any]:
        """Get a registered component instance."""
        metadata = self.components.get(component_type)
        return metadata.instance if metadata else None

    def get_component_metadata(self, component_type: ComponentType) -> Optional[ComponentMetadata]:
        """Get component metadata."""
        return self.components.get(component_type)

    async def initialize_all_components(self) -> Dict[ComponentType, bool]:
        """
        Initialize all registered components in dependency order.

        Returns:
            Dictionary mapping component types to success status
        """
        async with self.integration_lock:
            results = {}

            # Sort components by initialization order
            sorted_components = sorted(
                self.components.items(), key=lambda x: x[1].initialization_order
            )

            for component_type, metadata in sorted_components:
                try:
                    await self._initialize_component(component_type, metadata)
                    results[component_type] = True
                except Exception as e:
                    logger.error(f"Failed to initialize {component_type.value}: {e}")
                    metadata.state = ComponentState.FAILED
                    metadata.last_error = e
                    metadata.error_count += 1
                    results[component_type] = False

            return results

    async def _initialize_component(
        self, component_type: ComponentType, metadata: ComponentMetadata
    ):
        """Initialize a single component."""
        logger.info(f"Initializing component: {component_type.value}")

        # Check dependencies are ready
        await self._wait_for_dependencies(component_type)

        # Set state
        metadata.state = ComponentState.INITIALIZING

        # Initialize component
        if hasattr(metadata.instance, "initialize"):
            await metadata.instance.initialize()

        # Set up integrations
        await self._setup_component_integrations(component_type, metadata)

        # Mark as ready
        metadata.state = ComponentState.READY

        # Emit event
        await self._emit_event(
            "component_initialized",
            component_type,
            {"initialization_order": metadata.initialization_order},
        )

        logger.info(f"Component {component_type.value} initialized successfully")

    async def _wait_for_dependencies(self, component_type: ComponentType, timeout: float = 30.0):
        """Wait for component dependencies to be ready."""
        metadata = self.components[component_type]

        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ready = True

            for dep_type in metadata.dependencies:
                dep_metadata = self.components.get(dep_type)
                if not dep_metadata or dep_metadata.state != ComponentState.READY:
                    all_ready = False
                    break

            if all_ready:
                return

            await asyncio.sleep(0.1)

        # Check for required dependencies that are missing or not ready
        missing_required = []
        for dep in self.dependencies:
            if dep.dependent == component_type and dep.required:
                if dep.dependency not in self.components:
                    missing_required.append(dep.dependency)
                elif self.components[dep.dependency].state != ComponentState.READY:
                    missing_required.append(dep.dependency)

        if missing_required:
            raise RuntimeError(f"Required dependencies not ready: {missing_required}")

    async def _setup_component_integrations(
        self, component_type: ComponentType, metadata: ComponentMetadata
    ):
        """Set up integrations between components."""
        instance = metadata.instance

        # Set up common integrations based on component type
        if component_type == ComponentType.LLM:
            await self._setup_llm_integrations(instance)
        elif component_type == ComponentType.PERSONALITY:
            await self._setup_personality_integrations(instance)
        elif component_type == ComponentType.VOICE:
            await self._setup_voice_integrations(instance)
        elif component_type == ComponentType.TOOLS:
            await self._setup_tools_integrations(instance)
        elif component_type == ComponentType.WEBSOCKET:
            await self._setup_websocket_integrations(instance)

    async def _setup_llm_integrations(self, llm_manager):
        """Set up LLM manager integrations."""
        # Integrate with memory
        memory_manager = self.get_component(ComponentType.MEMORY)
        if memory_manager and hasattr(llm_manager, "set_memory_manager"):
            await llm_manager.set_memory_manager(memory_manager)

        # Integrate with personality
        personality_manager = self.get_component(ComponentType.PERSONALITY)
        if personality_manager and hasattr(llm_manager, "set_personality_manager"):
            await llm_manager.set_personality_manager(personality_manager)

        # Integrate with tools
        tools_manager = self.get_component(ComponentType.TOOLS)
        if tools_manager and hasattr(llm_manager, "set_tools_manager"):
            await llm_manager.set_tools_manager(tools_manager)

    async def _setup_personality_integrations(self, personality_manager):
        """Set up personality manager integrations."""
        # Integrate with memory
        memory_manager = self.get_component(ComponentType.MEMORY)
        if memory_manager and hasattr(personality_manager, "set_memory_manager"):
            await personality_manager.set_memory_manager(memory_manager)

    async def _setup_voice_integrations(self, voice_manager):
        """Set up voice manager integrations."""
        # Integrate with memory
        memory_manager = self.get_component(ComponentType.MEMORY)
        if memory_manager and hasattr(voice_manager, "set_memory_manager"):
            await voice_manager.set_memory_manager(memory_manager)

        # Integrate with LLM
        llm_manager = self.get_component(ComponentType.LLM)
        if llm_manager and hasattr(voice_manager, "set_llm_manager"):
            await voice_manager.set_llm_manager(llm_manager)

        # Integrate with personality
        personality_manager = self.get_component(ComponentType.PERSONALITY)
        if personality_manager and hasattr(voice_manager, "set_personality_manager"):
            await voice_manager.set_personality_manager(personality_manager)

        # Integrate with tools
        tools_manager = self.get_component(ComponentType.TOOLS)
        if tools_manager and hasattr(voice_manager, "set_tools_manager"):
            await voice_manager.set_tools_manager(tools_manager)

    async def _setup_tools_integrations(self, tools_manager):
        """Set up tools manager integrations."""
        # Integrate with memory
        memory_manager = self.get_component(ComponentType.MEMORY)
        if memory_manager and hasattr(tools_manager, "set_memory_manager"):
            await tools_manager.set_memory_manager(memory_manager)

    async def _setup_websocket_integrations(self, websocket_server):
        """Set up WebSocket server integrations."""
        # Integrate with all available components
        for component_type, metadata in self.components.items():
            if component_type != ComponentType.WEBSOCKET and metadata.state == ComponentState.READY:
                if hasattr(metadata.instance, "set_websocket_integration"):
                    try:
                        await metadata.instance.set_websocket_integration(websocket_server)
                    except Exception as e:
                        logger.warning(
                            f"Failed to set WebSocket integration for {component_type.value}: {e}"
                        )

    async def shutdown_all_components(self) -> Dict[ComponentType, bool]:
        """
        Shutdown all components in reverse dependency order.

        Returns:
            Dictionary mapping component types to success status
        """
        async with self.integration_lock:
            results = {}

            # Sort components by reverse initialization order
            sorted_components = sorted(
                self.components.items(), key=lambda x: x[1].initialization_order, reverse=True
            )

            for component_type, metadata in sorted_components:
                try:
                    await self._shutdown_component(component_type, metadata)
                    results[component_type] = True
                except Exception as e:
                    logger.error(f"Failed to shutdown {component_type.value}: {e}")
                    results[component_type] = False

            return results

    async def _shutdown_component(self, component_type: ComponentType, metadata: ComponentMetadata):
        """Shutdown a single component."""
        logger.info(f"Shutting down component: {component_type.value}")

        metadata.state = ComponentState.SHUTTING_DOWN

        # Shutdown component
        if hasattr(metadata.instance, "shutdown"):
            await metadata.instance.shutdown()
        elif hasattr(metadata.instance, "cleanup"):
            await metadata.instance.cleanup()

        metadata.state = ComponentState.SHUTDOWN

        # Emit event
        await self._emit_event("component_shutdown", component_type)

        logger.info(f"Component {component_type.value} shutdown successfully")

    def add_event_handler(
        self, event_type: str, handler: Callable[[IntegrationEvent], Awaitable[None]]
    ):
        """Add an event handler for integration events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def _emit_event(
        self, event_type: str, component_type: ComponentType, data: Dict[str, Any] = None
    ):
        """Emit an integration event."""
        event = IntegrationEvent(event_type, component_type, data)

        handlers = self.event_handlers.get(event_type, [])
        if handlers:
            await asyncio.gather(*[handler(event) for handler in handlers], return_exceptions=True)

    def get_integration_status(self) -> Dict[str, Any]:
        """Get overall integration status."""
        total_components = len(self.components)
        ready_components = sum(
            1 for m in self.components.values() if m.state == ComponentState.READY
        )
        failed_components = sum(
            1 for m in self.components.values() if m.state == ComponentState.FAILED
        )

        return {
            "total_components": total_components,
            "ready_components": ready_components,
            "failed_components": failed_components,
            "integration_health": "healthy" if failed_components == 0 else "degraded",
            "components": {
                comp_type.value: {
                    "state": metadata.state.value,
                    "dependencies": [dep.value for dep in metadata.dependencies],
                    "dependents": [dep.value for dep in metadata.dependents],
                    "error_count": metadata.error_count,
                    "last_error": str(metadata.last_error) if metadata.last_error else None,
                }
                for comp_type, metadata in self.components.items()
            },
        }
