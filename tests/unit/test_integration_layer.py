"""
Tests for the Component Integration Layer.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.coda.core.integration import (
    ComponentIntegrationLayer, ComponentType, ComponentState,
    ComponentDependency, ComponentMetadata, IntegrationEvent
)


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.shutdown_called = False
        
    async def initialize(self):
        """Mock initialize method."""
        self.initialized = True
        
    async def shutdown(self):
        """Mock shutdown method."""
        self.shutdown_called = True
        
    def get_health_status(self):
        """Mock health status method."""
        return {"status": "healthy", "name": self.name}


class TestComponentIntegrationLayer:
    """Test the component integration layer."""
    
    @pytest.fixture
    def integration_layer(self):
        """Create integration layer for testing."""
        return ComponentIntegrationLayer()
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return {
            ComponentType.EVENT_BUS: MockComponent("event_bus"),
            ComponentType.MEMORY: MockComponent("memory"),
            ComponentType.LLM: MockComponent("llm"),
            ComponentType.PERSONALITY: MockComponent("personality"),
            ComponentType.TOOLS: MockComponent("tools"),
        }
    
    def test_component_registration(self, integration_layer, mock_components):
        """Test component registration."""
        # Register a component
        integration_layer.register_component(
            ComponentType.MEMORY, 
            mock_components[ComponentType.MEMORY]
        )
        
        # Verify registration
        assert ComponentType.MEMORY in integration_layer.components
        metadata = integration_layer.components[ComponentType.MEMORY]
        assert metadata.component_type == ComponentType.MEMORY
        assert metadata.instance == mock_components[ComponentType.MEMORY]
        assert metadata.state == ComponentState.REGISTERED
    
    def test_dependency_calculation(self, integration_layer):
        """Test dependency calculation."""
        # Memory should depend on event bus
        memory_deps = integration_layer._get_component_dependencies(ComponentType.MEMORY)
        assert ComponentType.EVENT_BUS in memory_deps
        
        # LLM should depend on event bus
        llm_deps = integration_layer._get_component_dependencies(ComponentType.LLM)
        assert ComponentType.EVENT_BUS in llm_deps
    
    def test_initialization_order(self, integration_layer):
        """Test initialization order calculation."""
        # Event bus should have lowest order (0)
        event_bus_order = integration_layer._calculate_initialization_order(ComponentType.EVENT_BUS)
        assert event_bus_order == 0
        
        # Memory should come after event bus
        memory_order = integration_layer._calculate_initialization_order(ComponentType.MEMORY)
        assert memory_order > event_bus_order
        
        # LLM should come after memory
        llm_order = integration_layer._calculate_initialization_order(ComponentType.LLM)
        assert llm_order >= memory_order
    
    def test_get_component(self, integration_layer, mock_components):
        """Test component retrieval."""
        # Register component
        integration_layer.register_component(
            ComponentType.MEMORY, 
            mock_components[ComponentType.MEMORY]
        )
        
        # Retrieve component
        retrieved = integration_layer.get_component(ComponentType.MEMORY)
        assert retrieved == mock_components[ComponentType.MEMORY]
        
        # Test non-existent component
        non_existent = integration_layer.get_component(ComponentType.VOICE)
        assert non_existent is None
    
    @pytest.mark.asyncio
    async def test_single_component_initialization(self, integration_layer, mock_components):
        """Test initializing a single component."""
        # Register event bus first (dependency)
        integration_layer.register_component(
            ComponentType.EVENT_BUS, 
            mock_components[ComponentType.EVENT_BUS]
        )
        
        # Register memory component
        integration_layer.register_component(
            ComponentType.MEMORY, 
            mock_components[ComponentType.MEMORY]
        )
        
        # Mark event bus as ready (simulate it being initialized)
        integration_layer.components[ComponentType.EVENT_BUS].state = ComponentState.READY
        
        # Initialize memory component
        metadata = integration_layer.components[ComponentType.MEMORY]
        await integration_layer._initialize_component(ComponentType.MEMORY, metadata)
        
        # Verify initialization
        assert mock_components[ComponentType.MEMORY].initialized
        assert metadata.state == ComponentState.READY
    
    @pytest.mark.asyncio
    async def test_full_initialization_flow(self, integration_layer, mock_components):
        """Test full component initialization flow."""
        # Register all components
        for comp_type, component in mock_components.items():
            integration_layer.register_component(comp_type, component)
        
        # Initialize all components
        results = await integration_layer.initialize_all_components()
        
        # Verify all components were initialized successfully
        for comp_type in mock_components.keys():
            assert results[comp_type] is True
            assert mock_components[comp_type].initialized
            assert integration_layer.components[comp_type].state == ComponentState.READY
    
    @pytest.mark.asyncio
    async def test_shutdown_flow(self, integration_layer, mock_components):
        """Test component shutdown flow."""
        # Register and initialize components
        for comp_type, component in mock_components.items():
            integration_layer.register_component(comp_type, component)
            integration_layer.components[comp_type].state = ComponentState.READY
        
        # Shutdown all components
        results = await integration_layer.shutdown_all_components()
        
        # Verify all components were shutdown successfully
        for comp_type in mock_components.keys():
            assert results[comp_type] is True
            assert mock_components[comp_type].shutdown_called
            assert integration_layer.components[comp_type].state == ComponentState.SHUTDOWN
    
    def test_integration_status(self, integration_layer, mock_components):
        """Test integration status reporting."""
        # Register components
        for comp_type, component in mock_components.items():
            integration_layer.register_component(comp_type, component)
        
        # Get status
        status = integration_layer.get_integration_status()
        
        # Verify status structure
        assert "total_components" in status
        assert "ready_components" in status
        assert "failed_components" in status
        assert "integration_health" in status
        assert "components" in status
        
        assert status["total_components"] == len(mock_components)
        assert status["ready_components"] == 0  # None initialized yet
        assert status["integration_health"] == "healthy"  # No failures
    
    @pytest.mark.asyncio
    async def test_event_handling(self, integration_layer):
        """Test integration event handling."""
        events_received = []
        
        async def event_handler(event: IntegrationEvent):
            events_received.append(event)
        
        # Add event handler
        integration_layer.add_event_handler("test_event", event_handler)
        
        # Emit event
        await integration_layer._emit_event("test_event", ComponentType.MEMORY, {"test": "data"})
        
        # Verify event was received
        assert len(events_received) == 1
        event = events_received[0]
        assert event.event_type == "test_event"
        assert event.component_type == ComponentType.MEMORY
        assert event.data["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_dependency_waiting(self, integration_layer, mock_components):
        """Test dependency waiting mechanism."""
        # Register components
        integration_layer.register_component(
            ComponentType.EVENT_BUS, 
            mock_components[ComponentType.EVENT_BUS]
        )
        integration_layer.register_component(
            ComponentType.MEMORY, 
            mock_components[ComponentType.MEMORY]
        )
        
        # Memory depends on event bus, but event bus is not ready
        # This should timeout quickly for testing
        with pytest.raises(RuntimeError):
            await integration_layer._wait_for_dependencies(ComponentType.MEMORY, timeout=0.1)
        
        # Mark event bus as ready
        integration_layer.components[ComponentType.EVENT_BUS].state = ComponentState.READY
        
        # Now dependency waiting should succeed
        await integration_layer._wait_for_dependencies(ComponentType.MEMORY, timeout=0.1)
    
    def test_component_metadata(self, integration_layer, mock_components):
        """Test component metadata management."""
        # Register component
        integration_layer.register_component(
            ComponentType.MEMORY, 
            mock_components[ComponentType.MEMORY]
        )
        
        # Get metadata
        metadata = integration_layer.get_component_metadata(ComponentType.MEMORY)
        
        # Verify metadata
        assert metadata is not None
        assert metadata.component_type == ComponentType.MEMORY
        assert metadata.instance == mock_components[ComponentType.MEMORY]
        assert metadata.state == ComponentState.REGISTERED
        assert metadata.error_count == 0
        assert metadata.last_error is None
        
        # Test non-existent component
        non_existent_metadata = integration_layer.get_component_metadata(ComponentType.VOICE)
        assert non_existent_metadata is None
