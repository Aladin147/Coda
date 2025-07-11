"""
Tests for WebSocket Integration with Component Integration Layer.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.coda.core.websocket_integration import ComponentWebSocketIntegration
from src.coda.core.integration import (
    ComponentIntegrationLayer, ComponentType, ComponentState, IntegrationEvent
)
from src.coda.interfaces.websocket.server import CodaWebSocketServer


class MockWebSocketServer:
    """Mock WebSocket server for testing."""
    
    def __init__(self):
        self.broadcast_calls = []
        self.running = False
    
    async def broadcast(self, event):
        """Mock broadcast method."""
        self.broadcast_calls.append(event)
    
    async def start(self):
        """Mock start method."""
        self.running = True
    
    async def stop(self):
        """Mock stop method."""
        self.running = False


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


class TestComponentWebSocketIntegration:
    """Test WebSocket integration with component integration layer."""
    
    @pytest.fixture
    def integration_layer(self):
        """Create integration layer for testing."""
        return ComponentIntegrationLayer()
    
    @pytest.fixture
    def mock_websocket_server(self):
        """Create mock WebSocket server."""
        return MockWebSocketServer()
    
    @pytest.fixture
    def websocket_integration(self, integration_layer, mock_websocket_server):
        """Create WebSocket integration for testing."""
        integration = ComponentWebSocketIntegration(integration_layer)
        integration.set_websocket_server(mock_websocket_server)
        return integration
    
    def test_initialization(self, integration_layer):
        """Test WebSocket integration initialization."""
        integration = ComponentWebSocketIntegration(integration_layer)
        
        assert integration.integration_layer == integration_layer
        assert integration.websocket_server is None
        assert not integration.is_active
        assert not integration.event_handlers_registered
        assert integration.sequence_number == 0
    
    def test_set_websocket_server(self, websocket_integration, mock_websocket_server):
        """Test setting WebSocket server."""
        assert websocket_integration.websocket_server == mock_websocket_server
    
    @pytest.mark.asyncio
    async def test_start_integration(self, websocket_integration, mock_websocket_server):
        """Test starting WebSocket integration."""
        await websocket_integration.start()
        
        assert websocket_integration.is_active
        assert websocket_integration.event_handlers_registered
        
        # Should broadcast initial system status
        assert len(mock_websocket_server.broadcast_calls) > 0
    
    @pytest.mark.asyncio
    async def test_stop_integration(self, websocket_integration):
        """Test stopping WebSocket integration."""
        await websocket_integration.start()
        assert websocket_integration.is_active
        
        await websocket_integration.stop()
        assert not websocket_integration.is_active
    
    @pytest.mark.asyncio
    async def test_component_initialized_event(self, websocket_integration, integration_layer, mock_websocket_server):
        """Test handling component initialized events."""
        # Start integration
        await websocket_integration.start()
        
        # Register a mock component
        mock_component = MockComponent("test_component")
        integration_layer.register_component(ComponentType.MEMORY, mock_component)
        
        # Create integration event
        event = IntegrationEvent(
            event_type="component_initialized",
            component_type=ComponentType.MEMORY,
            data={"initialization_order": 1}
        )
        
        # Handle the event
        await websocket_integration._handle_component_initialized(event)
        
        # Check that event was broadcasted
        assert len(mock_websocket_server.broadcast_calls) >= 1
        
        # Find the component status event
        component_events = [call for call in mock_websocket_server.broadcast_calls 
                          if hasattr(call, 'type') and call.type.value == 'component_status']
        assert len(component_events) > 0
        
        event_data = component_events[0]
        assert event_data.component_type == "memory"
        assert event_data.state == "initialized"
    
    @pytest.mark.asyncio
    async def test_component_failed_event(self, websocket_integration, integration_layer, mock_websocket_server):
        """Test handling component failed events."""
        # Start integration
        await websocket_integration.start()
        
        # Register a mock component
        mock_component = MockComponent("test_component")
        integration_layer.register_component(ComponentType.LLM, mock_component)
        
        # Simulate component failure
        metadata = integration_layer.get_component_metadata(ComponentType.LLM)
        metadata.error_count = 1
        metadata.last_error = Exception("Test error")
        
        # Create integration event
        event = IntegrationEvent(
            event_type="component_failed",
            component_type=ComponentType.LLM,
            data={"error": "Test error"}
        )
        
        # Handle the event
        await websocket_integration._handle_component_failed(event)
        
        # Check that error event was broadcasted
        error_events = [call for call in mock_websocket_server.broadcast_calls 
                       if hasattr(call, 'type') and call.type.value == 'component_error']
        assert len(error_events) > 0
        
        event_data = error_events[0]
        assert event_data.component_type == "llm"
        assert event_data.state == "failed"
        assert event_data.error_count == 1
    
    @pytest.mark.asyncio
    async def test_system_status_broadcast(self, websocket_integration, integration_layer, mock_websocket_server):
        """Test system status broadcasting."""
        # Start integration
        await websocket_integration.start()
        
        # Register some components
        mock_component1 = MockComponent("component1")
        mock_component2 = MockComponent("component2")
        integration_layer.register_component(ComponentType.MEMORY, mock_component1)
        integration_layer.register_component(ComponentType.LLM, mock_component2)
        
        # Clear previous broadcasts
        mock_websocket_server.broadcast_calls.clear()
        
        # Broadcast system status
        await websocket_integration._broadcast_system_status()
        
        # Check that system status was broadcasted
        system_events = [call for call in mock_websocket_server.broadcast_calls 
                        if hasattr(call, 'type') and call.type.value == 'system_status']
        assert len(system_events) > 0
        
        event_data = system_events[0]
        assert event_data.total_components == 2
        assert hasattr(event_data, 'components')
    
    @pytest.mark.asyncio
    async def test_get_component_status(self, websocket_integration, integration_layer):
        """Test getting component status."""
        # Register a component
        mock_component = MockComponent("test_component")
        integration_layer.register_component(ComponentType.MEMORY, mock_component)
        
        # Get specific component status
        status = await websocket_integration.get_component_status("memory")
        assert status["component_type"] == "memory"
        assert "status" in status
        
        # Get all components status
        all_status = await websocket_integration.get_component_status()
        assert "system_status" in all_status
        assert "timestamp" in all_status
        
        # Test non-existent component
        not_found = await websocket_integration.get_component_status("nonexistent")
        assert "error" in not_found
    
    @pytest.mark.asyncio
    async def test_event_history(self, websocket_integration, mock_websocket_server):
        """Test event history tracking."""
        # Start integration
        await websocket_integration.start()

        # Clear initial events from start()
        websocket_integration.component_events.clear()

        # Simulate some events with proper data structure
        status_data = {"component_type": "memory", "state": "initialized"}
        error_data = {"component_type": "llm", "error_message": "Test error", "error_count": 1}
        await websocket_integration._broadcast_component_event("component_status", status_data)
        await websocket_integration._broadcast_component_event("component_error", error_data)

        # Get event history
        history = await websocket_integration.get_event_history(10)
        assert len(history) == 2
        
        # Check event structure
        for event in history:
            assert "event_type" in event
            assert "data" in event
            assert "timestamp" in event
    
    def test_integration_metrics(self, websocket_integration):
        """Test getting integration metrics."""
        metrics = websocket_integration.get_integration_metrics()
        
        assert "is_active" in metrics
        assert "websocket_server_available" in metrics
        assert "event_handlers_registered" in metrics
        assert "total_events_processed" in metrics
        assert "timestamp" in metrics
        
        # Initially not active
        assert not metrics["is_active"]
        assert metrics["websocket_server_available"]  # Mock server is set
        assert not metrics["event_handlers_registered"]
    
    @pytest.mark.asyncio
    async def test_sequence_number_increment(self, websocket_integration, mock_websocket_server):
        """Test that sequence numbers increment correctly."""
        await websocket_integration.start()
        
        initial_seq = websocket_integration.sequence_number
        
        # Broadcast an event
        await websocket_integration._broadcast_component_event("component_status", {"test": "data"})
        
        # Sequence number should have incremented
        assert websocket_integration.sequence_number == initial_seq + 1
    
    @pytest.mark.asyncio
    async def test_integration_without_websocket_server(self, integration_layer):
        """Test integration behavior without WebSocket server."""
        integration = ComponentWebSocketIntegration(integration_layer)
        
        # Should not start without WebSocket server
        await integration.start()
        assert not integration.is_active
        
        # Should handle events gracefully
        await integration._broadcast_component_event("component_status", {"test": "data"})
        # Should not raise exceptions
    
    @pytest.mark.asyncio
    async def test_event_storage_limit(self, websocket_integration, mock_websocket_server):
        """Test event history storage limit."""
        await websocket_integration.start()
        
        # Set a small limit for testing
        websocket_integration.max_event_history = 3
        
        # Generate more events than the limit
        for i in range(5):
            await websocket_integration._broadcast_component_event("component_status", {
                "component_type": "memory",
                "state": "initialized",
                "event": i
            })
        
        # Should only keep the last 3 events
        history = await websocket_integration.get_event_history(10)
        assert len(history) == 3
        
        # Should be the most recent events
        assert history[0]["data"]["event"] == 2
        assert history[1]["data"]["event"] == 3
        assert history[2]["data"]["event"] == 4
