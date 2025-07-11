"""
Tests for Dashboard Server.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock
from pathlib import Path

from src.coda.interfaces.dashboard.server import CodaDashboardServer
from src.coda.core.integration import ComponentIntegrationLayer, ComponentType
from src.coda.core.websocket_integration import ComponentWebSocketIntegration


class MockIntegrationLayer:
    """Mock integration layer for testing."""
    
    def __init__(self):
        self.components = {}
    
    def get_integration_status(self):
        return {
            "total_components": 3,
            "ready_components": 2,
            "failed_components": 1,
            "integration_health": "degraded",
            "components": {
                "memory": {"state": "ready"},
                "llm": {"state": "ready"},
                "voice": {"state": "failed"}
            }
        }


class MockWebSocketIntegration:
    """Mock WebSocket integration for testing."""
    
    def __init__(self):
        self.component_events = []
    
    async def get_event_history(self, limit=50):
        return [
            {
                "event_type": "component_status",
                "data": {"component_type": "memory", "state": "initialized"},
                "timestamp": "2024-01-01T12:00:00"
            },
            {
                "event_type": "system_status",
                "data": {"integration_health": "healthy"},
                "timestamp": "2024-01-01T12:01:00"
            }
        ]
    
    def get_integration_metrics(self):
        return {
            "is_active": True,
            "websocket_server_available": True,
            "event_handlers_registered": True,
            "total_events_processed": 42
        }


class TestCodaDashboardServer:
    """Test the dashboard server."""
    
    @pytest.fixture
    def dashboard_server(self):
        """Create dashboard server for testing."""
        return CodaDashboardServer(host="localhost", port=8081)
    
    @pytest.fixture
    def mock_integration_layer(self):
        """Create mock integration layer."""
        return MockIntegrationLayer()
    
    @pytest.fixture
    def mock_websocket_integration(self):
        """Create mock WebSocket integration."""
        return MockWebSocketIntegration()
    
    def test_initialization(self, dashboard_server):
        """Test dashboard server initialization."""
        assert dashboard_server.host == "localhost"
        assert dashboard_server.port == 8081
        assert not dashboard_server.running
        assert dashboard_server.integration_layer is None
        assert dashboard_server.websocket_integration is None
    
    def test_set_integrations(self, dashboard_server, mock_integration_layer, mock_websocket_integration):
        """Test setting integration components."""
        dashboard_server.set_integration_layer(mock_integration_layer)
        dashboard_server.set_websocket_integration(mock_websocket_integration)
        
        assert dashboard_server.integration_layer == mock_integration_layer
        assert dashboard_server.websocket_integration == mock_websocket_integration
    
    @pytest.mark.asyncio
    async def test_start_stop_server(self, dashboard_server):
        """Test starting and stopping the server."""
        # Start server
        await dashboard_server.start()
        assert dashboard_server.running
        
        # Stop server
        await dashboard_server.stop()
        assert not dashboard_server.running
    
    def test_get_dashboard_url(self, dashboard_server):
        """Test getting dashboard URL."""
        url = dashboard_server.get_dashboard_url()
        assert url == "http://localhost:8081"
    
    def test_get_status(self, dashboard_server, mock_integration_layer, mock_websocket_integration):
        """Test getting server status."""
        dashboard_server.set_integration_layer(mock_integration_layer)
        dashboard_server.set_websocket_integration(mock_websocket_integration)
        
        status = dashboard_server.get_status()
        
        assert "running" in status
        assert "host" in status
        assert "port" in status
        assert "dashboard_url" in status
        assert "integrations" in status
        assert status["integrations"]["integration_layer"] is True
        assert status["integrations"]["websocket_integration"] is True
    
    @pytest.mark.asyncio
    async def test_api_status_without_integration(self, dashboard_server):
        """Test status API without integration layer."""
        await dashboard_server.start()
        
        # Create mock request
        mock_request = Mock()
        
        try:
            response = await dashboard_server._api_status(mock_request)
            assert response.status == 503
            
            # Parse response body
            response_data = json.loads(response.text)
            assert "error" in response_data
            assert "Integration layer not available" in response_data["error"]
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_api_status_with_integration(self, dashboard_server, mock_integration_layer):
        """Test status API with integration layer."""
        dashboard_server.set_integration_layer(mock_integration_layer)
        await dashboard_server.start()
        
        # Create mock request
        mock_request = Mock()
        
        try:
            response = await dashboard_server._api_status(mock_request)
            assert response.status == 200
            
            # Parse response body
            response_data = json.loads(response.text)
            assert "total_components" in response_data
            assert "integration_health" in response_data
            assert "dashboard_server" in response_data
            assert response_data["total_components"] == 3
            assert response_data["integration_health"] == "degraded"
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_api_components(self, dashboard_server, mock_integration_layer):
        """Test components API."""
        dashboard_server.set_integration_layer(mock_integration_layer)
        await dashboard_server.start()
        
        # Create mock request
        mock_request = Mock()
        
        try:
            response = await dashboard_server._api_components(mock_request)
            assert response.status == 200
            
            # Parse response body
            response_data = json.loads(response.text)
            assert "components" in response_data
            assert "timestamp" in response_data
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_api_events(self, dashboard_server, mock_websocket_integration):
        """Test events API."""
        dashboard_server.set_websocket_integration(mock_websocket_integration)
        await dashboard_server.start()
        
        # Create mock request with query parameters
        mock_request = Mock()
        mock_request.query = {"limit": "10"}
        
        try:
            response = await dashboard_server._api_events(mock_request)
            assert response.status == 200
            
            # Parse response body
            response_data = json.loads(response.text)
            assert "events" in response_data
            assert "count" in response_data
            assert len(response_data["events"]) == 2
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_api_metrics(self, dashboard_server, mock_integration_layer, mock_websocket_integration):
        """Test metrics API."""
        dashboard_server.set_integration_layer(mock_integration_layer)
        dashboard_server.set_websocket_integration(mock_websocket_integration)
        await dashboard_server.start()
        
        # Create mock request
        mock_request = Mock()
        
        try:
            response = await dashboard_server._api_metrics(mock_request)
            assert response.status == 200
            
            # Parse response body
            response_data = json.loads(response.text)
            assert "dashboard_server" in response_data
            assert "websocket_integration" in response_data
            assert "integration_layer" in response_data
            assert "timestamp" in response_data
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_api_control_refresh(self, dashboard_server):
        """Test control API refresh action."""
        await dashboard_server.start()
        
        # Create mock request
        mock_request = Mock()
        mock_request.match_info = {"action": "refresh"}
        
        try:
            response = await dashboard_server._api_control(mock_request)
            assert response.status == 200
            
            # Parse response body
            response_data = json.loads(response.text)
            assert response_data["action"] == "refresh"
            assert response_data["status"] == "completed"
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_api_control_clear_events(self, dashboard_server, mock_websocket_integration):
        """Test control API clear events action."""
        dashboard_server.set_websocket_integration(mock_websocket_integration)
        await dashboard_server.start()
        
        # Add some events
        mock_websocket_integration.component_events = ["event1", "event2"]
        
        # Create mock request
        mock_request = Mock()
        mock_request.match_info = {"action": "clear_events"}
        
        try:
            response = await dashboard_server._api_control(mock_request)
            assert response.status == 200
            
            # Parse response body
            response_data = json.loads(response.text)
            assert response_data["action"] == "clear_events"
            assert response_data["status"] == "completed"
            
            # Verify events were cleared
            assert len(mock_websocket_integration.component_events) == 0
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_api_control_unknown_action(self, dashboard_server):
        """Test control API with unknown action."""
        await dashboard_server.start()
        
        # Create mock request
        mock_request = Mock()
        mock_request.match_info = {"action": "unknown_action"}
        
        try:
            response = await dashboard_server._api_control(mock_request)
            assert response.status == 400
            
            # Parse response body
            response_data = json.loads(response.text)
            assert "error" in response_data
            assert "Unknown action" in response_data["error"]
            
        finally:
            await dashboard_server.stop()
    
    @pytest.mark.asyncio
    async def test_health_check(self, dashboard_server):
        """Test health check endpoint."""
        # Test when not running
        response = await dashboard_server._health_check(Mock())
        assert response.status == 503
        
        response_data = json.loads(response.text)
        assert response_data["status"] == "unhealthy"
        
        # Test when running
        await dashboard_server.start()
        
        try:
            response = await dashboard_server._health_check(Mock())
            assert response.status == 200
            
            response_data = json.loads(response.text)
            assert response_data["status"] == "healthy"
            assert response_data["dashboard_server"]["running"] is True
            
        finally:
            await dashboard_server.stop()
    
    def test_dashboard_files_exist(self):
        """Test that dashboard files exist."""
        dashboard_dir = Path(__file__).parent.parent.parent / "src" / "coda" / "interfaces" / "dashboard"
        
        index_file = dashboard_dir / "index.html"
        js_file = dashboard_dir / "dashboard.js"
        
        assert index_file.exists(), "Dashboard index.html file should exist"
        assert js_file.exists(), "Dashboard JavaScript file should exist"
