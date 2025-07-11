"""
Advanced Connection Pooling System for Coda.

Provides intelligent connection pooling with health monitoring,
automatic scaling, and performance optimization.
"""

import asyncio
import logging
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states."""

    IDLE = "idle"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool."""

    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: float = 30.0
    health_check_interval: float = 60.0
    max_idle_time: float = 300.0


@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""

    connection_id: str
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    health_score: float = 1.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=10))
    error_count: int = 0


class PooledConnection:
    """Wrapper for pooled connections with health monitoring."""

    def __init__(self, connection_id: str, session: aiohttp.ClientSession):
        self.info = ConnectionInfo(
            connection_id=connection_id, created_at=datetime.now(), last_used=datetime.now()
        )
        self.session = session
        self._pool_ref: Optional[weakref.ref] = None

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a request using this connection."""
        start_time = time.time()

        try:
            self.info.state = ConnectionState.ACTIVE
            self.info.use_count += 1
            self.info.last_used = datetime.now()

            response = await self.session.request(method, url, **kwargs)

            # Record response time
            response_time = (time.time() - start_time) * 1000  # ms
            self.info.response_times.append(response_time)

            # Update health score based on response
            if response.status < 400:
                self.info.health_score = min(1.0, self.info.health_score + 0.1)
            else:
                self.info.health_score = max(0.0, self.info.health_score - 0.2)
                self.info.error_count += 1

            self.info.state = ConnectionState.IDLE
            return response

        except Exception as e:
            # Handle connection errors
            self.info.error_count += 1
            self.info.health_score = max(0.0, self.info.health_score - 0.3)
            self.info.state = ConnectionState.UNHEALTHY

            logger.warning(f"Connection {self.info.connection_id} error: {e}")
            raise

    async def health_check(self, health_url: str) -> bool:
        """Perform health check on this connection."""
        try:
            async with self.session.get(
                health_url, timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                is_healthy = response.status == 200

                if is_healthy:
                    self.info.health_score = min(1.0, self.info.health_score + 0.05)
                    self.info.state = ConnectionState.IDLE
                else:
                    self.info.health_score = max(0.0, self.info.health_score - 0.1)
                    self.info.state = ConnectionState.UNHEALTHY

                return is_healthy

        except Exception as e:
            self.info.health_score = max(0.0, self.info.health_score - 0.2)
            self.info.state = ConnectionState.UNHEALTHY
            logger.debug(f"Health check failed for connection {self.info.connection_id}: {e}")
            return False

    async def close(self):
        """Close this connection."""
        self.info.state = ConnectionState.CLOSED
        if not self.session.closed:
            await self.session.close()

    def get_avg_response_time(self) -> float:
        """Get average response time for this connection."""
        if not self.info.response_times:
            return 0.0
        return sum(self.info.response_times) / len(self.info.response_times)


class ConnectionPool:
    """
    Advanced connection pool with health monitoring and auto-scaling.

    Features:
    - Automatic connection health monitoring
    - Dynamic pool sizing based on load
    - Connection reuse optimization
    - Graceful degradation on failures
    """

    def __init__(self, service_name: str, config: "ConnectionPoolConfig"):
        self.service_name = service_name
        self.config = config

        # Connection management
        self.connections: Dict[str, PooledConnection] = {}
        self.idle_connections: deque = deque()
        self.active_connections: int = 0
        self.max_connections = config.max_connections

        # Health monitoring
        self.health_check_url: Optional[str] = None
        self.last_health_check = datetime.now()

        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0

        # Pool state
        self.running = False
        self.health_check_task: Optional[asyncio.Task] = None

        logger.info(f"ConnectionPool created for service: {service_name}")

    async def initialize(self):
        """Initialize the connection pool."""
        logger.info(f"Initializing connection pool for {self.service_name}")

        # Create initial connections
        for i in range(self.config.min_connections):
            await self._create_connection()

        # Start health monitoring
        self.running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Connection pool initialized with {len(self.connections)} connections")

    async def get_connection(self) -> PooledConnection:
        """Get a connection from the pool."""
        # Try to get an idle connection
        while self.idle_connections:
            connection = self.idle_connections.popleft()

            # Check if connection is still healthy
            if connection.info.state == ConnectionState.IDLE and connection.info.health_score > 0.5:
                self.active_connections += 1
                return connection
            else:
                # Remove unhealthy connection
                await self._remove_connection(connection)

        # No idle connections available, create new one if possible
        if len(self.connections) < self.max_connections:
            connection = await self._create_connection()
            if connection:
                self.active_connections += 1
                return connection

        # Pool is full, wait for a connection to become available
        return await self._wait_for_connection()

    async def return_connection(self, connection: PooledConnection):
        """Return a connection to the pool."""
        if connection.info.connection_id in self.connections:
            self.active_connections = max(0, self.active_connections - 1)

            # Check if connection is still healthy
            if (
                connection.info.state != ConnectionState.UNHEALTHY
                and connection.info.health_score > 0.3
            ):
                connection.info.state = ConnectionState.IDLE
                self.idle_connections.append(connection)
            else:
                # Remove unhealthy connection
                await self._remove_connection(connection)

    async def _create_connection(self) -> Optional[PooledConnection]:
        """Create a new connection."""
        try:
            connection_id = f"{self.service_name}_{len(self.connections)}"

            # Create aiohttp session with optimized settings
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout, connect=10.0)

            connector = aiohttp.TCPConnector(
                limit=100, limit_per_host=30, keepalive_timeout=300, enable_cleanup_closed=True
            )

            session = aiohttp.ClientSession(timeout=timeout, connector=connector)

            connection = PooledConnection(connection_id, session)
            connection._pool_ref = weakref.ref(self)

            self.connections[connection_id] = connection
            self.idle_connections.append(connection)

            logger.debug(f"Created new connection: {connection_id}")
            return connection

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return None

    async def _remove_connection(self, connection: PooledConnection):
        """Remove a connection from the pool."""
        connection_id = connection.info.connection_id

        if connection_id in self.connections:
            del self.connections[connection_id]

            # Remove from idle queue if present
            try:
                self.idle_connections.remove(connection)
            except ValueError:
                pass  # Connection not in idle queue

            await connection.close()
            logger.debug(f"Removed connection: {connection_id}")

    async def _wait_for_connection(self) -> PooledConnection:
        """Wait for a connection to become available."""
        start_time = time.time()

        while time.time() - start_time < self.config.connection_timeout:
            if self.idle_connections:
                return await self.get_connection()

            await asyncio.sleep(0.1)  # Brief wait

        raise TimeoutError(f"No connections available in pool for {self.service_name}")

    async def _health_check_loop(self):
        """Background health check loop."""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self):
        """Perform health checks on all connections."""
        if not self.health_check_url:
            return

        unhealthy_connections = []

        for connection in self.connections.values():
            if connection.info.state == ConnectionState.IDLE:
                is_healthy = await connection.health_check(self.health_check_url)

                if not is_healthy:
                    unhealthy_connections.append(connection)

        # Remove unhealthy connections
        for connection in unhealthy_connections:
            await self._remove_connection(connection)

        # Ensure minimum connections
        while len(self.connections) < self.config.min_connections:
            await self._create_connection()

        self.last_health_check = datetime.now()

    async def optimize(self):
        """Optimize pool performance."""
        # Remove old idle connections
        current_time = datetime.now()
        idle_timeout = timedelta(seconds=self.config.idle_timeout)

        connections_to_remove = []
        for connection in list(self.idle_connections):
            if current_time - connection.info.last_used > idle_timeout:
                connections_to_remove.append(connection)

        for connection in connections_to_remove:
            await self._remove_connection(connection)

        # Ensure minimum connections
        while len(self.connections) < self.config.min_connections:
            await self._create_connection()

    async def reduce_queue_size(self):
        """Reduce queue size for CPU optimization."""
        # This could implement queue size reduction logic

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_requests = self.total_requests
        success_rate = (self.successful_requests / total_requests) if total_requests > 0 else 1.0

        return {
            "service_name": self.service_name,
            "total_connections": len(self.connections),
            "idle_connections": len(self.idle_connections),
            "active_connections": self.active_connections,
            "max_connections": self.max_connections,
            "total_requests": total_requests,
            "success_rate": success_rate,
            "avg_response_time_ms": self.avg_response_time,
            "last_health_check": self.last_health_check.isoformat(),
        }

    async def close(self):
        """Close the connection pool."""
        logger.info(f"Closing connection pool for {self.service_name}")

        self.running = False

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for connection in list(self.connections.values()):
            await connection.close()

        self.connections.clear()
        self.idle_connections.clear()

        logger.info(f"Connection pool closed for {self.service_name}")
