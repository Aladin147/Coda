"""
Parallel Voice Processing Pipeline

This module implements a parallel processing pipeline for voice conversations,
enabling concurrent processing of multiple voice requests for optimal performance.
"""

import asyncio
import logging
import multiprocessing as mp
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .context_integration import ContextConfig
from .llm_integration import VoiceLLMConfig
from .mode_manager import ProcessingModeManager, ProcessingModeType
from .models import ConversationState, VoiceMessage, VoiceResponse
from .moshi_client import MoshiConfig
from .performance_optimizer import OptimizationConfig

logger = logging.getLogger("coda.voice.parallel_processor")


class ProcessingPriority(str, Enum):
    """Processing priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ProcessingStatus(str, Enum):
    """Processing status for requests."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingRequest:
    """A voice processing request."""

    request_id: str
    voice_message: VoiceMessage
    conversation_state: Optional[ConversationState]
    requested_mode: Optional[ProcessingModeType]
    priority: ProcessingPriority
    timeout_seconds: float
    callback: Optional[Callable[[VoiceResponse], None]]

    # Tracking
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: ProcessingStatus = ProcessingStatus.QUEUED

    def __lt__(self, other):
        """Enable comparison for priority queue."""
        if not isinstance(other, ProcessingRequest):
            return NotImplemented
        # Compare by creation time if same priority
        return self.created_at < other.created_at

    @property
    def age_seconds(self) -> float:
        """Get age of request in seconds."""
        return time.time() - self.created_at

    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing."""

    # Worker configuration
    max_workers: int = 4
    worker_type: str = "thread"  # "thread" or "process"

    # Queue configuration
    max_queue_size: int = 100
    priority_queue_enabled: bool = True

    # Performance settings
    batch_processing_enabled: bool = False
    batch_size: int = 3
    batch_timeout_seconds: float = 0.1

    # Resource management
    memory_limit_mb: int = 2048
    cpu_limit_percent: float = 80.0

    # Timeout settings
    default_timeout_seconds: float = 10.0
    max_timeout_seconds: float = 30.0

    # Load balancing
    enable_load_balancing: bool = True
    worker_health_check_interval: float = 5.0


class WorkerPool:
    """Manages a pool of workers for parallel processing."""

    def __init__(self, config: ParallelProcessingConfig):
        self.config = config
        self.workers: Dict[str, Any] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.is_running = False

    async def initialize(self) -> None:
        """Initialize the worker pool."""
        try:
            if self.config.worker_type == "process":
                self.executor = ProcessPoolExecutor(
                    max_workers=self.config.max_workers, mp_context=mp.get_context("spawn")
                )
            else:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.config.max_workers, thread_name_prefix="voice_worker"
                )

            # Initialize worker stats
            for i in range(self.config.max_workers):
                worker_id = f"worker_{i}"
                self.worker_stats[worker_id] = {
                    "requests_processed": 0,
                    "total_processing_time": 0.0,
                    "last_activity": time.time(),
                    "status": "idle",
                }

            self.is_running = True
            logger.info(
                f"WorkerPool initialized with {self.config.max_workers} {self.config.worker_type} workers"
            )

        except Exception as e:
            logger.error(f"Failed to initialize worker pool: {e}")
            raise

    async def submit_request(
        self, request: ProcessingRequest, processor_func: Callable
    ) -> asyncio.Future:
        """Submit a processing request to the worker pool."""

        if not self.executor:
            raise RuntimeError("Worker pool not initialized")

        # Wrap the processing function for async execution
        def wrapped_processor():
            try:
                # This would be the actual processing logic
                # For now, we'll simulate processing
                return self._simulate_processing(request)
            except Exception as e:
                logger.error(f"Worker processing failed: {e}")
                raise

        # Submit to executor
        future = asyncio.get_running_loop().run_in_executor(self.executor, wrapped_processor)

        return future

    def _simulate_processing(self, request: ProcessingRequest) -> VoiceResponse:
        """Simulate voice processing (placeholder for actual processing)."""

        # Simulate processing time based on priority
        processing_time = {
            ProcessingPriority.URGENT: 0.1,
            ProcessingPriority.HIGH: 0.2,
            ProcessingPriority.NORMAL: 0.3,
            ProcessingPriority.LOW: 0.5,
        }.get(request.priority, 0.3)

        time.sleep(processing_time)

        # Create mock response
        return VoiceResponse(
            response_id=f"parallel_{request.request_id}",
            conversation_id=request.voice_message.conversation_id,
            message_id=request.voice_message.message_id,
            text_content=f"Parallel processed response for: {request.voice_message.text_content[:50]}...",
            audio_data=b"",
            processing_mode=request.voice_message.processing_mode,
            total_latency_ms=processing_time * 1000,
            response_relevance=0.8,
        )

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "total_workers": len(self.worker_stats),
            "worker_type": self.config.worker_type,
            "is_running": self.is_running,
            "worker_details": self.worker_stats.copy(),
        }

    async def cleanup(self) -> None:
        """Clean up the worker pool."""
        try:
            self.is_running = False

            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None

            self.workers.clear()
            self.worker_stats.clear()

            logger.info("WorkerPool cleanup completed")

        except Exception as e:
            logger.error(f"Worker pool cleanup failed: {e}")


class ParallelVoiceProcessor:
    """
    Parallel voice processing pipeline for optimal performance.

    Features:
    - Concurrent request processing
    - Priority - based queue management
    - Load balancing across workers
    - Batch processing optimization
    - Resource monitoring and limits
    """

    def __init__(
        self,
        config: ParallelProcessingConfig,
        moshi_config: MoshiConfig,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig,
    ):
        """Initialize the parallel voice processor."""
        self.config = config
        self.mode_manager: Optional[ProcessingModeManager] = None
        self.worker_pool: Optional[WorkerPool] = None

        # Store configurations for mode manager
        self.moshi_config = moshi_config
        self.context_config = context_config
        self.llm_config = llm_config
        self.optimization_config = optimization_config

        # Request management
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=config.max_queue_size
        )
        self.active_requests: Dict[str, ProcessingRequest] = {}
        self.completed_requests: Dict[str, ProcessingRequest] = {}

        # Performance tracking
        self.processing_stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "average_latency_ms": 0.0,
            "queue_depth": 0,
            "active_workers": 0,
        }

        # Background tasks
        self.processor_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None

        logger.info("ParallelVoiceProcessor initialized")

    async def initialize(self) -> None:
        """Initialize the parallel processor."""
        try:
            # Initialize mode manager
            self.mode_manager = ProcessingModeManager(
                self.moshi_config, self.context_config, self.llm_config, self.optimization_config
            )
            await self.mode_manager.initialize()

            # Initialize worker pool
            self.worker_pool = WorkerPool(self.config)
            await self.worker_pool.initialize()

            # Start background processing
            self.processor_task = asyncio.create_task(self._process_requests())
            self.monitor_task = asyncio.create_task(self._monitor_performance())

            logger.info("ParallelVoiceProcessor initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize ParallelVoiceProcessor: {e}")
            raise

    async def submit_request(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None,
        requested_mode: Optional[ProcessingModeType] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
        callback: Optional[Callable[[VoiceResponse], None]] = None,
    ) -> str:
        """
        Submit a voice processing request.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state
            requested_mode: Specific processing mode to use
            priority: Processing priority
            timeout_seconds: Request timeout
            callback: Optional callback for response

        Returns:
            Request ID for tracking
        """

        request_id = str(uuid.uuid4())

        # Create processing request
        request = ProcessingRequest(
            request_id=request_id,
            voice_message=voice_message,
            conversation_state=conversation_state,
            requested_mode=requested_mode,
            priority=priority,
            timeout_seconds=timeout_seconds or self.config.default_timeout_seconds,
            callback=callback,
            created_at=time.time(),
        )

        try:
            # Add to priority queue
            priority_value = {
                ProcessingPriority.URGENT: 0,
                ProcessingPriority.HIGH: 1,
                ProcessingPriority.NORMAL: 2,
                ProcessingPriority.LOW: 3,
            }.get(priority, 2)

            await self.request_queue.put((priority_value, request))

            # Track request
            self.active_requests[request_id] = request
            self.processing_stats["total_requests"] += 1
            self.processing_stats["queue_depth"] = self.request_queue.qsize()

            logger.debug(f"Submitted request {request_id} with priority {priority}")

            return request_id

        except asyncio.QueueFull:
            logger.warning(f"Request queue full, rejecting request {request_id}")
            raise RuntimeError("Processing queue is full")

    async def get_response(self, request_id: str, timeout: Optional[float] = None) -> VoiceResponse:
        """
        Get response for a submitted request.

        Args:
            request_id: The request ID
            timeout: Timeout for waiting

        Returns:
            Voice response
        """

        timeout = timeout or self.config.default_timeout_seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if request is completed
            if request_id in self.completed_requests:
                request = self.completed_requests[request_id]

                if request.status == ProcessingStatus.COMPLETED:
                    # Return the response (stored in request metadata)
                    return getattr(request, "response", None)
                elif request.status == ProcessingStatus.FAILED:
                    raise RuntimeError(f"Request {request_id} failed")
                elif request.status == ProcessingStatus.CANCELLED:
                    raise RuntimeError(f"Request {request_id} was cancelled")

            # Wait a bit before checking again
            await asyncio.sleep(0.01)

        # Timeout
        if request_id in self.active_requests:
            await self.cancel_request(request_id)

        raise asyncio.TimeoutError(f"Request {request_id} timed out")

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or active request."""

        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            request.status = ProcessingStatus.CANCELLED
            request.completed_at = time.time()

            # Move to completed requests
            self.completed_requests[request_id] = request
            del self.active_requests[request_id]

            logger.info(f"Cancelled request {request_id}")
            return True

        return False

    async def _process_requests(self) -> None:
        """Background task to process requests from the queue."""

        while True:
            try:
                # Get request from queue with timeout
                try:
                    priority, request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Check if request is still valid
                if request.request_id not in self.active_requests:
                    continue

                # Check timeout
                if request.age_seconds > request.timeout_seconds:
                    await self.cancel_request(request.request_id)
                    continue

                # Start processing
                request.status = ProcessingStatus.PROCESSING
                request.started_at = time.time()

                try:
                    # Process the request
                    response = await self._process_single_request(request)

                    # Mark as completed
                    request.status = ProcessingStatus.COMPLETED
                    request.completed_at = time.time()
                    setattr(request, "response", response)

                    # Call callback if provided
                    if request.callback:
                        try:
                            request.callback(response)
                        except Exception as e:
                            logger.warning(f"Callback failed for request {request.request_id}: {e}")

                    # Update stats
                    self.processing_stats["completed_requests"] += 1

                except Exception as e:
                    logger.error(f"Processing failed for request {request.request_id}: {e}")
                    request.status = ProcessingStatus.FAILED
                    request.completed_at = time.time()
                    self.processing_stats["failed_requests"] += 1

                # Move to completed requests
                self.completed_requests[request.request_id] = request
                del self.active_requests[request.request_id]

                # Update queue depth
                self.processing_stats["queue_depth"] = self.request_queue.qsize()

            except Exception as e:
                logger.error(f"Request processing loop error: {e}")
                await asyncio.sleep(0.1)

    async def _process_single_request(self, request: ProcessingRequest) -> VoiceResponse:
        """Process a single voice request."""

        if not self.mode_manager:
            raise RuntimeError("Mode manager not initialized")

        # Process using mode manager
        response = await self.mode_manager.process_voice_message(
            request.voice_message, request.conversation_state, request.requested_mode
        )

        return response

    async def _monitor_performance(self) -> None:
        """Background task to monitor performance."""

        while True:
            try:
                # Update performance stats
                active_count = len(self.active_requests)
                completed_count = self.processing_stats["completed_requests"]

                if completed_count > 0:
                    # Calculate average latency from completed requests
                    total_latency = 0.0
                    latency_count = 0

                    for request in list(self.completed_requests.values())[-100:]:  # Last 100
                        if request.processing_time_seconds:
                            total_latency += request.processing_time_seconds * 1000
                            latency_count += 1

                    if latency_count > 0:
                        self.processing_stats["average_latency_ms"] = total_latency / latency_count

                self.processing_stats["active_workers"] = active_count

                # Clean up old completed requests
                current_time = time.time()
                old_requests = [
                    req_id
                    for req_id, req in self.completed_requests.items()
                    if current_time - req.completed_at > 300  # 5 minutes
                ]

                for req_id in old_requests:
                    del self.completed_requests[req_id]

                await asyncio.sleep(self.config.worker_health_check_interval)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(1.0)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""

        stats = self.processing_stats.copy()
        stats.update(
            {
                "queue_size": self.request_queue.qsize(),
                "active_requests": len(self.active_requests),
                "completed_requests_cached": len(self.completed_requests),
                "worker_stats": self.worker_pool.get_worker_stats() if self.worker_pool else {},
            }
        )

        return stats

    async def cleanup(self) -> None:
        """Clean up the parallel processor."""

        try:
            # Cancel background tasks
            if self.processor_task:
                self.processor_task.cancel()
                try:
                    await self.processor_task
                except asyncio.CancelledError:
                    pass

            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            # Cancel all active requests
            for request_id in list(self.active_requests.keys()):
                await self.cancel_request(request_id)

            # Clean up components
            if self.worker_pool:
                await self.worker_pool.cleanup()

            if self.mode_manager:
                await self.mode_manager.cleanup()

            # Clear data structures
            self.active_requests.clear()
            self.completed_requests.clear()

            logger.info("ParallelVoiceProcessor cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
