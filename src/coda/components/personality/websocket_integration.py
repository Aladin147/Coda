"""
WebSocket integration for the personality system.

This module provides WebSocketPersonalityManager that extends the base PersonalityManager
with real-time event broadcasting capabilities.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .interfaces import WebSocketPersonalityInterface
from .manager import PersonalityManager
from .models import (
    PersonalityAdjustment,
    PersonalityConfig,
    PersonalityEvent,
)

logger = logging.getLogger("coda.personality.websocket")


class WebSocketPersonalityManager(PersonalityManager, WebSocketPersonalityInterface):
    """
    Personality manager with WebSocket integration for real-time event broadcasting.

    Features:
    - All base PersonalityManager functionality
    - Real-time personality events via WebSocket
    - Personality state broadcasting
    - Adjustment event notifications
    - Topic change notifications
    - Session state updates
    """

    def __init__(
        self, config: Optional[PersonalityConfig] = None, session_id: Optional[str] = None
    ):
        """
        Initialize the WebSocket-enabled personality manager.

        Args:
            config: Configuration for personality system
            session_id: Optional session ID
        """
        super().__init__(config, session_id)
        self.websocket_integration = None

        logger.info("WebSocketPersonalityManager initialized")

    async def set_websocket_integration(self, integration: Any) -> None:
        """
        Set WebSocket integration for event broadcasting.

        Args:
            integration: WebSocket integration instance
        """
        self.websocket_integration = integration
        logger.info("WebSocket integration configured for personality system")

    async def broadcast_personality_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a personality event via WebSocket.

        Args:
            event_type: Type of personality event
            data: Event data
        """
        if self.websocket_integration:
            try:
                # Create personality event
                event = PersonalityEvent(
                    event_type=event_type,
                    data=data,
                    session_id=self.session_manager.session_state.session_id,
                )

                # Broadcast via WebSocket integration
                await self.websocket_integration.server.broadcast_event(
                    f"personality_{event_type}",
                    event.model_dump(),
                    high_priority=True,
                    session_id=event.session_id,
                )

            except Exception as e:
                logger.error(f"Failed to broadcast personality event {event_type}: {e}")

    async def broadcast_personality_state(self) -> None:
        """Broadcast current personality state."""
        state = self.get_personality_state()
        await self.broadcast_personality_event("state_update", state)

    async def broadcast_adjustment(self, adjustment: PersonalityAdjustment) -> None:
        """
        Broadcast a personality adjustment event.

        Args:
            adjustment: Personality adjustment to broadcast
        """
        await self.broadcast_personality_event(
            "trait_adjustment",
            {
                "trait": adjustment.trait_type.value,
                "old_value": adjustment.old_value,
                "new_value": adjustment.new_value,
                "delta": adjustment.delta,
                "reason": adjustment.reason,
                "confidence": adjustment.confidence,
                "timestamp": adjustment.timestamp.isoformat(),
            },
        )

    # Override methods to add WebSocket events

    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input with WebSocket event broadcasting."""
        start_time = time.time()

        results = await super().process_user_input(user_input)

        # Broadcast topic change if detected
        topic_context = results.get("topic_context", {})
        if topic_context.get("current_topic"):
            await self.broadcast_personality_event(
                "topic_detected",
                {
                    "topic": topic_context["current_topic"],
                    "category": topic_context["category"],
                    "confidence": topic_context["confidence"],
                    "keywords": topic_context["keywords"],
                    "processing_time_ms": (time.time() - start_time) * 1000,
                },
            )

        # Broadcast personality adjustments
        adjustments = results.get("personality_adjustments", {})
        for trait_name, adjustment_data in adjustments.items():
            await self.broadcast_personality_event(
                "trait_adjustment",
                {
                    "trait": trait_name,
                    "adjustment": adjustment_data,
                    "trigger": "user_input_processing",
                },
            )

        # Broadcast session state changes
        session_state = results.get("session_state", {})
        if session_state.get("in_closure_mode"):
            await self.broadcast_personality_event(
                "session_closure",
                {
                    "reason": session_state.get("closure_reason"),
                    "duration_minutes": session_state.get("duration_seconds", 0) / 60,
                    "turn_count": session_state.get("turn_count", 0),
                },
            )

        return results

    async def process_assistant_response(self, response: str) -> Dict[str, Any]:
        """Process assistant response with WebSocket event broadcasting."""
        results = await super().process_assistant_response(response)

        # Broadcast behavior pattern detection
        behavior_patterns = results.get("behavior_patterns", [])
        if behavior_patterns:
            await self.broadcast_personality_event(
                "behavior_patterns_detected",
                {
                    "patterns_count": len(behavior_patterns),
                    "patterns": behavior_patterns,
                    "response_length": results.get("response_length", 0),
                },
            )

        # Broadcast lore updates
        lore_updates = results.get("lore_updates", {})
        if lore_updates.get("new_triggers_added", 0) > 0:
            await self.broadcast_personality_event("lore_updated", lore_updates)

        return results

    async def process_feedback(
        self, feedback: str, feedback_type: str = "general"
    ) -> Dict[str, Any]:
        """Process user feedback with WebSocket event broadcasting."""
        results = await super().process_feedback(feedback, feedback_type)

        # Broadcast feedback processing
        await self.broadcast_personality_event(
            "feedback_processed",
            {
                "feedback_type": feedback_type,
                "feedback_length": len(feedback),
                "adjustments_applied": len(results.get("personality_adjustments", {})),
                "learning_updated": results.get("learning_updated", False),
            },
        )

        # Broadcast specific adjustments
        adjustments = results.get("personality_adjustments", {})
        for trait_name, adjustment_data in adjustments.items():
            await self.broadcast_personality_event(
                "feedback_adjustment",
                {
                    "trait": trait_name,
                    "adjustment": adjustment_data,
                    "feedback_type": feedback_type,
                },
            )

        return results

    async def apply_learned_adjustments(self) -> Dict[str, Any]:
        """Apply learned adjustments with WebSocket event broadcasting."""
        results = await super().apply_learned_adjustments()

        # Broadcast learning application
        await self.broadcast_personality_event(
            "learning_applied",
            {
                "adjustments_count": results.get("adjustments_applied", 0),
                "adjustments": results.get("adjustments", {}),
            },
        )

        return results

    # Additional WebSocket-specific methods

    async def broadcast_personality_analytics(self) -> None:
        """Broadcast comprehensive personality analytics."""
        analytics = self.get_analytics()
        await self.broadcast_personality_event("analytics_update", analytics)

    async def broadcast_trait_evolution(self, trait_name: str, days: int = 7) -> None:
        """Broadcast trait evolution data for visualization."""
        # Get trait adjustment history for the specified trait
        trait_adjustments = [
            adj for adj in self.adjustment_history if adj.trait_type.value == trait_name
        ]

        # Create evolution data
        evolution_data = []
        current_value = self.parameters.get_trait_value(
            getattr(self.parameters.PersonalityTraitType, trait_name.upper())
        )

        for adj in trait_adjustments[-20:]:  # Last 20 adjustments
            evolution_data.append(
                {
                    "timestamp": adj.timestamp.isoformat(),
                    "value": adj.new_value,
                    "delta": adj.delta,
                    "reason": adj.reason,
                    "confidence": adj.confidence,
                }
            )

        await self.broadcast_personality_event(
            "trait_evolution",
            {
                "trait": trait_name,
                "current_value": current_value,
                "evolution_data": evolution_data,
                "days": days,
            },
        )

    async def broadcast_session_summary(self) -> None:
        """Broadcast session summary when session ends."""
        summary = self.session_manager.generate_session_summary()
        await self.broadcast_personality_event("session_summary", summary)

    async def broadcast_behavior_insights(self) -> None:
        """Broadcast behavioral learning insights."""
        behavior_profile = self.behavioral_conditioner.get_behavior_profile()
        learning_stats = self.behavioral_conditioner.get_learning_stats()

        insights = {
            "learned_preferences": {
                k.value: v for k, v in behavior_profile.user_preferences.items()
            },
            "confidence_scores": {
                k.value: v for k, v in behavior_profile.confidence_scores.items()
            },
            "learning_stats": learning_stats,
            "detected_patterns": [
                p.model_dump() for p in behavior_profile.detected_patterns[-5:]
            ],  # Last 5 patterns
        }

        await self.broadcast_personality_event("behavior_insights", insights)

    async def broadcast_lore_usage(self) -> None:
        """Broadcast personal lore usage statistics."""
        lore_stats = self.personal_lore.get_lore_usage_stats()
        await self.broadcast_personality_event("lore_usage", lore_stats)

    async def broadcast_topic_distribution(self) -> None:
        """Broadcast topic distribution for the session."""
        topic_stats = self.topic_awareness.get_topic_stats()
        await self.broadcast_personality_event("topic_distribution", topic_stats)

    def reset_session(self, new_session_id: str = None) -> None:
        """Reset session with WebSocket notification."""
        old_session_id = self.session_manager.session_state.session_id
        super().reset_session(new_session_id)

        # Broadcast session reset
        if self.websocket_integration:
            import asyncio

            asyncio.create_task(
                self.broadcast_personality_event(
                    "session_reset",
                    {
                        "old_session_id": old_session_id,
                        "new_session_id": self.session_manager.session_state.session_id,
                    },
                )
            )

    async def start_personality_monitoring(self, interval_seconds: int = 30) -> None:
        """Start periodic personality state broadcasting."""
        import asyncio

        async def monitor_loop():
            while True:
                try:
                    await self.broadcast_personality_state()
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in personality monitoring loop: {e}")
                    await asyncio.sleep(interval_seconds)

        # Start monitoring task
        asyncio.create_task(monitor_loop())
        logger.info(f"Started personality monitoring with {interval_seconds}s interval")

    async def trigger_personality_snapshot(self) -> Dict[str, Any]:
        """Trigger a complete personality snapshot for debugging."""
        snapshot = {
            "timestamp": time.time(),
            "personality_state": self.get_personality_state(),
            "analytics": self.get_analytics(),
            "session_analytics": self.session_manager.get_session_analytics(),
            "behavior_profile": self.behavioral_conditioner.get_behavior_profile().model_dump(),
            "topic_context": self.topic_awareness.get_current_topic().model_dump(),
            "lore_stats": self.personal_lore.get_lore_usage_stats(),
        }

        await self.broadcast_personality_event("personality_snapshot", snapshot)
        return snapshot
