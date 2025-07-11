"""
Session Management System for Coda.

This module provides comprehensive session management capabilities including
session creation, tracking, persistence, and memory context management.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("coda.session_manager")


class SessionManager:
    """
    Manages conversation sessions with persistence and memory context.

    Features:
    - Session creation and lifecycle management
    - Persistent session storage
    - Memory context per session
    - Session metadata tracking
    - Cleanup of expired sessions
    """

    def __init__(self, storage_path: Optional[Path] = None, session_timeout_hours: int = 24):
        """
        Initialize session manager.

        Args:
            storage_path: Path to store session data (default: data/sessions)
            session_timeout_hours: Hours after which inactive sessions expire
        """
        self.storage_path = storage_path or Path("data/sessions")
        self.session_timeout = timedelta(hours=session_timeout_hours)

        # In-memory session storage
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, bool] = {}

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"SessionManager initialized with storage at {self.storage_path}")

    async def initialize(self):
        """Initialize session manager and load existing sessions."""
        await self._load_sessions()
        await self._cleanup_expired_sessions()
        logger.info(f"SessionManager initialized with {len(self.sessions)} sessions")

    async def create_session(
        self, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation session.

        Args:
            session_id: Optional custom session ID
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create session data
        session_data = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0,
            "status": "active",
            "metadata": metadata or {},
            "memory_context": {},
            "conversation_history": [],
        }

        # Store in memory and mark as active
        self.sessions[session_id] = session_data
        self.active_sessions[session_id] = True

        # Persist to storage
        await self._save_session(session_id)

        logger.info(f"Created session: {session_id}")
        return session_id

    async def end_session(self, session_id: str) -> bool:
        """
        End a conversation session.

        Args:
            session_id: Session to end

        Returns:
            True if session was ended, False if not found
        """
        if session_id not in self.sessions:
            return False

        # Update session status
        self.sessions[session_id]["status"] = "ended"
        self.sessions[session_id]["ended_at"] = datetime.now().isoformat()
        self.active_sessions[session_id] = False

        # Persist changes
        await self._save_session(session_id)

        logger.info(f"Ended session: {session_id}")
        return True

    async def update_session_activity(self, session_id: str) -> bool:
        """
        Update session last activity timestamp.

        Args:
            session_id: Session to update

        Returns:
            True if updated, False if session not found
        """
        if session_id not in self.sessions:
            return False

        self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
        await self._save_session(session_id)
        return True

    async def add_message_to_session(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to session history.

        Args:
            session_id: Target session
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional message metadata

        Returns:
            True if added, False if session not found
        """
        if session_id not in self.sessions:
            return False

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.sessions[session_id]["conversation_history"].append(message)
        self.sessions[session_id]["message_count"] += 1
        await self.update_session_activity(session_id)

        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID."""
        return self.sessions.get(session_id)

    def list_sessions(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        List all sessions.

        Args:
            active_only: If True, only return active sessions

        Returns:
            List of session data
        """
        if active_only:
            return [
                session_data
                for session_id, session_data in self.sessions.items()
                if self.active_sessions.get(session_id, False)
            ]
        return list(self.sessions.values())

    def get_session_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Target session
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        if session_id not in self.sessions:
            return []

        history = self.sessions[session_id]["conversation_history"]
        if limit:
            return history[-limit:]
        return history

    async def _save_session(self, session_id: str):
        """Save session data to persistent storage."""
        if session_id not in self.sessions:
            return

        session_file = self.storage_path / f"{session_id}.json"
        try:
            with open(session_file, "w") as f:
                json.dump(self.sessions[session_id], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

    async def _load_sessions(self):
        """Load sessions from persistent storage."""
        try:
            for session_file in self.storage_path.glob("*.json"):
                session_id = session_file.stem
                try:
                    with open(session_file, "r") as f:
                        session_data = json.load(f)

                    self.sessions[session_id] = session_data
                    # Mark as active if status is active and not expired
                    is_active = session_data.get(
                        "status"
                    ) == "active" and not self._is_session_expired(session_data)
                    self.active_sessions[session_id] = is_active

                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        expired_sessions = []

        for session_id, session_data in self.sessions.items():
            if self._is_session_expired(session_data):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            if self.active_sessions.get(session_id, False):
                await self.end_session(session_id)
                logger.info(f"Expired session: {session_id}")

    def _is_session_expired(self, session_data: Dict[str, Any]) -> bool:
        """Check if a session has expired."""
        try:
            last_activity = datetime.fromisoformat(session_data["last_activity"])
            return datetime.now() - last_activity > self.session_timeout
        except (KeyError, ValueError):
            return True  # Consider invalid sessions as expired
