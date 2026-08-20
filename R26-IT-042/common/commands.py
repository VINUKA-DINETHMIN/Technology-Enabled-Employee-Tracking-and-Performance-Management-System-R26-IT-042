"""
R26-IT-042 — Employee Activity Monitoring System
common/commands.py

Handles remote command polling for the employee app.
"""

from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

from common.database import MongoDBClient

logger = logging.getLogger(__name__)

class CommandPoller:
    """
    Polls the 'commands' collection for instructions from the admin.
    """

    def __init__(
        self,
        user_id: str,
        db_client: MongoDBClient,
        shutdown_event: threading.Event,
        interval_sec: int = 15
    ) -> None:
        self._user_id = user_id
        self._db = db_client
        self._shutdown = shutdown_event
        self._interval = interval_sec
        self._handlers: dict[str, Callable] = {}

    def register_handler(self, command_type: str, handler: Callable) -> None:
        """Register a function to handle a specific command type."""
        self._handlers[command_type] = handler

    def start(self) -> None:
        """Start the polling loop in a background thread."""
        logger.info("Command poller started for user: %s", self._user_id)
        while not self._shutdown.is_set():
            try:
                self._poll()
            except Exception as exc:
                logger.error("Command polling error: %s", exc)
            
            # Sleep in small increments to remain responsive to shutdown
            for _ in range(self._interval):
                if self._shutdown.is_set():
                    break
                time.sleep(1)

    def _poll(self) -> None:
        if not self._db or not self._db.is_connected:
            return

        col = self._db.get_collection("commands")
        if col is None:
            return

        now = datetime.utcnow().isoformat()
        
        # Find pending commands for this user that haven't expired
        query = {
            "target_user_id": self._user_id,
            "status": "pending",
            "expires_at": {"$gt": now}
        }
        
        commands = list(col.find(query))
        for cmd in commands:
            cmd_id = cmd.get("command_id")
            cmd_type = cmd.get("command_type")
            
            logger.info("Received command: %s (ID: %s)", cmd_type, cmd_id)
            
            handler = self._handlers.get(cmd_type)
            if handler:
                try:
                    # Mark as processing
                    col.update_one({"command_id": cmd_id}, {"$set": {"status": "processing", "started_at": now}})
                    
                    # Execute
                    handler(cmd)
                    
                    # Mark as completed
                    col.update_one({"command_id": cmd_id}, {"$set": {"status": "completed", "completed_at": datetime.utcnow().isoformat()}})
                except Exception as exc:
                    logger.error("Failed to execute command %s: %s", cmd_id, exc)
                    col.update_one({"command_id": cmd_id}, {"$set": {"status": "failed", "error": str(exc)}})
            else:
                logger.warning("No handler for command type: %s", cmd_type)
                col.update_one({"command_id": cmd_id}, {"$set": {"status": "ignored", "reason": "unsupported_command_type"}})
