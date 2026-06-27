"""Abstract transport layer for c3po.

The transport layer abstracts how c3po communicates with r2d2.
A single concrete implementation (WebSocket via rosbridge) is provided.
"""

from __future__ import annotations

import abc
from typing import Any


class Transport(abc.ABC):
    """Abstract interface for communication with an r2d2 server."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the connection to the server."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Close the connection and clean up."""
        ...

    @abc.abstractmethod
    def call_service(
        self, service: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a ROS 2 service and return the response dict."""
        ...

    @abc.abstractmethod
    def subscribe(self, topic: str, msg_type: str, callback: Any) -> None:
        """Subscribe to a ROS 2 topic.  *callback* receives parsed message dicts."""
        ...

    @abc.abstractmethod
    def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic."""
        ...

    @abc.abstractmethod
    def publish(self, topic: str, msg_type: str, msg: dict[str, Any]) -> None:
        """Publish a message to a ROS 2 topic."""
        ...
