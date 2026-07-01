"""Abstract transport layer for c3po.

Defines the interface that any transport must satisfy to communicate
with an r2d2 server.  The single concrete implementation is
:class:`WebSocketTransport`.
"""

from __future__ import annotations

import abc
from typing import Any


class Transport(abc.ABC):
    """Abstract interface for communicating with an r2d2 server."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the connection to the server."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Close the connection and release resources."""
        ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport is currently connected."""
        ...

    @abc.abstractmethod
    def send_json(self, msg: Any) -> None:
        """Encode *msg* (a protocol dataclass) as JSON and send it
        as a WebSocket text frame."""
        ...

    @abc.abstractmethod
    def send_binary(self, data: bytes) -> None:
        """Send raw *data* as a WebSocket binary frame."""
        ...

    @abc.abstractmethod
    def recv(self, timeout: float | None = None) -> Any:
        """Return the next message from the server.

        Returns a protocol dataclass (e.g. :class:`Observation`) for JSON
        messages, or a :class:`BinaryFrame` for camera data.

        Blocks until a message is available or *timeout* expires.

        Raises
        ------
        RobotDisconnectedError
            If the connection is lost or *timeout* expires without a message.
        """
        ...
