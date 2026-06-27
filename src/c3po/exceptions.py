"""Exceptions raised by the c3po client library."""

from __future__ import annotations


class RobotNotFoundError(RuntimeError):
    """Raised when no /describe service responds within the timeout."""


class RobotDisconnectedError(RuntimeError):
    """Raised when the WebSocket connection to the robot station is lost."""


class IncompatibleVersionError(RuntimeError):
    """Raised when the r2d2 server reports an incompatible API version."""
