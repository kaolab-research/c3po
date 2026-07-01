"""Exceptions raised by the c3po client library."""

from __future__ import annotations


class RobotNotFoundError(RuntimeError):
    """Raised when the robot station cannot be discovered."""


class RobotDisconnectedError(RuntimeError):
    """Raised when the connection to the robot station is lost."""


class IncompatibleVersionError(RuntimeError):
    """Raised when the r2d2 server reports an incompatible API version."""


class ProtocolError(RuntimeError):
    """Raised when a protocol message cannot be decoded."""
