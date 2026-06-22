"""Exceptions raised by the c3po client library."""

from __future__ import annotations


class RobotNotFoundError(RuntimeError):
    """Raised when no /describe service is discovered within the timeout."""


class RobotDisconnectedError(RuntimeError):
    """Raised when the ROS 2 connection to the robot station is lost."""
