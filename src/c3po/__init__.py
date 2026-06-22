"""c3po — The human-facing protocol droid.

A lightweight Python client library for interacting with N‑Droids robot stations
via a simple :class:`RobotClient` API.
"""

from c3po.exceptions import RobotDisconnectedError, RobotNotFoundError
from c3po.robot_client import RobotClient

__all__ = ["RobotClient", "RobotNotFoundError", "RobotDisconnectedError"]
