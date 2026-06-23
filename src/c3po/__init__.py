"""c3po — The human-facing protocol droid.

A lightweight Python client library for interacting with N‑Droids robot stations
via a simple :class:`RobotClient` API.

Usage::

    from c3po import RobotClient
"""

__all__ = ["RobotClient", "RobotNotFoundError", "RobotDisconnectedError"]


def __getattr__(name: str):
    if name == "RobotClient":
        from c3po.robot_client import RobotClient as _rc

        return _rc
    if name == "RobotNotFoundError":
        from c3po.exceptions import RobotNotFoundError as _rnf

        return _rnf
    if name == "RobotDisconnectedError":
        from c3po.exceptions import RobotDisconnectedError as _rdc

        return _rdc
    raise AttributeError(f"module 'c3po' has no attribute {name!r}")
