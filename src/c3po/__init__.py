"""c3po — The human-facing protocol droid.

A lightweight Python client library for interacting with N‑Droids robot stations
via a simple :class:`Robot` API.

Usage::

    from c3po import Robot

    with Robot() as robot:
        obs = robot.reset()
        while True:
            action = policy(obs)
            obs = robot.step(action)
"""

__all__ = [
    "Robot",
    "Recorder",
    "ReplayClient",
    "RobotNotFoundError",
    "RobotDisconnectedError",
    "IncompatibleVersionError",
    "to_lerobot_format",
    "to_lerobot_action",
]


def __getattr__(name: str):
    if name == "Robot":
        from c3po.robot import Robot as _r

        return _r
    if name == "Recorder":
        from c3po._recorder import Recorder as _rec

        return _rec
    if name == "ReplayClient":
        from c3po._recorder import ReplayClient as _replay

        return _replay
    if name == "RobotNotFoundError":
        from c3po.exceptions import RobotNotFoundError as _rnf

        return _rnf
    if name == "RobotDisconnectedError":
        from c3po.exceptions import RobotDisconnectedError as _rdc

        return _rdc
    if name == "IncompatibleVersionError":
        from c3po.exceptions import IncompatibleVersionError as _ive

        return _ive
    if name == "to_lerobot_format":
        from c3po._lerobot import to_lerobot_format as _tlf

        return _tlf
    if name == "to_lerobot_action":
        from c3po._lerobot import to_lerobot_action as _tla

        return _tla
    raise AttributeError(f"module 'c3po' has no attribute {name!r}")
