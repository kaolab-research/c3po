"""Recording context manager for c3po.

Provides :class:`Recording` — a context manager that wraps a robot
session with start/stop recording commands and a built-in keyboard
listener for episode control (``q`` / ``n`` / ``r``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from c3po._keyboard import KeyboardListener

if TYPE_CHECKING:
    from c3po.robot import Robot


class Recording:
    """Context manager for recording a dataset with keyboard controls.

    Usage::

        with robot.recording("session_001") as rec:
            while not rec.stop:
                obs = robot.reset()
                while not rec.stop and not rec.done:
                    obs = robot.step(policy(obs))
                if rec.rerecord:
                    robot.cancel_episode()
                    rec.clear()

    Parameters
    ----------
    robot:
        A connected :class:`Robot` instance.
    name:
        Dataset name passed to r2d2's ``start_recording`` command.
    """

    def __init__(self, robot: Robot, name: str) -> None:
        self._robot = robot
        self.name = name
        self._keyboard = KeyboardListener()
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Flags (delegated to keyboard listener)
    # ------------------------------------------------------------------

    @property
    def stop(self) -> bool:
        """``True`` after the user presses ``q`` (quit / stop recording)."""
        return self._keyboard.stop

    @property
    def done(self) -> bool:
        """``True`` after the user presses ``n`` (end current episode)."""
        return self._keyboard.done

    @property
    def rerecord(self) -> bool:
        """``True`` after the user presses ``r`` (discard and re-record)."""
        return self._keyboard.rerecord

    @property
    def episode_count(self) -> int:
        """Number of completed episodes (incremented each ``reset()``)."""
        return self._episode_count

    def clear(self) -> None:
        """Reset ``done`` and ``rerecord`` flags.  ``stop`` is never reset."""
        self._keyboard.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Recording:
        self._robot.start_recording(self.name)
        self._keyboard.start()
        return self

    def __exit__(self, *args) -> None:
        try:
            self._robot.stop_recording()
        except Exception:
            pass
        self._keyboard.stop_listener()
