"""Unit tests for the _ObservationBuffer synchronisation primitive."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from c3po._buffer import _ObservationBuffer
from c3po.exceptions import RobotDisconnectedError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed(buffer: _ObservationBuffer, keys: list[str], delay: float = 0.0) -> None:
    """Push one value per key into *buffer*, optionally after *delay* seconds."""
    if delay:
        time.sleep(delay)
    for k in keys:
        buffer.update(k, np.array([1.0]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestObservationBuffer:
    """Unit tests for _ObservationBuffer."""

    # -- blocks until all topics arrive -----------------------------------

    def test_blocks_until_all_topics(self) -> None:
        """get() does not return before every configured key has data."""
        buf = _ObservationBuffer()
        buf.configure(["joint_position", "joint_velocity", "front_rgb"])

        buf.update("joint_position", np.array([0.5]))
        buf.update("joint_velocity", np.array([0.1]))

        # Feed the missing key from another thread after a short delay.
        def _deliver_last() -> None:
            time.sleep(0.1)
            buf.update("front_rgb", np.zeros((480, 640, 3), dtype=np.uint8))

        t = threading.Thread(target=_deliver_last)
        t.start()

        obs = buf.get(timeout=1.0)
        t.join()

        assert set(obs.keys()) == {"joint_position", "joint_velocity", "front_rgb"}

    # -- returns a consistent snapshot ------------------------------------

    def test_snapshot_contains_latest_values(self) -> None:
        """get() returns the most recent value for every key."""
        buf = _ObservationBuffer()
        buf.configure(["joint_position", "joint_velocity"])

        buf.update("joint_position", np.array([0.1]))
        buf.update("joint_position", np.array([0.2]))  # overwrite
        buf.update("joint_velocity", np.array([0.5]))

        obs = buf.get(timeout=1.0)
        assert obs["joint_position"][0] == pytest.approx(0.2)
        assert obs["joint_velocity"][0] == pytest.approx(0.5)

    # -- clears received state between calls ------------------------------

    def test_clears_between_calls(self) -> None:
        """After get() returns, the next get() blocks until fresh data."""
        buf = _ObservationBuffer()
        buf.configure(["joint_position"])

        # First call: data is available, returns immediately.
        buf.update("joint_position", np.array([1.0]))
        obs1 = buf.get(timeout=0.5)
        assert obs1["joint_position"][0] == 1.0

        # Second call: no new data — must time out.
        with pytest.raises(RobotDisconnectedError):
            buf.get(timeout=0.1)

    # -- timeout raises the right error -----------------------------------

    def test_timeout_raises_robot_disconnected(self) -> None:
        """A timed-out get() raises RobotDisconnectedError."""
        buf = _ObservationBuffer()
        buf.configure(["joint_position"])

        # Never deliver data — timeout should fire.
        with pytest.raises(RobotDisconnectedError, match="Timeout"):
            buf.get(timeout=0.05)

    # -- configure resets expected keys -----------------------------------

    def test_configure_resets_keys(self) -> None:
        """Calling configure() twice replaces the expected set."""
        buf = _ObservationBuffer()
        buf.configure(["joint_position"])
        buf.update("joint_position", np.array([0.0]))
        buf.get(timeout=0.5)  # clears

        buf.configure(["front_rgb"])
        buf.update("front_rgb", np.zeros((1, 1, 3), dtype=np.uint8))
        obs = buf.get(timeout=0.5)
        assert "front_rgb" in obs
