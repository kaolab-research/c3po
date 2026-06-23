"""Thread-safe observation buffer with timestamp alignment.

Used internally by :class:`RobotClient` to collect data from multiple
ROS 2 subscription callbacks (running in the executor thread) and deliver
a consistent snapshot to the user-facing ``get_observation()`` call.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from c3po.exceptions import RobotDisconnectedError


class _ObservationBuffer:
    """Thread-safe buffer that aligns incoming ROS 2 observations.

    Subscription callbacks (running in the executor thread) push data into
    this buffer.  The user-facing ``get_observation()`` blocks until every
    expected topic has delivered at least one new value since the last call.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._expected: set[str] = set()
        self._latest: dict[str, np.ndarray] = {}
        self._received: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, topic_keys: list[str]) -> None:
        """Set the complete set of keys that must arrive each cycle."""
        with self._condition:
            self._expected = set(topic_keys)

    def update(self, key: str, value: np.ndarray) -> None:
        """Push a single observation value (called from a subscriber)."""
        with self._condition:
            self._latest[key] = value
            self._received.add(key)
            self._condition.notify_all()

    def get(self, timeout: float | None = None) -> dict[str, np.ndarray]:
        """Block until every expected key has new data, then return a snapshot.

        Parameters
        ----------
        timeout:
            Maximum time to wait in seconds.  If *None*, waits indefinitely.

        Returns
        -------
        dict
            Snapshot of the latest value for every configured key.

        Raises
        ------
        RobotDisconnectedError
            If *timeout* is not None and the deadline expires before all
            keys have been received.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._condition:
            while True:
                missing = self._expected - self._received
                if not missing:
                    result = dict(self._latest)
                    self._received.clear()
                    return result
                if timeout is not None:
                    assert deadline is not None  # guaranteed above
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RobotDisconnectedError(
                            f"Timeout waiting for topics: {missing}"
                        )
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()
