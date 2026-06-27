"""Unit tests for the ObservationBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from c3po._buffer import ObservationBuffer


class TestObservationBuffer:
    """Tests for the non-blocking ObservationBuffer."""

    def test_get_returns_empty_initially(self) -> None:
        buf = ObservationBuffer()
        assert buf.get() == {}

    def test_get_returns_latest_values(self) -> None:
        buf = ObservationBuffer()
        buf.update("joint_position", np.array([0.5]))
        buf.update("joint_velocity", np.array([0.01]))
        buf.update("joint_position", np.array([0.6]))  # overwrite

        obs = buf.get()
        assert obs["joint_position"][0] == pytest.approx(0.6)
        assert obs["joint_velocity"][0] == pytest.approx(0.01)

    def test_get_does_not_block(self) -> None:
        buf = ObservationBuffer()
        # Should return immediately, even with no data.
        obs = buf.get()
        assert obs == {}

    def test_get_returns_snapshot_not_live_dict(self) -> None:
        buf = ObservationBuffer()
        buf.update("a", np.array([1.0]))
        obs1 = buf.get()
        buf.update("a", np.array([2.0]))
        # obs1 should still have the old value.
        assert obs1["a"][0] == pytest.approx(1.0)

    def test_wait_until_any_returns_immediately_if_data_present(self) -> None:
        buf = ObservationBuffer()
        buf.update("a", np.array([1.0]))
        assert buf.wait_until_any(timeout=0.5) is True

    def test_wait_until_any_blocks_until_data(self) -> None:
        import threading
        import time

        buf = ObservationBuffer()
        result_container: list[bool] = []

        def _waiter() -> None:
            result_container.append(buf.wait_until_any(timeout=2.0))

        t = threading.Thread(target=_waiter)
        t.start()
        time.sleep(0.1)
        buf.update("a", np.array([1.0]))
        t.join(timeout=1.0)

        assert result_container == [True]

    def test_wait_until_any_times_out(self) -> None:
        buf = ObservationBuffer()
        assert buf.wait_until_any(timeout=0.05) is False

    def test_clear_removes_all_values(self) -> None:
        buf = ObservationBuffer()
        buf.update("a", np.array([1.0]))
        buf.clear()
        assert buf.get() == {}

    def test_wait_until_any_clears_event_after_return(self) -> None:
        buf = ObservationBuffer()
        buf.update("a", np.array([1.0]))
        # First wait succeeds.
        assert buf.wait_until_any(timeout=0.1) is True
        # Second wait should time out (no new data).
        assert buf.wait_until_any(timeout=0.05) is False
