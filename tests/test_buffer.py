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


class TestObservationBufferJpeg:
    """Tests for update_from_frame — JPEG decode and storage."""

    @staticmethod
    def _make_jpeg() -> bytes:
        """Return a tiny valid JPEG as bytes."""
        from io import BytesIO

        from PIL import Image

        rng = __import__("numpy").random.default_rng()
        img = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype="uint8"))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def test_update_from_frame_stores_rgb_array(self) -> None:
        buf = ObservationBuffer()
        jpeg = self._make_jpeg()
        buf.update_from_frame("front_rgb", jpeg)
        obs = buf.get()
        assert "front_rgb" in obs
        arr = obs["front_rgb"]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8
        assert arr.ndim == 3
        assert arr.shape[2] == 3  # RGB

    def test_update_from_frame_overwrites_previous(self) -> None:
        buf = ObservationBuffer()
        jpeg1 = self._make_jpeg()
        jpeg2 = self._make_jpeg()
        buf.update_from_frame("front_rgb", jpeg1)
        first = buf.get()["front_rgb"].copy()
        buf.update_from_frame("front_rgb", jpeg2)
        second = buf.get()["front_rgb"]
        assert not np.array_equal(first, second)

    def test_update_mixed_joint_state_and_jpeg(self) -> None:
        buf = ObservationBuffer()
        buf.update("follower/joint_position", np.array([0.1, 0.2, 0.3]))
        buf.update_from_frame("front_rgb", self._make_jpeg())
        obs = buf.get()
        assert "follower/joint_position" in obs
        assert "front_rgb" in obs
        np.testing.assert_array_equal(obs["follower/joint_position"], [0.1, 0.2, 0.3])
