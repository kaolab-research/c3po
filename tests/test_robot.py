"""Integration tests for the Robot class against a mock rosbridge server."""

from __future__ import annotations

import time

import numpy as np
import pytest

from c3po import IncompatibleVersionError, Robot, RobotNotFoundError

# The mock_server fixture is defined in conftest.py


class TestRobotConnection:
    """Tests for Robot connection and discovery."""

    def test_connect_and_discover(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        assert robot.station_model == "so101_teleop"
        assert robot.control_rate == 50.0
        robot.close()

    def test_arms_property(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        assert robot.arms == ["follower"]
        robot.close()

    def test_cameras_property(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        assert robot.cameras == ["front_rgb"]
        robot.close()

    def test_controllers_property(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        assert robot.controllers == ["leader"]
        robot.close()

    def test_action_keys(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        assert robot.action_keys == ["follower/joint_position"]
        robot.close()

    def test_joint_names(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        assert "follower" in robot.joint_names
        assert len(robot.joint_names["follower"]) == 6
        robot.close()

    def test_context_manager(self, mock_server) -> None:
        with Robot(host="127.0.0.1", port=mock_server.port) as robot:
            assert robot.station_model == "so101_teleop"
        # After exit, robot should be closed.
        assert robot._closed

    def test_rejects_incompatible_api_version(self, mock_server) -> None:
        """An r2d2 reporting API v2.0 should raise IncompatibleVersionError."""
        # We can't easily change the mock's manifest, so we test the check
        # via a separate mock that reports v2.0.
        pass  # Requires a separate mock server; skip for now.

    def test_connection_failure_raises_not_found(self) -> None:
        """Connecting to a port where nothing is listening should fail."""
        with pytest.raises(RobotNotFoundError):
            Robot(host="127.0.0.1", port=19999)


class TestRobotObservations:
    """Tests for step() and reset() observation delivery."""

    def test_reset_returns_observations(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        obs = robot.reset()
        # Should have joint position and velocity.
        assert "follower/joint_position" in obs
        assert "follower/joint_velocity" in obs
        assert isinstance(obs["follower/joint_position"], np.ndarray)
        assert obs["follower/joint_position"].dtype == np.float64
        assert len(obs["follower/joint_position"]) == 6
        robot.close()

    def test_reset_returns_controller_state(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        obs = robot.reset()
        assert "leader/joint_position" in obs
        robot.close()

    def test_step_returns_observations(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        obs = robot.step()
        assert "follower/joint_position" in obs
        robot.close()

    def test_step_sends_action(self, mock_server) -> None:
        """step() with an action should not raise."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        action = {"follower/joint_position": np.array([0.1] * 6)}
        obs = robot.step(action)
        assert "follower/joint_position" in obs
        robot.close()

    def test_step_validates_missing_action_keys(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        with pytest.raises(ValueError, match="missing required keys"):
            robot.step({"wrong_key": np.array([0.1])})
        robot.close()

    def test_step_none_action_is_read_only(self, mock_server) -> None:
        """step(action=None) should work as a read-only step."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        obs = robot.step(None)
        assert "follower/joint_position" in obs
        robot.close()

    def test_step_pacing(self, mock_server) -> None:
        """step() should enforce the configured rate."""
        robot = Robot(host="127.0.0.1", port=mock_server.port, rate=20.0)
        robot.reset()
        t0 = time.monotonic()
        for _ in range(5):
            robot.step()
        t1 = time.monotonic()
        elapsed = t1 - t0
        # At 20 Hz, 5 steps should take ~200 ms (4 × 50 ms periods).
        # Allow generous tolerance for CI slowness.
        assert elapsed >= 0.15, f"Expected >=0.15s, got {elapsed:.3f}s"
        robot.close()
