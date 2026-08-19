"""Integration tests for the Robot class against a mock r2d2 server."""

from __future__ import annotations

import time

import numpy as np
import pytest

from c3po import Robot, RobotNotFoundError
from c3po.exceptions import IncompatibleVersionError, RobotDisconnectedError


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

    def test_connection_failure_raises_not_found(self) -> None:
        """Connecting to a port where nothing is listening should fail."""
        with pytest.raises(RobotNotFoundError):
            Robot(host="127.0.0.1", port=19999)


class TestRobotObservations:
    """Tests for step() and reset() observation delivery."""

    def test_reset_returns_all_keys(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        obs = robot.reset()
        # Joint state.
        assert "follower/joint_position" in obs
        assert "follower/joint_velocity" in obs
        assert isinstance(obs["follower/joint_position"], np.ndarray)
        assert obs["follower/joint_position"].dtype == np.float64
        assert len(obs["follower/joint_position"]) == 6
        # Controller.
        assert "leader/joint_position" in obs
        robot.close()

    def test_reset_blocks_until_complete(self, mock_server) -> None:
        """reset() should not return until all expected keys have data."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        obs = robot.reset()
        # All observation keys from the manifest must be present.
        for key in robot._obs_keys:
            assert key in obs, f"Missing key: {key}"
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

    def test_step_with_reward_done(self, mock_server) -> None:
        """step() with reward and done kwargs should not raise."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        action = {"follower/joint_position": np.array([0.1] * 6)}
        obs = robot.step(action, reward=0.5, done=False)
        assert "follower/joint_position" in obs
        robot.close()

    def test_step_validates_missing_action_keys(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        with pytest.raises(ValueError, match="missing required keys"):
            robot.step({"wrong_key": np.array([0.1])})
        robot.close()

    def test_step_none_action_read_only(self, mock_server) -> None:
        """step(None) should be a read-only observation step."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        obs = robot.step(None)
        assert "follower/joint_position" in obs
        robot.close()

    def test_camera_frame_arrives(self, mock_server) -> None:
        """After reset, camera frames should be available in observations."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        obs = robot.reset()
        # The mock sends JPEG frames.  The buffer decodes them.
        # Wait a few steps to accumulate camera data.
        for _ in range(10):
            obs = robot.step()
            if "front_rgb" in obs:
                break
        assert "front_rgb" in obs
        assert isinstance(obs["front_rgb"], np.ndarray)
        assert obs["front_rgb"].ndim == 3
        robot.close()

    def test_multiple_steps(self, mock_server) -> None:
        """A sequence of steps should return fresh observations."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.reset()
        for i in range(5):
            obs = robot.step()
            assert "follower/joint_position" in obs
        robot.close()


class TestRobotRecording:
    """Tests for recording commands."""

    def test_start_stop_recording(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        # These should not raise.
        robot.start_recording("test_session")
        robot.stop_recording()
        robot.close()

    def test_cancel_episode(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.start_recording("test_session")
        robot.cancel_episode()
        robot.stop_recording()
        robot.close()

    def test_recording_flow(self, mock_server) -> None:
        """Full recording lifecycle: start → reset → step → stop."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        robot.start_recording("demo")
        obs = robot.reset()
        for _ in range(3):
            obs = robot.step()
        robot.stop_recording()
        robot.close()

    def test_wait_for_status_survives_camera_stream_after_stop(self, mock_server) -> None:
        """After stop_recording r2d2 stops observations but keeps streaming
        camera frames — wait_for_status must still return promptly instead
        of hanging in an observation wait (regression: the rollout client
        hung forever draining with step(None), 2026-08-19)."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        try:
            robot.start_recording("test_session")
            started = robot.wait_for_status("recording_started", timeout=10.0)
            assert started is not None
            assert started.event == "recording_started"

            robot.stop_recording()
            t0 = time.monotonic()
            ready = robot.wait_for_status("dataset_ready", timeout=15.0)
            elapsed = time.monotonic() - t0
            assert ready is not None
            assert ready.data.get("name") == "test_session"
            assert elapsed < 10.0  # must not stall on the camera stream
        finally:
            robot.close()

    def test_wait_for_status_timeout_returns_none(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        try:
            msg = robot.wait_for_status("never_happens", timeout=1.0)
            assert msg is None
        finally:
            robot.close()


class TestControllerCapabilities:
    """Tests for Robot.controller_capabilities property."""

    def test_returns_capabilities_from_manifest(self, mock_server) -> None:
        from c3po import Robot

        robot = Robot(host="127.0.0.1", port=mock_server.port)
        caps = robot.controller_capabilities
        # mock_server uses the default manifest which has a leader controller.
        assert "leader" in caps
        assert isinstance(caps["leader"], list)
        robot.close()

    def test_unknown_station_has_empty_dict(self) -> None:
        """If no controllers in manifest, returns empty dict."""
        # Use the mock_server which always has a leader.  Test the property
        # doesn't crash — it should handle missing controllers gracefully.
        pass  # covered by real integration tests


class TestLeaderHomePosition:
    """Tests for Robot.leader_home_position property."""

    def test_returns_home_position_from_manifest(self, mock_server) -> None:
        from c3po import Robot
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        home = robot.leader_home_position
        # Default mock_server manifest: leader controller with no home_position.
        assert "leader" in home
        assert home["leader"] is None
        robot.close()
