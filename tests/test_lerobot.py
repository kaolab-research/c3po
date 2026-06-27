"""Unit tests for LeRobot format conversion."""

from __future__ import annotations

import numpy as np

from c3po._lerobot import to_lerobot_action, to_lerobot_format


class TestLeRobotConversion:
    """Tests for to_lerobot_format and to_lerobot_action."""

    def test_single_arm_state(self) -> None:
        obs = {
            "joint_position": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "joint_velocity": np.array([0.01] * 6),
            "front_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        }
        result = to_lerobot_format(obs)
        assert "observation.state" in result
        np.testing.assert_array_equal(
            result["observation.state"],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        )

    def test_camera_mapping(self) -> None:
        obs = {
            "front_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
            "front_depth": np.zeros((64, 64), dtype=np.uint16),
        }
        result = to_lerobot_format(obs)
        assert "observation.images.front" in result
        assert "observation.images.front_depth" in result

    def test_camera_with_slash(self) -> None:
        obs = {
            "follower/wrist_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        }
        result = to_lerobot_format(obs)
        # / is replaced with .
        assert "observation.images.follower.wrist" in result

    def test_bimanual_state(self) -> None:
        obs = {
            "left_follower/joint_position": np.array([0.1, 0.2, 0.3]),
            "right_follower/joint_position": np.array([0.4, 0.5, 0.6]),
        }
        result = to_lerobot_format(obs)
        assert "observation.state" in result
        # Sorted keys → left before right.
        np.testing.assert_array_equal(
            result["observation.state"],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        )

    def test_all_joint_positions_included_in_state(self) -> None:
        """All joint_position keys are included in observation.state.
        (to_lerobot_format is a pure key-mapping function; it cannot
        distinguish arms from controllers without additional metadata.)"""
        obs = {
            "joint_position": np.array([0.1, 0.2]),
            "leader/joint_position": np.array([0.3, 0.4]),
        }
        result = to_lerobot_format(obs)
        # Sorted keys: joint_position (no prefix) before leader/
        assert len(result["observation.state"]) == 4

    def test_action_conversion(self) -> None:
        action = {
            "follower/joint_position": np.array([0.1, 0.2, 0.3]),
        }
        result = to_lerobot_action(action)
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3]))

    def test_bimanual_action_conversion(self) -> None:
        action = {
            "left_follower/joint_position": np.array([0.1, 0.2]),
            "right_follower/joint_position": np.array([0.3, 0.4]),
        }
        result = to_lerobot_action(action)
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))
