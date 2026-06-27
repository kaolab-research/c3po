"""Unit tests for manifest parsing."""

from __future__ import annotations

import json

from c3po._manifest import build_observation_keys, parse_manifest

# ---------------------------------------------------------------------------
# Sample manifest
# ---------------------------------------------------------------------------

_MANIFEST = {
    "api_version": "1.0",
    "station_model": "so101_teleop",
    "control_rate": 50.0,
    "arms": [
        {
            "name": "follower",
            "joint_count": 6,
            "joint_names": ["s", "l", "e", "wf", "wr", "g"],
            "command_mode": "joint_position",
        }
    ],
    "cameras": [
        {"name": "front_rgb", "resolution": [1280, 720]},
        {"name": "follower/wrist_rgb", "resolution": [640, 480]},
    ],
    "controllers": [
        {
            "name": "leader",
            "type": "joint_position",
            "joint_count": 6,
            "joint_names": ["s", "l", "e", "wf", "wr", "g"],
        }
    ],
}


def _make_raw(manifest: dict) -> dict:
    """Build a raw /describe response dict from a manifest."""
    return {
        "robot_model": manifest.get("station_model", "unknown"),
        "joint_count": manifest["arms"][0]["joint_count"]
        if manifest.get("arms")
        else 0,
        "control_rate": manifest.get("control_rate", 50.0),
        "camera_names": [c["name"] for c in manifest.get("cameras", [])],
        "command_mode": manifest["arms"][0]["command_mode"]
        if manifest.get("arms")
        else "",
        "joint_names": manifest["arms"][0]["joint_names"]
        if manifest.get("arms")
        else [],
        "manifest_json": json.dumps(manifest),
    }


class TestParseManifest:
    """Tests for parse_manifest()."""

    def test_parses_station_model(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["station_model"] == "so101_teleop"

    def test_parses_control_rate(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["control_rate"] == 50.0

    def test_parses_arm_names(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["arm_names"] == ["follower"]

    def test_parses_camera_names(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["camera_names"] == ["front_rgb", "follower/wrist_rgb"]

    def test_parses_controller_names(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["controller_names"] == ["leader"]

    def test_parses_action_keys(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["action_keys"] == ["follower/joint_position"]

    def test_parses_joint_names(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["joint_names"]["follower"] == ["s", "l", "e", "wf", "wr", "g"]

    def test_parses_api_version(self) -> None:
        result = parse_manifest(_make_raw(_MANIFEST))
        assert result["api_version"] == "1.0"

    def test_single_arm_no_prefix(self) -> None:
        """Single-arm station with empty name → no prefix in action keys."""
        manifest = {
            **dict(_MANIFEST),
            "arms": [
                {
                    "name": "",
                    "joint_count": 6,
                    "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                    "command_mode": "joint_position",
                }
            ],
        }
        raw = _make_raw(manifest)
        result = parse_manifest(raw)
        assert result["action_keys"] == ["joint_position"]

    def test_bimanual_station(self) -> None:
        manifest = {
            **dict(_MANIFEST),
            "arms": [
                {
                    "name": "left_follower",
                    "joint_count": 6,
                    "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                    "command_mode": "joint_position",
                },
                {
                    "name": "right_follower",
                    "joint_count": 6,
                    "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                    "command_mode": "joint_position",
                },
            ],
        }
        raw = _make_raw(manifest)
        result = parse_manifest(raw)
        assert set(result["action_keys"]) == {
            "left_follower/joint_position",
            "right_follower/joint_position",
        }

    def test_backward_compat_flat_fields(self) -> None:
        """Legacy /describe response without manifest_json."""
        raw = {
            "robot_model": "so101",
            "joint_count": 6,
            "control_rate": 100.0,
            "camera_names": ["front_rgb"],
            "command_mode": "joint_position",
            "joint_names": ["a", "b", "c", "d", "e", "f"],
            "manifest_json": "{}",  # empty JSON
        }
        result = parse_manifest(raw)
        assert result["arm_names"] == [""]
        assert result["action_keys"] == ["joint_position"]


class TestBuildObservationKeys:
    """Tests for build_observation_keys()."""

    def test_single_arm_with_cameras_and_controller(self) -> None:
        manifest = parse_manifest(_make_raw(_MANIFEST))
        keys = build_observation_keys(manifest)
        expected = [
            "follower/joint_position",
            "follower/joint_velocity",
            "front_rgb",
            "follower/wrist_rgb",
            "leader/joint_position",
        ]
        assert keys == expected

    def test_no_arms(self) -> None:
        manifest = parse_manifest(
            _make_raw({**_MANIFEST, "arms": [], "controllers": []})
        )
        keys = build_observation_keys(manifest)
        assert keys == ["front_rgb", "follower/wrist_rgb"]
