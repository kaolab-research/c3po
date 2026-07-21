"""Unit tests for manifest parsing."""

from __future__ import annotations

from c3po._manifest import build_observation_keys, parse_manifest

# ---------------------------------------------------------------------------
# Sample manifest (as received from describe_response.manifest)
# ---------------------------------------------------------------------------

_MANIFEST: dict = {
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


class TestParseManifest:
    """Tests for parse_manifest()."""

    def test_parses_station_model(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["station_model"] == "so101_teleop"

    def test_parses_control_rate(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["control_rate"] == 50.0

    def test_parses_arm_names(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["arm_names"] == ["follower"]

    def test_parses_camera_names(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["camera_names"] == ["front_rgb", "follower/wrist_rgb"]

    def test_parses_controller_names(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["controller_names"] == ["leader"]

    def test_parses_action_keys(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["action_keys"] == ["follower/joint_position"]

    def test_parses_joint_names(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["joint_names"]["follower"] == ["s", "l", "e", "wf", "wr", "g"]

    def test_parses_api_version(self) -> None:
        result = parse_manifest(_MANIFEST)
        assert result["api_version"] == "1.0"

    def test_single_arm_no_prefix(self) -> None:
        """Single-arm station with empty name → no prefix in action keys."""
        manifest = {
            **_MANIFEST,
            "arms": [
                {
                    "name": "",
                    "joint_count": 6,
                    "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                    "command_mode": "joint_position",
                }
            ],
        }
        result = parse_manifest(manifest)
        assert result["action_keys"] == ["joint_position"]

    def test_bimanual_station(self) -> None:
        manifest = {
            **_MANIFEST,
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
        result = parse_manifest(manifest)
        assert set(result["action_keys"]) == {
            "left_follower/joint_position",
            "right_follower/joint_position",
        }

    def test_defaults_on_missing_fields(self) -> None:
        """Manifest with minimal fields should get sensible defaults."""
        result = parse_manifest({})
        assert result["api_version"] == "0.0"
        assert result["station_model"] == "unknown"
        assert result["control_rate"] == 50.0
        assert result["arm_names"] == []
        assert result["action_keys"] == []

    def test_empty_arms(self) -> None:
        """Station with no arms should have empty action_keys."""
        result = parse_manifest({**_MANIFEST, "arms": [], "controllers": []})
        assert result["arm_names"] == []
        assert result["action_keys"] == []

    def test_controller_without_joint_names(self) -> None:
        """Controllers may omit joint_names (e.g. HID devices)."""
        manifest = {
            **_MANIFEST,
            "controllers": [{"name": "gamepad", "type": "button_state"}],
        }
        result = parse_manifest(manifest)
        assert result["controller_names"] == ["gamepad"]

    # -- streaming resolution (Phase 9) -------------------------------------

    def test_camera_stream_resolution_preserved(self) -> None:
        """stream_resolution in camera entries is preserved during parsing."""
        manifest = {
            **_MANIFEST,
            "cameras": [
                {
                    "name": "front_rgb",
                    "resolution": [720, 1280],
                    "stream_resolution": [480, 854],
                }
            ],
        }
        result = parse_manifest(manifest)
        cam = result["cameras"][0]
        assert cam["name"] == "front_rgb"
        assert cam["resolution"] == [720, 1280]
        assert cam.get("stream_resolution") == [480, 854]

    def test_camera_stream_resolution_omitted_when_absent(self) -> None:
        """Cameras without stream_resolution should not have the key."""
        result = parse_manifest(_MANIFEST)
        for cam in result["cameras"]:
            assert "stream_resolution" not in cam


class TestBuildObservationKeys:
    """Tests for build_observation_keys()."""

    def test_single_arm_with_cameras_and_controller(self) -> None:
        manifest = parse_manifest(_MANIFEST)
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
        manifest = parse_manifest({**_MANIFEST, "arms": [], "controllers": []})
        keys = build_observation_keys(manifest)
        assert keys == ["front_rgb", "follower/wrist_rgb"]

    def test_hierarchical_camera_names(self) -> None:
        """Camera names with '/' (e.g. follower/wrist_rgb) are preserved."""
        manifest = parse_manifest(_MANIFEST)
        keys = build_observation_keys(manifest)
        assert "follower/wrist_rgb" in keys
