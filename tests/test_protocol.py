"""Tests for the N-Droids wire protocol — JSON messages and binary frames.

These tests define the protocol contract. Both c3po and r2d2 must satisfy them.
"""

from __future__ import annotations

import json

import pytest
from c3po._protocol import (
    # Message types
    Action,
    BinaryFrame,
    CancelEpisode,
    DescribeRequest,
    DescribeResponse,
    Encoding,
    Observation,
    # Errors
    ProtocolError,
    StartRecording,
    StopRecording,
    # Functions
    decode_binary_frame,
    decode_json_message,
    encode_binary_frame,
    encode_json_message,
)

# ---------------------------------------------------------------------------
# JSON message roundtrip tests
# ---------------------------------------------------------------------------


class TestJsonMessages:
    """Every message type must survive a create → encode → decode → compare cycle."""

    def test_describe_request_roundtrip(self) -> None:
        msg = DescribeRequest()
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.type == "describe"

    def test_describe_response_roundtrip(self) -> None:
        manifest = {
            "api_version": "1.0",
            "station_model": "so101_teleop",
            "control_rate": 50.0,
            "arms": [],
            "cameras": [],
            "controllers": [],
        }
        msg = DescribeResponse(manifest=manifest)
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.type == "describe_response"
        assert decoded.manifest == manifest

    def test_action_message_roundtrip(self) -> None:
        msg = Action(data={"follower/joint_position": [0.1, -0.3, 0.5, -0.2, 0.0, 0.0]})
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.type == "action"
        assert decoded.data == {
            "follower/joint_position": [0.1, -0.3, 0.5, -0.2, 0.0, 0.0]
        }
        assert decoded.reward == 0.0
        assert decoded.done is False

    def test_action_with_reward_done_roundtrip(self) -> None:
        msg = Action(
            data={"follower/joint_position": [0.1, 0.2]},
            reward=1.0,
            done=True,
        )
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.data == {"follower/joint_position": [0.1, 0.2]}
        assert decoded.reward == 1.0
        assert decoded.done is True

    def test_observation_message_roundtrip(self) -> None:
        msg = Observation(
            data={
                "follower/joint_position": [0.1, -0.3, 0.5],
                "follower/joint_velocity": [0.01, -0.02, 0.0],
                "leader/joint_position": [0.12, -0.28, 0.52],
            }
        )
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.type == "observation"
        assert decoded.data == msg.data

    def test_start_recording_roundtrip(self) -> None:
        msg = StartRecording(name="session_001")
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.type == "start_recording"
        assert decoded.name == "session_001"

    def test_stop_recording_roundtrip(self) -> None:
        msg = StopRecording()
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.type == "stop_recording"

    def test_cancel_episode_roundtrip(self) -> None:
        msg = CancelEpisode()
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.type == "cancel_episode"

    def test_unknown_message_type_raises(self) -> None:
        raw = json.dumps({"type": "nonexistent", "foo": "bar"})
        with pytest.raises(ProtocolError, match="Unknown message type"):
            decode_json_message(raw)

    def test_message_without_type_field_raises(self) -> None:
        raw = json.dumps({"foo": "bar"})
        with pytest.raises(ProtocolError, match="missing"):
            decode_json_message(raw)


# ---------------------------------------------------------------------------
# Binary frame tests
# ---------------------------------------------------------------------------


class TestBinaryFrames:
    """Binary camera frames must survive encode → decode with exact fidelity."""

    def test_frame_header_roundtrip(self) -> None:
        """A frame with empty JPEG payload still roundtrips its metadata."""
        frame = BinaryFrame(
            camera_name="front_rgb",
            timestamp_ns=1234567890,
            width=1280,
            height=720,
            encoding=Encoding.JPEG_RGB,
            jpeg_bytes=b"",
        )
        raw = encode_binary_frame(frame)
        decoded = decode_binary_frame(raw)
        assert decoded.camera_name == "front_rgb"
        assert decoded.timestamp_ns == 1234567890
        assert decoded.width == 1280
        assert decoded.height == 720
        assert decoded.encoding == Encoding.JPEG_RGB
        assert decoded.jpeg_bytes == b""

    def test_frame_with_jpeg_payload(self) -> None:
        """JPEG bytes survive the roundtrip byte-for-byte."""
        jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0x00] * 100 + [0xFF, 0xD9])
        frame = BinaryFrame(
            camera_name="follower/wrist_rgb",
            timestamp_ns=999999999999999,
            width=640,
            height=480,
            encoding=Encoding.JPEG_RGB,
            jpeg_bytes=jpeg,
        )
        raw = encode_binary_frame(frame)
        decoded = decode_binary_frame(raw)
        assert decoded.camera_name == "follower/wrist_rgb"
        assert decoded.jpeg_bytes == jpeg

    def test_encoding_constants_are_distinct(self) -> None:
        """JPEG_RGB and JPEG_DEPTH must have different values."""
        assert Encoding.JPEG_RGB != Encoding.JPEG_DEPTH
        assert Encoding.JPEG_RGB == 0
        assert Encoding.JPEG_DEPTH == 1

    def test_malformed_header_raises(self) -> None:
        """A message shorter than the minimum header length raises ProtocolError."""
        too_short = b"\x00\x00\x00"  # 3 bytes — can't even read name_len
        with pytest.raises(ProtocolError, match="too short"):
            decode_binary_frame(too_short)
