"""Tests for MockR2D2Server — the mock r2d2 used by c3po integration tests."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import websocket
from conftest import MOCK_MANIFEST, MOCK_MANIFEST_BIMANUAL, MockR2D2Server

from c3po._protocol import (
    Action,
    CancelEpisode,
    DescribeRequest,
    Observation,
    StartRecording,
    StatusMessage,
    StopRecording,
    decode_binary_frame,
    decode_json_message,
    encode_json_message,
)

# ---------------------------------------------------------------------------
# Helpers: a minimal c3po-style client for direct mock testing
# ---------------------------------------------------------------------------


def _ws_connect(url: str) -> websocket.WebSocket:
    """Connect to *url* and return the WebSocket."""
    ws = websocket.WebSocket()
    ws.connect(url)
    return ws


def _recv_json(ws: websocket.WebSocket, timeout: float = 2.0):
    """Receive the next text frame and decode it as a protocol message.

    Skips binary frames (camera data) until a text frame arrives.
    """
    ws.settimeout(timeout)
    while True:
        opcode, data = ws.recv_data()
        if opcode == websocket.ABNF.OPCODE_TEXT:
            return decode_json_message(data.decode("utf-8"))
        # OPCODE_BINARY: skip — camera frames are tested separately.


def _recv_binary(ws: websocket.WebSocket, timeout: float = 2.0) -> bytes:
    """Receive one binary frame and return its raw bytes."""
    ws.settimeout(timeout)
    # websocket-client returns bytes for binary frames, str for text
    opcode, data = ws.recv_data()
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMockR2D2:
    """Verify that MockR2D2Server correctly speaks the N-Droids protocol."""

    def test_handshake_returns_manifest(self, mock_server) -> None:
        """Sending describe must return a describe_response with the manifest."""
        ws = _ws_connect(mock_server.url)
        try:
            ws.send(encode_json_message(DescribeRequest()))
            resp = _recv_json(ws)
            assert resp.type == "describe_response"
            assert resp.manifest == MOCK_MANIFEST
        finally:
            ws.close()

    def test_streams_observations(self, mock_server) -> None:
        """After handshake, the server must stream observation messages."""
        ws = _ws_connect(mock_server.url)
        try:
            ws.send(encode_json_message(DescribeRequest()))
            # Discard the describe_response.
            _recv_json(ws)

            # Wait for at least one observation.
            obs = _recv_json(ws)
            assert isinstance(obs, Observation)
            assert "follower/joint_position" in obs.data
            assert "follower/joint_velocity" in obs.data
            assert "leader/joint_position" in obs.data
            assert len(obs.data["follower/joint_position"]) == 6
        finally:
            ws.close()

    def test_streams_binary_frames(self, mock_server) -> None:
        """After handshake, the server must stream binary JPEG frames
        for each camera in the manifest."""
        ws = _ws_connect(mock_server.url)
        try:
            ws.send(encode_json_message(DescribeRequest()))
            _recv_json(ws)  # describe_response

            # Receive at least one binary frame.
            # (Observations and binary frames are interleaved — skip JSON
            #  frames until we get a binary one.)
            for _ in range(50):
                opcode, data = ws.recv_data()
                if opcode == websocket.ABNF.OPCODE_BINARY:
                    break
            else:
                pytest.fail("No binary frame received within 50 messages")

            frame = decode_binary_frame(data)
            assert frame.camera_name == "front_rgb"
            assert frame.encoding == 0  # JPEG_RGB
            assert len(frame.payload) > 100  # non-trivial JPEG
        finally:
            ws.close()

    def test_echoes_actions(self, mock_server) -> None:
        """The server must accept action messages without error."""
        ws = _ws_connect(mock_server.url)
        try:
            ws.send(encode_json_message(DescribeRequest()))
            _recv_json(ws)  # describe_response

            action = Action(data={"follower/joint_position": [0.1] * 6})
            ws.send(encode_json_message(action))

            # Should not raise — just verify we can still receive observations.
            obs = _recv_json(ws)
            assert isinstance(obs, Observation)
        finally:
            ws.close()

    def test_recording_commands_acked(self, mock_server) -> None:
        """start_recording / stop_recording / cancel_episode must not error.

        Mirrors real r2d2: the mock pushes recording_started on start and
        recording_stopped + dataset_ready on stop (after which observations
        stop but camera frames keep flowing)."""
        ws = _ws_connect(mock_server.url)
        try:
            ws.send(encode_json_message(DescribeRequest()))
            _recv_json(ws)

            ws.send(encode_json_message(StartRecording(name="test")))
            ws.send(encode_json_message(CancelEpisode()))
            ws.send(encode_json_message(StopRecording()))

            # Drain (observations may be queued ahead of the statuses).
            events = []
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and len(events) < 3:
                msg = _recv_json(ws)
                if isinstance(msg, StatusMessage):
                    events.append(msg.event)
            assert "recording_started" in events
            assert "recording_stopped" in events
            assert "dataset_ready" in events
        finally:
            ws.close()

    def test_configurable_control_rate(self) -> None:
        """A server with control_rate=100 must produce messages at ~100 Hz."""
        server = MockR2D2Server(control_rate=100.0)
        server.start()
        try:
            ws = _ws_connect(server.url)
            ws.send(encode_json_message(DescribeRequest()))
            _recv_json(ws)  # describe_response

            import time

            t0 = time.monotonic()
            count = 0
            while time.monotonic() - t0 < 0.5:
                _recv_json(ws)
                count += 1
            ws.close()

            # At 100 Hz for 0.5 s expect ~50 messages.  Allow generous tolerance.
            assert count >= 20, (
                f"Expected >=20 observations in 0.5s at 100 Hz, got {count}"
            )
        finally:
            server.stop()

    def test_bimanual_manifest(self) -> None:
        """A bimanual ALOHA manifest must produce observations with
        left_ and right_ prefixed keys."""
        server = MockR2D2Server(manifest=MOCK_MANIFEST_BIMANUAL)
        server.start()
        try:
            ws = _ws_connect(server.url)
            ws.send(encode_json_message(DescribeRequest()))
            resp = _recv_json(ws)
            assert resp.manifest == MOCK_MANIFEST_BIMANUAL

            obs = _recv_json(ws)
            assert "left_follower/joint_position" in obs.data
            assert "right_follower/joint_position" in obs.data
            assert "left_leader/joint_position" in obs.data
            assert "right_leader/joint_position" in obs.data
        finally:
            ws.close()
            server.stop()
