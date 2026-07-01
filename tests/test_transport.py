"""Tests for the WebSocket transport layer speaking the N-Droids custom protocol."""

from __future__ import annotations

import time

import pytest

from c3po._protocol import (
    Action,
    BinaryFrame,
    DescribeRequest,
    DescribeResponse,
    Observation,
    encode_json_message,
)
from c3po._transport._websocket import WebSocketTransport
from c3po.exceptions import RobotDisconnectedError, RobotNotFoundError

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSocketTransport:
    """Integration tests against MockR2D2Server."""

    def test_connect_to_mock(self, mock_server) -> None:
        transport = WebSocketTransport(mock_server.url)
        transport.connect()
        assert transport.is_connected
        transport.close()

    def test_send_json_receives_response(self, mock_server) -> None:
        """Sending a describe request must return a DescribeResponse."""
        transport = WebSocketTransport(mock_server.url)
        transport.connect()

        transport.send_json(DescribeRequest())
        resp = transport.recv(timeout=2.0)

        assert isinstance(resp, DescribeResponse)
        assert resp.manifest["station_model"] == "so101_teleop"
        transport.close()

    def test_recv_json_observation(self, mock_server) -> None:
        """After handshake, the transport must receive Observation messages."""
        transport = WebSocketTransport(mock_server.url)
        transport.connect()

        transport.send_json(DescribeRequest())
        transport.recv(timeout=2.0)  # describe_response

        obs = transport.recv(timeout=2.0)
        assert isinstance(obs, Observation)
        assert "follower/joint_position" in obs.data
        transport.close()

    def test_recv_binary_frame(self, mock_server) -> None:
        """After handshake, the transport must receive BinaryFrame objects
        for camera data."""
        transport = WebSocketTransport(mock_server.url)
        transport.connect()

        transport.send_json(DescribeRequest())
        transport.recv(timeout=2.0)  # describe_response

        # Receive until we get a binary frame (skipping observations).
        for _ in range(50):
            msg = transport.recv(timeout=2.0)
            if isinstance(msg, BinaryFrame):
                break
        else:
            pytest.fail("No binary frame received within 50 messages")

        assert msg.camera_name == "front_rgb"
        assert msg.encoding == 0  # JPEG_RGB
        assert len(msg.jpeg_bytes) > 100
        transport.close()

    def test_recv_binary_frame_reassembles_jpeg(self, mock_server) -> None:
        """JPEG bytes must survive the transport byte-for-byte."""
        transport = WebSocketTransport(mock_server.url)
        transport.connect()

        transport.send_json(DescribeRequest())
        transport.recv(timeout=2.0)  # describe_response

        for _ in range(50):
            msg = transport.recv(timeout=2.0)
            if isinstance(msg, BinaryFrame):
                break
        else:
            pytest.fail("No binary frame received")

        # The JPEG should start with the SOI marker.
        assert msg.jpeg_bytes[:2] == b"\xff\xd8"
        # And end with EOI.
        assert msg.jpeg_bytes[-2:] == b"\xff\xd9"
        transport.close()

    def test_close_cleanly(self, mock_server) -> None:
        """Closing the transport must not raise."""
        transport = WebSocketTransport(mock_server.url)
        transport.connect()
        transport.close()
        assert not transport.is_connected

    def test_connection_refused_raises(self) -> None:
        """Connecting to a port where nothing is listening must raise
        RobotNotFoundError."""
        transport = WebSocketTransport("ws://127.0.0.1:19999")
        with pytest.raises(RobotNotFoundError):
            transport.connect()

    def test_disconnect_detected(self, mock_server) -> None:
        """If the server stops, recv() must raise RobotDisconnectedError."""
        transport = WebSocketTransport(mock_server.url)
        transport.connect()

        transport.send_json(DescribeRequest())
        transport.recv(timeout=2.0)  # describe_response

        # Stop the server.
        mock_server.stop()

        # Next recv should eventually fail.
        with pytest.raises(RobotDisconnectedError):
            # The mock might have already sent some observations into the
            # queue before stopping.  Drain them, then the next recv fails.
            for _ in range(100):
                transport.recv(timeout=0.5)

    def test_rapid_send_recv(self, mock_server) -> None:
        """Rapid send/recv should not drop messages or corrupt state."""
        transport = WebSocketTransport(mock_server.url)
        transport.connect()

        transport.send_json(DescribeRequest())
        transport.recv(timeout=2.0)  # describe_response

        # Send several actions in quick succession.
        for i in range(10):
            action = Action(data={"follower/joint_position": [float(i)] * 6})
            transport.send_json(action)

        # We should still be able to receive observations.
        obs = transport.recv(timeout=2.0)
        assert isinstance(obs, Observation)
        transport.close()
