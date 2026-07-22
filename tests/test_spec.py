"""Tests for c3po Robot.spec() method.

Phase 11: runtime introspection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from c3po._protocol import SpecRequest, SpecResponse


class TestRobotSpec:
    """Tests for Robot.spec() — sends SpecRequest, returns SpecResponse."""

    def test_spec_sends_request_and_returns_response(self) -> None:
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport") as mock_cls:
            expected = SpecResponse(
                effective_control_rate=48.7,
                uptime_seconds=3600.0,
                cameras=[
                    {
                        "name": "front_rgb",
                        "capture_resolution": [720, 1280],
                        "stream_resolution": [480, 854],
                        "effective_fps": 28.3,
                        "frames_sent": 12345,
                    }
                ],
                recording={
                    "active": True,
                    "dataset_name": "test",
                    "total_episodes": 5,
                    "total_frames": 1250,
                },
                watchdog_trigger_count=0,
            )
            mock_transport = MagicMock()
            mock_transport.is_connected = True
            mock_transport.recv.return_value = expected
            mock_cls.return_value = mock_transport

            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._transport = mock_transport

            result = robot.spec()

            assert mock_transport.send_json.called
            sent_msg = mock_transport.send_json.call_args[0][0]
            assert isinstance(sent_msg, SpecRequest)
            assert isinstance(result, SpecResponse)
            assert result.effective_control_rate == pytest.approx(48.7)

    def test_spec_when_not_recording(self) -> None:
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport") as mock_cls:
            mock_transport = MagicMock()
            mock_transport.is_connected = True
            mock_transport.recv.return_value = SpecResponse(
                effective_control_rate=50.0,
                uptime_seconds=120.0,
                cameras=[],
                recording=None,
                watchdog_trigger_count=0,
            )
            mock_cls.return_value = mock_transport

            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._transport = mock_transport

            result = robot.spec()
            assert result.recording is None

    def test_spec_drains_stale_messages(self) -> None:
        """spec() drains Observations until it finds the SpecResponse."""
        from c3po.robot import Robot
        from c3po._protocol import Observation

        with patch("c3po.robot.WebSocketTransport") as mock_cls:
            expected = SpecResponse(
                effective_control_rate=48.7,
                uptime_seconds=100.0,
                cameras=[],
                recording=None,
                watchdog_trigger_count=0,
            )
            # Simulate: stale Observation arrives first, then SpecResponse.
            mock_transport = MagicMock()
            mock_transport.is_connected = True
            mock_transport.recv.side_effect = [
                Observation(data={"joint_position": [0.1]}),
                expected,
            ]
            mock_cls.return_value = mock_transport

            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._transport = mock_transport

            result = robot.spec()
            assert isinstance(result, SpecResponse)
            assert result.effective_control_rate == pytest.approx(48.7)
            # recv() should have been called twice (drained the stale message).
            assert mock_transport.recv.call_count == 2


class TestSpecStr:
    """Tests for SpecResponse.__str__ — human-readable formatting."""

    def test_str_includes_key_sections(self) -> None:
        spec = SpecResponse(
            effective_control_rate=48.7,
            uptime_seconds=3662.0,
            cameras=[
                {
                    "name": "front_rgb",
                    "capture_resolution": [720, 1280],
                    "stream_resolution": [480, 854],
                    "effective_fps": 28.3,
                    "frames_sent": 12345,
                },
                {
                    "name": "wrist_rgb",
                    "capture_resolution": [480, 640],
                    "stream_resolution": [480, 640],
                    "effective_fps": 29.8,
                    "frames_sent": 12300,
                },
            ],
            recording={
                "active": True,
                "dataset_name": "session_001",
                "total_episodes": 5,
                "total_frames": 1250,
            },
            watchdog_trigger_count=0,
        )
        output = str(spec)

        assert "r2d2" in output.lower()
        assert "48.7" in output
        assert "1h 1m" in output
        assert "front_rgb" in output
        assert "720p" in output and "480p" in output
        assert "28.3" in output
        assert "12345" in output
        assert "session_001" in output
        assert "5 episodes" in output
        assert "1250 frames" in output

    def test_str_no_cameras(self) -> None:
        spec = SpecResponse(
            effective_control_rate=50.0,
            uptime_seconds=10.0,
            cameras=[],
            recording=None,
            watchdog_trigger_count=0,
        )
        output = str(spec)
        assert "(none)" in output
        assert "not recording" in output

    def test_str_watchdog_triggers(self) -> None:
        spec = SpecResponse(
            effective_control_rate=50.0,
            uptime_seconds=10.0,
            cameras=[],
            recording=None,
            watchdog_trigger_count=3,
        )
        output = str(spec)
        assert "3 trigger" in output
