"""Tests for c3po Robot.spec() method and format_spec() formatter.

Phase 11: runtime introspection.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from c3po._protocol import SpecRequest, SpecResponse


class TestRobotSpec:
    """Tests for Robot.spec() — sends SpecRequest, returns SpecResponse."""

    def test_spec_sends_request_and_returns_response(self) -> None:
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport") as mock_cls:
            mock_transport = MagicMock()
            mock_transport.is_connected = True
            mock_transport.recv.return_value = SpecResponse(
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
            mock_cls.return_value = mock_transport

            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._transport = mock_transport

            result = robot.spec()

            # Verify SpecRequest was sent.
            assert mock_transport.send_json.called
            sent_msg = mock_transport.send_json.call_args[0][0]
            assert isinstance(sent_msg, SpecRequest)

            # Verify response.
            assert isinstance(result, SpecResponse)
            assert result.effective_control_rate == pytest.approx(48.7)
            assert len(result.cameras) == 1
            assert result.recording is not None

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


class TestFormatSpec:
    """Tests for format_spec() — human-readable spec output."""

    def test_format_includes_key_sections(self) -> None:
        from c3po._spec import format_spec

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
        output = format_spec(spec)

        assert "r2d2" in output.lower()
        assert "48.7" in output
        assert "1h 1m" in output  # 3662s = 1h 1m 2s
        assert "front_rgb" in output
        assert "720" in output and "480" in output
        assert "28.3" in output
        assert "12345" in output
        assert "session_001" in output
        assert "5" in output
        assert "1250" in output

    def test_format_no_cameras(self) -> None:
        from c3po._spec import format_spec

        spec = SpecResponse(
            effective_control_rate=50.0,
            uptime_seconds=10.0,
            cameras=[],
            recording=None,
            watchdog_trigger_count=0,
        )
        output = format_spec(spec)
        assert "(none)" in output.lower()
        assert "not recording" in output.lower()

    def test_format_watchdog_triggers(self) -> None:
        from c3po._spec import format_spec

        spec = SpecResponse(
            effective_control_rate=50.0,
            uptime_seconds=10.0,
            cameras=[],
            recording=None,
            watchdog_trigger_count=3,
        )
        output = format_spec(spec)
        assert "3" in output
        assert "trigger" in output.lower()
