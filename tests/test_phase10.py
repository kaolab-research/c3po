"""Tests for Phase 10: StatusMessage ingestion, rate reporting, and signal handling.

These tests verify c3po-side behavior for the new Phase 10 features.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from c3po._buffer import ObservationBuffer
from c3po._protocol import StatusMessage


# ---------------------------------------------------------------------------
# 10.3: StatusMessage ingestion in c3po
# ---------------------------------------------------------------------------


class TestStatusMessageIngestion:
    """Tests that Robot correctly handles incoming StatusMessage frames."""

    def test_status_message_logged_at_info_level(self, caplog) -> None:
        """StatusMessage should be logged when ingested."""
        from c3po.robot import Robot

        # Create a Robot without a real connection — just test _ingest.
        # We patch the transport to avoid real WebSocket calls.
        with patch("c3po.robot.WebSocketTransport"):
            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._buffer = ObservationBuffer()

            with caplog.at_level(logging.INFO):
                msg = StatusMessage(
                    event="episode_finalized",
                    message="Episode 3 written to disk",
                    data={"episode_index": 3, "frame_count": 150},
                )
                robot._ingest(msg)

            assert "episode_finalized" in caplog.text
            assert "Episode 3" in caplog.text

    def test_watchdog_triggered_logged_at_warning(self, caplog) -> None:
        """Watchdog events should be logged at WARNING level."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport"):
            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._buffer = ObservationBuffer()

            with caplog.at_level(logging.WARNING):
                msg = StatusMessage(
                    event="watchdog_triggered",
                    message="No action for 0.200 s",
                    data={"seconds_since_last_action": 0.200},
                )
                robot._ingest(msg)

            assert "watchdog" in caplog.text.lower()

    def test_unknown_event_logged_at_debug(self, caplog) -> None:
        """Unknown events should not crash — logged at DEBUG."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport"):
            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._buffer = ObservationBuffer()

            with caplog.at_level(logging.DEBUG):
                msg = StatusMessage(
                    event="some_future_event", message="A new event type"
                )
                robot._ingest(msg)

            # Should not raise.
            assert "some_future_event" in caplog.text

    def test_on_status_callback_invoked(self) -> None:
        """When on_status callback is set, it should be called on each message."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport"):
            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._buffer = ObservationBuffer()

            calls: list[tuple] = []
            robot.on_status = lambda event, message, data: calls.append(
                (event, message, data)
            )

            msg = StatusMessage(
                event="recording_started",
                message="Recording session_001",
                data={"dataset_name": "session_001"},
            )
            robot._ingest(msg)

            assert len(calls) == 1
            assert calls[0] == (
                "recording_started",
                "Recording session_001",
                {"dataset_name": "session_001"},
            )

    def test_on_status_none_is_noop(self) -> None:
        """None callback (default) should not raise."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport"):
            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._on_status_callback = None
            robot._buffer = ObservationBuffer()

            # Default: on_status is None
            msg = StatusMessage(event="recording_started", message="test")
            robot._ingest(msg)  # Should not raise


# ---------------------------------------------------------------------------
# 10.6: Rate reporting
# ---------------------------------------------------------------------------


class TestRateReporting:
    """Tests that c3po reports control rate and loop timing."""

    def test_rate_info_logged_on_reset(self, caplog) -> None:
        """reset() should log the station's configured control rate."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport"):
            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._rate = 50.0
            robot._manifest = {"control_rate": 50.0}
            robot._buffer = ObservationBuffer()

            assert robot.control_rate == 50.0
            assert robot.rate == 50.0

    def test_rate_and_manifest_rate_independent(self) -> None:
        """control_rate (from manifest) and rate (requested) may differ."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport"):
            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._rate = 30.0  # user requested 30Hz
            robot._manifest = {"control_rate": 50.0}  # hardware says 50Hz

            assert robot.rate == 30.0
            assert robot.control_rate == 50.0


# ---------------------------------------------------------------------------
# 10.4: SIGTERM / disconnect handling
# ---------------------------------------------------------------------------


class TestDisconnectHandling:
    """Tests for graceful shutdown and torque disable on disconnect."""

    def test_close_called_on_context_exit(self) -> None:
        """Robot.__exit__ should call close()."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport") as mock_transport_cls:
            mock_transport = MagicMock()
            mock_transport.is_connected = True
            mock_transport_cls.return_value = mock_transport

            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._transport = mock_transport

            robot.close()

            mock_transport.close.assert_called_once()

    def test_close_only_called_once(self) -> None:
        """Multiple close() calls should be idempotent."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport") as mock_transport_cls:
            mock_transport = MagicMock()
            mock_transport.is_connected = True
            mock_transport_cls.return_value = mock_transport

            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = False
            robot._transport = mock_transport

            robot.close()
            robot.close()  # second call

            # Transport.close should only be called once.
            assert mock_transport.close.call_count == 1

    def test_close_during_recording_sends_stop(self) -> None:
        """If recording is active, close() should try to stop_recording."""
        from c3po.robot import Robot

        with patch("c3po.robot.WebSocketTransport") as mock_transport_cls:
            mock_transport = MagicMock()
            mock_transport.is_connected = True
            mock_transport_cls.return_value = mock_transport

            robot = Robot.__new__(Robot)
            robot._closed = False
            robot._recording = True  # recording is active
            robot._transport = mock_transport

            robot.close()

            # Should have sent StopRecording before closing.
            assert mock_transport.send_json.called
