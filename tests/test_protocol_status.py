"""Tests for StatusMessage — server→client asynchronous notifications.

Phase 10 adds a StatusMessage type for r2d2 to push lifecycle events
(episode finalized, recording started, watchdog triggered, etc.) to
c3po outside the control loop cadence.
"""

from __future__ import annotations

import pytest

from c3po._protocol import (
    ProtocolError,
    StatusMessage,
    decode_json_message,
    encode_json_message,
)


class TestStatusMessage:
    """Roundtrip and validation tests for StatusMessage."""

    def test_roundtrip_basic(self) -> None:
        msg = StatusMessage(event="episode_finalized", message="Episode 3 written to disk")
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert isinstance(decoded, StatusMessage)
        assert decoded.event == "episode_finalized"
        assert decoded.message == "Episode 3 written to disk"
        assert decoded.data == {}

    def test_roundtrip_with_data(self) -> None:
        msg = StatusMessage(
            event="episode_finalized",
            message="Episode 5 completed",
            data={"episode_index": 5, "frame_count": 250, "duration_ms": 5.2},
        )
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert isinstance(decoded, StatusMessage)
        assert decoded.event == "episode_finalized"
        assert decoded.data == {
            "episode_index": 5,
            "frame_count": 250,
            "duration_ms": 5.2,
        }

    def test_roundtrip_empty_data(self) -> None:
        """data field defaults to empty dict."""
        msg = StatusMessage(event="recording_started", message="Recording session_001")
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.data == {}

    def test_known_events(self) -> None:
        """All documented event strings roundtrip correctly."""
        events = [
            "episode_finalized",
            "recording_started",
            "recording_stopped",
            "watchdog_triggered",
            "cycle_overrun",
            "dataset_ready",
        ]
        for event in events:
            msg = StatusMessage(event=event, message=f"Test {event}")
            raw = encode_json_message(msg)
            decoded = decode_json_message(raw)
            assert decoded.event == event

    def test_type_field_is_status(self) -> None:
        msg = StatusMessage(event="test", message="hello")
        raw = encode_json_message(msg)
        assert '"type"' in raw and '"status"' in raw

    def test_missing_event_raises(self) -> None:
        import json

        raw = json.dumps({"type": "status", "message": "no event field"})
        with pytest.raises(ProtocolError):
            decode_json_message(raw)
