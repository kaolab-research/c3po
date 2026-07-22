"""Tests for SpecRequest / SpecResponse protocol messages.

Phase 11 adds runtime introspection: c3po can request live metrics
(effective control rate, camera stats, recording state, watchdog count)
from r2d2 via a spec_request / spec_response exchange.
"""

from __future__ import annotations

import pytest

from c3po._protocol import (
    ProtocolError,
    SpecRequest,
    SpecResponse,
    decode_json_message,
    encode_json_message,
)


class TestSpecRequest:
    def test_roundtrip(self) -> None:
        msg = SpecRequest()
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert isinstance(decoded, SpecRequest)
        assert decoded.type == "spec_request"


class TestSpecResponse:
    def test_roundtrip_with_cameras(self) -> None:
        msg = SpecResponse(
            effective_control_rate=48.7,
            uptime_seconds=3600.0,
            cameras=[
                {
                    "name": "front_rgb",
                    "capture_resolution": [720, 1280],
                    "stream_resolution": [480, 854],
                    "effective_fps": 28.3,
                    "frames_sent": 12345,
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
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert isinstance(decoded, SpecResponse)
        assert decoded.effective_control_rate == pytest.approx(48.7)
        assert decoded.uptime_seconds == pytest.approx(3600.0)
        assert len(decoded.cameras) == 1
        assert decoded.cameras[0]["name"] == "front_rgb"
        assert decoded.cameras[0]["effective_fps"] == pytest.approx(28.3)
        assert decoded.recording is not None
        assert decoded.recording["active"] is True
        assert decoded.watchdog_trigger_count == 0

    def test_roundtrip_with_null_recording(self) -> None:
        """recording is None when no recording is active."""
        msg = SpecResponse(
            effective_control_rate=50.0,
            uptime_seconds=10.0,
            cameras=[],
            recording=None,
            watchdog_trigger_count=0,
        )
        raw = encode_json_message(msg)
        decoded = decode_json_message(raw)
        assert decoded.recording is None

    def test_missing_required_field_raises(self) -> None:
        import json

        raw = json.dumps({"type": "spec_response", "effective_control_rate": 50.0})
        with pytest.raises(ProtocolError):
            decode_json_message(raw)
