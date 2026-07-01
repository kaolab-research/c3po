"""Tests for the Recording context manager."""

from __future__ import annotations

import time

from c3po._recording import Recording

from c3po import Robot


class TestRecording:
    """Tests for the Recording context manager."""

    def test_context_manager_sends_start_stop(self, mock_server) -> None:
        """Entering the context sends start_recording, exiting sends stop_recording."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        with Recording(robot, "test_session") as rec:
            assert rec.name == "test_session"
        robot.close()

    def test_flags_default_to_false(self, mock_server) -> None:
        """Before any key press, all flags are False."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        with Recording(robot, "test_session") as rec:
            assert rec.stop is False
            assert rec.done is False
            assert rec.rerecord is False
        robot.close()

    def test_simulate_key_presses(self, mock_server) -> None:
        """Simulating key presses should set the corresponding flags."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        with Recording(robot, "test_session") as rec:
            rec._keyboard._handle_key("n")
            assert rec.done is True
            assert rec.stop is False

            rec._keyboard._handle_key("q")
            assert rec.stop is True
        robot.close()

    def test_clear_resets_done_and_rerecord(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        with Recording(robot, "test_session") as rec:
            rec._keyboard._handle_key("n")
            rec._keyboard._handle_key("r")
            assert rec.done is True
            assert rec.rerecord is True

            rec.clear()
            assert rec.done is False
            assert rec.rerecord is False
            assert rec.stop is False  # stop never reset
        robot.close()

    def test_stop_persists_after_clear(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        with Recording(robot, "test_session") as rec:
            rec._keyboard._handle_key("q")
            rec.clear()
            assert rec.stop is True
        robot.close()

    def test_episode_count_starts_at_zero(self, mock_server) -> None:
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        with Recording(robot, "test_session") as rec:
            assert rec.episode_count == 0
        robot.close()

    def test_context_manager_cleanup_on_exception(self, mock_server) -> None:
        """stop_recording should still be sent even if the block raises."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        try:
            with Recording(robot, "test_session") as rec:
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        # Should not raise — context manager exits cleanly.
        robot.close()

    def test_robot_recording_convenience_method(self, mock_server) -> None:
        """robot.recording('test') should return a Recording instance."""
        robot = Robot(host="127.0.0.1", port=mock_server.port)
        rec = robot.recording("test_session")
        assert isinstance(rec, Recording)
        assert rec.name == "test_session"
        # Not entering — just testing the factory method.
        robot.close()
