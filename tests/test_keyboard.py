"""Tests for the keyboard listener."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest
from c3po._keyboard import KeyboardListener


def _has_tty() -> bool:
    """Return True if stdin is a real TTY (skip keyboard tests otherwise)."""
    return sys.stdin.isatty()


class TestKeyboardListenerParser:
    """Tests for key → flag mapping logic (no TTY required)."""

    @staticmethod
    def _make_listener() -> KeyboardListener:
        kl = KeyboardListener()
        # Don't start the background thread — just test the parser.
        return kl

    def test_q_sets_stop(self) -> None:
        kl = self._make_listener()
        kl.simulate_key("q")
        assert kl.stop is True
        assert kl.done is False
        assert kl.rerecord is False

    def test_n_sets_done(self) -> None:
        kl = self._make_listener()
        kl.simulate_key("n")
        assert kl.stop is False
        assert kl.done is True
        assert kl.rerecord is False

    def test_r_sets_rerecord(self) -> None:
        kl = self._make_listener()
        kl.simulate_key("r")
        assert kl.stop is False
        assert kl.done is False
        assert kl.rerecord is True

    def test_uppercase_keys_work(self) -> None:
        kl = self._make_listener()
        kl.simulate_key("Q")
        assert kl.stop is True
        kl.simulate_key("N")
        assert kl.done is True
        kl.simulate_key("R")
        assert kl.rerecord is True

    def test_unknown_keys_ignored(self) -> None:
        kl = self._make_listener()
        for key in ("a", "z", "1", " ", "\x1b"):
            kl.simulate_key(key)
        assert kl.stop is False
        assert kl.done is False
        assert kl.rerecord is False

    def test_clear_resets_done_and_rerecord(self) -> None:
        kl = self._make_listener()
        kl.simulate_key("n")
        kl.simulate_key("r")
        assert kl.done is True
        assert kl.rerecord is True
        kl.clear()
        assert kl.done is False
        assert kl.rerecord is False
        # stop is terminal — never reset by clear().
        assert kl.stop is False  # (we didn't press q)

    def test_stop_is_terminal(self) -> None:
        """stop is never reset by clear()."""
        kl = self._make_listener()
        kl.simulate_key("q")
        assert kl.stop is True
        kl.clear()
        assert kl.stop is True  # stop persists

    def test_multiple_keys_accumulate(self) -> None:
        """Pressing n then q should set both flags."""
        kl = self._make_listener()
        kl.simulate_key("n")
        kl.simulate_key("q")
        assert kl.done is True
        assert kl.stop is True


class TestKeyboardListenerTTY:
    """Tests that require a real TTY (integration tests)."""

    @pytest.mark.skipif(not _has_tty(), reason="stdin is not a TTY")
    def test_context_manager_does_not_crash(self) -> None:
        """Context manager should not raise on a real TTY."""
        with KeyboardListener() as kl:
            # Just verify the thread started.
            assert kl._thread is not None
            assert kl._thread.is_alive()
        # After exit, thread should be stopped.
        assert not kl._thread.is_alive()

    @pytest.mark.skipif(not _has_tty(), reason="stdin is not a TTY")
    def test_terminal_restored_after_context(self) -> None:
        """Terminal settings should be restored after the context block."""
        import termios

        orig = termios.tcgetattr(sys.stdin.fileno())
        with KeyboardListener():
            pass
        restored = termios.tcgetattr(sys.stdin.fileno())
        # The terminal attributes should be identical.
        assert orig == restored
