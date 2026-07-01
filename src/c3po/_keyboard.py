"""Keyboard listener for interactive recording controls.

Captures single-key presses (``q`` / ``n`` / ``r``) from the terminal
and sets boolean flags on a shared object.  Used by the ``Recording``
context manager to let the researcher control episode boundaries
during data collection.

Pure stdlib on Unix (``termios`` + ``tty`` + ``select``).
Graceful no-op on Windows.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------

_IS_UNIX = sys.platform != "win32"

if _IS_UNIX:
    import atexit
    import select
    import termios
    import tty

# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------


class KeyboardListener:
    """Background thread that reads single-char key presses from stdin.

    Parameters
    ----------
    input_fd:
        File descriptor to read from.  Defaults to ``sys.stdin.fileno()``.
        Override for testing with a pipe.

    Attributes
    ----------
    stop:
        ``True`` after the user presses ``q`` (quit / stop recording).
        Terminal — never reset by :meth:`clear`.
    done:
        ``True`` after the user presses ``n`` (end current episode).
        Reset by :meth:`clear`.
    rerecord:
        ``True`` after the user presses ``r`` (discard and re-record).
        Reset by :meth:`clear`.
    """

    def __init__(self, input_fd: int | None = None) -> None:
        self._input_fd: int | None = input_fd  # resolved lazily in start()
        self._thread: threading.Thread | None = None
        self._running = False
        self._orig_attrs: list | None = None
        self._lock = threading.Lock()

        # Flags — set by key presses.
        self._stop = False
        self._done = False
        self._rerecord = False

    # ------------------------------------------------------------------
    # Public flags (read-only properties for external consumers)
    # ------------------------------------------------------------------

    @property
    def stop(self) -> bool:
        with self._lock:
            return self._stop

    @stop.setter
    def stop(self, value: bool) -> None:
        with self._lock:
            self._stop = value

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    @done.setter
    def done(self, value: bool) -> None:
        with self._lock:
            self._done = value

    @property
    def rerecord(self) -> bool:
        with self._lock:
            return self._rerecord

    @rerecord.setter
    def rerecord(self, value: bool) -> None:
        with self._lock:
            self._rerecord = value

    def clear(self) -> None:
        """Reset ``done`` and ``rerecord`` flags.  ``stop`` is never reset."""
        with self._lock:
            self._done = False
            self._rerecord = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> KeyboardListener:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop_listener()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background listener thread."""
        if not _IS_UNIX:
            return  # Windows: no-op

        if not sys.stdin.isatty():
            logger.debug("stdin is not a TTY — keyboard listener disabled")
            return

        # Resolve the input fd lazily (so __init__ works under pytest).
        if self._input_fd is None:
            self._input_fd = sys.stdin.fileno()

        self._running = True

        # Save terminal settings and switch to raw mode.
        self._orig_attrs = termios.tcgetattr(self._input_fd)
        tty.setraw(self._input_fd)

        # Restore terminal on normal exit.
        atexit.register(self._restore_terminal)

        self._thread = threading.Thread(
            target=self._listen, daemon=True, name="c3po-kbd"
        )
        self._thread.start()

    def stop_listener(self) -> None:
        """Stop the listener thread and restore terminal settings."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        self._restore_terminal()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _listen(self) -> None:
        """Background loop: read stdin byte-by-byte and dispatch."""
        if self._input_fd is None:
            return
        try:
            while self._running:
                # Non-blocking check: is there data on stdin?
                ready, _, _ = select.select([self._input_fd], [], [], 0.1)
                if not ready:
                    continue
                char = os.read(self._input_fd, 1)
                if char:
                    self._handle_key(char.decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            pass  # fd closed or terminal gone

    def _handle_key(self, key: str) -> None:
        """Map a single-character key press to a flag."""
        key = key.lower()
        with self._lock:
            if key == "q":
                self._stop = True
            elif key == "n":
                self._done = True
            elif key == "r":
                self._rerecord = True
            # Other keys are intentionally ignored.

    def _restore_terminal(self) -> None:
        """Restore the terminal to its original settings."""
        if self._orig_attrs is None:
            return
        try:
            termios.tcsetattr(self._input_fd, termios.TCSADRAIN, self._orig_attrs)
        except (termios.error, OSError):
            pass
        self._orig_attrs = None
