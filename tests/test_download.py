"""Tests for Phase 12.3: Robot.download_dataset().

Uses a lightweight mock HTTP server that serves a fake LeRobot dataset
tree, plus the mock WebSocket transport to provide dataset_ready URLs.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socketserver
import tempfile
import threading
import time
from pathlib import Path

import pytest

from c3po import Robot


# ---------------------------------------------------------------------------
# Mock HTTP server serving a fake LeRobot dataset
# ---------------------------------------------------------------------------


def _make_fake_dataset(root: Path, name: str) -> Path:
    """Create a minimal fake LeRobot dataset directory tree."""
    ds = root / name
    ds.mkdir(parents=True)

    # data/chunk-000/episode_000000.parquet
    chunk = ds / "data" / "chunk-000"
    chunk.mkdir(parents=True)
    (chunk / "episode_000000.parquet").write_bytes(
        b"PARQUET" + os.urandom(100)
    )

    # meta/info.json
    meta = ds / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "robot_type": "so101_teleop",
        "fps": 30.0,
        "total_episodes": 1,
        "total_frames": 100,
    }))

    # videos/chunk-000/observation.images.front_rgb/episode_000000.mp4
    vid_dir = ds / "videos" / "chunk-000" / "observation.images.front_rgb"
    vid_dir.mkdir(parents=True)
    (vid_dir / "episode_000000.mp4").write_bytes(
        b"MP4" + os.urandom(200)
    )

    return ds


class _MockHTTPServer:
    """Tiny HTTP server serving a directory on a random port."""

    def __init__(self, directory: str) -> None:
        import functools

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=directory,
        )
        self._httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dataset() -> Path:
    """Create a temporary fake LeRobot dataset served over HTTP."""
    tmp = tempfile.mkdtemp()
    root = Path(tmp)
    _make_fake_dataset(root, "test_ds")
    return root


@pytest.fixture
def fake_ds_url(fake_dataset: Path) -> str:
    """Return a string that download_dataset() can use as a URL.

    download_dataset constructs the URL from _dataset_urls or falls back
    to host-based construction.  For testing we inject the URL directly.
    """
    # Not used directly — the test will construct the full URL
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDownloadDataset:
    """Tests for Robot.download_dataset()."""

    @pytest.fixture(autouse=True)
    def _setup(self, fake_dataset: Path, mock_server) -> None:
        """Start an HTTP server serving the fake dataset."""
        self._ds_root = fake_dataset
        self._http = _MockHTTPServer(str(fake_dataset))
        self._http.start()
        self._mock = mock_server
        time.sleep(0.1)  # let HTTP server start

    def teardown_method(self) -> None:
        if hasattr(self, "_http"):
            self._http.stop()

    def _make_robot(self) -> Robot:
        """Create a Robot connected to the mock WebSocket server."""
        robot = Robot(host=self._mock.host, port=self._mock.port)
        # Inject the dataset URL so download_dataset can find it.
        robot._dataset_urls["test_ds"] = (
            f"http://127.0.0.1:{self._http.port}/test_ds/"
        )
        return robot

    def test_download_creates_directory_tree(self) -> None:
        """download_dataset should recreate the full LeRobot tree locally."""
        robot = self._make_robot()
        with tempfile.TemporaryDirectory() as dest:
            result = robot.download_dataset("test_ds", dest, show_progress=False)
            assert result.is_dir()
            assert (result / "meta" / "info.json").is_file()
            assert (result / "data" / "chunk-000" / "episode_000000.parquet").is_file()
            vid = (
                result
                / "videos"
                / "chunk-000"
                / "observation.images.front_rgb"
                / "episode_000000.mp4"
            )
            assert vid.is_file()
        robot.close()

    def test_download_uses_client_host_not_advertised_host(self) -> None:
        """The advertised URL's host is the station's static IP — the client
        must download via the host it actually reached r2d2 on (regression:
        --record-dest downloads silently following the wrong host)."""
        robot = Robot(host=self._mock.host, port=self._mock.port)
        robot._dataset_urls["test_ds"] = (
            f"http://10.42.0.1:{self._http.port}/test_ds/"  # unreachable host
        )
        with tempfile.TemporaryDirectory() as dest:
            result = robot.download_dataset("test_ds", dest, show_progress=False)
            assert (result / "meta" / "info.json").is_file()
        robot.close()

    def test_download_parquet_content_matches(self) -> None:
        """Downloaded files should have the same content as the originals."""
        robot = self._make_robot()
        with tempfile.TemporaryDirectory() as dest:
            result = robot.download_dataset("test_ds", dest, show_progress=False)
            downloaded = (
                result / "data" / "chunk-000" / "episode_000000.parquet"
            ).read_bytes()
            original = (
                self._ds_root
                / "test_ds"
                / "data"
                / "chunk-000"
                / "episode_000000.parquet"
            ).read_bytes()
            assert downloaded == original
        robot.close()

    def test_download_video_content_matches(self) -> None:
        """Downloaded MP4 files should match."""
        robot = self._make_robot()
        with tempfile.TemporaryDirectory() as dest:
            result = robot.download_dataset("test_ds", dest, show_progress=False)
            downloaded = (
                result
                / "videos"
                / "chunk-000"
                / "observation.images.front_rgb"
                / "episode_000000.mp4"
            ).read_bytes()
            original = (
                self._ds_root
                / "test_ds"
                / "videos"
                / "chunk-000"
                / "observation.images.front_rgb"
                / "episode_000000.mp4"
            ).read_bytes()
            assert downloaded == original
        robot.close()

    def test_missing_dataset_raises_runtime_error(self) -> None:
        """Requesting a non-existent dataset should raise RuntimeError."""
        robot = self._make_robot()
        robot._dataset_urls["bad_ds"] = (
            f"http://127.0.0.1:{self._http.port}/nonexistent/"
        )
        with tempfile.TemporaryDirectory() as dest:
            with pytest.raises(RuntimeError, match="Failed to reach"):
                robot.download_dataset("bad_ds", dest, show_progress=False)
        robot.close()

    def test_download_with_default_dest(self, tmp_path: Path) -> None:
        """Calling with only the dataset name defaults to cwd."""
        robot = self._make_robot()
        cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = robot.download_dataset("test_ds", show_progress=False)
            assert result.resolve() == (tmp_path / "test_ds").resolve()
            assert result.is_dir()
            assert (result / "meta" / "info.json").is_file()
        finally:
            os.chdir(cwd)
        robot.close()
