"""Human-readable formatting for SpecResponse.

Usage::

    from c3po import Robot
    from c3po._spec import format_spec

    with Robot() as robot:
        print(format_spec(robot.spec()))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c3po._protocol import SpecResponse


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m {secs}s"


def _fmt_resolution(res: list[int]) -> str:
    """Format [height, width] as a concise string."""
    h, w = res[0], res[1]
    if h >= 720:
        return f"{h}p"
    if h == 480 and w <= 854:
        return "480p"
    return f"{w}×{h}"


def format_spec(spec: SpecResponse) -> str:
    """Return a human-readable multiline string from a SpecResponse.

    Parameters
    ----------
    spec:
        A :class:`SpecResponse` returned by :meth:`Robot.spec`.

    Returns
    -------
    str
        Formatted string suitable for printing.
    """
    lines = []
    sep = "─" * 42

    lines.append(f"╭{sep}╮")
    lines.append(f"│ {'r2d2 Station Spec':^40} │")
    lines.append(f"├{sep}┤")

    # Uptime and rate.
    rate_str = f"{spec.effective_control_rate:.1f} Hz"
    lines.append(f"│ {'Uptime:':<14} {_fmt_duration(spec.uptime_seconds):<24} │")
    lines.append(f"│ {'Control rate:':<14} {rate_str:<24} │")

    # Cameras.
    lines.append(f"│ {'Cameras:':<40} │")
    if not spec.cameras:
        lines.append(f"│   (none){' ':<33} │")
    else:
        for cam in spec.cameras:
            name = cam["name"]
            cap = _fmt_resolution(cam.get("capture_resolution", [0, 0]))
            stream = _fmt_resolution(cam.get("stream_resolution", [0, 0]))
            fps = cam.get("effective_fps", 0.0)
            sent = cam.get("frames_sent", 0)
            arrow = "→" if cap != stream else "="
            cam_line = (
                f"  {name:<15} {cap} {arrow} {stream} @ {fps:.1f} fps"
            )
            # Pad to 40 chars.
            cam_line = cam_line[:38]
            lines.append(f"│ {cam_line:<40} │")
            lines.append(f"│   ({sent} frames sent){' ':<21} │")

    # Recording.
    if spec.recording is not None:
        rec = spec.recording
        lines.append(f"│ {'Recording:':<40} │")
        lines.append(
            f"│   {rec['dataset_name']:<19} "
            f"ep={rec['total_episodes']} fr={rec['total_frames']}"
            f"{' ':<3} │"
        )
    else:
        lines.append(f"│ {'Recording:':<14} (not recording){' ':<22} │")

    # Watchdog.
    wd_str = f"{spec.watchdog_trigger_count} trigger(s)"
    lines.append(f"│ {'Watchdog:':<14} {wd_str:<24} │")

    lines.append(f"╰{sep}╯")
    return "\n".join(lines)
