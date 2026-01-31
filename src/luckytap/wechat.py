"""Find and activate the WeChat window via AppleScript and Quartz."""

from __future__ import annotations

import logging
import subprocess

import Quartz

log = logging.getLogger(__name__)


def is_wechat_running() -> bool:
    """Check whether WeChat is currently running."""
    apps = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )
    return any(w.get("kCGWindowOwnerName") == "WeChat" for w in apps)


def get_wechat_window() -> tuple[int, int, int, int] | None:
    """Return (x, y, width, height) of the main WeChat window, or None."""
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )
    for w in windows:
        if w.get("kCGWindowOwnerName") != "WeChat":
            continue
        bounds = w.get("kCGWindowBounds")
        if not bounds:
            continue
        width = int(bounds["Width"])
        height = int(bounds["Height"])
        # Skip tiny windows (menu-bar extras, tooltips, etc.)
        if width < 200 or height < 200:
            continue
        return (
            int(bounds["X"]),
            int(bounds["Y"]),
            width,
            height,
        )
    return None


def activate_wechat() -> None:
    """Bring WeChat to the front using NSRunningApplication."""
    from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace

    workspace = NSWorkspace.sharedWorkspace()
    for app in workspace.runningApplications():
        if app.localizedName() == "WeChat":
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            return
    # Fallback to AppleScript if NSWorkspace doesn't find it
    subprocess.run(
        ["osascript", "-e", 'tell application "WeChat" to activate'],
        check=False,
        capture_output=True,
    )
