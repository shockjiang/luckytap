"""CGEvent mouse clicks and keyboard events."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import Quartz

log = logging.getLogger(__name__)

CLICK_MARKER_SIZE = 15  # half-size of the marker box in pixels


def click(x: int, y: int, purpose: str = "", frame: np.ndarray | None = None) -> None:
    """Perform a mouse click at the given screen coordinates."""
    point = Quartz.CGPointMake(x, y)
    log.debug("Click at (%d, %d) in order to: [%s]", x, y, purpose)

    if frame is not None:
        vis = frame.copy()
        s = CLICK_MARKER_SIZE
        cv2.rectangle(vis, (x - s, y - s), (x + s, y + s), (0, 0, 255), 2)
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, s, 2)
        fname = f"click-{purpose.replace(' ', '_')}.jpg"
        cv2.imwrite(fname, vis)
        log.debug("Saved click visualization to %s", fname)

    # Mouse down
    event = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    time.sleep(0.02)

    # Mouse up
    event = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def scroll(x: int, y: int, dy: int) -> None:
    """Scroll at the given screen coordinates.

    dy > 0 scrolls up (towards older messages), dy < 0 scrolls down.
    """
    log.debug("Scroll at (%d, %d) dy=%d", x, y, dy)
    point = Quartz.CGPointMake(x, y)

    # Move the mouse to the target location first so the scroll event
    # is delivered to the correct window.
    move = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventMouseMoved, point, Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move)
    time.sleep(0.02)

    # Use line-based scrolling (kCGScrollEventUnitLine = 1) which apps
    # like WeChat handle reliably.
    event = Quartz.CGEventCreateScrollWheelEvent(
        None, Quartz.kCGScrollEventUnitLine, 1, dy
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def press_escape() -> None:
    """Press and release the Escape key."""
    log.debug("Pressing Escape")
    # Key down
    event = Quartz.CGEventCreateKeyboardEvent(None, 53, True)  # 53 = Escape
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    time.sleep(0.02)

    # Key up
    event = Quartz.CGEventCreateKeyboardEvent(None, 53, False)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
