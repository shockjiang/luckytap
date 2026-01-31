"""Screen capture of the WeChat window region."""

from __future__ import annotations

import logging

import numpy as np
import Quartz
from Cocoa import NSBitmapImageRep

from luckytap.config import CHAT_AREA_LEFT_RATIO

log = logging.getLogger(__name__)


def get_retina_scale() -> float:
    """Return the Retina backing scale factor for the main screen."""
    try:
        from AppKit import NSScreen

        return float(NSScreen.mainScreen().backingScaleFactor())
    except Exception:
        return 1.0


def capture_region(x: int, y: int, w: int, h: int) -> np.ndarray | None:
    """Capture a screen region and return it as a BGR numpy array.

    Uses CGWindowListCreateImage for fast, in-process capture.
    """
    rect = Quartz.CGRectMake(x, y, w, h)
    image = Quartz.CGWindowListCreateImage(
        rect,
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
        Quartz.kCGWindowImageShouldBeOpaque,
    )
    if image is None:
        log.warning("CGWindowListCreateImage returned None — check Screen Recording permission")
        return None

    bitmap = NSBitmapImageRep.alloc().initWithCGImage_(image)
    width = bitmap.pixelsWide()
    height = bitmap.pixelsHigh()
    raw = bitmap.bitmapData()

    if raw is None:
        return None

    # NSBitmapImageRep from CGImage uses ARGB byte order on macOS
    buf = np.frombuffer(
        raw, dtype=np.uint8, count=width * height * 4
    ).reshape(height, width, 4)

    # Convert ARGB → BGR for OpenCV
    bgr = np.empty((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = buf[:, :, 3]  # B
    bgr[:, :, 1] = buf[:, :, 2]  # G
    bgr[:, :, 2] = buf[:, :, 1]  # R
    return bgr


def capture_chat_area(
    wx: int, wy: int, ww: int, wh: int
) -> tuple[np.ndarray | None, int, int]:
    """Capture the chat area (right portion) of the WeChat window.

    Returns (image, offset_x, offset_y) where offsets are in screen coords.
    """
    left = int(ww * CHAT_AREA_LEFT_RATIO)
    cx = wx + left
    cw = ww - left
    img = capture_region(cx, wy, cw, wh)
    return img, cx, wy
