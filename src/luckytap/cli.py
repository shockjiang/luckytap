"""CLI entry point and main detection loop."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from luckytap import __version__
from luckytap.capture import capture_chat_area, capture_region, get_retina_scale
from luckytap.clicker import click, scroll
from luckytap.config import (
    CHAT_AREA_LEFT_RATIO,
    CONVERSATION_SWITCH_DELAY,
    DEDUP_DISTANCE,
    DISMISS_DELAY,
    MATCH_THRESHOLD,
    NUM_CHATS,
    OPEN_BUTTON_DELAY,
    POLL_INTERVAL,
    SCROLL_AMOUNT,
    SCROLL_DELAY,
    SCROLL_STEPS,
    SELF_MSG_X_RATIO,
    SIDEBAR_ROW_HEIGHT,
    SIDEBAR_TOP_OFFSET,
    WINDOW_REQUERY_INTERVAL,
)
from luckytap.detect import detect_matches, load_templates
from luckytap.wechat import activate_wechat, get_wechat_window, is_wechat_running

log = logging.getLogger("luckytap")


def _quantize(x: int, y: int, distance: int = DEDUP_DISTANCE) -> tuple[int, int]:
    """Round coordinates to a grid to deduplicate nearby detections."""
    return (x // distance * distance, y // distance * distance)


def _is_new(qx: int, qy: int, seen: set[tuple[int, int]]) -> bool:
    return (qx, qy) not in seen


def _click_conversation(wx: int, wy: int, ww: int, index: int, frame=None) -> None:
    """Click the Nth conversation in the WeChat sidebar (0-based index)."""
    sidebar_center_x = wx + int(ww * CHAT_AREA_LEFT_RATIO) // 2
    row_y = wy + SIDEBAR_TOP_OFFSET + index * SIDEBAR_ROW_HEIGHT + SIDEBAR_ROW_HEIGHT // 2
    log.debug("Clicking conversation %d at (%d, %d)", index, sidebar_center_x, row_y)
    click(sidebar_center_x, row_y, f"swith conversation {index}",frame=frame)


def _capture_template_mode(args: argparse.Namespace) -> None:
    """Interactive helper: capture a region of the screen and save as a template."""
    print("=== Template Capture Mode ===")
    print("1. Open WeChat and navigate to a chat with a visible red envelope.")
    print("2. Press Enter when ready...")
    input()

    win = get_wechat_window()
    if win is None:
        print("ERROR: Could not find WeChat window. Is WeChat open?", file=sys.stderr)
        sys.exit(1)

    wx, wy, ww, wh = win
    print(f"WeChat window found at ({wx}, {wy}) size {ww}x{wh}")

    frame = capture_region(wx, wy, ww, wh)
    if frame is None:
        print("ERROR: Screen capture failed. Check Screen Recording permission.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.template_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshot_path = out_dir / "wechat_screenshot.png"
    cv2.imwrite(str(screenshot_path), frame)
    print(f"Full screenshot saved to: {screenshot_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {screenshot_path} in an image editor")
    print("  2. Crop the red envelope icon and save as: templates/hongbao_icon.png")
    print("  3. Crop the 'Open' button (开) and save as: templates/open_button.png")
    print("  4. Run: python -m luckytap --dry-run -v")


def _run_loop(args: argparse.Namespace) -> None:
    """Main detection + click loop."""
    if not is_wechat_running():
        print("ERROR: WeChat is not running.", file=sys.stderr)
        sys.exit(1)

    activate_wechat()
    time.sleep(0.5)

    win = get_wechat_window()
    if win is None:
        print("ERROR: Could not find WeChat window.", file=sys.stderr)
        sys.exit(1)

    wx, wy, ww, wh = win
    log.info("WeChat window at (%d, %d) size %dx%d", wx, wy, ww, wh)

    scale = get_retina_scale()
    log.info("Retina scale factor: %.1f", scale)

    template_dir = Path(args.template_dir)
    all_templates = load_templates(template_dir)
    if not all_templates:
        print(
            f"ERROR: No templates found in {template_dir}. "
            "Run with --capture-template first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Split templates by purpose:
    #   detect_templates — for finding red envelopes in the chat area
    #   open_templates   — for clicking buttons in the open dialog
    detect_templates = [(n, t) for n, t in all_templates if n.startswith("hongbao_icon")]
    open_templates = [(n, t) for n, t in all_templates if n.startswith("open_button")]
    log.info(
        "Templates: %d for detection (%s), %d for open dialog (%s)",
        len(detect_templates), [n for n, _ in detect_templates],
        len(open_templates), [n for n, _ in open_templates],
    )

    seen: set[tuple[int, int]] = set()
    last_window_check = 0.0
    interval = args.interval
    threshold = args.threshold
    dry_run = args.dry_run

    if dry_run:
        log.info("DRY RUN mode — will detect but not click")

    num_chats = args.num_chats
    scroll_steps = args.scroll_steps
    scan = args.scan_chats

    log.info("Starting detection loop (interval=%.3fs, threshold=%.2f)", interval, threshold)
    # Global Escape key listener using Quartz event tap (works even when not focused)
    quit_event = threading.Event()

    def _global_key_listener() -> None:
        import Quartz
        from PyObjCTools import AppHelper

        def _callback(_proxy, event_type, event, _refcon):
            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            if keycode == 53:  # Escape
                quit_event.set()
                AppHelper.stopEventLoop()
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
            _callback,
            None,
        )
        if tap is None:
            log.warning("Could not create event tap — check Accessibility permission")
            return

        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(loop, source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(tap, True)
        AppHelper.runConsoleEventLoop()

    listener = threading.Thread(target=_global_key_listener, daemon=True)
    listener.start()

    if scan:
        log.info("Scan mode: cycling through top %d conversations", num_chats)
        print(f"Scanning top {num_chats} conversations for red envelopes... Press Esc or Ctrl+C to stop.")
    else:
        print("Watching for red envelopes... Press Esc or Ctrl+C to stop.")

    try:
        while not quit_event.is_set():
            now = time.monotonic()

            # Periodically re-query window position and ensure WeChat is foreground
            if now - last_window_check > WINDOW_REQUERY_INTERVAL:
                activate_wechat()
                new_win = get_wechat_window()
                if new_win is not None:
                    if new_win != (wx, wy, ww, wh):
                        wx, wy, ww, wh = new_win
                        log.info("Window moved to (%d, %d) size %dx%d", wx, wy, ww, wh)
                else:
                    log.warning("WeChat window not found — retrying...")
                    time.sleep(1)
                    continue
                last_window_check = now

            if scan:
                # Ensure WeChat is foreground before interacting
                activate_wechat()
                time.sleep(0.1)

                # Cycle through the top N conversations
                for chat_idx in range(num_chats):
                    if quit_event.is_set():
                        break

                    _click_conversation(wx, wy, ww, chat_idx, frame=None)
                    time.sleep(CONVERSATION_SWITCH_DELAY)

                    # Check the currently visible chat area
                    _check_and_open(
                        wx, wy, ww, wh, detect_templates, open_templates, threshold, scale,
                        seen, dry_run, chat_idx,
                    )

                    # Scroll up through chat history to find more red packets
                    chat_center_x = wx + int(ww * (CHAT_AREA_LEFT_RATIO + 1) / 2)
                    chat_center_y = wy + wh // 2
                    scrolled = 0
                    for scroll_i in range(scroll_steps):
                        if quit_event.is_set():
                            break
                        scroll(chat_center_x, chat_center_y, SCROLL_AMOUNT)
                        scrolled += 1
                        time.sleep(SCROLL_DELAY)
                        _check_and_open(
                            wx, wy, ww, wh, detect_templates, open_templates, threshold, scale,
                            seen, dry_run, chat_idx,
                        )

                    # Scroll back down to the latest messages before switching.
                    # No detection needed — these areas were already checked
                    # during the upward scroll.
                    if scrolled:
                        for _ in range(scrolled):
                            scroll(chat_center_x, chat_center_y, -SCROLL_AMOUNT)
                        time.sleep(SCROLL_DELAY)

                # Clear seen set after a full scan round so envelopes in
                # conversations can be re-detected on the next cycle
                seen.clear()
            else:
                _check_and_open(
                    wx, wy, ww, wh, detect_templates, open_templates, threshold, scale,
                    seen, dry_run,
                )

            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped.")


def _check_and_open(
    wx: int,
    wy: int,
    ww: int,
    wh: int,
    detect_templates: list[tuple[str, np.ndarray]],
    open_templates: list[tuple[str, np.ndarray]],
    threshold: float,
    scale: float,
    seen: set[tuple[int, int]],
    dry_run: bool,
    chat_idx: int | None = None,
) -> None:
    """Capture the chat area, detect red envelopes, and open them."""
    frame, offset_x, offset_y = capture_chat_area(wx, wy, ww, wh)
    if frame is None:
        log.warning("Capture failed, skipping...")
        return

    matches = detect_matches(frame, detect_templates, threshold=0.99)
    frame_w = frame.shape[1]

    for name, mx, my, mw, mh, conf, scale in matches:
        # Skip red packets on the right side (sent by self)
        center_x = mx + mw / 2
        if center_x > frame_w * SELF_MSG_X_RATIO:
            log.debug("Ignoring self-sent red packet at x=%.0f (%.0f%% of frame width)",
                      center_x, center_x / frame_w * 100)
            continue

        screen_x = offset_x + int(center_x / scale)
        screen_y = offset_y + int((my + mh / 2) / scale)

        qpos = _quantize(screen_x, screen_y)
        if not _is_new(*qpos, seen):
            continue

        seen.add(qpos)
        prefix = f"[chat {chat_idx}] " if chat_idx is not None else ""
        log.info(
            "Match '%s' at screen (%d, %d) confidence=%.3f",
            name, screen_x, screen_y, conf,
        )

        if dry_run:
            continue

        # Step 1: Click the red envelope
        click(screen_x, screen_y, "detect and click red bag", frame=frame)
        time.sleep(OPEN_BUTTON_DELAY)

        # Step 2: Click through the open dialog (two buttons)
        _try_click_open_buttons(
            wx, wy, ww, wh, open_templates, threshold, scale
        )

        # Step 3: Dismiss any dialog by clicking the chat area background.
        # Avoid press_escape() since the global Escape listener would quit the app.
        dismiss_x = wx + int(ww * (CHAT_AREA_LEFT_RATIO + 1) / 2)
        dismiss_y = wy + 30  # top of chat area, unlikely to hit anything
        # click(dismiss_x, dismiss_y, "dismiss dialog", frame=frame)
        time.sleep(DISMISS_DELAY)

        # UI has changed after clicking; remaining match coordinates are stale.
        # Break and let the outer loop re-detect with a fresh screenshot.
        break


def _try_click_open_buttons(
    wx: int,
    wy: int,
    ww: int,
    wh: int,
    open_templates: list[tuple[str, np.ndarray]],
    threshold: float,
    scale: float,
) -> None:
    frame = capture_region(wx, wy, ww, wh)

    matches = detect_matches(frame, open_templates, threshold=threshold)
    if not matches:
        log.debug("no matches found on open stage")
        cv2.imwrite('error-open-nomatch.jpg', frame)
        breakpoint()
        return
    

    name, mx, my, mw, mh, conf, scale = matches[0]
    log.info(
        "Match '%s' at screen (%d, %d) confidence=%.3f",
        name, mx, my, conf,
    )
    sx = wx + int((mx + mw / 2) / scale)
    sy = wy + int((my + mh / 2) / scale)
    click(sx, sy, f"open {name}", frame=frame)
    time.sleep(OPEN_BUTTON_DELAY)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="luckytap",
        description="Automate WeChat red envelope (红包) grabbing on macOS",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {POLL_INTERVAL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help=f"Template match threshold 0.0-1.0 (default: {MATCH_THRESHOLD})",
    )
    parser.add_argument(
        "--template-dir",
        default="templates",
        help="Directory containing template images (default: templates)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect envelopes but do not click",
    )
    parser.add_argument(
        "--scan-chats",
        action="store_true",
        help="Cycle through the top N sidebar conversations to check each one",
    )
    parser.add_argument(
        "--num-chats",
        type=int,
        default=NUM_CHATS,
        help=f"Number of conversations to scan (default: {NUM_CHATS}, used with --scan-chats)",
    )
    parser.add_argument(
        "--scroll-steps",
        type=int,
        default=SCROLL_STEPS,
        help=f"Number of scroll steps per conversation to find older red packets (default: {SCROLL_STEPS})",
    )
    parser.add_argument(
        "--capture-template",
        action="store_true",
        help="Capture a WeChat screenshot to help create template images",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

    if args.capture_template:
        _capture_template_mode(args)
    else:
        _run_loop(args)
