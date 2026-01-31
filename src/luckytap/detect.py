"""OpenCV template matching for red envelope detection."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from luckytap.config import MATCH_THRESHOLD

log = logging.getLogger(__name__)


def load_templates(template_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Load all PNG templates from a directory.

    Returns list of (name, bgr_image) tuples.
    Templates are assumed to be at native capture resolution (Retina pixels)
    since they are cropped from screenshots taken by --capture-template.
    """
    templates: list[tuple[str, np.ndarray]] = []
    if not template_dir.is_dir():
        log.warning("Template directory does not exist: %s", template_dir)
        return templates

    for p in sorted(template_dir.glob("*.png")):
        # Skip screenshots saved by --capture-template
        if "screenshot" in p.stem.lower():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("Could not load template: %s", p)
            continue
        templates.append((p.stem, img))
        log.info("Loaded template '%s' (%dx%d)", p.stem, img.shape[1], img.shape[0])

    return templates


def detect_matches(
    frame: np.ndarray,
    templates: list[tuple[str, np.ndarray]],
    threshold: float = MATCH_THRESHOLD,
) -> list[tuple[str, int, int, int, int, float]]:
    """Run template matching on a frame.

    Returns list of (template_name, x, y, w, h, confidence) for each match
    above the threshold. Coordinates are relative to the frame.
    """
    results: list[tuple[str, int, int, int, int, float]] = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, tmpl in templates:
        th, tw = tmpl.shape[:2]
        if tw > frame.shape[1] or th > frame.shape[0]:
            continue

        gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        match_result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCOEFF_NORMED)

        locations = np.where(match_result >= threshold)
        for pt_y, pt_x in zip(*locations):
            confidence = float(match_result[pt_y, pt_x])
            results.append((name, int(pt_x), int(pt_y), tw, th, confidence))

    # visualize the matching bbox and save to {name}.jpg
    if results:
        vis = frame.copy()
        for name, x, y, w, h, conf in results:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis, f"{name} {conf:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        fout_name = f'matches-{name}.jpg'
        cv2.imwrite(fout_name, vis)
        log.debug(f"Saved match visualization to {fout_name}")
    # Non-maximum suppression: keep only the best match in each cluster
    return _nms(results)


def _nms(
    detections: list[tuple[str, int, int, int, int, float]],
    overlap_thresh: float = 0.5,
) -> list[tuple[str, int, int, int, int, float]]:
    """Simple non-maximum suppression on detection boxes."""
    if not detections:
        return []

    # Sort by confidence descending
    detections.sort(key=lambda d: d[5], reverse=True)
    keep: list[tuple[str, int, int, int, int, float]] = []

    for det in detections:
        _, dx, dy, dw, dh, _ = det
        suppress = False
        for _, kx, ky, kw, kh, _ in keep:
            # Compute IoU
            x1 = max(dx, kx)
            y1 = max(dy, ky)
            x2 = min(dx + dw, kx + kw)
            y2 = min(dy + dh, ky + kh)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_d = dw * dh
            area_k = kw * kh
            union = area_d + area_k - inter
            if union > 0 and inter / union > overlap_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(det)

    return keep
