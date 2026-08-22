"""Interactive windows: calibration clicking and the live drill display.

The calibration window has a magnifier, because the whole system's accuracy
rests on four clicks. At 1/5 scale the A-zone is 1.2" x 2.2", so a two-pixel
click error is a real fraction of the scoring zone, and a loupe makes that
error avoidable rather than merely regrettable.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import TargetFrame
from .targets import ScaledTarget

WINDOW = "dryfire"
LOUPE = 180
LOUPE_ZOOM = 5

CORNER_PROMPTS = (
    "TOP-LEFT corner of the A-zone",
    "TOP-RIGHT corner of the A-zone",
    "BOTTOM-RIGHT corner of the A-zone",
    "BOTTOM-LEFT corner of the A-zone",
)


def _draw_loupe(canvas: np.ndarray, frame: np.ndarray, mx: int, my: int) -> None:
    """Magnified inset around the cursor with a crosshair, so the exact pixel
    being selected is visible."""
    h, w = frame.shape[:2]
    half = LOUPE // (2 * LOUPE_ZOOM)
    x0, y0 = max(0, mx - half), max(0, my - half)
    x1, y1 = min(w, mx + half), min(h, my + half)
    if x1 <= x0 or y1 <= y0:
        return
    patch = cv2.resize(
        frame[y0:y1, x0:x1], (LOUPE, LOUPE), interpolation=cv2.INTER_NEAREST
    )
    cv2.line(patch, (LOUPE // 2, 0), (LOUPE // 2, LOUPE), (0, 255, 255), 1)
    cv2.line(patch, (0, LOUPE // 2), (LOUPE, LOUPE // 2), (0, 255, 255), 1)
    cv2.rectangle(patch, (0, 0), (LOUPE - 1, LOUPE - 1), (255, 255, 255), 1)

    px = w - LOUPE - 12 if mx < w // 2 else 12
    canvas[12 : 12 + LOUPE, px : px + LOUPE] = patch


def click_calibration(
    frame: np.ndarray, target: ScaledTarget
) -> list[tuple[float, float]] | None:
    """Ask for the four A-zone corners. Returns None if cancelled.

    The A-zone is used rather than the target outline because it is engraved
    into the target's own face, so it is exactly coplanar with the surface being
    scored — see ``geometry`` for why that matters.
    """
    points: list[tuple[float, float]] = []
    mouse = {"x": 0, "y": 0}

    def on_mouse(event, x, y, flags, _):  # noqa: ANN001
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = frame.copy()
        for i, (px, py) in enumerate(points):
            cv2.drawMarker(
                canvas, (int(px), int(py)), (0, 255, 0), cv2.MARKER_CROSS, 22, 2
            )
            cv2.putText(
                canvas, str(i + 1), (int(px) + 10, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        if len(points) == 4:
            cv2.polylines(
                canvas, [np.array(points, np.int32)], True, (0, 255, 0), 2
            )

        if len(points) < 4:
            msg = f"Click {CORNER_PROMPTS[len(points)]}  ({len(points)}/4)"
        else:
            msg = "ENTER to accept · U to undo · ESC to cancel"
        _banner(canvas, msg, f"{target.describe()}")
        _draw_loupe(canvas, frame, mouse["x"], mouse["y"])

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(WINDOW)
            return None
        if key in (ord("u"), ord("U")) and points:
            points.pop()
        if key in (13, 10) and len(points) == 4:
            cv2.destroyWindow(WINDOW)
            return points


def _banner(canvas: np.ndarray, line1: str, line2: str = "") -> None:
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - 74), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    cv2.putText(
        canvas, line1, (16, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
    )
    if line2:
        cv2.putText(
            canvas, line2, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (170, 200, 255), 1,
        )


def draw_overlay(
    canvas: np.ndarray,
    tf: TargetFrame,
    target: ScaledTarget,
    shots: list[tuple[float, float, str]],
    header: str,
    subheader: str = "",
    show_zones: bool = True,
) -> np.ndarray:
    """Draw the scoring geometry and the shots so far onto a live frame."""
    if show_zones:
        spec = target.spec
        for zone in reversed(spec.zones):
            colour = (90, 90, 90)
            if zone.name == spec.primary_zone:
                colour = (0, 220, 255)
            if zone.kind == "rect":
                hw, hh = (zone.w / 2) * target.scale, (zone.h / 2) * target.scale
                cx, cy = zone.cx * target.scale, zone.cy * target.scale
                pts = [
                    (cx - hw, cy + hh), (cx + hw, cy + hh),
                    (cx + hw, cy - hh), (cx - hw, cy - hh),
                ]
                poly = np.array([tf.to_image(x, y) for x, y in pts], np.int32)
                cv2.polylines(canvas, [poly], True, colour, 2)
            elif zone.kind == "circle":
                ring = [
                    tf.to_image(
                        zone.r * target.scale * np.cos(a),
                        zone.r * target.scale * np.sin(a),
                    )
                    for a in np.linspace(0, 2 * np.pi, 48)
                ]
                cv2.polylines(canvas, [np.array(ring, np.int32)], True, colour, 2)

    for i, (x, y, zone) in enumerate(shots):
        px, py = tf.to_image(x, y)
        hit = zone == target.spec.primary_zone
        colour = (0, 255, 0) if hit else (60, 120, 255)
        cv2.circle(canvas, (int(px), int(py)), 9, colour, 2)
        cv2.circle(canvas, (int(px), int(py)), 2, colour, -1)
        cv2.putText(
            canvas, str(i + 1), (int(px) + 12, int(py) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
        )

    _banner(canvas, header, subheader)
    return canvas


def scoreboard(
    width: int,
    height: int,
    target: ScaledTarget,
    shots: list[tuple[float, float, str]],
) -> np.ndarray:
    """A rectified plan view of the target — what the group actually looks like,
    with the camera's perspective removed."""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    spec = target.spec
    half_w = target.printed_width / 2
    half_h = target.printed_height / 2
    ppi = min(width / (2 * half_w * 1.15), height / (2 * half_h * 1.05))

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(width / 2 + x * ppi), int(height / 2 - y * ppi)

    for zone in reversed(spec.zones):
        colour = (0, 200, 235) if zone.name == spec.primary_zone else (70, 70, 70)
        if zone.kind == "rect":
            hw, hh = (zone.w / 2) * target.scale, (zone.h / 2) * target.scale
            cx, cy = zone.cx * target.scale, zone.cy * target.scale
            cv2.rectangle(canvas, to_px(cx - hw, cy + hh), to_px(cx + hw, cy - hh), colour, 1)
        elif zone.kind == "circle":
            cv2.circle(canvas, to_px(0, 0), int(zone.r * target.scale * ppi), colour, 1)

    for i, (x, y, zone) in enumerate(shots):
        colour = (0, 255, 0) if zone == spec.primary_zone else (60, 120, 255)
        cv2.circle(canvas, to_px(x, y), 6, colour, -1)
        cv2.putText(
            canvas, str(i + 1), (to_px(x, y)[0] + 8, to_px(x, y)[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
        )
    return canvas
