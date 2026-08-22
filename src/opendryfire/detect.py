"""Laser blob detection and shot assembly.

Two problems live here, and they are different:

1. *Blob detection* — is there a laser dot in this frame, and where?
2. *Shot assembly* — a ~100ms Strikeman pulse spans 3 frames at 30fps and 6 at
   60fps, so consecutive detections must collapse into one shot.

On the detection test: a <5mW 635nm dot on white paper saturates its own core to
white, so the *peak pixel* is often not red at all — only the halo is. Testing
red dominance at the brightest pixel (what ShootOFF does) therefore fails on a
close, bright dot. We test dominance over the blob's dilated neighborhood
instead, which catches the halo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class Blob:
    """A laser candidate in one frame, in pixel coordinates."""

    x: float
    y: float
    area: int
    peak: int  # peak red-channel value, 0-255
    redness: float  # mean(R) - max(mean(G), mean(B)) over the neighborhood
    frame_index: int
    timestamp: float


@dataclass
class Shot:
    """One trigger press: a run of consecutive frames containing the same dot.

    ``x``/``y`` come from the *first* frame of the run, which is the instant the
    firing pin struck. Later frames record where the muzzle drifted during the
    ~100ms the laser stayed lit — useful as a settle/follow-through signal, but
    not the hit point.
    """

    x: float
    y: float
    t_first: float
    t_last: float
    frames: int
    trace: list[tuple[float, float]] = field(default_factory=list)
    peak: int = 0

    @property
    def duration(self) -> float:
        return self.t_last - self.t_first

    @property
    def drift(self) -> float:
        """Pixel distance from the first sample to the farthest later sample."""
        if len(self.trace) < 2:
            return 0.0
        x0, y0 = self.trace[0]
        return max(float(np.hypot(x - x0, y - y0)) for x, y in self.trace[1:])


@dataclass
class DetectorConfig:
    """Thresholds.

    The defaults are measured, not guessed. In a warm-lit room with cherry wood
    and a bamboo table — a red-dominant background, i.e. the hard case — a
    sweep over a real capture gave:

        threshold=230, margin=12  ->  852 false blobs per frame
        threshold=248, margin=40  ->   50
        threshold=248, margin=60  ->    0

    Warm ambient surfaces top out around 48 redness because incandescent light
    is broadband; a 635nm laser is monochromatic and lands far above that. So a
    high red *margin* separates them cleanly while a high brightness threshold
    alone does not. ``calibrate_scene`` re-derives both against the actual room,
    since these numbers are specific to one camera and one set of lamps.
    """

    bright_threshold: int = 248  # min red-channel value to be a candidate
    red_margin: float = 55.0  # min mean(R) - max(mean(G), mean(B)) in the halo
    min_area: int = 2  # px; below this it is sensor noise
    max_area: int = 4000  # px; above this it is a lamp or a window
    dilate_px: int = 5  # halo ring width for the red-dominance test
    # Shot assembly
    gap_frames: int = 2  # allowed dropped frames inside one pulse
    link_radius_px: float = 40.0  # used to pick among blobs, not to split shots
    # Longest a single pulse may last. A Strikeman pulse is ~100ms; 350ms is
    # generous headroom while staying far below the time a slide rack takes,
    # so it can never merge two genuine presses.
    max_pulse_s: float = 0.35
    refractory_s: float = 0.25  # ignore new shots for this long after one ends

    def to_dict(self) -> dict:
        return {
            "bright_threshold": self.bright_threshold,
            "red_margin": self.red_margin,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "dilate_px": self.dilate_px,
            "gap_frames": self.gap_frames,
            "link_radius_px": self.link_radius_px,
            "max_pulse_s": self.max_pulse_s,
            "refractory_s": self.refractory_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class LaserDetector:
    """Stateful detector: feed it frames in order, get Shots out."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self._frame_index = 0
        self._open_shot: Shot | None = None
        self._last_seen_frame = -999
        self._last_pos: tuple[float, float] | None = None
        self._suppress_until = 0.0

    # -- per-frame blob finding ------------------------------------------------

    def find_blobs(self, frame: np.ndarray, timestamp: float = 0.0) -> list[Blob]:
        """Find laser candidates in a single BGR frame."""
        b, g, r = cv2.split(frame)
        mask = (r >= self.cfg.bright_threshold).astype(np.uint8) * 255
        # A 3x3 open removes hot pixels without eating a 2-3px dot.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        blobs: list[Blob] = []
        for i in range(1, n):  # 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self.cfg.min_area <= area <= self.cfg.max_area):
                continue
            component = (labels == i).astype(np.uint8)
            # Dilate to include the red halo around a saturated white core.
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.cfg.dilate_px * 2 + 1,) * 2
            )
            halo = cv2.dilate(component, k)
            mean_b = float(cv2.mean(b, mask=halo)[0])
            mean_g = float(cv2.mean(g, mask=halo)[0])
            mean_r = float(cv2.mean(r, mask=halo)[0])
            redness = mean_r - max(mean_g, mean_b)
            if redness < self.cfg.red_margin:
                continue
            # Intensity-weighted centroid for sub-pixel accuracy.
            cx, cy = _weighted_centroid(r, component)
            blobs.append(
                Blob(
                    x=cx,
                    y=cy,
                    area=area,
                    peak=int(r[component > 0].max()),
                    redness=redness,
                    frame_index=self._frame_index,
                    timestamp=timestamp,
                )
            )
        blobs.sort(key=lambda bl: bl.area, reverse=True)
        return blobs

    # -- shot assembly ---------------------------------------------------------

    def push(self, frame: np.ndarray, timestamp: float) -> Shot | None:
        """Feed one frame. Returns a Shot on the frame where a pulse *ends*.

        Shot boundaries are decided by **time**, not by how far the dot moved.
        A laser cartridge emits exactly one pulse per trigger press, and a
        striker-fired pistol cannot fire again until the slide is cycled — so
        two shots are physically seconds apart, and every blob inside the pulse
        window belongs to the shot already open however far it travelled.

        An earlier version split a shot whenever the dot jumped further than
        ``link_radius_px`` between frames, on the theory that a distant dot must
        be a new shot. It is not: a muzzle that dips or swings during the ~100ms
        the laser is lit moves the dot a long way, and one press was scored as
        two hits. Distance now only chooses *which* blob to follow when several
        are visible.
        """
        blobs = self.find_blobs(frame, timestamp)
        self._frame_index += 1

        if self._open_shot is not None:
            within_pulse = (
                timestamp - self._open_shot.t_first
            ) <= self.cfg.max_pulse_s
            best = self._nearest(blobs)
            if best is not None and within_pulse:
                self._open_shot.t_last = timestamp
                self._open_shot.frames += 1
                self._open_shot.trace.append((best.x, best.y))
                self._open_shot.peak = max(self._open_shot.peak, best.peak)
                self._last_seen_frame = self._frame_index
                self._last_pos = (best.x, best.y)
                return None
            # Either the dot went away, or the pulse has run past any plausible
            # length. Close once the dot has been gone for long enough that a
            # dropped frame cannot explain it.
            gap = self._frame_index - self._last_seen_frame
            if not within_pulse or gap > self.cfg.gap_frames:
                return self._close_open_shot()
            return None

        # No shot open: start one, unless we are still inside the refractory
        # window that follows the previous shot.
        if blobs and timestamp >= self._suppress_until:
            best = blobs[0]
            self._open_shot = Shot(
                x=best.x,
                y=best.y,
                t_first=timestamp,
                t_last=timestamp,
                frames=1,
                trace=[(best.x, best.y)],
                peak=best.peak,
            )
            self._last_seen_frame = self._frame_index
            self._last_pos = (best.x, best.y)
        return None

    def _nearest(self, blobs: list[Blob]) -> Blob | None:
        """The blob most likely to be the dot we are already tracking.

        Nearest to its last position, falling back to the largest. This is what
        ``link_radius_px`` is for now — picking among candidates, never deciding
        that a shot has ended.
        """
        if not blobs:
            return None
        if self._last_pos is None:
            return blobs[0]
        lx, ly = self._last_pos
        return min(blobs, key=lambda b: float(np.hypot(b.x - lx, b.y - ly)))

    def flush(self) -> Shot | None:
        """Close any pulse still open at end of stream."""
        return self._close_open_shot()

    def _close_open_shot(self) -> Shot | None:
        shot, self._open_shot = self._open_shot, None
        if shot is not None:
            self._suppress_until = shot.t_last + self.cfg.refractory_s
        self._last_pos = None
        return shot


# A real 635nm dot on cream PLA reaches a halo redness of roughly 120-200. A
# margin above this cap would be cleared by *nothing*, laser included, so
# raising it further trades a false-positive problem for silent blindness —
# strictly the worse failure, because a missed hit looks like a missed shot.
MAX_USABLE_MARGIN = 120.0


def calibrate_scene(
    frames: list[np.ndarray],
    mask: np.ndarray | None = None,
    safety: float = 1.25,
    max_margin: float = MAX_USABLE_MARGIN,
) -> tuple[DetectorConfig, dict]:
    """Derive thresholds that see nothing in *this* room with no laser present.

    Takes frames of the static scene and finds the loosest settings that still
    yield zero false blobs, then backs off by ``safety``. Looser is better:
    every point of extra margin is headroom the real laser has to clear, so
    tightening beyond what the room requires only costs sensitivity.

    Pass ``mask`` (from ``TargetFrame.scoring_mask``) to restrict calibration to
    the target plane — the room outside it is never scored, so a red LED on a
    charger in the corner should not be allowed to drive thresholds up.

    If a single bright red object forces the margin past ``max_margin``, the
    threshold is capped and the report says so, with the offender's location.
    Capping is the honest move: an uncapped margin silently stops detecting the
    laser, and under-detection is indistinguishable from not shooting.
    """
    if not frames:
        raise ValueError("need at least one frame of the scene")

    worst_bright = 0
    worst_redness = 0.0
    offender: tuple[int, int, int, int] | None = None

    for frame in frames:
        b, g, r = cv2.split(frame.astype(np.int16))
        redness = r - np.maximum(g, b)
        if mask is not None:
            keep = mask > 0
            r = np.where(keep, r, 0)
            redness = np.where(keep, redness, 0)
        # The brightest thing that is also meaningfully red is what we must clear.
        candidate = (r >= 200) & (redness >= 20)
        if not candidate.any():
            continue
        worst_bright = max(worst_bright, int(r[candidate].max()))
        frame_worst = float(redness[candidate].max())
        if frame_worst > worst_redness:
            worst_redness = frame_worst
            ys, xs = np.nonzero(candidate & (redness >= frame_worst - 2))
            if len(xs):
                offender = (
                    int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1,
                )

    wanted = max(worst_redness * safety, 35.0)
    capped = wanted > max_margin
    cfg = DetectorConfig(
        bright_threshold=min(int(max(worst_bright, 200)), 250),
        red_margin=min(wanted, max_margin),
    )

    det = LaserDetector(cfg)
    false_blobs = sum(
        len(det.find_blobs(cv2.bitwise_and(f, f, mask=mask) if mask is not None else f))
        for f in frames
    )
    report = {
        "frames": len(frames),
        "brightest_red_pixel": worst_bright,
        "max_ambient_redness": round(worst_redness, 1),
        "chosen_threshold": cfg.bright_threshold,
        "chosen_margin": round(cfg.red_margin, 1),
        "false_blobs_after": false_blobs,
        "masked": mask is not None,
        "capped": capped,
        "wanted_margin": round(wanted, 1),
        "offender_bbox": offender,
    }
    return cfg, report


def describe_calibration(report: dict) -> list[str]:
    """Turn a calibration report into advice, rather than raw numbers."""
    out: list[str] = []
    if report["capped"]:
        box = report.get("offender_bbox")
        where = f" around x={box[0]}-{box[0] + box[2]}, y={box[1]}-{box[1] + box[3]}" if box else ""
        out.append(
            f"Something in frame is nearly as red as a laser "
            f"(redness {report['max_ambient_redness']:.0f}){where}."
        )
        out.append(
            f"Clearing it would need a margin of {report['wanted_margin']:.0f}, "
            f"which the laser itself may not reach — capped at "
            f"{report['chosen_margin']:.0f} instead."
        )
        out.append(
            "Fix it at the source, in order of preference: remove the red light "
            "from view (an LED on a charger or power strip is the usual "
            "culprit), or run `dryfire calibrate` first and re-run autocal so "
            "the mask excludes everything off-target."
        )
    if report["false_blobs_after"]:
        out.append(
            f"{report['false_blobs_after']} false blobs remain across "
            f"{report['frames']} frames. The audio gate will reject them as "
            f"unpaired, but calibrating the target plane first is better."
        )
    if not out:
        out.append("Clean: nothing in this room reads as a laser.")
    return out


def _weighted_centroid(channel: np.ndarray, component: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid of one connected component."""
    ys, xs = np.nonzero(component)
    w = channel[ys, xs].astype(np.float64)
    total = w.sum()
    if total <= 0:
        return float(xs.mean()), float(ys.mean())
    return float((xs * w).sum() / total), float((ys * w).sum() / total)
