"""Laser blob detection and shot assembly.

Two problems live here, and they are different:

1. *Blob detection* — is there a laser dot in this frame, and where?
2. *Shot assembly* — a ~100ms pulse spans 3 frames at 30fps and 6 at 60fps, so
   consecutive detections must collapse into one shot.

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
    # (x, y, w, h) of the component before dilation. Carried so calibration can
    # point at what it measured without recomputing components — the duplicate
    # implementation is what let the two disagree about what "redness" meant.
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


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
    # Largest above-threshold footprint seen during the pulse, in px. Carried
    # because the size of the dot is the thing ``max_area`` is judged against,
    # and until it was measured the ceiling was only ever a guess.
    area: int = 0

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


@dataclass(frozen=True)
class OversizedRegion:
    """A red-dominant region that passed every test except the size ceiling.

    Dropping these silently is how a bright, close laser becomes invisible: the
    splash it makes is larger than ``max_area``, so the real dot is discarded
    and whatever fragment of it survives at the edge gets scored instead. They
    are kept so the tools can say it out loud, and so ``spike`` can measure the
    dot the room actually produces.
    """

    area: int
    redness: float
    peak: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    frame_index: int
    timestamp: float


@dataclass
class DotMeasurement:
    """What the laser actually looks like in this room, measured not assumed.

    ``max_area`` exists to separate a laser from a lamp, but it was a fixed
    4000px guess while the real footprint depends on the cartridge, the
    distance, the surface and the exposure the camera happened to choose. On
    the reference setup at 9ft the splash measured over 9,500px — twice the
    ceiling — so the real dot was discarded every time and a fragment of its
    edge was scored instead (DESIGN.md 17). This is that number, taken from
    real presses by ``spike``.

    ``areas`` are the largest above-threshold footprint per flash, at the
    ``bright_threshold`` in force when they were measured — a higher threshold
    later means a smaller footprint, so the measurement stays conservative; a
    lower one means the footprint is larger than recorded and it must be
    measured again.
    """

    flashes: int
    areas: list[int] = field(default_factory=list)
    peaks: list[int] = field(default_factory=list)
    rednesses: list[float] = field(default_factory=list)
    bright_threshold: int = 0
    measured: str = ""

    @property
    def largest(self) -> int:
        return max(self.areas) if self.areas else 0

    @property
    def median_area(self) -> int:
        return int(np.median(self.areas)) if self.areas else 0

    @property
    def radius_px(self) -> int:
        """Radius of a disc with the median measured area — the size a probe
        dot has to be for ``sees_reference_dot`` to mean anything."""
        return int(round(np.sqrt(max(self.median_area, 1) / np.pi)))

    def rejected_by(self, max_area: int) -> int:
        """How many measured flashes a given ceiling would throw away."""
        return sum(1 for a in self.areas if a > max_area)

    def lines(self) -> list[str]:
        if not self.areas:
            return ["no laser flash was measured"]
        return [
            f"flashes measured: {self.flashes}",
            f"footprint: median {self.median_area} px, "
            f"largest {self.largest} px, at threshold {self.bright_threshold}",
            f"equivalent radius: {self.radius_px} px",
            f"peak red: median {int(np.median(self.peaks))}" if self.peaks else "",
            f"redness: median {np.median(self.rednesses):.0f}, "
            f"min {min(self.rednesses):.0f}" if self.rednesses else "",
        ]

    def to_dict(self) -> dict:
        return {
            "flashes": self.flashes,
            "areas": list(self.areas),
            "peaks": list(self.peaks),
            "rednesses": [round(v, 2) for v in self.rednesses],
            "bright_threshold": self.bright_threshold,
            "measured": self.measured,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DotMeasurement":
        return cls(
            flashes=int(d.get("flashes", 0)),
            areas=[int(a) for a in d.get("areas", [])],
            peaks=[int(a) for a in d.get("peaks", [])],
            rednesses=[float(a) for a in d.get("rednesses", [])],
            bright_threshold=int(d.get("bright_threshold", 0)),
            measured=str(d.get("measured", "")),
        )


def measure_dot(
    detector: "LaserDetector", shots: list[Shot], measured: str = ""
) -> DotMeasurement:
    """Group everything a detector saw into flashes and measure them.

    Samples come from both sides of the size ceiling: regions it rejected for
    being too big, and shots it accepted. Both are the same physical thing —
    a trigger press lighting the target — and which side of the ceiling a press
    lands on is exactly the fact being measured.
    """
    samples: list[tuple[float, int, int, float | None]] = [
        (o.timestamp, o.area, o.peak, o.redness) for o in detector.oversized
    ]
    samples += [(s.t_first, s.area, s.peak, None) for s in shots if s.area]
    samples.sort()

    gap = detector.cfg.max_pulse_s
    flashes: list[list[tuple[float, int, int, float | None]]] = []
    for sample in samples:
        if flashes and sample[0] - flashes[-1][-1][0] <= gap:
            flashes[-1].append(sample)
        else:
            flashes.append([sample])

    areas, peaks, rednesses = [], [], []
    for flash in flashes:
        areas.append(max(f[1] for f in flash))
        peaks.append(max(f[2] for f in flash))
        reds = [f[3] for f in flash if f[3] is not None]
        if reds:
            rednesses.append(max(reds))
    return DotMeasurement(
        flashes=len(flashes),
        areas=areas,
        peaks=peaks,
        rednesses=rednesses,
        bright_threshold=detector.cfg.bright_threshold,
        measured=measured,
    )


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
    # Longest a single pulse may last. Common cartridges emit ~100ms; 350ms is
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
        # Red regions rejected for size, and the number of frames containing
        # one. Both are evidence that the ceiling, not the room, is the limit.
        self.oversized: list[OversizedRegion] = []
        self.oversized_frames = 0

    @property
    def open_shot(self) -> Shot | None:
        """The pulse currently in progress, if any.

        ``push`` only *returns* a Shot once the pulse has ended, which is
        several frames after the dot went dark — so the frame a caller holds
        when it receives a Shot never contains the laser. Anything that wants
        the pixels at the moment of the break (a saved crop, an overlay) has to
        act when this becomes non-None instead.
        """
        return self._open_shot

    # -- per-frame blob finding ------------------------------------------------

    def find_blobs(self, frame: np.ndarray, timestamp: float = 0.0) -> list[Blob]:
        """Find laser candidates in a single BGR frame.

        The size ceiling is applied *after* the red-dominance test, not before.
        A region that is red enough to be the laser but too big to score is not
        a thing to drop quietly — it is the laser, over-exposed — so it is
        recorded in ``oversized`` and reported by the tools.
        """
        b, g, r = cv2.split(frame)
        mask = (r >= self.cfg.bright_threshold).astype(np.uint8) * 255
        # A 3x3 open removes hot pixels without eating a 2-3px dot.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        blobs: list[Blob] = []
        too_big = False
        for i in range(1, n):  # 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.cfg.min_area:
                continue
            bbox = (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            )
            redness = _halo_redness(b, g, r, labels, i, bbox, self.cfg.dilate_px)
            if redness < self.cfg.red_margin:
                continue
            component = (labels == i).astype(np.uint8)
            peak = int(r[component > 0].max())
            if area > self.cfg.max_area:
                too_big = True
                self.oversized.append(
                    OversizedRegion(
                        area=area,
                        redness=redness,
                        peak=peak,
                        centroid=(float(centroids[i][0]), float(centroids[i][1])),
                        bbox=bbox,
                        frame_index=self._frame_index,
                        timestamp=timestamp,
                    )
                )
                continue
            # Intensity-weighted centroid for sub-pixel accuracy.
            cx, cy = _weighted_centroid(r, component)
            blobs.append(
                Blob(
                    x=cx,
                    y=cy,
                    area=area,
                    peak=peak,
                    redness=redness,
                    frame_index=self._frame_index,
                    timestamp=timestamp,
                    bbox=bbox,
                )
            )
        if too_big:
            self.oversized_frames += 1
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
                self._open_shot.area = max(self._open_shot.area, best.area)
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
                area=best.area,
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


# Ceiling on the red margin, in the same units the margin is *tested* in:
# the mean redness of a blob's dilated halo, not the redness of its peak pixel.
# The two differ by a factor of two or more, because dilating a small dot pulls
# in surrounding background — which is exactly what makes the halo test good at
# rejecting point sources, and exactly what makes a per-pixel number the wrong
# thing to threshold against.
#
# Measured, halo-mean, against real captures:
#
#     synthetic reference dot (the model below)     66
#     real 635nm dot on cream PLA, JPEG round-trip  57   (understated by JPEG)
#     real 635nm dots, known to clear a 55 margin  >=55
#     small red LED on a charger                    39
#     bright warm bamboo surface                     0   (excluded by max_area)
#
# So a margin above ~55 is above what a real dot has been observed to produce,
# and raising it further trades a visible false-positive problem for invisible
# blindness — strictly the worse failure, because a missed hit is
# indistinguishable from not having fired.
MAX_USABLE_MARGIN = 55.0

# Floor. An empty room needs *some* margin, or sensor noise that happens to be
# warm would qualify.
MIN_MARGIN = 35.0

# The weakest dot calibration must still be able to see: a saturated white core
# inside an additive red halo. Same model the synthetic tests use, so what the
# suite exercises and what calibration verifies cannot drift apart.
REFERENCE_DOT_CORE_PX = 3
REFERENCE_DOT_HALO_PX = 9
REFERENCE_DOT_GAIN = 220.0


def _stamp_reference_dot(
    frame: np.ndarray, cx: int, cy: int, halo_px: int = REFERENCE_DOT_HALO_PX
) -> None:
    """Render a reference laser dot into ``frame``, in place.

    Additive, because that is what a laser does to a surface: the halo adds red
    to whatever was already there, and the core saturates. A dot stamped on a
    blown-out background therefore comes out *harder* to detect, which is the
    honest result — an over-exposed room really does hide the laser.

    ``halo_px`` is the dot's radius. It is a parameter because the default 9px
    describes a *small* dot, and a check that only ever probes for a small dot
    cannot notice a ceiling that rejects the large one this room produces —
    ``spike`` measures the real radius and ``calibrate_scene`` passes it here.
    """
    halo_px = max(1, int(halo_px))
    core_px = max(1, round(halo_px * REFERENCE_DOT_CORE_PX / REFERENCE_DOT_HALO_PX))
    h, w = frame.shape[:2]
    x0, x1 = max(0, cx - halo_px), min(w, cx + halo_px + 1)
    y0, y1 = max(0, cy - halo_px), min(h, cy + halo_px + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xs - cx, ys - cy)

    patch = frame[y0:y1, x0:x1]
    halo = dist <= halo_px
    falloff = np.clip(1.0 - dist[halo] / halo_px, 0.0, 1.0)
    red = patch[..., 2].astype(np.float64)
    red[halo] = np.clip(red[halo] + falloff * REFERENCE_DOT_GAIN, 0, 255)
    patch[..., 2] = red.astype(np.uint8)
    patch[dist <= core_px] = 255


def sees_reference_dot(
    frame: np.ndarray,
    cfg: DetectorConfig,
    mask: np.ndarray | None = None,
    halo_px: int = REFERENCE_DOT_HALO_PX,
) -> bool:
    """Would ``cfg`` still detect a laser in this room?

    The question `calibrate_scene` used to leave unasked. Counting false blobs
    only proves a config is not *over*-sensitive, and a config that detects
    nothing at all passes that check perfectly — which is how a room can be
    reported "clean" by a detector that has been rendered blind.

    So a reference dot is stamped into a real captured frame and put through the
    real ``find_blobs``. Same code path, same units, no separate model of what
    detection means.
    """
    probe = frame.copy()
    h, w = probe.shape[:2]
    if mask is not None and (mask > 0).any():
        ys, xs = np.nonzero(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
    else:
        cx, cy = w // 2, h // 2
    halo_px = max(1, int(halo_px))
    _stamp_reference_dot(probe, cx, cy, halo_px)
    if mask is not None:
        probe = cv2.bitwise_and(probe, probe, mask=mask)
    found = LaserDetector(cfg).find_blobs(probe)
    return any(float(np.hypot(bl.x - cx, bl.y - cy)) <= halo_px for bl in found)


def calibrate_scene(
    frames: list[np.ndarray],
    mask: np.ndarray | None = None,
    safety: float = 1.25,
    max_margin: float = MAX_USABLE_MARGIN,
    dot: DotMeasurement | None = None,
    area_safety: float = 1.5,
) -> tuple[DetectorConfig, dict]:
    """Derive thresholds that see nothing in *this* room with no laser present.

    Takes frames of the static scene and finds the loosest settings that still
    yield zero false blobs, then backs off by ``safety``. Looser is better:
    every point of extra margin is headroom the real laser has to clear, so
    tightening beyond what the room requires only costs sensitivity.

    Pass ``mask`` (from ``TargetFrame.scoring_mask``) to restrict calibration to
    the target plane — the room outside it is never scored, so a red LED on a
    charger in the corner should not be allowed to drive thresholds up.

    Two things this gets right that are easy to get wrong:

    *The margin is measured the way it is tested.* Both numbers come out of
    ``find_blobs`` itself, so "redness" means one thing. Deriving the margin
    from the maximum *per-pixel* redness instead — while ``find_blobs`` tests
    the *mean over a dilated halo* — inflates it by a factor of two or more,
    and picks fights with objects the brightness threshold has already
    excluded. A warm bamboo surface measured 78 per-pixel and 42 halo-mean; the
    resulting margin of 98 could not be cleared by any real dot, and the room
    was still reported clean because a blind detector finds no false positives.

    *The result is verified to still see a laser.* If a single bright red object
    forces the margin past ``max_margin``, the threshold is capped and the
    report says so, with the offender's location. Capping is the honest move:
    an uncapped margin silently stops detecting the laser, and under-detection
    is indistinguishable from not shooting.

    Pass ``dot`` — a real measurement from ``spike`` — to size the area ceiling
    against the laser instead of guessing it. The ceiling then sits between the
    biggest splash the laser actually made (with ``area_safety`` headroom) and
    the biggest red thing in the room, and the blindness probe uses a dot of
    the measured radius rather than a 9px one it was always going to find.
    Without a measurement the default ceiling stands and the report says it is
    an assumption.
    """
    if not frames:
        raise ValueError("need at least one frame of the scene")

    def prepared(frame: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(frame, frame, mask=mask) if mask is not None else frame

    # Step 1 — brightness. The threshold has to sit above the brightest thing
    # in the room that is also meaningfully red, so those pixels never become
    # candidates in the first place.
    worst_bright = 0
    for frame in frames:
        b, g, r = cv2.split(prepared(frame).astype(np.int16))
        redness = r - np.maximum(g, b)
        candidate = (r >= 200) & (redness >= 20)
        if candidate.any():
            worst_bright = max(worst_bright, int(r[candidate].max()))
    bright_threshold = min(int(max(worst_bright, 200)), 250)

    # Step 2 — the area ceiling, before the margin, because the margin is
    # derived from the blobs the ceiling lets through. Its job is to separate
    # the laser from a lamp, so it needs to know how big each of those is: the
    # laser from ``dot``, the room from the frames themselves.
    h, w = frames[0].shape[:2]
    uncapped = LaserDetector(
        DetectorConfig(
            bright_threshold=bright_threshold,
            red_margin=MIN_MARGIN,
            max_area=w * h,
        )
    )
    room_red_area = 0
    for frame in frames:
        for blob in uncapped.find_blobs(prepared(frame)):
            room_red_area = max(room_red_area, blob.area)

    default_area = DetectorConfig().max_area
    area_overlap = False
    if dot is not None and dot.largest:
        floor = int(np.ceil(dot.largest * area_safety))
        if room_red_area > floor:
            # Room for both: sit between them rather than at either edge.
            max_area = int(round(np.sqrt(floor * room_red_area)))
        else:
            max_area = floor
        area_overlap = room_red_area >= dot.largest
    else:
        max_area = default_area

    # Step 3 — margin, measured by the detector itself at that threshold. Only
    # blobs that survive the brightness and area filters can matter, and their
    # redness is the halo mean the margin will actually be compared against.
    probe = LaserDetector(
        DetectorConfig(
            bright_threshold=bright_threshold,
            red_margin=float("-inf"),
            max_area=max_area,
        )
    )
    worst_redness = 0.0
    offender: tuple[int, int, int, int] | None = None
    for frame in frames:
        for blob in probe.find_blobs(prepared(frame)):
            if blob.redness > worst_redness:
                worst_redness = blob.redness
                offender = blob.bbox

    wanted = max(worst_redness * safety, MIN_MARGIN)
    capped = wanted > max_margin
    cfg = DetectorConfig(
        bright_threshold=bright_threshold,
        red_margin=min(wanted, max_margin),
        max_area=max_area,
    )

    det = LaserDetector(cfg)
    false_blobs = sum(len(det.find_blobs(prepared(f))) for f in frames)
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
        "chosen_max_area": cfg.max_area,
        "room_red_area": room_red_area,
        "dot_measured": dot is not None and bool(dot.areas),
        "dot_area": dot.largest if dot else 0,
        "dot_radius_px": dot.radius_px if dot else 0,
        "dot_flashes_rejected": dot.rejected_by(cfg.max_area) if dot else 0,
        "area_overlap": area_overlap,
        # A footprint measured at a higher threshold than the one chosen here
        # understates the real one, so the measurement no longer bounds it.
        "dot_threshold_dropped": bool(
            dot is not None
            and dot.bright_threshold
            and cfg.bright_threshold < dot.bright_threshold
        ),
        "sees_reference_dot": sees_reference_dot(
            frames[0], cfg, mask, dot.radius_px if dot and dot.areas
            else REFERENCE_DOT_HALO_PX
        ),
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
    if not report.get("sees_reference_dot", True):
        out.append(
            f"BLIND: with threshold {report['chosen_threshold']} and margin "
            f"{report['chosen_margin']:.0f}, a reference laser dot stamped into "
            f"this scene is NOT detected. Do not shoot against this calibration "
            f"— every miss would look like a shot you did not take."
        )
        out.append(
            "The room is almost certainly too bright: the background is close "
            "enough to saturation that a dot adds nothing on top of it. Dim the "
            "lights, move the camera nearer the target, or point a lamp away "
            "from the target face."
        )
    if report["false_blobs_after"]:
        out.append(
            f"{report['false_blobs_after']} false blobs remain across "
            f"{report['frames']} frames. The audio gate will reject them as "
            f"unpaired, but calibrating the target plane first is better."
        )
    if report.get("dot_flashes_rejected"):
        out.append(
            f"{report['dot_flashes_rejected']} of the measured flashes are still "
            f"bigger than the {report['chosen_max_area']}px ceiling and would not "
            f"be scored. Something in the room is forcing it down — mask to the "
            f"target plane with `dryfire calibrate`, or take the red object out "
            f"of view."
        )
    if report.get("area_overlap"):
        out.append(
            "The laser and the brightest red thing in the room are the same size, "
            "so no ceiling can separate them by area alone. Mask to the target "
            "plane with `dryfire calibrate`, or remove the red object from view."
        )
    if report.get("dot_threshold_dropped"):
        out.append(
            "This calibration uses a lower brightness threshold than the one the "
            "dot was measured at, so the real footprint is now larger than the "
            "measurement and the ceiling no longer bounds it. Re-run "
            "`dryfire spike` to re-measure."
        )
    if not out:
        out.append("Clean: nothing in this room reads as a laser.")
    # Said either way, because the size ceiling is the one threshold that used
    # to be a guess wearing a measurement's clothes.
    if report.get("dot_measured", False):
        out.append(
            f"Size ceiling: max_area {report['chosen_max_area']}px, from a "
            f"measured dot of {report['dot_area']}px (biggest red thing in the "
            f"room: {report['room_red_area']}px)."
        )
    else:
        out.append(
            f"Size ceiling: max_area {report.get('chosen_max_area', 0)}px, which "
            f"is an assumption — no laser has been measured in this room. A dot "
            f"bigger than that is discarded and a fragment of its edge scored in "
            f"its place. Run `dryfire spike` to measure yours."
        )
    return out


def _halo_redness(
    b: np.ndarray,
    g: np.ndarray,
    r: np.ndarray,
    labels: np.ndarray,
    label: int,
    bbox: tuple[int, int, int, int],
    dilate_px: int,
) -> float:
    """Mean(R) - max(mean(G), mean(B)) over a component's dilated halo.

    This is *the* definition of redness in this codebase — deriving it any
    other way is what shipped a margin no laser could clear (DESIGN.md 10a).
    It is computed on a window around the component rather than the whole
    frame, which is the same pixels (the halo reaches at most ``dilate_px``
    beyond the bounding box) at a fraction of the cost. That matters now that
    every candidate is measured before the size ceiling is applied.
    """
    x, y, w, h = bbox
    pad = dilate_px + 1
    y0, y1 = max(0, y - pad), min(labels.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(labels.shape[1], x + w + pad)
    comp = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    halo = cv2.dilate(comp, k)
    mean_b = float(cv2.mean(b[y0:y1, x0:x1], mask=halo)[0])
    mean_g = float(cv2.mean(g[y0:y1, x0:x1], mask=halo)[0])
    mean_r = float(cv2.mean(r[y0:y1, x0:x1], mask=halo)[0])
    return mean_r - max(mean_g, mean_b)


def _weighted_centroid(channel: np.ndarray, component: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid of one connected component."""
    ys, xs = np.nonzero(component)
    w = channel[ys, xs].astype(np.float64)
    total = w.sum()
    if total <= 0:
        return float(xs.mean()), float(ys.mean())
    return float((xs * w).sum() / total), float((ys * w).sum() / total)
