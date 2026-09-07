"""Mapping camera pixels to target-plane inches.

A homography maps one *plane* to another, so every reference point used to build
it must lie in the same physical plane as the scoring surface. This is the one
place where a plausible-looking setup silently produces wrong numbers:

    Markers on the table, target standing vertically  -> WRONG
    Markers on a backer board behind the target       -> wrong by the parallax
                                                         of the offset
    Points on the target's own face                   -> correct

With the camera ~1m away and 20 degrees off-axis, a marker plane sitting 20mm
behind the target face displaces a hit by roughly 7mm — about 0.3", which is a
quarter of a 1/5-scale A-zone. That is invisible in the output and would corrupt
a whole phase of training, so the default reference is the target's *own*
engraved A-zone, which cannot be out of plane with itself.

ArUco remains available for flat setups (a paper target taped to a wall with
markers printed on the same sheet), where coplanarity is automatic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .targets import ScaledTarget

ARUCO_DICT = cv2.aruco.DICT_4X4_50
# Marker ids, in the order top-left, top-right, bottom-right, bottom-left.
ARUCO_IDS = (0, 1, 2, 3)

# Half width/height of each target at full scale, in inches — used to size the
# scoring mask. USPSA Metric is 460 x 760 mm; the B-8's outermost 5-ring is
# 19.68" across.
TARGET_HALF_EXTENTS: dict[str, tuple[float, float]] = {
    "uspsa": (9.06, 14.96),
    "b8": (9.84, 9.84),
}


@dataclass
class TargetFrame:
    """A calibrated mapping from image pixels to printed-target inches.

    Target-space origin is the A-zone centre, +x right, +y **up** (image y runs
    down, so the homography flips it). Units are inches on the *printed* target;
    convert to simulated-distance inches with ``ScaledTarget.to_sim_inches``.
    """

    homography: np.ndarray  # 3x3, image px -> target inches
    reference: str  # what was clicked/detected, for the record
    image_size: tuple[int, int]
    target_key: str = ""
    scale: float = 1.0
    stand_distance_ft: float = 0.0
    # Points used to build it, kept so a calibration can be audited or redrawn.
    image_points: tuple[tuple[float, float], ...] = ()
    target_points: tuple[tuple[float, float], ...] = ()
    drift_signature: list[float] = field(default_factory=list)

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.homography)

    def to_target(self, px: float, py: float) -> tuple[float, float]:
        """Image pixel -> printed-target inches."""
        v = self.homography @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        """Printed-target inches -> image pixel."""
        v = self.inverse @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def reprojection_error(self) -> float:
        """RMS error, in inches, of the calibration points through the fit.

        With exactly 4 points the homography is exact and this is ~0; it only
        becomes meaningful with 5+ points. Reported so a bad fit is visible.
        """
        if len(self.image_points) < 4:
            return 0.0
        errs = []
        for (px, py), (tx, ty) in zip(self.image_points, self.target_points):
            mx, my = self.to_target(px, py)
            errs.append(np.hypot(mx - tx, my - ty))
        return float(np.sqrt(np.mean(np.square(errs))))

    def scoring_mask(
        self, shape: tuple[int, int], margin_frac: float = 0.25
    ) -> np.ndarray:
        """A mask covering the target plus a proportional margin, in image space.

        This is what removes the room from the problem. The background is warm
        cherry wood and a bamboo table, both red-dominant; masking to the target
        plane means they are never examined at all.

        The margin is a *fraction* of the target, not a fixed number of inches.
        An absolute margin cannot work across scales: two inches around a 1/3
        target is a sensible border, while the same two inches around a 3.6"
        wide 1/5 target more than doubles it and pulls the wall back into frame.

        The margin exists so a shot that just misses is still recorded — the
        plan asks "all 5 in the A-zone? Y/N", and a near miss is a different
        datum from no detection at all.
        """
        h, w = shape[:2]
        spec = TARGET_HALF_EXTENTS.get(self.target_key, (9.06, 14.96))
        half_w = spec[0] * self.scale * (1 + margin_frac)
        half_h = spec[1] * self.scale * (1 + margin_frac)
        corners = [
            (-half_w, half_h),
            (half_w, half_h),
            (half_w, -half_h),
            (-half_w, -half_h),
        ]
        pts = np.array([self.to_image(x, y) for x, y in corners], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        return mask

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "homography": self.homography.tolist(),
            "reference": self.reference,
            "image_size": list(self.image_size),
            "target_key": self.target_key,
            "scale": self.scale,
            "stand_distance_ft": self.stand_distance_ft,
            "image_points": [list(p) for p in self.image_points],
            "target_points": [list(p) for p in self.target_points],
            "drift_signature": self.drift_signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetFrame":
        return cls(
            homography=np.array(d["homography"], dtype=np.float64),
            reference=d.get("reference", ""),
            image_size=tuple(d.get("image_size", (0, 0))),  # type: ignore[arg-type]
            target_key=d.get("target_key", ""),
            scale=d.get("scale", 1.0),
            stand_distance_ft=d.get("stand_distance_ft", 0.0),
            image_points=tuple(tuple(p) for p in d.get("image_points", ())),  # type: ignore[misc]
            target_points=tuple(tuple(p) for p in d.get("target_points", ())),  # type: ignore[misc]
            drift_signature=d.get("drift_signature", []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "TargetFrame":
        return cls.from_dict(json.loads(path.read_text()))


def frame_size_problem(
    tf: TargetFrame, size: tuple[int, int]
) -> str | None:
    """Why frames of ``size`` cannot be scored through ``tf``, or None.

    A calibration maps *these* pixels to inches, so it is valid only for the
    frame geometry it was built in. The camera does not reliably deliver the
    format it was asked for and its own property read-back does not reveal that
    (see ``camera.Camera._configure``), so the check has to be made against
    real frames, here, before anything is scored.

    The two formats are not the same view at two sizes. Measured on the
    reference hardware, the 16:9 stream is the 4:3 one scaled 0.9 with ~120
    rows cut from the top and bottom; scoring 16:9 pixels through a 4:3
    calibration moved the A-zone centre by 0.90", on a zone 1.18" wide. That is
    a whole-zone error which looks like an ordinary high-left group, which is
    why this refuses instead of warning.
    """
    have, want = tuple(size), tuple(tf.image_size)
    if 0 in have or 0 in want or have == want:
        return None
    return (
        f"this calibration was built on {want[0]}x{want[1]} frames but the "
        f"camera is delivering {have[0]}x{have[1]}. Those are different views, "
        f"not one view at two sizes, so every hit would be scored through the "
        f"wrong geometry.\n"
        f"  Fix the format (`dryfire probe` shows what each camera grants), or "
        f"recalibrate at this size with `dryfire calibrate`."
    )


def calibration_format_problem(
    requested: tuple[int, int], delivered: tuple[int, int]
) -> str | None:
    """Why a calibration must not be *built* on ``delivered``, or None.

    §16 stopped the wrong format being scored through; it did not stop one
    being written down. A calibrate run that silently lands in the device's
    default format saves a valid-looking `TargetFrame` whose ``image_size`` is
    a view the shooter never asked for, and the reprojection error is perfect
    because the corners really were clicked in it. The evidence that anything
    is wrong exists only at that moment — every later command sees a good file
    and a stream that disagrees with it, and blames the stream.

    So the refusal belongs here, before the corners are clicked: a format the
    driver did not grant is not a format to record geometry in. Asking for the
    delivered size explicitly is the escape hatch, exactly as a bare
    ``--camera 1`` is for the name-to-index check.
    """
    want, have = tuple(requested), tuple(delivered)
    if 0 in want or 0 in have or want == have:
        return None
    return (
        f"refusing to calibrate on a {have[0]}x{have[1]} stream: "
        f"{want[0]}x{want[1]} was asked for and the camera granted something "
        f"else. Those are different views, not one view at two sizes, so a "
        f"calibration built here would record geometry for a format you did "
        f"not choose — and every later run would either be refused for not "
        f"matching it, or scored through it.\n"
        f"  Format negotiation loses this on roughly one open in eight "
        f"(DESIGN.md 16), so simply running `dryfire calibrate` again usually "
        f"takes it.\n"
        f"  If {have[0]}x{have[1]} is the format you actually want, ask for it:"
        f"  --width {have[0]} --height {have[1]}"
    )


def build_homography(
    image_points: list[tuple[float, float]],
    target_points: list[tuple[float, float]],
    max_residual_in: float = 0.05,
) -> np.ndarray:
    """Least-squares homography from image pixels to target inches.

    ``cv2.findHomography`` does not reliably return None for degenerate input —
    give it four collinear points and it hands back a matrix that maps none of
    them correctly. Since a bad calibration produces plausible-looking but wrong
    inches forever after, the fit is verified here by reprojecting the very
    points it was built from and rejecting it if they do not land.
    """
    if len(image_points) < 4:
        raise ValueError("need at least 4 point correspondences")
    src = np.array(image_points, dtype=np.float64)
    dst = np.array(target_points, dtype=np.float64)

    if _quad_area(src) < 1e-6 or _quad_area(dst) < 1e-9:
        raise ValueError(
            "calibration points are collinear or coincident — click the four "
            "A-zone corners in order (top-left, top-right, bottom-right, "
            "bottom-left), not along one edge"
        )

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None or not np.all(np.isfinite(H)):
        raise RuntimeError("homography fit failed; re-run calibration")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    residual = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    if residual > max_residual_in:
        raise RuntimeError(
            f"homography does not fit its own calibration points "
            f"(residual {residual:.3f}\" > {max_residual_in}\"). The points are "
            f"probably degenerate or clicked out of order."
        )
    return H


def _quad_area(pts: np.ndarray) -> float:
    """Shoelace area of the polygon through the points, as a degeneracy test.
    Collinear or coincident points enclose no area."""
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def azone_reference_points(target: ScaledTarget) -> list[tuple[float, float]]:
    """The four A-zone corners in printed-target inches, clockwise from
    top-left. This is the default calibration reference because it is engraved
    into the target's own face and so is exactly coplanar with the scoring
    surface."""
    z = target.spec.zone(target.spec.primary_zone)
    if z.kind == "circle":
        r = z.r * target.scale
        return [(-r, r), (r, r), (r, -r), (-r, -r)]
    hw = (z.w / 2) * target.scale
    hh = (z.h / 2) * target.scale
    return [(-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh)]


def calibrate_from_points(
    image_points: list[tuple[float, float]],
    target: ScaledTarget,
    image_size: tuple[int, int],
    reference: str = "a-zone corners",
) -> TargetFrame:
    """Build a TargetFrame from four clicked A-zone corners."""
    tgt = azone_reference_points(target)
    if len(image_points) != len(tgt):
        raise ValueError(f"expected {len(tgt)} points, got {len(image_points)}")
    H = build_homography(image_points, tgt)
    return TargetFrame(
        homography=H,
        reference=reference,
        image_size=image_size,
        target_key=target.spec.key,
        scale=target.scale,
        stand_distance_ft=target.stand_distance_ft,
        image_points=tuple(image_points),
        target_points=tuple(tgt),
    )


# -- ArUco path ---------------------------------------------------------------


def detect_aruco(frame: np.ndarray) -> dict[int, np.ndarray]:
    """Return {marker_id: 4x2 corner array} for every marker found."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def calibrate_from_aruco(
    frame: np.ndarray,
    marker_positions: dict[int, tuple[float, float]],
) -> TargetFrame | None:
    """Build a TargetFrame from markers whose centres are at known target-space
    inches. Requires all four; a partial detection returns None rather than a
    degraded fit."""
    found = detect_aruco(frame)
    have = [i for i in marker_positions if i in found]
    if len(have) < 4:
        return None
    img_pts = [tuple(found[i].mean(axis=0)) for i in have]
    tgt_pts = [marker_positions[i] for i in have]
    H = build_homography(img_pts, tgt_pts)  # type: ignore[arg-type]
    return TargetFrame(
        homography=H,
        reference=f"aruco {sorted(have)}",
        image_size=(frame.shape[1], frame.shape[0]),
        image_points=tuple(img_pts),  # type: ignore[arg-type]
        target_points=tuple(tgt_pts),
    )


def make_marker_sheet(
    out_path: Path,
    target: ScaledTarget,
    marker_in: float = 1.0,
    dpi: int = 300,
) -> dict[int, tuple[float, float]]:
    """Render a printable sheet of four ArUco markers around the target area.

    Only correct for *flat* setups — a paper target on the same sheet, taped to
    a wall. Do not use it with a standing 3D printed target: the markers would
    sit in a different plane from the target face. See the module docstring.
    """
    half_w = target.printed_width / 2 + marker_in
    half_h = target.printed_height / 2 + marker_in
    positions = {
        ARUCO_IDS[0]: (-half_w, half_h),
        ARUCO_IDS[1]: (half_w, half_h),
        ARUCO_IDS[2]: (half_w, -half_h),
        ARUCO_IDS[3]: (-half_w, -half_h),
    }
    sheet_w = int((2 * half_w + marker_in * 2) * dpi)
    sheet_h = int((2 * half_h + marker_in * 2) * dpi)
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
    dic = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    px = int(marker_in * dpi)

    for mid, (x, y) in positions.items():
        img = cv2.aruco.generateImageMarker(dic, mid, px)
        cx = int((x + half_w + marker_in) * dpi)
        cy = int((half_h + marker_in - y) * dpi)
        x0, y0 = cx - px // 2, cy - px // 2
        sheet[y0 : y0 + px, x0 : x0 + px] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return positions


# -- drift detection ----------------------------------------------------------


def signature(frame: np.ndarray, mask: np.ndarray | None = None) -> list[float]:
    """A cheap descriptor of the scene, used to notice the tripod being bumped.

    A downsampled grayscale grid; comparing it frame to frame catches a shifted
    camera without needing feature matching.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float64) / 255.0).ravel().tolist()


def drift_score(a: list[float], b: list[float]) -> float:
    """RMS difference between two signatures. Empirically, >0.06 means the
    camera or the target moved enough to invalidate the calibration."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.array(a), np.array(b)
    return float(np.sqrt(np.mean((va - vb) ** 2)))


DRIFT_THRESHOLD = 0.06
