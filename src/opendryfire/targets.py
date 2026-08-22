"""Target geometry and scoring, in real-world inches.

Everything here is defined at **full scale**. Scaling to a printed 1/3 or 1/5
target and converting to simulated distance happens in ``ScaledTarget``, so the
scoring rules are written once and the scale factor cannot be applied twice.

Sources for the numbers:

* USPSA/IPSC Metric target — overall 460 x 760 mm (18.11" x 29.92"), A-zone
  150 x 280 mm (5.906" x 11.024"). The A-zone is the number that matters: at
  1/5 scale it gives 1.18" x 2.20", the size commonly quoted for scaled
  practice targets.
  https://rules.uspsa.org/uspsa/appendix/B1
* NRA B-8 rings (diameters, inches): X 1.695, 10 3.36, 9 5.54, 8 8.00,
  7 11.00, 6 14.80, 5 19.68. The black is the 9-ring outward, i.e. 5.54".
  https://www.indecorous.com/bullseye/rings.html

The A-zone rectangle and the B-8 rings are exact. The USPSA *silhouette outline*
is a documented approximation used only for on-target/off-target calls, never
for scoring — see ``SILHOUETTE_IS_APPROXIMATE``. Measure any printed or 3D
printed target with a ruler before trusting a number from it; a silently
mis-scaled target corrupts every figure downstream and looks entirely normal
while doing so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

# The outline is reconstructed from published overall dimensions rather than
# from the rulebook's vector art, so it is good to a few percent at the edges.
SILHOUETTE_IS_APPROXIMATE = True

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Zone:
    """A scoring region in target-space inches, origin at the A-zone center,
    +x right, +y up."""

    name: str
    points: int
    # Either a rectangle (w, h) centered at (cx, cy), or a circle radius.
    kind: Literal["rect", "circle", "poly"] = "rect"
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    r: float = 0.0
    poly: tuple[tuple[float, float], ...] = ()

    def contains(self, x: float, y: float) -> bool:
        if self.kind == "rect":
            return (
                abs(x - self.cx) <= self.w / 2 and abs(y - self.cy) <= self.h / 2
            )
        if self.kind == "circle":
            return math.hypot(x - self.cx, y - self.cy) <= self.r
        return _point_in_poly(x, y, self.poly)


@dataclass(frozen=True)
class TargetSpec:
    """A target at full scale, in inches."""

    key: str
    name: str
    width: float
    height: float
    # Ordered best-first; the first zone that contains the point wins.
    zones: tuple[Zone, ...]
    # The zone that counts as "clean" for pass/fail drills.
    primary_zone: str
    max_points: int = 0
    notes: str = ""

    def score(self, x: float, y: float) -> tuple[str, int]:
        """Score a hit at target-space inches. Returns (zone_name, points)."""
        for z in self.zones:
            if z.contains(x, y):
                return z.name, z.points
        return "miss", 0

    def zone(self, name: str) -> Zone:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(f"{self.key} has no zone {name!r}")


# -- USPSA / IPSC Metric ------------------------------------------------------

_A_W, _A_H = 150 / MM_PER_INCH, 280 / MM_PER_INCH  # 5.906" x 11.024" (exact)
_USPSA_W, _USPSA_H = 460 / MM_PER_INCH, 760 / MM_PER_INCH  # 18.11" x 29.92"

# A-zone center sits above the target's mid-height; the body extends further
# down than up. Offsets below place the A-zone so the outline lands correctly.
_A_CENTER_FROM_TOP = 11.6  # inches from the top edge to the A-zone center

USPSA_METRIC = TargetSpec(
    key="uspsa",
    name="USPSA/IPSC Metric",
    width=_USPSA_W,
    height=_USPSA_H,
    primary_zone="A",
    zones=(
        Zone("A", 5, "rect", 0.0, 0.0, _A_W, _A_H),
        # Head A-zone sits above the shoulders.
        Zone("A-head", 5, "rect", 0.0, 9.06, 3.94, 2.36),
        Zone("C", 3, "rect", 0.0, 0.0, 11.02, 17.72),
        Zone("D", 1, "rect", 0.0, -1.5, _USPSA_W, 23.0),
    ),
    max_points=5,
    notes="A-zone is exact (150x280mm). Outline and C/D are approximate.",
)


# -- NRA B-8 ------------------------------------------------------------------

_B8_RINGS: tuple[tuple[str, int, float], ...] = (
    ("X", 10, 1.695),
    ("10", 10, 3.36),
    ("9", 9, 5.54),  # edge of the black
    ("8", 8, 8.00),
    ("7", 7, 11.00),
    ("6", 6, 14.80),
    ("5", 5, 19.68),
)

B8 = TargetSpec(
    key="b8",
    name="NRA B-8",
    width=_B8_RINGS[-1][2],
    height=_B8_RINGS[-1][2],
    primary_zone="9",
    zones=tuple(
        Zone(nm, pts, "circle", 0.0, 0.0, r=dia / 2) for nm, pts, dia in _B8_RINGS
    ),
    max_points=10,
    notes="Black is the 9-ring outward (5.54\" diameter).",
)

# Diameter of the black on a B-8, for "was it in the black" calls.
B8_BLACK_DIAMETER = 5.54

TARGETS: dict[str, TargetSpec] = {t.key: t for t in (USPSA_METRIC, B8)}


# -- Scaling ------------------------------------------------------------------


@dataclass(frozen=True)
class ScaledTarget:
    """A physically printed target at some scale, stood at some distance.

    The trick that makes scaled practice work: a 1/5 scale target at 9 ft
    subtends the same angle as a full one at 15 yd, so a group measured here
    converts directly into the group you would shoot at 15 yd.
    ``sim_distance_yd`` is derived, never typed in by hand, so the two can
    never disagree.
    """

    spec: TargetSpec
    scale: float  # 1/3 -> 0.3333, 1/5 -> 0.2
    stand_distance_ft: float

    @property
    def sim_distance_yd(self) -> float:
        """Distance this setup simulates, in yards."""
        return (self.stand_distance_ft / 3.0) / self.scale

    @property
    def printed_width(self) -> float:
        return self.spec.width * self.scale

    @property
    def printed_height(self) -> float:
        return self.spec.height * self.scale

    def score(self, x_in: float, y_in: float) -> tuple[str, int]:
        """Score a hit given *printed-target* inches; converts to full scale."""
        return self.spec.score(x_in / self.scale, y_in / self.scale)

    def to_sim_inches(self, printed_inches: float) -> float:
        """Convert a distance measured on the printed target into the
        equivalent distance at the simulated range."""
        return printed_inches / self.scale

    def describe(self) -> str:
        return (
            f"{self.spec.name} at 1/{round(1 / self.scale)} scale, "
            f"stood at {self.stand_distance_ft:g} ft "
            f"= {self.sim_distance_yd:.0f} yd equivalent "
            f"({self.printed_width:.2f}\" x {self.printed_height:.2f}\" printed)"
        )


# -- Group statistics ---------------------------------------------------------


@dataclass
class GroupStats:
    """Group statistics: size, placement, and shape."""

    n: int
    extreme_spread: float  # inches, printed scale
    sim_extreme_spread: float  # inches at the simulated distance
    centroid: tuple[float, float]
    centroid_offset: float
    mean_radius: float
    shape: str
    in_primary: int = 0

    def summary(self, zone: str = "A") -> str:
        return (
            f"{self.n} shots | group {self.sim_extreme_spread:.2f}\" "
            f"(sim) | {self.in_primary}/{self.n} in {zone} | {self.shape}"
        )


def group_stats(
    hits: list[tuple[float, float]],
    target: ScaledTarget | None = None,
) -> GroupStats:
    """Extreme spread, centroid and shape classification.

    Shape is classified into four buckets — round / low-left / strung vertical
    / off-center — because the fix differs by bucket: low-left is trigger,
    round-but-large is the wobble zone, off-center is a zero problem rather
    than a technique problem.
    """
    n = len(hits)
    if n == 0:
        return GroupStats(0, 0.0, 0.0, (0.0, 0.0), 0.0, 0.0, "no hits")

    xs = [h[0] for h in hits]
    ys = [h[1] for h in hits]
    cx, cy = sum(xs) / n, sum(ys) / n

    spread = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            spread = max(spread, math.hypot(xs[i] - xs[j], ys[i] - ys[j]))

    radii = [math.hypot(x - cx, y - cy) for x, y in hits]
    mean_radius = sum(radii) / n
    offset = math.hypot(cx, cy)

    scale = target.scale if target else 1.0
    return GroupStats(
        n=n,
        extreme_spread=spread,
        sim_extreme_spread=spread / scale,
        centroid=(cx, cy),
        centroid_offset=offset,
        mean_radius=mean_radius,
        shape=classify_shape(hits, (cx, cy), mean_radius),
        in_primary=sum(
            1
            for x, y in hits
            if target and target.score(x, y)[0] == target.spec.primary_zone
        )
        if target
        else 0,
    )


def classify_shape(
    hits: list[tuple[float, float]],
    centroid: tuple[float, float],
    mean_radius: float,
) -> str:
    """Classify a group into four diagnostic buckets.

    Off-center is tested first, because a tight group that is off-centre is a
    sight or zero problem rather than a technique one, and chasing it in dry
    fire is the wrong response.
    """
    n = len(hits)
    if n < 3:
        return "too few shots"

    cx, cy = centroid
    offset = math.hypot(cx, cy)

    # Displaced by more than its own dispersion => systematic, not random.
    if mean_radius > 1e-9 and offset > 1.5 * mean_radius:
        down_left = cx < 0 and cy < 0
        # Low-left specifically is the trigger-jerk signature (right-handed).
        if down_left and abs(cx) > 0.4 * offset and abs(cy) > 0.4 * offset:
            return "low-left (trigger — see Block A)"
        return "off-center (check zero, not technique)"

    # Elongation via the spread of the covariance ellipse.
    vx = sum((x - cx) ** 2 for x, _ in hits) / n
    vy = sum((y - cy) ** 2 for _, y in hits) / n
    vxy = sum((x - cx) * (y - cy) for x, y in hits) / n
    tr, det = vx + vy, vx * vy - vxy * vxy
    disc = max(tr * tr / 4 - det, 0.0)
    lam1 = tr / 2 + math.sqrt(disc)
    lam2 = max(tr / 2 - math.sqrt(disc), 1e-12)
    elongation = math.sqrt(lam1 / lam2)

    if elongation > 2.0:
        vertical = vy > vx
        return "strung vertical" if vertical else "strung horizontal"
    return "round"


def _point_in_poly(x: float, y: float, poly: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint:
                inside = not inside
    return inside
