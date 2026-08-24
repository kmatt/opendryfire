"""End-to-end: a rendered laser dot on a target, scored back into inches.

Each earlier test file checks one stage. This one checks that the stages are
wired together the right way round — the class of bug where every unit passes
and the system still reports a low-left group as high-right.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from opendryfire.detect import DetectorConfig, LaserDetector
from opendryfire.geometry import (
    azone_reference_points,
    calibrate_from_points,
    frame_size_problem,
)
from opendryfire.session import pair_events
from opendryfire.plans import load_plan
from opendryfire.targets import group_stats

PLAN_SETUPS = load_plan().setups

W, H = 1280, 960


def target_to_pixels(target, pts, pose="oblique"):
    """Fake camera with a fixed pose, mapping target inches to pixels."""
    hw = target.printed_width / 2 + 1.0
    hh = target.printed_height / 2 + 1.0
    src = np.array([(-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh)], np.float32)
    dst = {
        "frontal": np.array([(430, 130), (850, 130), (850, 830), (430, 830)], np.float32),
        "oblique": np.array([(410, 160), (865, 120), (890, 815), (445, 860)], np.float32),
    }[pose]
    Hm = cv2.getPerspectiveTransform(src, dst)
    out = []
    for x, y in pts:
        v = Hm @ np.array([x, y, 1.0])
        out.append((float(v[0] / v[2]), float(v[1] / v[2])))
    return out


def render_scene(dot_px=None, warm_background=True):
    """A frame resembling the real setup: cream PLA target, warm wood behind.

    The background is deliberately red-dominant, which is the hard case — it is
    what the actual room looks like and what naive red-channel thresholding
    fails on.
    """
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :] = (28, 42, 78) if warm_background else (40, 40, 40)
    if warm_background:
        cv2.rectangle(frame, (0, 0), (W, 320), (30, 58, 110), -1)  # cherry wood
        cv2.rectangle(frame, (0, 700), (W, H), (75, 120, 160), -1)  # bamboo table
    # Cream PLA target: measured around R=147 G=131 B=103.
    poly = np.array(target_to_pixels(PLAN_SETUPS["f-15"],
                                     [(-1.9, 3.1), (1.9, 3.1), (1.9, -3.1), (-1.9, -3.1)]),
                    np.int32)
    cv2.fillConvexPoly(frame, poly, (103, 131, 147))

    if dot_px is not None:
        cx, cy = int(dot_px[0]), int(dot_px[1])
        ys, xs = np.mgrid[0:H, 0:W]
        dist = np.hypot(xs - cx, ys - cy)
        halo = dist <= 9
        fall = np.clip(1 - dist[halo] / 9, 0, 1)
        r = frame[..., 2].astype(np.float32)
        r[halo] = np.clip(r[halo] + fall * 240, 0, 255)
        frame[..., 2] = r.astype(np.uint8)
        frame[dist <= 3] = 255  # saturated core
    return frame


def calibrated(target, pose="oblique"):
    corners = target_to_pixels(target, azone_reference_points(target), pose)
    return calibrate_from_points(corners, target, (W, H))


@pytest.mark.parametrize("pose", ["frontal", "oblique"])
@pytest.mark.parametrize(
    "where",
    [(0.0, 0.0), (0.4, 0.7), (-0.45, -0.8), (0.5, -0.95)],
)
def test_a_rendered_hit_scores_back_to_where_it_was_placed(pose, where):
    target = PLAN_SETUPS["f-15"]
    tf = calibrated(target, pose)
    px = target_to_pixels(target, [where], pose)[0]

    det = LaserDetector()
    shot = None
    for i in range(6):
        lit = 2 <= i < 5
        s = det.push(render_scene(px if lit else None), timestamp=i / 30)
        shot = shot or s
    shot = shot or det.flush()

    assert shot is not None, "the laser was not detected against a warm background"
    got = tf.to_target(shot.x, shot.y)
    assert got[0] == pytest.approx(where[0], abs=0.05)
    assert got[1] == pytest.approx(where[1], abs=0.05)


def test_the_warm_room_alone_produces_no_shots():
    """The false-positive case: cherry wood and a bamboo table are red-dominant,
    and must still register nothing across a long idle run."""
    det = LaserDetector()
    shots = [det.push(render_scene(None), timestamp=i / 30) for i in range(90)]
    assert [s for s in shots if s] == []
    assert det.flush() is None


def test_a_hit_inside_the_a_zone_scores_a_and_outside_does_not():
    target = PLAN_SETUPS["f-15"]
    # A-zone at 1/5 scale is 1.18 x 2.20", so half-height is 1.10".
    assert target.score(0.0, 1.0)[0] == "A"
    assert target.score(0.0, 1.3)[0] != "A"
    assert target.score(0.5, 0.0)[0] == "A"
    assert target.score(0.8, 0.0)[0] != "A"


def test_a_low_left_group_is_measured_as_low_and_left():
    """The axis-flip check at system level: image y grows downward, target y
    grows upward. Getting this wrong inverts the plan's trigger diagnosis."""
    target = PLAN_SETUPS["f-15"]
    tf = calibrated(target, "oblique")
    placed = [(-0.35, -0.55), (-0.42, -0.48), (-0.38, -0.61), (-0.30, -0.52)]
    recovered = []
    for p in placed:
        px = target_to_pixels(target, [p], "oblique")[0]
        det = LaserDetector()
        shot = None
        for i in range(6):
            s = det.push(render_scene(px if 2 <= i < 5 else None), timestamp=i / 30)
            shot = shot or s
        shot = shot or det.flush()
        assert shot is not None
        recovered.append(tf.to_target(shot.x, shot.y))

    assert all(x < 0 for x, _ in recovered)
    assert all(y < 0 for _, y in recovered)
    stats = group_stats(recovered, target)
    assert "low-left" in stats.shape


def test_group_size_converts_to_the_simulated_distance():
    """Two hits 0.6 inches apart on a 1/5 target is a 3 inch group at 15 yd."""
    target = PLAN_SETUPS["f-15"]
    tf = calibrated(target, "oblique")
    recovered = []
    for p in [(-0.3, 0.0), (0.3, 0.0)]:
        px = target_to_pixels(target, [p], "oblique")[0]
        det = LaserDetector()
        shot = None
        for i in range(6):
            s = det.push(render_scene(px if 2 <= i < 5 else None), timestamp=i / 30)
            shot = shot or s
        shot = shot or det.flush()
        recovered.append(tf.to_target(shot.x, shot.y))

    stats = group_stats(recovered, target)
    assert stats.extreme_spread == pytest.approx(0.6, abs=0.05)
    assert stats.sim_extreme_spread == pytest.approx(3.0, abs=0.25)


def test_a_flash_without_a_click_does_not_become_a_shot():
    """System-level statement of the fusion rule: the camera alone cannot
    score, so a reflection cannot invent a hit."""
    flashes = [(12.0, 600.0, 400.0, 0.0, 3)]
    assert pair_events(flashes, [])[0][1] is None
    assert pair_events(flashes, [12.03])[0][1] is not None


def test_scoring_mask_covers_the_target_without_swallowing_the_room():
    target = PLAN_SETUPS["f-15"]
    tf = calibrated(target, "oblique")
    mask = tf.scoring_mask((H, W))

    # The whole target face is inside.
    for probe in [(0, 0), (-1.7, 2.8), (1.7, 2.8), (-1.7, -2.8), (1.7, -2.8)]:
        px, py = tf.to_image(*probe)
        assert mask[int(py), int(px)] == 255, f"{probe} fell outside the mask"

    # The room is not: far corners excluded, and most of the frame discarded.
    assert mask[5, 5] == 0
    assert mask[H - 5, W - 5] == 0
    covered = float((mask > 0).mean())
    assert covered < 0.35, f"mask covers {covered:.0%} of frame — too loose"


def test_mask_margin_scales_with_the_target():
    """A fixed inch margin would be a sane border on a 1/3 target and a
    frame-filling one on a 1/5 target."""
    third = calibrated(PLAN_SETUPS["b-draw"], "frontal").scoring_mask((H, W))
    fifth = calibrated(PLAN_SETUPS["f-15"], "frontal").scoring_mask((H, W))
    # Both are rendered by the same fake camera, so the 1/5 mask must be the
    # smaller of the two in target-inch terms — but both cover a similar share
    # of frame because the fake camera normalises the target to the same pixels.
    assert 0.05 < (third > 0).mean() < 0.35
    assert 0.05 < (fifth > 0).mean() < 0.35


# -- camera format ------------------------------------------------------------
#
# The camera does not reliably deliver the format it was asked for, and its own
# property read-back echoes the request rather than the truth (test_camera). So
# a session can open in 16:9 against a calibration built in 4:3. This is the
# round trip that says what that costs, because nothing about the output looks
# wrong when it happens — it reads as an ordinary high group.


def project_4_3(target, pts, pose="oblique"):
    """The fake camera above, at the 1920x1440 format the plan asks for."""
    return [(x * 1.5, y * 1.5) for x, y in target_to_pixels(target, pts, pose)]


def as_16_9(px, py):
    """Where a 4:3 pixel appears in the 16:9 stream of the same scene.

    Measured on the reference hardware by aligning frames captured back to back
    in both formats (normalized cross-correlation 0.998): the 16:9 stream is
    the 4:3 view scaled 0.9, cutting ~120 rows off the top and the bottom and
    showing ~107px more on each side. DESIGN.md 16.
    """
    return 0.9 * px + 96, 0.9 * py - 108


@pytest.mark.parametrize("pose", ["frontal", "oblique"])
def test_a_16_9_stream_scored_through_a_4_3_calibration_lands_elsewhere(pose):
    target = PLAN_SETUPS["f-15"]
    tf = calibrate_from_points(
        project_4_3(target, azone_reference_points(target), pose),
        target, (1920, 1440),
    )
    azone_half_width = target.spec.zone(target.spec.primary_zone).w * target.scale / 2

    for probe in [(0.0, 0.0), (0.3, 0.6), (-0.4, -0.7)]:
        (px, py), = project_4_3(target, [probe], pose)
        scored = tf.to_target(*as_16_9(px, py))
        moved = float(np.hypot(scored[0] - probe[0], scored[1] - probe[1]))
        assert moved > azone_half_width, (
            f"{probe} moved only {moved:.2f}\" — if the formats were this close "
            f"the guard would not be worth its weight"
        )

    # And it is not merely a number that shifts: a centred A-zone hit is
    # reported as a different zone entirely.
    (px, py), = project_4_3(target, [(0.0, 0.0)], pose)
    assert target.score(0.0, 0.0)[0] == "A"
    assert target.score(*tf.to_target(*as_16_9(px, py)))[0] != "A"


def test_the_guard_refuses_exactly_that_stream():
    target = PLAN_SETUPS["f-15"]
    tf = calibrate_from_points(
        project_4_3(target, azone_reference_points(target)), target, (1920, 1440)
    )
    assert frame_size_problem(tf, (1920, 1440)) is None
    assert frame_size_problem(tf, (1920, 1080)) is not None
