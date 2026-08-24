"""Geometry tests.

The important ones are the round-trips: a synthetic camera view of a target at
a known pose, scored back into inches, must land where it started. That is the
only way to catch a scale or axis-flip error, which would otherwise look
perfectly plausible in the output and quietly corrupt months of numbers.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from opendryfire.geometry import (
    azone_reference_points,
    build_homography,
    calibrate_from_points,
    drift_score,
    frame_size_problem,
    signature,
)
from opendryfire.plans import load_plan
from opendryfire.targets import USPSA_METRIC, ScaledTarget

PLAN_SETUPS = load_plan().setups


def project(target: ScaledTarget, pts_in, pose="oblique"):
    """Fake camera: map target inches to pixels with a known perspective."""
    hw = target.printed_width / 2 + 1
    hh = target.printed_height / 2 + 1
    src = np.array(
        [(-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh)], dtype=np.float32
    )
    if pose == "frontal":
        dst = np.array(
            [(400, 100), (900, 100), (900, 900), (400, 900)], dtype=np.float32
        )
    else:  # off to one side, as the camera actually sits
        dst = np.array(
            [(380, 140), (910, 95), (940, 880), (415, 930)], dtype=np.float32
        )
    H = cv2.getPerspectiveTransform(src, dst)
    out = []
    for x, y in pts_in:
        v = H @ np.array([x, y, 1.0])
        out.append((float(v[0] / v[2]), float(v[1] / v[2])))
    return out


@pytest.mark.parametrize("pose", ["frontal", "oblique"])
@pytest.mark.parametrize("setup_key", ["b-draw", "f-15", "f-25"])
def test_round_trip_recovers_the_original_inches(pose, setup_key):
    target = PLAN_SETUPS[setup_key]
    corners_in = azone_reference_points(target)
    corners_px = project(target, corners_in, pose)
    frame = calibrate_from_points(corners_px, target, (1280, 1024))

    probes = [(0.0, 0.0), (0.3, 0.5), (-0.4, -0.6), (0.55, -0.9)]
    probes_px = project(target, probes, pose)
    for (want_x, want_y), (px, py) in zip(probes, probes_px):
        got_x, got_y = frame.to_target(px, py)
        assert got_x == pytest.approx(want_x, abs=0.01)
        assert got_y == pytest.approx(want_y, abs=0.01)


def test_y_axis_points_up_not_down():
    """Image y grows downward. If the flip is missed, every low-left group is
    reported as high-left and the Block A diagnosis inverts."""
    target = PLAN_SETUPS["f-15"]
    corners_px = project(target, azone_reference_points(target), "frontal")
    frame = calibrate_from_points(corners_px, target, (1280, 1024))

    top_px = project(target, [(0.0, 1.0)], "frontal")[0]
    bottom_px = project(target, [(0.0, -1.0)], "frontal")[0]
    assert top_px[1] < bottom_px[1]  # sanity: up is a smaller pixel row
    assert frame.to_target(*top_px)[1] > 0
    assert frame.to_target(*bottom_px)[1] < 0


def test_azone_corners_match_the_published_dimensions():
    """1/5 scale must reproduce the 1.2 x 2.2 inch A-zone the plan quotes."""
    target = PLAN_SETUPS["f-15"]
    pts = azone_reference_points(target)
    width = pts[1][0] - pts[0][0]
    height = pts[0][1] - pts[3][1]
    assert width == pytest.approx(1.18, abs=0.02)
    assert height == pytest.approx(2.20, abs=0.02)


def test_scoring_is_scale_aware():
    """The same printed-inch offset is a different score at different scales,
    because the printed target is a different size."""
    f15 = PLAN_SETUPS["f-15"]  # 1/5
    b = PLAN_SETUPS["b-draw"]  # 1/3
    # 0.9" above centre: outside a 1/5 A-zone (half-height 1.10"), inside 1/3.
    assert f15.score(0.0, 1.2)[0] != "A"
    assert b.score(0.0, 1.2)[0] == "A"


def test_simulated_distance_is_derived_not_typed():
    assert PLAN_SETUPS["f-15"].sim_distance_yd == pytest.approx(15.0, abs=0.01)
    assert PLAN_SETUPS["f-25"].sim_distance_yd == pytest.approx(25.0, abs=0.01)
    assert PLAN_SETUPS["b-draw"].sim_distance_yd == pytest.approx(7.0, abs=0.01)
    assert PLAN_SETUPS["test"].sim_distance_yd == pytest.approx(10.0, abs=0.01)


def test_group_measured_at_one_fifth_scale_converts_to_sim_inches():
    """A 0.6 inch group on a 1/5 target is a 3 inch group at 15 yards — the
    number the plan's Distance Card compares against."""
    assert PLAN_SETUPS["f-15"].to_sim_inches(0.6) == pytest.approx(3.0, abs=0.01)


def test_degenerate_points_raise_rather_than_returning_nonsense():
    with pytest.raises((ValueError, RuntimeError)):
        build_homography(
            [(0, 0), (1, 1), (2, 2), (3, 3)],  # collinear
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        )


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        build_homography([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])


def test_reprojection_error_is_small_for_a_clean_fit():
    target = PLAN_SETUPS["f-15"]
    corners_px = project(target, azone_reference_points(target), "oblique")
    frame = calibrate_from_points(corners_px, target, (1280, 1024))
    assert frame.reprojection_error() < 0.01


def test_calibration_survives_a_save_load_round_trip(tmp_path):
    target = PLAN_SETUPS["f-25"]
    corners_px = project(target, azone_reference_points(target), "oblique")
    frame = calibrate_from_points(corners_px, target, (1280, 1024))
    path = tmp_path / "cal.json"
    frame.save(path)
    from opendryfire.geometry import TargetFrame

    back = TargetFrame.load(path)
    probe = project(target, [(0.31, -0.47)], "oblique")[0]
    assert back.to_target(*probe) == pytest.approx(frame.to_target(*probe), abs=1e-9)
    assert back.scale == frame.scale


def structured_scene() -> np.ndarray:
    """A scene with large-scale structure, like a real room. Uniform noise is
    the wrong model: it averages to flat grey under downsampling, so a shifted
    copy looks identical to any low-resolution descriptor."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = (40, 45, 60)
    cv2.rectangle(img, (180, 90), (430, 400), (210, 215, 220), -1)  # the target
    cv2.rectangle(img, (250, 160), (350, 300), (120, 125, 130), 4)  # A-zone
    cv2.circle(img, (560, 60), 45, (200, 190, 120), -1)  # a lamp
    return img


def test_drift_signature_detects_a_moved_camera():
    scene = structured_scene()
    bumped = np.roll(scene, 45, axis=1)
    assert drift_score(signature(scene), signature(scene)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert drift_score(signature(scene), signature(bumped)) > 0.06


def test_drift_signature_tolerates_mere_noise_and_light_flicker():
    """It must not cry wolf on sensor noise, or every session would demand a
    recalibration it does not need."""
    rng = np.random.default_rng(3)
    scene = structured_scene()
    noisy = np.clip(
        scene.astype(np.int16) + rng.integers(-6, 7, scene.shape), 0, 255
    ).astype(np.uint8)
    assert drift_score(signature(scene), signature(noisy)) < 0.06


def test_scoring_mask_covers_the_target_and_excludes_the_room():
    target = PLAN_SETUPS["f-15"]
    corners_px = project(target, azone_reference_points(target), "oblique")
    frame = calibrate_from_points(corners_px, target, (1280, 1024))
    mask = frame.scoring_mask((1024, 1280))
    cx, cy = frame.to_image(0.0, 0.0)
    assert mask[int(cy), int(cx)] == 255  # target centre is inside
    assert mask[10, 10] == 0  # far corner of the room is not
    assert mask.mean() < 200  # most of the frame is excluded


# -- frame geometry -----------------------------------------------------------
#
# A calibration is only valid for the frame geometry it was built in, and the
# camera does not reliably deliver the format it was asked for — see
# ``test_camera``. These are the tests that stop that becoming a scored number.


def calibration_at(size):
    target = PLAN_SETUPS["f-15"]
    corners_px = project(target, azone_reference_points(target), "oblique")
    return calibrate_from_points(corners_px, target, size)


def test_the_format_it_was_built_in_is_accepted():
    tf = calibration_at((1920, 1440))
    assert frame_size_problem(tf, (1920, 1440)) is None


def test_a_different_format_is_refused_and_names_both_sizes():
    tf = calibration_at((1920, 1440))
    problem = frame_size_problem(tf, (1920, 1080))
    assert problem is not None
    assert "1920x1440" in problem and "1920x1080" in problem
    assert "recalibrate" in problem.lower()


def test_a_calibration_with_no_recorded_size_cannot_be_checked():
    """Silence here is not an all-clear, so the caller says so — the point is
    that this must not masquerade as a passed check."""
    tf = calibration_at((0, 0))
    assert frame_size_problem(tf, (1920, 1440)) is None


def test_an_unmeasured_stream_cannot_be_checked_either():
    tf = calibration_at((1920, 1440))
    assert frame_size_problem(tf, (0, 0)) is None
