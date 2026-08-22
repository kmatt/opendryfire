"""Detector tests against synthetic frames.

These exist so the detection logic can be validated without the physical
cartridge. They deliberately model the awkward part: a bright dot saturates its
own core to white, so the red signal lives in the halo, not the peak.
"""

from __future__ import annotations

import numpy as np
import pytest

from opendryfire.detect import DetectorConfig, LaserDetector, Shot


def make_frame(
    w: int = 1280,
    h: int = 720,
    bg: int = 60,
    dot: tuple[int, int] | None = None,
    core_radius: int = 3,
    halo_radius: int = 9,
) -> np.ndarray:
    """A dim room with an optional 635nm dot: saturated white core, red halo."""
    frame = np.full((h, w, 3), bg, dtype=np.uint8)
    if dot is None:
        return frame
    cx, cy = dot
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.hypot(xs - cx, ys - cy)

    halo = dist <= halo_radius
    falloff = np.clip(1.0 - (dist[halo] / halo_radius), 0, 1)
    frame[..., 2][halo] = np.clip(bg + falloff * 220, 0, 255).astype(np.uint8)  # R

    core = dist <= core_radius
    frame[core] = 255  # blown out to white, no red dominance at the peak
    return frame


def run_sequence(
    det: LaserDetector, dots: list[tuple[int, int] | None], fps: float = 60.0
) -> list[Shot]:
    """Drive the detector exactly the way the capture loop does: collect shots
    from ``push`` as they close, then whatever ``flush`` still holds."""
    shots: list[Shot] = []
    dt = 1 / fps
    for i, dot in enumerate(dots):
        if (s := det.push(make_frame(dot=dot), timestamp=i * dt)) is not None:
            shots.append(s)
    if (s := det.flush()) is not None:
        shots.append(s)
    return shots


def lit_frames(n: int, spans: list[tuple[int, int]], dot: tuple[int, int]) -> list:
    """n frames, with ``dot`` present across each [start, stop) span."""
    return [
        dot if any(a <= i < b for a, b in spans) else None for i in range(n)
    ]


def test_finds_a_dot_and_locates_it():
    det = LaserDetector()
    frame = make_frame(dot=(640, 360))
    blobs = det.find_blobs(frame, timestamp=0.0)
    assert len(blobs) == 1
    assert blobs[0].x == pytest.approx(640, abs=1.5)
    assert blobs[0].y == pytest.approx(360, abs=1.5)


def test_empty_frame_yields_nothing():
    det = LaserDetector()
    assert det.find_blobs(make_frame(), timestamp=0.0) == []


def test_rejects_a_white_light_that_is_not_red():
    """A lamp or a reflection is bright but colour-neutral."""
    det = LaserDetector()
    frame = make_frame()
    frame[300:340, 300:340] = 255  # neutral white patch
    assert det.find_blobs(frame, timestamp=0.0) == []


def test_rejects_a_blob_larger_than_a_laser_dot():
    det = LaserDetector(DetectorConfig(max_area=500))
    frame = make_frame(dot=(640, 360), core_radius=40, halo_radius=60)
    assert det.find_blobs(frame, timestamp=0.0) == []


def test_hundred_ms_pulse_at_30fps_is_exactly_one_shot():
    """The real failure mode: 3 frames of the same dot scoring as 3 hits."""
    shots = run_sequence(
        LaserDetector(), lit_frames(30, [(10, 13)], (640, 360)), fps=30
    )
    assert len(shots) == 1
    assert shots[0].frames == 3
    assert shots[0].duration == pytest.approx(2 / 30, abs=1e-6)


def test_hundred_ms_pulse_at_60fps_is_exactly_one_shot():
    shots = run_sequence(
        LaserDetector(), lit_frames(60, [(20, 26)], (400, 300)), fps=60
    )
    assert len(shots) == 1
    assert shots[0].frames == 6


def test_hit_point_comes_from_the_first_frame_not_the_average():
    """The break is frame one. Later frames are muzzle drift and must not
    contaminate the scored position."""
    positions = [(400, 300), (404, 306), (409, 313), (413, 319)]
    dots = [None] * 5 + positions + [None] * 11
    shots = run_sequence(LaserDetector(), dots)
    assert len(shots) == 1
    assert shots[0].x == pytest.approx(400, abs=1.5)
    assert shots[0].y == pytest.approx(300, abs=1.5)
    assert shots[0].drift == pytest.approx(np.hypot(13, 19), abs=2.0)


def test_a_dropped_frame_mid_pulse_does_not_split_the_shot():
    dots = [
        (640, 360) if i in {10, 11, 13, 14} else None for i in range(30)
    ]  # frame 12 dropped by the driver
    shots = run_sequence(LaserDetector(), dots)
    assert len(shots) == 1
    assert shots[0].frames == 4


def test_refractory_window_suppresses_an_immediate_second_pulse():
    shots = run_sequence(
        LaserDetector(DetectorConfig(refractory_s=0.25)),
        lit_frames(60, [(10, 16), (20, 26)], (640, 360)),  # 2nd pulse ~66ms later
    )
    assert len(shots) == 1


def test_two_genuinely_separate_presses_are_two_shots():
    """A Glock needs a slide rack between presses, so real shots are seconds
    apart. They must not be collapsed by the refractory window."""
    shots = run_sequence(
        LaserDetector(), lit_frames(300, [(10, 16), (200, 206)], (640, 360))
    )
    assert len(shots) == 2


# -- scene calibration --------------------------------------------------------


def test_calibration_is_clean_in_an_empty_room():
    from opendryfire.detect import calibrate_scene, describe_calibration

    frames = [make_frame() for _ in range(5)]
    cfg, report = calibrate_scene(frames)
    assert report["false_blobs_after"] == 0
    assert not report["capped"]
    assert "Clean" in " ".join(describe_calibration(report))


def test_one_very_red_light_does_not_blind_the_detector():
    """A red LED on a charger is nearly as red as a laser. Raising the margin
    past it would stop the laser being detected too, so the margin is capped
    and the problem reported instead of silently traded away."""
    from opendryfire.detect import MAX_USABLE_MARGIN, calibrate_scene, describe_calibration

    frames = []
    for _ in range(5):
        f = make_frame()
        f[600:606, 1100:1106] = (10, 10, 250)  # small, intense red LED
        frames.append(f)

    cfg, report = calibrate_scene(frames)
    assert report["capped"] is True
    assert cfg.red_margin <= MAX_USABLE_MARGIN
    advice = " ".join(describe_calibration(report))
    assert "red" in advice.lower()
    assert report["offender_bbox"] is not None


def test_calibration_reports_where_the_offender_is():
    """So it can be identified and moved, rather than just tolerated."""
    from opendryfire.detect import calibrate_scene

    frames = []
    for _ in range(3):
        f = make_frame()
        f[500:504, 200:204] = (5, 5, 252)
        frames.append(f)
    _, report = calibrate_scene(frames)
    x, y, w, h = report["offender_bbox"]
    assert 190 <= x <= 210
    assert 490 <= y <= 510


def test_masking_removes_an_off_target_light_from_consideration():
    """The real fix: the charger is outside the target plane, so once the
    scoring mask exists the light is never examined at all."""
    import numpy as np

    from opendryfire.detect import calibrate_scene

    frames = []
    for _ in range(3):
        f = make_frame()
        f[600:606, 1100:1106] = (10, 10, 250)  # red LED, off to one side
        frames.append(f)

    mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    mask[150:520, 400:900] = 255  # target plane only; excludes the LED

    _, unmasked = calibrate_scene(frames)
    _, masked = calibrate_scene(frames, mask=mask)
    assert unmasked["capped"] is True
    assert masked["capped"] is False
    assert masked["chosen_margin"] < unmasked["chosen_margin"]


def test_config_survives_a_round_trip():
    from opendryfire.detect import DetectorConfig

    cfg = DetectorConfig(bright_threshold=251, red_margin=88.5)
    back = DetectorConfig.from_dict(cfg.to_dict())
    assert back.bright_threshold == 251
    assert back.red_margin == 88.5


# -- a moving dot is still one shot -------------------------------------------


def moving_sequence(det: LaserDetector, positions, fps=30.0):
    """Drive the detector with a dot that moves frame to frame."""
    shots = []
    dt = 1 / fps
    for i, dot in enumerate(positions):
        if (s := det.push(make_frame(dot=dot), timestamp=i * dt)) is not None:
            shots.append(s)
    if (s := det.flush()) is not None:
        shots.append(s)
    return shots


def test_a_dot_that_jumps_far_mid_pulse_is_still_one_shot():
    """Reported from a real session: one press scored as two hits because the
    muzzle moved while the laser was lit. Distance must not split a shot."""
    positions = [None, None] + [(400, 300), (520, 380), (660, 470)] + [None] * 10
    shots = moving_sequence(LaserDetector(), positions)
    assert len(shots) == 1
    assert shots[0].frames == 3


def test_the_scored_position_is_still_the_first_frame_after_a_jump():
    """The break is frame one; the drift afterwards must not move the score."""
    positions = [None, None] + [(400, 300), (520, 380), (660, 470)] + [None] * 10
    shot = moving_sequence(LaserDetector(), positions)[0]
    assert shot.x == pytest.approx(400, abs=2.0)
    assert shot.y == pytest.approx(300, abs=2.0)


def test_a_large_jump_is_recorded_as_drift_rather_than_discarded():
    """The movement is real information — it just is not a second shot."""
    positions = [None, None] + [(400, 300), (520, 380), (660, 470)] + [None] * 10
    shot = moving_sequence(LaserDetector(), positions)[0]
    assert shot.drift == pytest.approx(np.hypot(260, 170), abs=5.0)


def test_a_pulse_longer_than_any_real_one_is_closed_not_extended_forever():
    """A continuously lit dot — a stuck cartridge, or a red light switched on —
    must not accumulate into one enormous shot."""
    det = LaserDetector()
    positions = [None] * 2 + [(400, 300)] * 60  # 2 seconds lit at 30fps
    shots = moving_sequence(det, positions)
    assert shots
    assert shots[0].duration <= DetectorConfig().max_pulse_s + 0.05


def test_two_real_presses_separated_by_a_rack_are_still_two_shots():
    """The fix must not merge genuine shots. A slide rack takes far longer than
    the pulse window."""
    positions = (
        [None] * 5 + [(400, 300)] * 3 + [None] * 40 + [(500, 350)] * 3 + [None] * 5
    )
    shots = moving_sequence(LaserDetector(), positions)
    assert len(shots) == 2


def test_a_dot_reappearing_within_the_refractory_window_is_not_a_new_shot():
    positions = [None] * 3 + [(400, 300)] * 3 + [None] * 2 + [(700, 500)] * 3 + [None] * 5
    shots = moving_sequence(LaserDetector(), positions)
    assert len(shots) == 1
