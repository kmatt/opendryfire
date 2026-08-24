"""Detector tests against synthetic frames.

These exist so the detection logic can be validated without the physical
cartridge. They deliberately model the awkward part: a bright dot saturates its
own core to white, so the red signal lives in the halo, not the peak.
"""

from __future__ import annotations

import numpy as np
import pytest

from opendryfire.detect import (
    DetectorConfig,
    LaserDetector,
    Shot,
    describe_calibration,
)


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
    """A red LED on a charger is nearly as red as a laser.

    What matters is not *how* the detector survives it — it is that the config
    calibration hands back can still see a laser. Asserting the mechanism
    (''the margin got capped'') instead of the property is what let a blinding
    margin ship: the cap was measured in per-pixel redness while the margin is
    tested in halo-mean redness, so the two never met.

    Dilating a 6x6 LED by 5px pulls in enough dark surround to drag its halo
    mean well below a real dot's, so no capping is needed here at all.
    """
    from opendryfire.detect import MAX_USABLE_MARGIN, calibrate_scene

    frames = []
    for _ in range(5):
        f = make_frame()
        f[600:606, 1100:1106] = (10, 10, 250)  # small, intense red LED
        frames.append(f)

    cfg, report = calibrate_scene(frames)
    assert cfg.red_margin <= MAX_USABLE_MARGIN
    assert report["offender_bbox"] is not None
    assert report["sees_reference_dot"] is True
    assert LaserDetector(cfg).find_blobs(make_frame(dot=(640, 360)))
    assert report["false_blobs_after"] == 0


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
    assert masked["chosen_margin"] < unmasked["chosen_margin"]
    assert masked["max_ambient_redness"] == 0.0  # the LED was never examined
    assert masked["offender_bbox"] is None
    # Both must still be able to see a laser; masking only buys sensitivity.
    assert masked["sees_reference_dot"] is True
    assert unmasked["sees_reference_dot"] is True


# -- the margin must be in the units it is tested in ---------------------------


def test_the_margin_is_measured_the_way_find_blobs_measures_redness():
    """The bug this pins: calibration used to threshold on the maximum
    *per-pixel* redness while ``find_blobs`` tests the *mean over a dilated
    halo*. Halo mean is far lower, so the margin came out inflated by a factor
    of two or more and no real dot could clear it."""
    from opendryfire.detect import calibrate_scene

    # A specular highlight on warm wood: per-pixel redness 78, halo mean far
    # lower once the surrounding dark background is dilated in.
    frames = []
    for _ in range(3):
        f = make_frame()
        f[500:508, 900:908] = (172, 172, 250)
        frames.append(f)

    per_pixel = 250 - 172
    _, report = calibrate_scene(frames)
    assert report["max_ambient_redness"] < per_pixel / 2, (
        "calibration is still reporting per-pixel redness, not halo mean"
    )
    # And the number it reports is the one find_blobs would produce.
    probe = LaserDetector(
        DetectorConfig(bright_threshold=report["chosen_threshold"],
                       red_margin=float("-inf"))
    )
    seen = max(b.redness for b in probe.find_blobs(frames[0]))
    assert report["max_ambient_redness"] == pytest.approx(seen, abs=0.1)


def test_a_warm_surface_does_not_produce_a_margin_no_laser_can_clear():
    """The reported failure, end to end: autocal ran, wrote a margin of ~98
    against warm wood, and every subsequent spike saw nothing while the report
    said the room was clean."""
    from opendryfire.detect import MAX_USABLE_MARGIN, calibrate_scene

    frames = []
    for _ in range(3):
        f = make_frame()
        f[500:508, 900:908] = (172, 172, 250)  # per-pixel redness 78
        frames.append(f)

    cfg, report = calibrate_scene(frames)
    assert cfg.red_margin <= MAX_USABLE_MARGIN
    assert report["sees_reference_dot"] is True
    assert LaserDetector(cfg).find_blobs(make_frame(dot=(640, 360)))

    # The margin the old per-pixel derivation would have chosen is blinding —
    # this is what makes the test a regression test rather than a tautology.
    blinding = DetectorConfig(bright_threshold=cfg.bright_threshold,
                              red_margin=78 * 1.25)
    assert not LaserDetector(blinding).find_blobs(make_frame(dot=(640, 360)))


def test_calibration_never_hands_back_a_config_that_cannot_see_a_laser():
    """The invariant. Whatever the room, the config that comes out must still
    detect a dot — a detector that sees nothing trivially has no false
    positives, so 'zero false blobs' alone can never establish this."""
    from opendryfire.detect import calibrate_scene

    scenes = {
        "empty room": lambda f: f,
        "red LED": lambda f: _paint(f, 600, 1100, 6, (10, 10, 250)),
        "warm highlight": lambda f: _paint(f, 500, 900, 8, (172, 172, 250)),
        "bright warm surface": lambda f: _paint(f, 400, 800, 140, (120, 195, 255)),
        "several red lights": lambda f: _paint(
            _paint(f, 600, 1100, 6, (10, 10, 250)), 200, 300, 5, (20, 20, 248)
        ),
    }
    for name, paint in scenes.items():
        frames = [paint(make_frame()) for _ in range(3)]
        cfg, report = calibrate_scene(frames)
        assert report["sees_reference_dot"] is True, f"blind config for {name}"
        assert LaserDetector(cfg).find_blobs(make_frame(dot=(640, 360))), (
            f"cannot see a real dot after calibrating against {name}"
        )


def _paint(frame, y, x, size, bgr):
    frame[y:y + size, x:x + size] = bgr
    return frame


def test_sees_reference_dot_is_false_for_a_margin_no_dot_can_clear():
    """The check that was missing. A margin of 100 is what autocal actually
    wrote in the reported failure; nothing in the old code noticed."""
    from opendryfire.detect import sees_reference_dot

    room = make_frame()
    assert sees_reference_dot(room, DetectorConfig()) is True
    assert sees_reference_dot(room, DetectorConfig(red_margin=100.0)) is False


def test_a_blind_calibration_is_reported_as_blind_and_never_as_clean():
    """The reported symptom was a cheerful 'Clean: nothing in this room reads
    as a laser' printed by a detector that could not see the laser either."""
    from opendryfire.detect import describe_calibration

    report = {
        "frames": 60,
        "brightest_red_pixel": 255,
        "max_ambient_redness": 42.0,
        "chosen_threshold": 250,
        "chosen_margin": 100.0,
        "false_blobs_after": 0,
        "masked": True,
        "capped": False,
        "wanted_margin": 100.0,
        "offender_bbox": None,
        "sees_reference_dot": False,
    }
    advice = " ".join(describe_calibration(report))
    assert "BLIND" in advice
    assert "Clean" not in advice


def test_a_shot_is_returned_long_after_its_dot_has_gone_dark():
    """Why anything wanting the pixels of the break must use ``open_shot``.

    ``push`` returns a Shot when the pulse *ends*, which is gap_frames+1 frames
    after the last sighting. A caller that crops the frame it is holding at that
    moment saves a picture of the target with no laser on it — which is how
    `dryfire spike` came to save eight crops of blank target face and give no
    way to tell a real detection from a false one.
    """
    lit = {2, 3, 4}
    det = LaserDetector(DetectorConfig())
    opened_on, returned_on = None, None
    for i in range(12):
        frame = make_frame(dot=(640, 360) if i in lit else None)
        was_open = det.open_shot is not None
        shot = det.push(frame, timestamp=i / 30.0)
        if not was_open and det.open_shot is not None:
            opened_on = i
        if shot is not None:
            returned_on = i
            closing_frame_has_dot = bool((frame[..., 2] >= 248).any())

    assert opened_on == min(lit), "open_shot must appear on the first lit frame"
    assert returned_on is not None and returned_on > max(lit)
    assert not closing_frame_has_dot, (
        "the frame push returns on has no dot — cropping it proves nothing"
    )


def test_open_shot_holds_the_break_position_while_the_pulse_runs():
    """The position is fixed at the break, so a crop taken when the shot opens
    is centred on the same pixel the finished Shot will report."""
    det = LaserDetector(DetectorConfig())
    det.push(make_frame(dot=(700, 400)), timestamp=0.0)
    assert det.open_shot is not None
    x, y = det.open_shot.x, det.open_shot.y

    for i in (1, 2):  # muzzle drifts while the laser is still lit
        det.push(make_frame(dot=(700 + 12 * i, 400 + 12 * i)), timestamp=i / 30.0)
    assert (det.open_shot.x, det.open_shot.y) == (x, y)

    shot = None
    for i in range(3, 10):
        shot = shot or det.push(make_frame(dot=None), timestamp=i / 30.0)
    assert shot is not None
    assert (shot.x, shot.y) == (x, y)
    assert shot.drift > 0  # the drift was recorded, not scored as a new hit


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


# -- the size ceiling ---------------------------------------------------------
#
# ``max_area`` was a fixed 4000px guess, and the real dot on the reference setup
# measured over 9,500px. So every press was discarded and whatever fragment of
# its edge survived was scored instead — a wrong hit position that looks exactly
# like an ordinary one (DESIGN.md 17). These tests are what stops that being
# silent again.


def splash_frame(w=1280, h=720, surface=150, centre=(640, 360), radius=60,
                 engraving=None):
    """A close, over-exposed dot on a bright surface.

    Modelled the way the real thing behaves: the incident light scales with
    what the surface reflects, so a dark engraved groove under the splash stays
    dark and cuts the bright region in two. ``surface=150`` is the cream PLA
    target as measured through the camera.
    """
    frame = np.full((h, w, 3), surface, dtype=np.uint8)
    if engraving is not None:
        x0, x1 = engraving
        frame[:, x0:x1] = 12  # an engraved groove reflects almost nothing
    cx, cy = centre
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.hypot(xs - cx, ys - cy)
    albedo = frame[..., 2].astype(np.float64) / 255.0
    falloff = np.clip((1.0 - dist / radius) * 3.0, 0, 1)
    red = frame[..., 2].astype(np.float64) + 255 * falloff * albedo
    frame[..., 2] = np.clip(red, 0, 255).astype(np.uint8)
    frame[dist <= radius / 3] = 255  # blown-out core
    return frame


def test_a_dot_too_big_for_the_ceiling_is_recorded_not_silently_dropped():
    det = LaserDetector(DetectorConfig(bright_threshold=200, red_margin=35))
    blobs = det.find_blobs(splash_frame(radius=60), timestamp=0.0)

    assert blobs == []  # nothing scoreable, which is correct
    assert det.oversized_frames == 1  # but it is on the record
    assert det.oversized[0].area > det.cfg.max_area
    assert det.oversized[0].redness >= det.cfg.red_margin


def test_a_dot_that_fits_the_ceiling_is_scored_and_not_recorded_as_oversized():
    det = LaserDetector(DetectorConfig(bright_threshold=200, red_margin=35))
    blobs = det.find_blobs(splash_frame(radius=20), timestamp=0.0)

    assert len(blobs) == 1
    assert blobs[0].x == pytest.approx(640, abs=3)
    assert det.oversized_frames == 0


def test_an_engraved_line_splits_a_splash_into_a_fragment_that_scores_wrong():
    """The whole bug in one frame: the splash is too big, a dark engraved line
    cuts a piece off it, and the piece is small enough to pass. The fragment is
    nowhere near the dot, and without the oversized record nothing says so."""
    det = LaserDetector(DetectorConfig(bright_threshold=200, red_margin=35))
    frame = splash_frame(radius=60, engraving=(660, 672))
    blobs = det.find_blobs(frame, timestamp=0.0)

    assert det.oversized_frames == 1, "the main splash must be on the record"
    assert blobs, "a fragment survived the ceiling"
    assert abs(blobs[0].x - 640) > 30, (
        "the fragment is off to the side of the dot — this is the wrong hit "
        "position the ceiling produces"
    )


def test_the_blindness_probe_can_fail_on_a_dot_the_size_this_room_produces():
    """``sees_reference_dot`` always probed with a 9px dot, whatever the room
    actually produced — so it passed while the real dot was being thrown away
    for size. Given the measured radius it can now fail."""
    from opendryfire.detect import sees_reference_dot

    # A ceiling low enough that only the area gate can reject the big dot —
    # both probes clear the brightness and redness gates identically.
    cfg = DetectorConfig(bright_threshold=200, red_margin=35, max_area=1000)
    scene = make_frame(bg=60)

    assert sees_reference_dot(scene, cfg), "a 9px dot fits and is found"
    assert not sees_reference_dot(scene, cfg, halo_px=60), (
        "a dot of this size does not fit under max_area, and a check that "
        "cannot notice that is the check that let this bug ship"
    )


def test_a_measured_dot_sizes_the_ceiling_that_would_have_rejected_it():
    from opendryfire.detect import DotMeasurement, calibrate_scene

    frames = [make_frame() for _ in range(3)]
    dot = DotMeasurement(flashes=10, areas=[9000, 11000, 10500],
                         peaks=[255] * 3, rednesses=[70, 66, 71],
                         bright_threshold=200)

    cfg, report = calibrate_scene(frames, dot=dot)

    assert report["dot_measured"]
    assert cfg.max_area >= 11000, "the ceiling must clear the biggest real flash"
    assert report["dot_flashes_rejected"] == 0
    assert report["sees_reference_dot"], "a dot of the measured size is found"
    assert "Size ceiling" in " ".join(describe_calibration(report))


def test_without_a_measurement_the_ceiling_is_reported_as_an_assumption():
    from opendryfire.detect import calibrate_scene

    cfg, report = calibrate_scene([make_frame() for _ in range(3)])

    assert not report["dot_measured"]
    assert cfg.max_area == DetectorConfig().max_area
    said = " ".join(describe_calibration(report))
    assert "assumption" in said and "spike" in said


def test_measure_dot_groups_frames_of_one_press_into_one_flash():
    from opendryfire.detect import measure_dot

    det = LaserDetector(DetectorConfig(bright_threshold=200, red_margin=35))
    # two presses, four frames apart in time, each lit for three frames
    for i in range(3):
        det.push(splash_frame(radius=60), timestamp=i / 30)
    for i in range(3):
        det.push(splash_frame(radius=60), timestamp=2.0 + i / 30)

    dot = measure_dot(det, [], measured="2026-08-23")
    assert dot.flashes == 2
    assert dot.largest > 4000
    assert dot.rejected_by(4000) == 2
    assert dot.radius_px > 30


def test_windowing_the_halo_did_not_change_what_redness_means():
    """Redness is now computed on a window around each component instead of the
    whole frame, so that every candidate can be measured before the size gate.
    It has to be the same number: one definition, or the margin and the test it
    is compared against drift apart (DESIGN.md 10a)."""
    import cv2
    from opendryfire.detect import _halo_redness

    frame = splash_frame(radius=20, engraving=(660, 672))
    b, g, r = cv2.split(frame)
    mask = (r >= 200).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    assert n > 1

    for i in range(1, n):
        bbox = (
            int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        # the old implementation: dilate the component across the whole frame
        component = (labels == i).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5 * 2 + 1,) * 2)
        halo = cv2.dilate(component, k)
        whole_frame = float(cv2.mean(r, mask=halo)[0]) - max(
            float(cv2.mean(g, mask=halo)[0]), float(cv2.mean(b, mask=halo)[0])
        )
        assert _halo_redness(b, g, r, labels, i, bbox, 5) == pytest.approx(
            whole_frame, abs=1e-9
        )
