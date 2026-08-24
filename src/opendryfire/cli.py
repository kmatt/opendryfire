"""Command line entry point.

    dryfire probe                 list cameras, mics, and measure real frame rate
    dryfire use-camera iphone     pick a camera by name and remember it
    dryfire spike                 the 30-second question: does it see the flash?
    dryfire calibrate             click the four A-zone corners, save the mapping
    dryfire run B                 run a block
    dryfire plan                  show the active plan
    dryfire audiocal              separate striker clicks from room noise
    dryfire sounds                preview the hit/miss chimes
    dryfire report                what the sessions say
    dryfire targets               show the geometry, and print marker sheets
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from .audio import (
    SAMPLE_RATE,
    LiveListener,
    OnsetConfig,
    ShotTimer,
    calibrate_click_floor,
)
from .camera import Camera, enumerate_cameras
from .detect import (
    DetectorConfig,
    DotMeasurement,
    LaserDetector,
    calibrate_scene,
    describe_calibration,
    measure_dot,
)
from .devices import (
    CameraMismatch,
    CameraNotFound,
    list_devices,
    preflight,
    resolve,
)
from .drills import ParLadder, coaching_note, evaluate_rep
from .feedback import Chimes
from .feedback import describe as describe_chimes
from .geometry import (
    DRIFT_THRESHOLD,
    TargetFrame,
    calibrate_from_points,
    drift_score,
    frame_size_problem,
    make_marker_sheet,
    signature,
)
from .plans import PlanError, bundled_plans, load_plan
from .session import Rep, Session, ShotRecord, calibration_path, sessions_dir
from .targets import TARGETS, group_stats
from .ui import click_calibration, draw_overlay, scoreboard
from .vault import VaultPaths, append_to_tracker, session_note

ROOT = Path.home() / ".dryfire"


def _load_dot_measurement() -> DotMeasurement | None:
    """The measured laser, if `spike` has ever measured one in this room."""
    path = ROOT / "dot.json"
    if not path.exists():
        return None
    try:
        return DotMeasurement.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _save_dot_measurement(dot: DotMeasurement) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "dot.json"
    path.write_text(json.dumps(dot.to_dict(), indent=2))
    return path


def _oversized_warning(frames: int, largest: int, cfg) -> list[str]:  # noqa: ANN001
    """What to say when the detector threw away a red region for being too big.

    This is the failure that has no symptom of its own: the splash is
    discarded, some fragment of its edge is scored in its place, and the number
    that comes out looks like an ordinary shot in an ordinary group.
    """
    if not frames:
        return []
    return [
        f"{frames} frames held a red region bigger than max_area "
        f"({cfg.max_area}px) — largest {largest}px.",
        "That region is the laser, over-exposed. It was NOT scored; anything "
        "scored at the same moment was a fragment of its edge, not the dot.",
        "Run `dryfire spike` to measure the dot, then `dryfire autocal` to size "
        "the ceiling to it.",
    ]


def _load_detector_config() -> DetectorConfig:
    p = ROOT / "detector.json"
    if p.exists():
        return DetectorConfig.from_dict(json.loads(p.read_text()))
    return DetectorConfig()


def _prefs() -> dict:
    p = ROOT / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_prefs(d: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "config.json").write_text(json.dumps(d, indent=2))


def _plan(args):  # noqa: ANN001, ANN201
    """The active plan, or exit with a readable message."""
    try:
        return load_plan(getattr(args, "plan", None), ROOT)
    except PlanError as exc:
        print(f"\nplan error: {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc


def _camera_spec(args) -> str | int:  # noqa: ANN001
    """Explicit --camera wins, then the saved preference, then index 0."""
    if getattr(args, "camera", None) is not None:
        return args.camera
    return _prefs().get("camera", 0)


def _open_camera(args, quiet: bool = False) -> Camera:  # noqa: ANN001
    spec = _camera_spec(args)
    try:
        cam = Camera(spec, args.width, args.height, args.fps)
    except (CameraNotFound, CameraMismatch) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc
    if not quiet:
        print(f"camera: {cam.name} (index {cam.index}) "
              f"{cam.info.width}x{cam.info.height}")
        for msg in _format_warnings(cam, args):
            print(f"  ! {msg}")
        for msg in preflight(cam.index):
            print(f"  ! {msg}")
    return cam


def _format_warnings(cam: Camera, args) -> list[str]:  # noqa: ANN001
    """What to say when the stream is not the format that was requested.

    The size printed above is measured from a delivered frame, so it can differ
    from both the request and the driver's own read-back. Both differences
    matter: the first changes the field of view a calibration is built in, and
    the second means nothing else the driver reports can be taken at face
    value.
    """
    out: list[str] = []
    if not cam.info.verified:
        out.append(
            "no frame arrived while opening the camera, so its size is the "
            "driver's claim and not a measurement"
        )
        return out
    if cam.info.frame_size != (args.width, args.height):
        out.append(
            f"asked for {args.width}x{args.height}; this device is sending "
            f"{cam.info.width}x{cam.info.height}"
        )
    if cam.info.size_is_a_lie():
        out.append(
            f"the driver reports {cam.info.claimed[0]}x{cam.info.claimed[1]} "
            f"while sending {cam.info.width}x{cam.info.height} — it is "
            f"echoing the request, not the format it configured"
        )
    return out


def _geometry_ok(size: tuple[int, int], tf: TargetFrame, *, fatal: bool) -> bool:
    """Check a live stream against the geometry its calibration was built in.

    Fatal wherever a hit becomes a number — a mismatch there corrupts the log
    silently. A warning in ``spike``, which scores nothing and is a diagnostic
    someone may legitimately run in any format.
    """
    if 0 in tuple(tf.image_size):
        print("  ! this calibration records no frame size, so the camera "
              "geometry cannot be verified. Recalibrate to record one.")
        return True
    problem = frame_size_problem(tf, size)
    if problem is None:
        return True
    if fatal:
        print(f"\n{problem}\n", file=sys.stderr)
        return False
    print(f"  ! {problem}")
    return True


# -- probe --------------------------------------------------------------------


def cmd_probe(args) -> int:  # noqa: ANN001
    print("Cameras")
    cams = enumerate_cameras(4)
    if not cams:
        print("  none opened. On macOS grant camera access to your terminal:")
        print("  System Settings > Privacy & Security > Camera")
        return 1
    current = _prefs().get("camera")
    for c in cams:
        with Camera(c.index, args.width, args.height, args.fps) as cam:
            measured = cam.measure_fps(60)
            star = " *selected" if current is not None and \
                _matches(current, cam) else ""
            print(f"  [{c.index}] {cam.name or 'unnamed'}{star}")
            print(f"       {cam.info.width}x{cam.info.height} "
                  f"measured {measured:.1f}fps (reported {cam.info.reported_fps:.0f})")
            manual = cam.try_manual_exposure()
            print(f"       manual exposure: {'available' if manual else 'NOT available'}")
    for msg in preflight(cams[0].index):
        print(f"  ! {msg}")
    if current is None:
        print("\n  No camera preference saved; using index 0.")
        print("  Set one with:  dryfire use-camera iphone")

    print("\nAudio")
    try:
        import sounddevice as sd

        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] in  {d['name']}")
            if d["max_output_channels"] > 0:
                print(f"  [{i}] out {d['name']}")
    except Exception as exc:  # pragma: no cover
        print(f"  audio unavailable: {exc}")
    return 0


def _matches(spec, cam) -> bool:  # noqa: ANN001
    """Whether a saved preference refers to this open camera."""
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec) == cam.index
    return str(spec).lower() in (cam.name or "").lower()


def cmd_use_camera(args) -> int:  # noqa: ANN001
    """Save which camera to use, by name, so it survives index shuffling."""
    devices = list_devices()
    if not devices:
        print("no cameras found", file=sys.stderr)
        return 1
    try:
        index = resolve(args.name, devices)
    except CameraNotFound as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    dev = next(d for d in devices if d.index == index)
    prefs = _prefs()
    # Store the *name*, not the index: Continuity Camera shifts indices as it
    # connects and disconnects, and a stale index silently selects the wrong
    # lens for a calibration built on the right one.
    prefs["camera"] = dev.name
    _save_prefs(prefs)
    print(f"camera set to: {dev.name} (currently index {dev.index})")
    if dev.is_continuity:
        print("\nContinuity Camera notes:")
        print("  - iPhone must be nearby, unlocked once, Wi-Fi + Bluetooth on")
        print("  - keep it plugged in and still; picking it up drops the feed")
        print("  - turn OFF Center Stage in Control Center > Video Effects")
        print("    (dryfire disables it per-session, but Control Center wins)")
        print("  - recalibrate after any reposition: `dryfire calibrate`")
    for msg in preflight(index, devices):
        print(f"  ! {msg}")
    return 0


# -- spike --------------------------------------------------------------------


def _crop_around(frame: np.ndarray, x: float, y: float, half: int = 60):
    """A square crop centred on a detection, clamped to the frame."""
    h, w = frame.shape[:2]
    cx, cy = int(x), int(y)
    return frame[max(0, cy - half): cy + half, max(0, cx - half): cx + half]


def cmd_spike(args) -> int:  # noqa: ANN001
    """Answer the only question that gates the whole project: in this room,
    with this camera, does a trigger press produce a detectable flash?"""
    cfg = _load_detector_config()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    det = LaserDetector(cfg)
    chimes = Chimes(volume=args.volume, enabled=not args.no_sound)
    shots: list = []

    print(f"Dry fire {args.expect} times at the target. ESC or q to stop early.")
    print(f"Thresholds: brightness>={cfg.bright_threshold} redness>={cfg.red_margin:.0f}")
    print("The tone means 'laser seen', NOT 'hit' — a spike has no target")
    print("geometry yet, so a shot on the wall sounds the same as one on the A-zone.")
    print("Saving a crop of every detection so false positives are visible.\n")

    pending: np.ndarray | None = None
    with _open_camera(args) as cam:
        # A spike scores nothing, so a format change cannot corrupt a number
        # here — but it silently changes the view, and the preview looking
        # unlike the calibration window is exactly the symptom that sent
        # someone looking. Say it out loud instead.
        cal = calibration_path(ROOT, getattr(args, "profile", "default"))
        if cal.exists():
            _geometry_ok(cam.info.frame_size, TargetFrame.load(cal), fatal=False)
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            r = cam.read()
            if r is None:
                continue
            frame, t = r
            was_open = det.open_shot is not None
            shot = det.push(frame, t)
            # Grab the crop on the frame where the pulse *opens*. By the time
            # push hands back a Shot the dot has been dark for several frames,
            # so cropping then saves a picture of the target with no laser on
            # it — which is exactly the evidence the crop exists to provide.
            if not was_open and det.open_shot is not None:
                pending = _crop_around(frame, det.open_shot.x, det.open_shot.y)
            if shot is not None:
                # A spike runs before calibration, so there is no target
                # geometry and nothing here knows a hit from a miss. Play the
                # neutral "seen" tone rather than "hit": a rising hit chime for
                # a shot that landed on the wall is actively misleading.
                chimes.play("seen")
                shots.append(shot)
                n = len(shots)
                crop = pending if pending is not None else _crop_around(
                    frame, shot.x, shot.y
                )
                pending = None
                if crop is not None and crop.size:
                    cv2.imwrite(str(out / f"hit_{n:02d}.jpg"), crop)
                print(f"  {n:2d}. ({shot.x:7.1f},{shot.y:7.1f})  "
                      f"{shot.frames} frames  {shot.duration * 1000:5.1f}ms  "
                      f"peak R={shot.peak}  drift={shot.drift:.1f}px")
            preview = frame.copy()
            for s in shots:
                cv2.circle(preview, (int(s.x), int(s.y)), 10, (0, 255, 0), 2)
            cv2.putText(preview, f"detections: {len(shots)}/{args.expect}",
                        (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("dryfire spike", preview)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
        final = det.flush()
        if final is not None:
            shots.append(final)
    chimes.stop()
    cv2.destroyAllWindows()

    print(f"\n{len(shots)} detections for {args.expect} presses.")
    print(f"Crops in {out} — check every one is a real hit before trusting the count.")

    # The measurement this whole command exists to be able to take: how big the
    # laser's footprint actually is in this room, at this distance, on this
    # surface. Everything that judges a blob by size was guessing until now.
    dot = measure_dot(det, shots, measured=date.today().isoformat())
    for line in _oversized_warning(det.oversized_frames,
                                   max((o.area for o in det.oversized), default=0),
                                   cfg):
        print(f"  ! {line}")
    print("\nLaser measurement")
    for line in dot.lines():
        if line:
            print(f"  {line}")
    if dot.areas:
        rejected = dot.rejected_by(cfg.max_area)
        print(f"  rejected by the current max_area ({cfg.max_area}px): "
              f"{rejected} of {dot.flashes}")
        path = _save_dot_measurement(dot)
        print(f"  saved to {path} — `dryfire autocal` sizes the ceiling from it")
        if dot.flashes != args.expect:
            print(f"  note: {dot.flashes} flashes measured for {args.expect} "
                  f"presses; the measurement is only as good as that match")

    if len(shots) == args.expect:
        print("\nClean. Detection is not the bottleneck; go to `dryfire calibrate`.")
    elif len(shots) < args.expect:
        print("\nUnder-detecting. Dim the room, or move the camera closer to the target.")
        print("Then lower --margin/--threshold via `dryfire autocal`.")
    else:
        print("\nOver-detecting. Run `dryfire autocal` to raise thresholds "
              "against this room.")
    return 0


# -- autocal ------------------------------------------------------------------


def cmd_autocal(args) -> int:  # noqa: ANN001
    """Derive detector thresholds from the room with no laser present."""
    print(f"Capturing {args.frames} frames of the static scene. Do not fire.")
    frames = []
    with _open_camera(args) as cam:
        for _ in range(15):
            cam.read()
        while len(frames) < args.frames:
            r = cam.read()
            if r is not None:
                frames.append(r[0])

    mask = None
    cal = calibration_path(ROOT, args.profile)
    if cal.exists() and not args.no_mask:
        tf = TargetFrame.load(cal)
        # The mask is drawn in the calibration's pixel space. In a different
        # format it lands somewhere else in the scene, and thresholds derived
        # through it describe the wrong patch of room.
        h, w = frames[0].shape[:2]
        if not _geometry_ok((w, h), tf, fatal=True):
            return 1
        mask = tf.scoring_mask(frames[0].shape)
        print("Restricting calibration to the calibrated target plane.")

    dot = _load_dot_measurement()
    if dot is None:
        print("No laser measurement found; the size ceiling will be an "
              "assumption. `dryfire spike` measures it.")
    else:
        print(f"Using the laser measured on {dot.measured or 'an earlier run'}: "
              f"{dot.flashes} flashes, largest {dot.largest}px.")

    cfg, report = calibrate_scene(frames, mask, dot=dot)
    print(f"  brightest red pixel : {report['brightest_red_pixel']}")
    print(f"  max ambient redness : {report['max_ambient_redness']}")
    print(f"  chosen threshold    : {report['chosen_threshold']}")
    print(f"  chosen margin       : {report['chosen_margin']}"
          + (f"  (capped from {report['wanted_margin']})" if report["capped"] else ""))
    print(f"  chosen max area     : {report['chosen_max_area']}"
          + (f"  (dot {report['dot_area']}px, room {report['room_red_area']}px)"
             if report["dot_measured"] else "  (assumed — no dot measured)"))
    print(f"  false blobs after   : {report['false_blobs_after']}")
    print(f"  masked to target    : {report['masked']}")

    print()
    for line in describe_calibration(report):
        print(f"  {line}")

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "detector.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"\nSaved to {ROOT / 'detector.json'}")
    return 0


# -- calibrate ----------------------------------------------------------------


def cmd_calibrate(args) -> int:  # noqa: ANN001
    plan = _plan(args)
    setup_key = args.setup or next(iter(plan.setups))
    try:
        target = plan.setup(setup_key)
    except PlanError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    if args.setup is None:
        print(f"no --setup given; using first in plan: {setup_key}")
    print(target.describe())
    print("Click the four corners of the engraved A-zone, in order.")

    with _open_camera(args) as cam:
        frame = None
        for _ in range(20):
            r = cam.read()
            if r is not None:
                frame = r[0]
        if frame is None:
            print("no frame from camera", file=sys.stderr)
            return 1

    pts = click_calibration(frame, target)
    if pts is None:
        print("cancelled")
        return 1

    try:
        tf = calibrate_from_points(pts, target, (frame.shape[1], frame.shape[0]))
    except (ValueError, RuntimeError) as exc:
        print(f"calibration rejected: {exc}", file=sys.stderr)
        return 1

    tf.drift_signature = signature(frame)
    path = calibration_path(ROOT, args.profile)
    tf.save(path)

    print(f"\nSaved {path}")
    print(f"  reference:  {tf.reference}")
    print(f"  reprojection error: {tf.reprojection_error():.4f}\"")
    w = target.printed_width
    print(f"  sanity: target should be {w:.2f}\" wide, "
          f"{target.printed_height:.2f}\" tall on the print")
    print(f"  1 pixel ~= {_inches_per_pixel(tf):.4f}\" at the target "
          f"({_inches_per_pixel(tf) / target.scale:.3f}\" at "
          f"{target.sim_distance_yd:.0f}yd equivalent)")
    return 0


def _inches_per_pixel(tf: TargetFrame) -> float:
    a = tf.to_target(0, 0)
    b = tf.to_target(100, 0)
    return float(np.hypot(b[0] - a[0], b[1] - a[1]) / 100)


# -- run ----------------------------------------------------------------------


def cmd_run(args) -> int:  # noqa: ANN001
    plan = _plan(args)
    try:
        block = plan.block(args.block)
    except PlanError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    target = block.target

    cal = calibration_path(ROOT, args.profile)
    if not cal.exists():
        print(f"no calibration at {cal}. Run `dryfire calibrate` first.", file=sys.stderr)
        return 1
    tf = TargetFrame.load(cal)
    if abs(tf.scale - target.scale) > 1e-6:
        print(f"calibration is for 1/{round(1 / tf.scale)} scale but block "
              f"{block.key} needs 1/{round(1 / target.scale)}. Recalibrate.",
              file=sys.stderr)
        return 1

    print(f"\n{block.name}")
    print(f"  {target.describe()}")
    if block.guidance:
        print(f"  {block.guidance}")
    if block.warning:
        print(f"  ! {block.warning}")

    if block.timed and not args.no_timer and not _audio_calibrated():
        print("  ! Shot timing is uncalibrated. Run `dryfire audiocal` first — "
              "the default click floor is conservative and may miss presses.")

    timed = block.timed and not args.no_timer
    if block.timed and args.no_timer:
        print("  (timer disabled by --no-timer)")
    if not block.timed and args.timer:
        print("  ! You asked for a par on an untimed block. The plan says to do "
              "this only after a month of untimed work.")
        timed = True

    session = Session.begin(block.key, block.setup_key, target)
    ladder = ParLadder(block.start_par, block.par_step, block.clean_to_advance,
                       block.of_reps) if (timed and block.ladder and block.start_par) else None
    if ladder:
        session.par_history.append(ladder.par)

    rc = _run_timed(args, block, target, tf, session, ladder) if timed else \
        _run_untimed(args, block, target, tf, session)

    if rc != 0:
        # The only way out with a non-zero code is a run that never started —
        # the camera geometry did not match the calibration. Saving the empty
        # session anyway would put a rep-less entry in the training log and
        # export it to the vault, which is a made-up record of a session that
        # did not happen.
        return rc
    _finish(session, target, args)
    return rc


def _run_timed(args, block, target, tf, session, ladder) -> int:  # noqa: ANN001
    """Beep, listen, look. One rep at a time."""
    timer = ShotTimer(config=_load_onset_config())
    cfg = _load_detector_config()
    # No listener is passed: in timed mode the chime plays *between* reps, and
    # each rep opens a fresh recording. The guard is time, not muting — see the
    # settle wait below.
    chimes = Chimes(volume=args.volume, enabled=not args.no_sound)
    mask = None
    # Each rep gets a fresh detector, so the count of rejected-for-size regions
    # has to be carried across them.
    oversized_frames, oversized_largest = 0, 0

    with _open_camera(args) as cam:
        if not _geometry_ok(cam.info.frame_size, tf, fatal=True):
            return 1
        for i in range(args.reps):
            par = ladder.par if ladder else block.start_par
            print(f"\nrep {i + 1}/{args.reps}  par {par:.2f}s  "
                  f"{ladder.status() if ladder else ''}")
            print("  ready... ", end="", flush=True)

            det = LaserDetector(cfg)
            flashes: list = []
            stop = time.perf_counter() + 1.2
            while time.perf_counter() < stop:
                r = cam.read()
                if r:
                    if mask is None:
                        mask = tf.scoring_mask(r[0].shape)
                    det.push(cv2.bitwise_and(r[0], r[0], mask=mask), r[1])

            rep_result = timer.run_rep(listen_seconds=args.listen)
            t_cam_end = time.perf_counter() + 0.35
            while time.perf_counter() < t_cam_end:
                r = cam.read()
                if r is None:
                    continue
                s = det.push(cv2.bitwise_and(r[0], r[0], mask=mask), r[1])
                if s is not None:
                    flashes.append(s)
            s = det.flush()
            if s is not None:
                flashes.append(s)

            rep = Rep(index=i, par=par, valid=rep_result.valid)
            if not rep_result.valid:
                print("  NO BEEP HEARD — rep discarded (check mic/speaker volume)")
            else:
                rep.first_shot_time = rep_result.first_shot_time
                for j, fl in enumerate(flashes[:1]):
                    x, y = tf.to_target(fl.x, fl.y)
                    zone, pts = target.score(x, y)
                    rep.shots.append(
                        ShotRecord(index=j, x=x, y=y, zone=zone, points=pts,
                                   t=rep.first_shot_time or 0.0,
                                   drift_px=fl.drift, frames=fl.frames)
                    )
                if not flashes:
                    print("  click heard but no laser seen — off-target, or "
                          "the cartridge did not fire")

            ok, why = evaluate_rep(rep, par)
            session.reps.append(rep)
            print(f"  {'CLEAN' if ok else 'no  '}  {why}")

            if rep.shots:
                sound = chimes.for_zone(
                    rep.shots[0].zone, target.spec.primary_zone
                )
            elif rep.valid:
                sound = "miss"
            else:
                sound = "reject"

            advanced = ladder and rep.valid and ladder.record(ok)
            if advanced:
                session.par_history.append(ladder.par)
                print(f"  >> par drops to {ladder.par:.2f}s")
                sound = "advance"
            chimes.play(sound)

            oversized_frames += det.oversized_frames
            oversized_largest = max(
                oversized_largest, max((o.area for o in det.oversized), default=0)
            )

            # Let the chime finish before the next rep opens its recording,
            # or the tail would be heard as that rep's striker click and the
            # draw time would come out near zero.
            _sleep(chimes.duration(sound) + 0.15)
    chimes.stop()
    for line in _oversized_warning(oversized_frames, oversized_largest, cfg):
        print(f"  ! {line}")
    return 0


def _sleep(seconds: float) -> None:
    end = time.perf_counter() + max(seconds, 0.0)
    while time.perf_counter() < end:
        time.sleep(0.01)


def _run_untimed(args, block, target, tf, session) -> int:  # noqa: ANN001
    """Block F: the camera scores, the clock stays out of it.

    A microphone still runs, but only to confirm each flash came from a striker
    fall — never to impose a par.
    """
    cfg = _load_detector_config()
    det = LaserDetector(cfg)
    # Restrict detection to the target plane. The room outside it is never
    # scored, so letting it be examined only creates chances to be wrong —
    # this is what keeps a red charger LED or a lamp out of the results.
    mask = None
    listener = LiveListener(config=_load_onset_config())
    listener.start()
    chimes = Chimes(volume=args.volume, enabled=not args.no_sound, listener=listener)
    shots: list[tuple[float, float, str]] = []
    clicks: list[float] = []
    baseline = None

    print(f"\nFire {args.reps} shots. No timer. ESC or q to finish.\n")
    if chimes.enabled:
        print("  sound: " + " / ".join(d for d in describe_chimes()))
        print()
    try:
        with _open_camera(args) as cam:
            if not _geometry_ok(cam.info.frame_size, tf, fatal=True):
                return 1
            while len(shots) < args.reps:
                r = cam.read()
                if r is None:
                    continue
                frame, t = r
                if baseline is None:
                    baseline = signature(frame)
                if mask is None:
                    mask = tf.scoring_mask(frame.shape)
                clicks += [listener.wall_time(c) for c in listener.poll()]

                flash = det.push(cv2.bitwise_and(frame, frame, mask=mask), t)
                if flash is not None:
                    near = [c for c in clicks if abs(c - flash.t_first) < 0.15]
                    if not near and not args.no_audio_gate:
                        print("  flash with no click — ignored (not a shot)")
                        chimes.play("reject")
                    else:
                        x, y = tf.to_target(flash.x, flash.y)
                        zone, pts = target.score(x, y)
                        shots.append((x, y, zone))
                        session.reps.append(
                            Rep(index=len(shots) - 1,
                                shots=[ShotRecord(index=len(shots) - 1, x=x, y=y,
                                                  zone=zone, points=pts, t=0.0,
                                                  drift_px=flash.drift,
                                                  frames=flash.frames)])
                        )
                        chimes.play(chimes.for_zone(zone, target.spec.primary_zone))
                        sim = target.to_sim_inches((x ** 2 + y ** 2) ** 0.5)
                        print(f"  {len(shots):2d}. {zone:6s} "
                              f"({x:+.2f},{y:+.2f})\" "
                              f"= {sim:.2f}\" off centre at "
                              f"{target.sim_distance_yd:.0f}yd")

                drift = drift_score(baseline, signature(frame))
                header = f"{block.name} — {len(shots)}/{args.reps}"
                sub = "" if drift < DRIFT_THRESHOLD else \
                    f"CAMERA MOVED (drift {drift:.3f}) — recalibrate, numbers are suspect"
                canvas = draw_overlay(frame.copy(), tf, target, shots, header, sub)
                board = scoreboard(360, canvas.shape[0], target, shots)
                cv2.imshow("dryfire", np.hstack([canvas, board]))
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    finally:
        chimes.stop()
        listener.stop()
        cv2.destroyAllWindows()
    for line in _oversized_warning(
        det.oversized_frames, max((o.area for o in det.oversized), default=0), cfg
    ):
        print(f"  ! {line}")
    return 0


def _finish(session: Session, target, args) -> None:  # noqa: ANN001
    print("\n" + "=" * 60)
    print(session.summary())
    for note in coaching_note(session, target):
        print(f"  {note}")

    path = sessions_dir(ROOT) / f"{session.started.replace(':', '')}-{session.block}.json"
    session.save(path)
    print(f"\nsaved {path}")

    vault = VaultPaths.from_config(_prefs())
    if vault is None:
        return  # Markdown export not configured; JSON above is the record.
    if not vault.exists():
        print(f"notes folder not found at {vault.root}; skipping export")
        return
    note_path, _ = session_note(session, target, vault, write=not args.dry_run)
    print(f"{'would write' if args.dry_run else 'wrote'} {note_path}")
    if args.tracker:
        tp, line = append_to_tracker(session, target, vault, write=not args.dry_run)
        print(f"{'would append' if args.dry_run else 'appended'} to {tp}: {line}")


# -- report / targets ---------------------------------------------------------


def cmd_report(args) -> int:  # noqa: ANN001
    plan = _plan(args)
    d = sessions_dir(ROOT)
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        print("no sessions yet")
        return 0
    print(f"{'date':<20} {'block':<6} {'reps':>5} {'best':>7} {'clean':>7} {'group@sim':>11}")
    for f in files[-args.limit:]:
        s = Session.load(f)
        target = plan.setups.get(s.setup_key)
        stats = s.stats(target) if target else None
        best = s.best_time()
        print(f"{s.started:<20} {s.block:<6} {len(s.reps):>5} "
              f"{(f'{best:.2f}s' if best else '—'):>7} "
              f"{f'{s.clean_count}/{len(s.reps)}':>7} "
              f"{(f'{stats.sim_extreme_spread:.2f}\"' if stats and stats.n else '—'):>11}")
    return 0


def cmd_targets(args) -> int:  # noqa: ANN001
    plan = _plan(args)
    print(f"Setups in plan: {plan.name}\n")
    for key, t in plan.setups.items():
        z = t.spec.zone(t.spec.primary_zone)
        print(f"  {key:8s} {t.describe()}")
        print(f"           {t.spec.primary_zone}-zone printed: "
              f"{z.w * t.scale:.2f}\" x {z.h * t.scale:.2f}\"")
    if args.markers:
        out = Path(args.markers).expanduser()
        target = plan.setup(args.setup or next(iter(plan.setups)))
        pos = make_marker_sheet(out, target)
        print(f"\nwrote marker sheet {out}")
        print("  ONLY for flat setups (paper target on the same sheet, on a wall).")
        print("  Do not use with a standing 3D printed target — the markers would")
        print("  sit in a different plane from the target face and skew every hit.")
        for mid, (x, y) in pos.items():
            print(f"    id {mid}: ({x:+.2f}, {y:+.2f})\"")
    return 0


def _audio_calibrated() -> bool:
    return (ROOT / "audio.json").exists()


def _load_onset_config() -> OnsetConfig:
    p = ROOT / "audio.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            return OnsetConfig(**{k: v for k, v in d.items()
                                  if k in OnsetConfig.__annotations__})
        except (json.JSONDecodeError, TypeError):
            pass
    return OnsetConfig()


def cmd_audiocal(args) -> int:  # noqa: ANN001
    """Separate striker clicks from room noise, by measuring both.

    A fixed floor cannot work across rooms: in one measured room the median
    high-passed level was 2e-5 while ordinary transients reached 3e-3.
    """
    import sounddevice as sd

    print(f"Phase 1 — stay quiet for {args.room_seconds:.0f}s. Recording room noise.")
    room = sd.rec(int(args.room_seconds * SAMPLE_RATE),
                  samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    print(f"\nPhase 2 — dry fire {args.clicks} times, evenly spaced, "
          f"over the next {args.click_seconds:.0f}s.")
    print("Gun clear, muzzle in a safe direction. Starting now.")
    clicks = sd.rec(int(args.click_seconds * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    try:
        cal = calibrate_click_floor(room[:, 0], clicks[:, 0], args.clicks)
    except ValueError as exc:
        print(f"\ncalibration failed: {exc}", file=sys.stderr)
        return 1

    print()
    for line in cal.lines():
        print(f"  {line}")

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "audio.json").write_text(
        json.dumps({"floor": cal.floor, "ratio": cal.ratio}, indent=2)
    )
    print(f"\nSaved to {ROOT / 'audio.json'}")
    if not cal.ok:
        print("\nRe-run after improving the setup; the current floor will miss shots.")
        return 1
    return 0


def cmd_sounds(args) -> int:  # noqa: ANN001
    """Play each chime so the outcomes can be learned before a session."""
    import time as _t

    from .feedback import CHIMES

    chimes = Chimes(volume=args.volume)
    if not chimes.enabled:
        print("no audio output available")
        return 1
    print("Playing each outcome sound.\n")
    for name, spec in CHIMES.items():
        print(f"  {name:8s} {spec.description}")
        chimes.play(name)
        _t.sleep(chimes.duration(name) + 0.55)
    chimes.stop()
    print("\nRising = good, falling = bad, quiet double blip = ignored.")
    return 0


def cmd_plan(args) -> int:  # noqa: ANN001
    plan = _plan(args)
    for line in plan.describe():
        print(line)
    print()
    print(f"source: {plan.source}")
    others = [p for p in bundled_plans() if p != plan.source]
    if others:
        print("bundled: " + ", ".join(p.stem for p in bundled_plans()))
    print("\nWrite your own: copy that file, edit it, then either pass")
    print("  --plan <path>   or   save it as ~/.dryfire/plan.toml")
    return 0


def cmd_vault(args) -> int:  # noqa: ANN001
    """Configure optional Markdown export of session notes."""
    prefs = _prefs()
    if args.path is None:
        vault = VaultPaths.from_config(prefs)
        if vault is None:
            print("Markdown export is off. Turn it on with:")
            print("  dryfire vault ~/path/to/notes")
            return 0
        print(f"notes folder : {vault.root}"
              f"{'' if vault.exists() else '   (MISSING)'}")
        print(f"sessions in  : {vault.sessions}")
        print(f"tracker      : {vault.tracker}")
        print(f"plan note    : {vault.plan_note or '(none)'}")
        return 0

    if args.path.lower() in ("off", "none", "disable"):
        for k in ("vault", "vault_sessions", "vault_tracker", "vault_plan_note"):
            prefs.pop(k, None)
        _save_prefs(prefs)
        print("Markdown export disabled.")
        return 0

    root = Path(args.path).expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1
    prefs["vault"] = str(root)
    if args.sessions:
        prefs["vault_sessions"] = args.sessions
    if args.tracker:
        prefs["vault_tracker"] = args.tracker
    if args.plan_note:
        prefs["vault_plan_note"] = args.plan_note
    _save_prefs(prefs)
    print(f"Markdown export on: {root}")
    print("Session notes are written after each run. Add --tracker to a run to")
    print("also append one line to the tracker note.")
    return 0


# -- argparse -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dryfire", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default=None,
                   help="path to a plan TOML (default: ~/.dryfire/plan.toml, "
                        "else the bundled example)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def camera_args(sp):
        sp.add_argument(
            "--camera", default=None,
            help="camera name or index, e.g. 'iphone' or 0. "
                 "Defaults to the saved preference (see `dryfire use-camera`).",
        )
        # 1920x1440 is a 4:3 mode the iPhone offers: 33% more vertical pixels
        # than 1080p, which is the dimension that matters for a tall silhouette.
        sp.add_argument("--width", type=int, default=1920)
        sp.add_argument("--height", type=int, default=1440)
        sp.add_argument("--fps", type=int, default=60)

    sp = sub.add_parser("probe", help="list cameras and mics, measure real fps")
    camera_args(sp)
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("use-camera", help="pick a camera by name and remember it")
    sp.add_argument("name", help="name or fragment, e.g. 'iphone' or 'MacBook'")
    sp.set_defaults(func=cmd_use_camera)

    sp = sub.add_parser("spike", help="does the camera see the laser flash?")
    camera_args(sp)
    sp.add_argument("--expect", type=int, default=10)
    sp.add_argument("--seconds", type=float, default=90.0)
    sp.add_argument("--out", default=str(ROOT / "spike"))
    sp.add_argument("--no-sound", action="store_true")
    sp.add_argument("--volume", type=float, default=0.35)
    sp.set_defaults(func=cmd_spike)

    sp = sub.add_parser("autocal", help="derive detector thresholds from the room")
    camera_args(sp)
    sp.add_argument("--frames", type=int, default=60)
    sp.add_argument("--profile", default="default")
    sp.add_argument("--no-mask", action="store_true")
    sp.set_defaults(func=cmd_autocal)

    sp = sub.add_parser("calibrate", help="click the A-zone corners")
    camera_args(sp)
    sp.add_argument("--setup", default=None,
                    help="setup key from the plan, e.g. f-15")
    sp.add_argument("--profile", default="default")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("run", help="run a block")
    camera_args(sp)
    sp.add_argument("block", help="block key from the plan (see `dryfire plan`)")
    sp.add_argument("--reps", type=int, default=10)
    sp.add_argument("--listen", type=float, default=6.0)
    sp.add_argument("--profile", default="default")
    sp.add_argument("--timer", action="store_true", help="force a par on an untimed block")
    sp.add_argument("--no-timer", action="store_true")
    sp.add_argument("--no-audio-gate", action="store_true",
                    help="score flashes without requiring a click (noisy rooms)")
    sp.add_argument("--no-sound", action="store_true",
                    help="silence the hit chimes")
    sp.add_argument("--volume", type=float, default=0.35,
                    help="chime volume, 0.0-1.0 (default 0.35)")
    sp.add_argument("--tracker", action="store_true",
                    help="also append a line to the tracker note "
                         "(needs `dryfire vault` configured)")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("audiocal",
                        help="measure room noise vs striker clicks, set the floor")
    sp.add_argument("--room-seconds", type=float, default=10.0)
    sp.add_argument("--clicks", type=int, default=6)
    sp.add_argument("--click-seconds", type=float, default=20.0)
    sp.set_defaults(func=cmd_audiocal)

    sp = sub.add_parser("sounds", help="preview the hit/miss chimes")
    sp.add_argument("--volume", type=float, default=0.35)
    sp.set_defaults(func=cmd_sounds)

    sp = sub.add_parser("plan", help="show the active plan and where it came from")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("vault", help="configure optional Markdown export")
    sp.add_argument("path", nargs="?", default=None,
                    help="notes folder, or 'off' to disable")
    sp.add_argument("--sessions", default=None, help="subfolder for session notes")
    sp.add_argument("--tracker", default=None, help="tracker note filename")
    sp.add_argument("--plan-note", default=None,
                    help="note to link sessions back to")
    sp.set_defaults(func=cmd_vault)

    sp = sub.add_parser("report", help="summarise saved sessions")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("targets", help="show geometry; optionally print markers")
    sp.add_argument("--markers", help="write an ArUco sheet to this path")
    sp.add_argument("--setup", default=None, help="setup key from the plan")
    sp.set_defaults(func=cmd_targets)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
