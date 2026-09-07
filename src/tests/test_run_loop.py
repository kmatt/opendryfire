"""The timed run loop: is the camera actually watching while the rep happens?

Every unit in this repo passed while ``dryfire run`` was structurally blind.
``_run_timed`` read the camera for 1.2s before the beep and 0.35s after the
listen window closed, and not once during the ~6s in between — which is exactly
when the draw happens. Nothing is buffered: measured on a Continuity Camera, a
6s gap followed by a 0.35s drain gives back 11 live frames, not the ~180 that
elapsed. So every frame of the pulse was gone and each rep printed "click heard
but no laser seen" over a shot that was dead centre.

That is the class of bug ``test_integration`` exists for, one level up: the
pixels->inches path was right, the click detector was right, and the system
still could not see. These drive ``_run_timed`` itself with a fake camera and a
fake timer, so they need no hardware.
"""

from __future__ import annotations

import time
import types

import pytest

from opendryfire import cli
from opendryfire.audio import SAMPLE_RATE, Click, TimedRep
from opendryfire.detect import DetectorConfig
from opendryfire.drills import ParLadder
from opendryfire.session import Session
from test_integration import PLAN_SETUPS, calibrated, render_scene, target_to_pixels

TARGET = PLAN_SETUPS["f-15"]
LISTEN = 0.4  # a short listen window; the real default is 6.0s
FRAME_DT = 1 / 120  # the fake camera's pacing
PLACED = (0.4, 0.7)  # where the shot lands, in printed-target inches

# Rendering the warm-room scene is expensive, so both states are built once.
_PX = target_to_pixels(TARGET, [PLACED], "oblique")[0]
_DARK = render_scene(None)
_LIT = render_scene(_PX)


class FakeCamera:
    """A camera that lights the dot during a chosen wall-clock window.

    ``read`` is paced rather than instant so the detector sees a plausible
    frame rate, and hands out real ``perf_counter`` timestamps — the same clock
    ``_run_timed`` uses to bound its loops.
    """

    def __init__(self, lit: tuple[float, float] | None = None) -> None:
        self.info = types.SimpleNamespace(frame_size=(_DARK.shape[1], _DARK.shape[0]))
        self.name, self.index = "fake", 0
        self._lit = lit  # (start, end) as perf_counter values, or None
        self.read_times: list[float] = []

    def read(self):
        time.sleep(FRAME_DT)
        t = time.perf_counter()
        self.read_times.append(t)
        lit = self._lit is not None and self._lit[0] <= t < self._lit[1]
        return (_LIT if lit else _DARK), t

    def light_from(self, start: float, duration: float) -> None:
        self._lit = (start, start + duration)

    def reads_between(self, t0: float, t1: float) -> int:
        return sum(1 for t in self.read_times if t0 < t < t1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeTimer:
    """Stands in for ShotTimer: blocks for the listen window, then reports a
    beep and one click at a fixed offset — the timing the audio would give."""

    instance: "FakeTimer | None" = None
    draw_time = 1.31  # seconds from beep to click, per the fake audio

    def __init__(self, *a, **kw) -> None:
        self.opened = self.closed = 0.0
        self.on_open = None
        FakeTimer.instance = self

    def run_rep(self, listen_seconds: float = 6.0, delay: float = 0.0) -> TimedRep:
        self.opened = time.perf_counter()
        if self.on_open is not None:
            self.on_open(self.opened)
        end = self.opened + listen_seconds
        while time.perf_counter() < end:
            time.sleep(0.005)
        self.closed = time.perf_counter()
        return TimedRep(
            beep_sample=0,
            clicks=[Click(sample=int(self.draw_time * SAMPLE_RATE), strength=9.0)],
            rate=SAMPLE_RATE,
        )


def _rig(monkeypatch, cam: FakeCamera):
    """Wire ``_run_timed`` up to the fakes and return its arguments."""
    monkeypatch.setattr(cli, "ShotTimer", FakeTimer)
    monkeypatch.setattr(cli, "_open_camera", lambda args, quiet=False: cam)
    # Never read the developer's own ~/.dryfire while testing.
    monkeypatch.setattr(cli, "_load_detector_config", DetectorConfig)
    monkeypatch.setattr(cli, "_sleep", lambda seconds: None)

    args = types.SimpleNamespace(
        reps=1, listen=LISTEN, volume=0.5, no_sound=True,
    )
    block = types.SimpleNamespace(name="B", start_par=2.5)
    session = Session.begin("B", "f-15", TARGET)
    return args, block, TARGET, calibrated(TARGET, "oblique"), session


def test_the_camera_is_read_while_the_rep_is_being_timed(monkeypatch):
    """The regression itself. The listen window is when the shot happens, so
    reading zero frames during it means the run can never see a laser."""
    cam = FakeCamera()
    args, block, target, tf, session = _rig(monkeypatch, cam)

    assert cli._run_timed(args, block, target, tf, session, None) == 0

    timer = FakeTimer.instance
    during = cam.reads_between(timer.opened, timer.closed)
    # At 120fps of pacing a 0.4s window holds ~48 frames; anything above a
    # handful proves the camera was being watched rather than parked.
    assert during > 10, (
        f"only {during} frames were read during the {LISTEN}s listen window — "
        "the camera is not being watched while the rep happens"
    )


def test_a_flash_during_the_draw_is_scored_where_it_landed(monkeypatch):
    """The whole point: a dot lit mid-window is found, and lands on the inch
    it was rendered at."""
    cam = FakeCamera()
    args, block, target, tf, session = _rig(monkeypatch, cam)
    # Light the dot 0.15s into the listen window for 120ms, as a cartridge does.
    FakeTimer.instance = None
    monkeypatch.setattr(
        cli, "ShotTimer",
        lambda *a, **kw: _timer_lighting(cam, 0.15, 0.12),
    )

    assert cli._run_timed(args, block, target, tf, session, None) == 0

    rep = session.reps[0]
    assert rep.valid
    assert rep.shots, "a dot lit during the draw was not detected"
    assert rep.shots[0].x == pytest.approx(PLACED[0], abs=0.05)
    assert rep.shots[0].y == pytest.approx(PLACED[1], abs=0.05)


def test_the_shot_time_comes_from_the_microphone_not_the_camera(monkeypatch):
    """The central split. The flash is rendered 0.15s into the window while the
    fake audio reports a 1.31s draw; the recorded time must be the audio's."""
    cam = FakeCamera()
    args, block, target, tf, session = _rig(monkeypatch, cam)
    monkeypatch.setattr(
        cli, "ShotTimer",
        lambda *a, **kw: _timer_lighting(cam, 0.15, 0.12),
    )

    cli._run_timed(args, block, target, tf, session, None)

    rep = session.reps[0]
    assert rep.first_shot_time == pytest.approx(FakeTimer.draw_time, abs=1e-6)
    assert rep.shots[0].t == pytest.approx(FakeTimer.draw_time, abs=1e-6)


def test_a_dot_lit_before_the_beep_is_not_scored_as_the_reps_shot(monkeypatch):
    """A false start is not a draw. The pre-beep warm-up settles the detector;
    what it sees there must not become the rep's hit."""
    cam = FakeCamera()
    args, block, target, tf, session = _rig(monkeypatch, cam)
    # Lit immediately, i.e. during the warm-up, and dark from then on.
    cam.light_from(time.perf_counter(), 0.12)

    cli._run_timed(args, block, target, tf, session, None)

    assert not session.reps[0].shots, "a flash before the beep was scored"


def _timer_lighting(cam: FakeCamera, after: float, duration: float) -> FakeTimer:
    """A FakeTimer that lights the fake camera once the window has opened."""
    timer = FakeTimer()
    timer.on_open = lambda t: cam.light_from(t + after, duration)
    return timer


# -- discarded reps are re-run, not counted -----------------------------------
#
# A rep with no beep was never measured, so there is nothing to judge. Counting
# it toward the block made a single discard enough to make a 10-rep block
# unwinnable: the ladder records only reps it can judge, so its window never
# filled and it abstained however well the block was shot. Twice on the
# reference setup, at a measured ~17% discard rate.


class ScriptedTimer(FakeTimer):
    """A timer that reports no beep on chosen attempt numbers."""

    def __init__(self, fail_on=()) -> None:
        super().__init__()
        self.fail_on = set(fail_on)
        self.attempts = 0

    def run_rep(self, listen_seconds: float = 6.0, delay: float = 0.0) -> TimedRep:
        self.attempts += 1
        rep = super().run_rep(listen_seconds, delay)
        if self.attempts in self.fail_on:
            return TimedRep(beep_sample=None, clicks=[], rate=SAMPLE_RATE)
        return rep


def _fast_rig(monkeypatch, cam, reps, fail_on=()):
    """As ``_rig``, but with the camera's settle windows shortened so a
    multi-rep block runs in seconds, and a timer that can miss the beep."""
    args, block, target, tf, session = _rig(monkeypatch, cam)
    args.reps = reps
    args.listen = 0.22
    monkeypatch.setattr(cli, "WARMUP_S", 0.05)
    monkeypatch.setattr(cli, "TAIL_S", 0.05)
    timer = ScriptedTimer(fail_on)
    timer.on_open = lambda t: cam.light_from(t + 0.04, 0.09)
    monkeypatch.setattr(cli, "ShotTimer", lambda *a, **kw: timer)
    return args, block, target, tf, session, timer


def test_a_discarded_rep_is_re_run_so_the_block_gets_the_reps_it_asked_for(
    monkeypatch,
):
    cam = FakeCamera()
    args, block, target, tf, session, timer = _fast_rig(
        monkeypatch, cam, reps=4, fail_on=(2,)
    )

    cli._run_timed(args, block, target, tf, session, None)

    assert timer.attempts == 5, "the discarded rep was not re-run"
    assert len(session.valid_reps) == 4
    # The discard stays in the log: it happened, and the record says so.
    assert len(session.reps) == 5
    assert [r.valid for r in session.reps] == [True, False, True, True, True]


def test_the_ladder_still_advances_when_a_rep_was_discarded(monkeypatch):
    """The bug this was reported for. Every valid rep is clean, so a full
    window exists and the par has to drop — the discard must not prevent it."""
    cam = FakeCamera()
    args, block, target, tf, session, timer = _fast_rig(
        monkeypatch, cam, reps=4, fail_on=(3,)
    )
    ladder = ParLadder(2.5, step=0.1, needed=3, window=4)

    cli._run_timed(args, block, target, tf, session, ladder)

    assert len(ladder.history) == 0, "history clears on a drop"
    assert ladder.advanced_at == [2.5]
    assert ladder.par == pytest.approx(2.4)
    assert session.par_history == [2.4]


def test_three_discards_in_a_row_stop_the_block(monkeypatch):
    """A dead microphone must not become an unbounded session, and must not
    look like a run of bad shooting either."""
    cam = FakeCamera()
    args, block, target, tf, session, timer = _fast_rig(
        monkeypatch, cam, reps=6, fail_on=(2, 3, 4)
    )

    cli._run_timed(args, block, target, tf, session, None)

    assert timer.attempts == 4, "it kept going after three straight discards"
    assert len(session.valid_reps) == 1


def test_a_block_that_never_fills_is_bounded_and_says_so(monkeypatch, capsys):
    """Discards spaced out enough to dodge the consecutive guard still cannot
    run forever, and a short block has to announce itself — the ladder's
    silence looks exactly like shooting that was not good enough."""
    cam = FakeCamera()
    args, block, target, tf, session, timer = _fast_rig(
        monkeypatch, cam, reps=3, fail_on=(1, 2, 4, 5)
    )

    cli._run_timed(args, block, target, tf, session, None)

    assert timer.attempts == 3 * cli.MAX_ATTEMPT_FACTOR
    assert len(session.valid_reps) == 2
    out = capsys.readouterr().out
    assert "SHORT" in out
    assert "cannot advance" in out


def test_the_summary_counts_valid_reps_and_names_the_discards(monkeypatch):
    """`8/10 clean` over ten attempts including a discard reads as missing the
    bar, and disagrees with the ladder, which judged nine."""
    cam = FakeCamera()
    args, block, target, tf, session, timer = _fast_rig(
        monkeypatch, cam, reps=3, fail_on=(2,)
    )

    cli._run_timed(args, block, target, tf, session, None)

    line = session.summary()
    assert "3 reps" in line
    assert "3/3 clean" in line
    assert "(1 discarded)" in line


def test_a_clean_block_reports_exactly_as_before(monkeypatch):
    cam = FakeCamera()
    args, block, target, tf, session, timer = _fast_rig(monkeypatch, cam, reps=3)

    cli._run_timed(args, block, target, tf, session, None)

    line = session.summary()
    assert "3 reps" in line and "3/3 clean" in line
    assert "discarded" not in line
