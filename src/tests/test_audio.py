"""Timing tests against synthesised audio.

The whole value of the acoustic approach is that draw time is exact, so these
assert to the millisecond. A synthetic Glock striker click is modelled as a
short broadband burst with a fast exponential decay, which is what a firing pin
on a laser cartridge's rubber back actually sounds like.
"""

from __future__ import annotations

import numpy as np
import pytest

from opendryfire.audio import (
    BEEP_DUR,
    SAMPLE_RATE,
    ClickDetector,
    OnsetConfig,
    ShotTimer,
    find_beep_onset,
    make_beep,
)

RATE = SAMPLE_RATE


def click_burst(rate: int = RATE, duration: float = 0.012, amp: float = 0.5) -> np.ndarray:
    """Broadband transient with a fast decay."""
    n = int(duration * rate)
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(n).astype(np.float32)
    envelope = np.exp(-np.arange(n) / (0.0015 * rate)).astype(np.float32)
    return (noise * envelope * amp).astype(np.float32)


def room_noise(n: int, level: float = 0.002, seed: int = 7) -> np.ndarray:
    """Low-level pink-ish room tone."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float32)
    # One-pole lowpass gives a plausible room rumble spectrum.
    out = np.zeros(n, dtype=np.float32)
    a = 0.995
    for i in range(1, n):
        out[i] = a * out[i - 1] + (1 - a) * white[i]
    out /= np.abs(out).max() + 1e-9
    return (out * level).astype(np.float32)


def scene(
    beep_at: float,
    click_times: list[float],
    total: float = 6.0,
    noise: float = 0.002,
    click_amp: float = 0.5,
) -> np.ndarray:
    """Build a full recorded buffer: room tone + beep + clicks."""
    n = int(total * RATE)
    buf = room_noise(n, noise)

    def mix(sig: np.ndarray, at: float) -> None:
        """Add a signal at a time, clipping anything past the buffer end so a
        deliberately out-of-range beep just means 'no beep'."""
        i = int(at * RATE)
        if i >= n:
            return
        k = min(len(sig), n - i)
        buf[i : i + k] += sig[:k]

    mix(make_beep(rate=RATE), beep_at)
    burst = click_burst(amp=click_amp)
    for t in click_times:
        mix(burst, t)
    return buf


def test_beep_onset_is_found_to_the_millisecond():
    buf = scene(beep_at=1.0, click_times=[])
    onset = find_beep_onset(buf, RATE)
    assert onset is not None
    assert onset / RATE == pytest.approx(1.0, abs=0.006)


def test_no_beep_returns_none_rather_than_a_wrong_time():
    """A silent speaker must invalidate the rep, never produce a plausible
    number that silently corrupts the log."""
    buf = room_noise(int(3 * RATE), 0.002)
    assert find_beep_onset(buf, RATE) is None


def test_single_click_is_detected():
    det = ClickDetector(RATE)
    buf = scene(beep_at=10.0, click_times=[1.0], total=2.0)  # beep out of range
    clicks = []
    for i in range(0, len(buf) - 256, 256):
        clicks += det.push(buf[i : i + 256])
    assert len(clicks) == 1
    assert clicks[0].seconds(RATE) == pytest.approx(1.0, abs=0.02)


def test_draw_time_is_accurate_to_a_few_milliseconds():
    """The headline number: beep to first click."""
    timer = ShotTimer(RATE)
    buf = scene(beep_at=1.0, click_times=[2.42])
    rep = timer.analyse(buf)
    assert rep.valid
    assert rep.first_shot_time == pytest.approx(1.42, abs=0.02)


def test_the_beep_itself_is_never_counted_as_a_shot():
    """The beep has a sharp onset too; without gating it reads as a click at
    t=0 and every draw time comes out near zero."""
    timer = ShotTimer(RATE)
    buf = scene(beep_at=1.0, click_times=[2.0])
    rep = timer.analyse(buf)
    assert rep.valid
    assert len(rep.clicks) == 1
    assert rep.first_shot_time == pytest.approx(1.0, abs=0.02)


def test_multiple_clicks_give_splits():
    timer = ShotTimer(RATE)
    buf = scene(beep_at=0.5, click_times=[1.5, 2.3, 3.4], total=5.0)
    rep = timer.analyse(buf)
    assert len(rep.clicks) == 3
    assert rep.first_shot_time == pytest.approx(1.0, abs=0.02)
    assert rep.splits == pytest.approx([0.8, 1.1], abs=0.03)


def test_refractory_window_does_not_double_count_one_click():
    det = ClickDetector(RATE, OnsetConfig(refractory_s=0.15))
    buf = scene(beep_at=10.0, click_times=[1.0], total=2.0)
    clicks = []
    for i in range(0, len(buf) - 256, 256):
        clicks += det.push(buf[i : i + 256])
    assert len(clicks) == 1


def test_louder_room_noise_does_not_manufacture_clicks():
    """Adaptive median/MAD thresholding should ride over a noisier room rather
    than firing constantly."""
    det = ClickDetector(RATE)
    buf = room_noise(int(4 * RATE), level=0.02, seed=99)
    clicks = []
    for i in range(0, len(buf) - 256, 256):
        clicks += det.push(buf[i : i + 256])
    assert clicks == []


def test_quiet_click_in_a_quiet_room_still_registers():
    det = ClickDetector(RATE)
    buf = scene(beep_at=10.0, click_times=[1.0], total=2.0, noise=0.001, click_amp=0.08)
    clicks = []
    for i in range(0, len(buf) - 256, 256):
        clicks += det.push(buf[i : i + 256])
    assert len(clicks) == 1


def test_invalid_rep_reports_none_not_a_number():
    timer = ShotTimer(RATE)
    buf = room_noise(int(3 * RATE), 0.002)  # no beep, no click
    rep = timer.analyse(buf)
    assert not rep.valid
    assert rep.first_shot_time is None


def test_beep_duration_constant_matches_generated_beep():
    assert len(make_beep(rate=RATE)) == int(BEEP_DUR * RATE)


# -- click-floor calibration --------------------------------------------------


def test_floor_is_placed_between_room_noise_and_real_clicks():
    from opendryfire.audio import calibrate_click_floor

    room = room_noise(int(6 * RATE), level=0.002, seed=3)
    firing = room_noise(int(10 * RATE), level=0.002, seed=4)
    burst = click_burst(amp=0.5)
    for t in (1.0, 2.5, 4.0, 5.5, 7.0):
        i = int(t * RATE)
        firing[i : i + len(burst)] += burst

    cal = calibrate_click_floor(room, firing, expected_clicks=5)
    assert cal.ok
    assert cal.room_max < cal.floor < cal.click_min
    assert cal.separation > 1.5


def test_calibrated_floor_actually_silences_the_room():
    from opendryfire.audio import ClickDetector, calibrate_click_floor

    room = room_noise(int(6 * RATE), level=0.002, seed=3)
    firing = room_noise(int(10 * RATE), level=0.002, seed=4)
    burst = click_burst(amp=0.5)
    for t in (1.0, 2.5, 4.0, 5.5, 7.0):
        firing[int(t * RATE) : int(t * RATE) + len(burst)] += burst
    cal = calibrate_click_floor(room, firing, expected_clicks=5)

    det = ClickDetector(RATE, cal.config())
    quiet = room_noise(int(6 * RATE), level=0.002, seed=77)
    found = []
    for i in range(0, len(quiet) - 256, 256):
        found += det.push(quiet[i : i + 256])
    assert found == []


def test_calibrated_floor_still_hears_the_clicks():
    from opendryfire.audio import ClickDetector, calibrate_click_floor

    room = room_noise(int(6 * RATE), level=0.002, seed=3)
    firing = room_noise(int(10 * RATE), level=0.002, seed=4)
    burst = click_burst(amp=0.5)
    times = (1.0, 2.5, 4.0, 5.5, 7.0)
    for t in times:
        firing[int(t * RATE) : int(t * RATE) + len(burst)] += burst
    cal = calibrate_click_floor(room, firing, expected_clicks=5)

    det = ClickDetector(RATE, cal.config())
    found = []
    for i in range(0, len(firing) - 256, 256):
        found += det.push(firing[i : i + 256])
    assert len(found) == len(times)


def hissy_room(n: int, level: float, seed: int) -> np.ndarray:
    """A room with real high-frequency content — a fan, or a laptop under load.

    Plain ``room_noise`` is lowpass rumble, which the click detector's 1.8kHz
    high-pass removes almost entirely; it cannot model a room that competes
    with a striker click. This can.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * level).astype(np.float32)


def test_indistinguishable_clicks_are_reported_rather_than_papered_over():
    """If a press is no louder than the room, say so. Silently choosing a
    threshold in the overlap would produce a mix of phantom and missed shots
    with no way to tell which."""
    from opendryfire.audio import calibrate_click_floor

    room = hissy_room(int(6 * RATE), level=0.05, seed=8)
    firing = hissy_room(int(8 * RATE), level=0.05, seed=9)
    burst = click_burst(amp=0.02)  # quieter than the hiss it sits in
    for t in (1.0, 3.0, 5.0):
        firing[int(t * RATE) : int(t * RATE) + len(burst)] += burst

    cal = calibrate_click_floor(room, firing, expected_clicks=3)
    assert not cal.ok
    assert "not clearly louder" in cal.note
    assert cal.floor > cal.room_max  # bias toward missing, never inventing


def test_a_hissy_room_still_calibrates_when_the_click_is_loud_enough():
    """The same difficult room, with a normal-strength press, must still work
    — the failure above should be about separation, not about hiss per se."""
    from opendryfire.audio import calibrate_click_floor

    room = hissy_room(int(6 * RATE), level=0.01, seed=21)
    firing = hissy_room(int(8 * RATE), level=0.01, seed=22)
    burst = click_burst(amp=0.6)
    for t in (1.0, 3.0, 5.0):
        firing[int(t * RATE) : int(t * RATE) + len(burst)] += burst

    cal = calibrate_click_floor(room, firing, expected_clicks=3)
    assert cal.ok
    assert cal.room_max < cal.floor < cal.click_min


def test_peak_picking_finds_separated_events_not_one_event_twice():
    from opendryfire.audio import block_energies, peak_events

    sig = room_noise(int(6 * RATE), level=0.001, seed=12)
    burst = click_burst(amp=0.4)
    for t in (1.0, 3.0, 5.0):
        sig[int(t * RATE) : int(t * RATE) + len(burst)] += burst
    peaks = peak_events(block_energies(sig, RATE), count=3)
    assert len(peaks) == 3
    assert min(peaks) > 0.01  # all three are real events, not noise


def test_too_short_a_recording_raises():
    from opendryfire.audio import calibrate_click_floor

    with pytest.raises(ValueError):
        calibrate_click_floor(np.zeros(10, np.float32), np.zeros(10, np.float32), 3)
