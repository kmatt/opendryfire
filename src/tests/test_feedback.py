"""Hit-feedback tests.

The one that matters is ``test_chime_is_not_heard_as_a_click``: the speakers
feed the microphone, so if a chime looks like a striker strike the system scores
a phantom shot, which plays another chime, which scores another shot. Everything
else here is secondary.
"""

from __future__ import annotations

import numpy as np
import pytest

from opendryfire.audio import ClickDetector, OnsetConfig
from opendryfire.feedback import (
    CHIMES,
    MAX_SAFE_FREQ,
    ChimeSpec,
    Chimes,
    Tone,
    render,
)

RATE = 48_000


def run_detector(signal: np.ndarray, config: OnsetConfig | None = None) -> list:
    det = ClickDetector(RATE, config)
    clicks = []
    for i in range(0, len(signal) - 256, 256):
        clicks += det.push(signal[i : i + 256])
    return clicks


def with_room_tone(chime: np.ndarray, pad: float = 1.0) -> np.ndarray:
    """Put a chime in a realistic quiet room, with lead-in so the detector's
    adaptive baseline has settled before the sound arrives."""
    rng = np.random.default_rng(11)
    n_pad = int(pad * RATE)
    noise = (rng.standard_normal(n_pad * 2 + len(chime)) * 0.002).astype(np.float32)
    noise[n_pad : n_pad + len(chime)] += chime
    return noise


@pytest.mark.parametrize("name", sorted(CHIMES))
def test_chime_is_not_heard_as_a_click(name):
    """Every chime, at the volume actually played, through the real detector.

    This is the unmuted case: even with the safety net removed, the sound
    itself must not read as a strike. The mute in ``Chimes.play`` is then
    defence in depth rather than the only thing standing between the speakers
    and a feedback loop.
    """
    chime = render(CHIMES[name], RATE, volume=0.35)
    assert run_detector(with_room_tone(chime)) == []


@pytest.mark.parametrize("name", sorted(CHIMES))
def test_muting_covers_even_an_unreasonably_loud_chime(name):
    """The safety net on its own, tested independently: at full scale, muted."""
    from opendryfire.audio import ClickDetector

    chime = render(CHIMES[name], RATE, volume=1.0)
    signal = with_room_tone(chime)
    det = ClickDetector(RATE)
    pad = int(1.0 * RATE)
    clicks = []
    muted = False
    for i in range(0, len(signal) - 256, 256):
        # Mute as soon as we reach the block the chime starts in. Testing for
        # an exact sample index would never fire, since the pad is not a
        # multiple of the block size.
        if not muted and i + 256 > pad:
            det.mute_until(det.samples_seen + len(chime) + int(0.12 * RATE))
            muted = True
        clicks += det.push(signal[i : i + 256])
    assert muted, "mute was never applied; the test would prove nothing"
    assert clicks == []


@pytest.mark.parametrize("name", sorted(CHIMES))
def test_chime_energy_sits_below_the_click_detectors_passband(name):
    """Structural version of the above: check the spectrum directly rather
    than trusting one detector configuration."""
    chime = render(CHIMES[name], RATE, volume=1.0)
    spectrum = np.abs(np.fft.rfft(chime))
    freqs = np.fft.rfftfreq(len(chime), 1 / RATE)
    total = float((spectrum**2).sum())
    above = float((spectrum[freqs >= 1800] ** 2).sum())
    assert above / total < 0.01, f"{name}: {above / total:.3%} of energy is above 1.8kHz"


def test_a_sharp_attack_would_be_heard_as_a_click():
    """Justifies the soft attack: the same tone with a rectangular onset is
    detected. This is why a naive 'ping' cannot be used."""
    n = int(0.15 * RATE)
    t = np.arange(n) / RATE
    harsh = (np.sin(2 * np.pi * 880 * t)).astype(np.float32)  # no ramp at all
    clicks = run_detector(with_room_tone(harsh))
    assert clicks, "expected the rectangular onset to trigger the detector"


def test_render_refuses_a_frequency_the_detector_would_hear():
    spec = ChimeSpec("bad", (Tone(3000.0, 0.1),))
    with pytest.raises(ValueError) as exc:
        render(spec, RATE)
    assert "ceiling" in str(exc.value)


def test_every_shipped_chime_is_under_the_safe_ceiling():
    for spec in CHIMES.values():
        for tone in spec.tones:
            assert tone.freq <= MAX_SAFE_FREQ, f"{spec.name} has a {tone.freq}Hz tone"


def test_chimes_are_normalised_to_the_requested_volume():
    for name in CHIMES:
        buf = render(CHIMES[name], RATE, volume=0.35)
        assert float(np.abs(buf).max()) == pytest.approx(0.35, abs=1e-4)


def test_outcomes_map_to_distinguishable_sounds():
    c = Chimes(enabled=False)
    assert c.for_zone("A", "A") == "hit"
    assert c.for_zone("C", "A") == "off"
    assert c.for_zone("miss", "A") == "miss"
    # B-8 scoring uses a different primary zone.
    assert c.for_zone("9", "9") == "hit"
    assert c.for_zone("7", "9") == "off"


def test_hit_and_miss_do_not_sound_alike():
    """They must be distinguishable without looking, which is the whole point."""
    hit = {t.freq for t in CHIMES["hit"].tones}
    miss = {t.freq for t in CHIMES["miss"].tones}
    assert not (hit & miss)
    assert max(hit) > max(miss)  # hit rises, miss falls


def test_disabled_chimes_are_silent_and_do_not_raise():
    c = Chimes(enabled=False)
    c.play("hit")
    c.play("nonexistent")
    c.stop()


def test_playing_mutes_the_listener_before_making_a_sound():
    """Ordering matters: muting after playback starts leaves a window in which
    the chime is sounding and the detector is still armed."""
    events = []

    class FakeListener:
        def mute_for(self, seconds):
            events.append(("mute", seconds))

    c = Chimes(enabled=False, listener=FakeListener())
    c.enabled = True
    c._rendered["hit"] = np.zeros(int(0.3 * RATE), dtype=np.float32)
    c.play("hit")

    assert events and events[0][0] == "mute"
    assert events[0][1] > 0.3  # covers the chime plus a tail


def test_mute_window_suppresses_detection_then_recovers():
    from opendryfire.audio import ClickDetector

    det = ClickDetector(RATE)
    rng = np.random.default_rng(5)
    quiet = (rng.standard_normal(256) * 0.002).astype(np.float32)
    for _ in range(40):
        det.push(quiet)

    det.mute_until(det.samples_seen + int(0.3 * RATE))

    burst = (rng.standard_normal(256) * 0.5).astype(np.float32)
    muted_hits = []
    for _ in range(int(0.3 * RATE) // 256):
        muted_hits += det.push(burst)
    assert muted_hits == []

    for _ in range(40):
        det.push(quiet)
    assert det.push(burst), "detector should hear again once the mute expires"


def test_spike_uses_a_neutral_tone_not_the_hit_tone():
    """A spike runs before calibration, so nothing knows a hit from a miss.
    Playing the rising 'hit' chime for a shot on the wall is misleading."""
    assert "seen" in CHIMES
    assert CHIMES["seen"].tones != CHIMES["hit"].tones
    assert len(CHIMES["seen"].tones) == 1  # flat, not rising


def test_the_neutral_tone_is_also_inaudible_to_the_click_detector():
    chime = render(CHIMES["seen"], RATE, volume=0.35)
    assert run_detector(with_room_tone(chime)) == []
