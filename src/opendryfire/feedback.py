"""Audible hit feedback.

The point is not decoration. During precision work you should be watching the
front sight, not a laptop screen. A distinct sound per outcome means you can
score a shot without ever looking away from the target.

Two constraints make this less trivial than playing a WAV file:

1. **The microphone is listening for striker clicks.** Anything the speakers
   emit is heard by the click detector. A naive "ping" is exactly what a click
   looks like — a sharp broadband transient — so it would be detected as a shot,
   which would then trigger another chime. That is a literal feedback loop.

2. **A sharp attack is broadband by definition.** The onset envelope, not the
   tone, is what creates high-frequency energy. So the chimes here use a soft
   raised-cosine attack of ~12ms, which keeps their energy at the fundamental.

Both defences are applied together: chime fundamentals stay well below the
click detector's 1.8kHz high-pass, *and* the detector is explicitly muted
across the chime's duration. The mute is exact and the spectral separation is a
matter of degree, so the mute is the primary mechanism — but the tones are kept
intrinsically safe too, because in timed drills the chime is played through a
different code path from the live listener.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

try:  # pragma: no cover - audio device may be absent in CI
    import sounddevice as sd

    HAVE_AUDIO = True
except Exception:  # pragma: no cover
    sd = None
    HAVE_AUDIO = False

SAMPLE_RATE = 48_000

# Ceiling on chime fundamentals, set by measurement rather than theory. A
# 4th-order Butterworth is not a brick wall, so tones near its 1.8kHz corner
# leave real residual energy in the click detector's passband. Measured peak
# block HF energy, as a ratio against a quiet room floor (detection fires at 6):
#
#     880Hz -> 7.1   (detected: too close to the corner)
#     587Hz -> 1.5
#     440Hz -> 0.5
#
# 700Hz keeps every shipped chime under ratio ~3. Do not raise it without
# re-running test_chime_is_not_heard_as_a_click.
MAX_SAFE_FREQ = 700.0
ATTACK_S = 0.012
RELEASE_S = 0.06


@dataclass(frozen=True)
class Tone:
    """One note in a chime."""

    freq: float
    duration: float
    delay: float = 0.0
    gain: float = 1.0


@dataclass(frozen=True)
class ChimeSpec:
    """A named sound, described as notes rather than a file."""

    name: str
    tones: tuple[Tone, ...]
    description: str = ""


# Deliberately far apart, so the outcome is obvious without looking. A rising
# interval reads as success and a falling one as failure to basically everyone,
# which is worth more than any amount of documentation.
CHIMES: dict[str, ChimeSpec] = {
    "hit": ChimeSpec(
        "hit",
        (Tone(523.3, 0.09), Tone(659.3, 0.15, delay=0.07)),
        "A-zone: rising third, C5 to E5",
    ),
    "off": ChimeSpec(
        "off",
        (Tone(440.0, 0.16, gain=0.85),),
        "on target, outside the scoring zone: single flat A4",
    ),
    "miss": ChimeSpec(
        "miss",
        (Tone(392.0, 0.10, gain=0.8), Tone(293.7, 0.18, delay=0.08, gain=0.8)),
        "off target: falling fourth, G4 to D4",
    ),
    "seen": ChimeSpec(
        "seen",
        (Tone(523.3, 0.11, gain=0.75),),
        "detection with no geometry yet (spike): neutral acknowledgement",
    ),
    "reject": ChimeSpec(
        "reject",
        (Tone(349.2, 0.07, gain=0.5), Tone(349.2, 0.07, delay=0.11, gain=0.5)),
        "flash with no click, ignored: quiet double blip",
    ),
    "advance": ChimeSpec(
        "advance",
        (
            Tone(392.0, 0.08),
            Tone(523.3, 0.08, delay=0.09),
            Tone(659.3, 0.20, delay=0.18),
        ),
        "par ladder dropped: three-note rise",
    ),
}


def render(spec: ChimeSpec, rate: int = SAMPLE_RATE, volume: float = 0.35) -> np.ndarray:
    """Render a chime to samples.

    Each tone gets a raised-cosine attack and release. The attack is the part
    that matters: a rectangular onset would splatter energy across the spectrum
    and be heard by the click detector as a strike.
    """
    total = max(t.delay + t.duration for t in spec.tones)
    out = np.zeros(int(total * rate) + 1, dtype=np.float32)

    for tone in spec.tones:
        if tone.freq > MAX_SAFE_FREQ:
            raise ValueError(
                f"{spec.name}: {tone.freq}Hz is above the {MAX_SAFE_FREQ}Hz "
                f"ceiling and would be audible to the click detector"
            )
        n = int(tone.duration * rate)
        t = np.arange(n) / rate
        wave = np.sin(2 * np.pi * tone.freq * t).astype(np.float32)

        env = np.ones(n, dtype=np.float32)
        a = min(int(ATTACK_S * rate), n // 2)
        r = min(int(RELEASE_S * rate), n // 2)
        if a:
            env[:a] = 0.5 * (1 - np.cos(np.pi * np.arange(a) / a))
        if r:
            env[-r:] = 0.5 * (1 + np.cos(np.pi * np.arange(r) / r))

        start = int(tone.delay * rate)
        out[start : start + n] += wave * env * tone.gain

    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * volume
    return out.astype(np.float32)


@dataclass
class Chimes:
    """Plays outcome sounds without letting them contaminate shot detection.

    Pass the session's ``LiveListener`` as ``listener`` and the detector is muted
    for the duration of each chime plus a small tail.
    """

    volume: float = 0.35
    enabled: bool = True
    rate: int = SAMPLE_RATE
    listener: object | None = None
    _rendered: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not HAVE_AUDIO:
            self.enabled = False
        for name, spec in CHIMES.items():
            self._rendered[name] = render(spec, self.rate, self.volume)

    def duration(self, name: str) -> float:
        buf = self._rendered.get(name)
        return len(buf) / self.rate if buf is not None else 0.0

    def play(self, name: str) -> None:
        """Play a chime without blocking the capture loop.

        Failures are swallowed: losing a sound is a cosmetic problem, and
        raising here would abort a training session over an audio device.
        """
        if not self.enabled:
            return
        buf = self._rendered.get(name)
        if buf is None:
            return

        # Mute the click detector first, so there is no window in which the
        # chime is already sounding but the detector is still armed.
        tail = self.duration(name) + 0.12
        listener = self.listener
        if listener is not None and hasattr(listener, "mute_for"):
            try:
                listener.mute_for(tail)
            except Exception:
                pass

        with self._lock:
            try:  # pragma: no cover - needs a device
                sd.play(buf, self.rate, blocking=False)
            except Exception:
                pass

    def for_zone(self, zone: str, primary: str) -> str:
        """Pick a chime from a scored zone."""
        if zone == primary:
            return "hit"
        if zone == "miss":
            return "miss"
        return "off"

    def stop(self) -> None:
        if not self.enabled:
            return
        try:  # pragma: no cover
            sd.stop()
        except Exception:
            pass


def describe() -> list[str]:
    """Human-readable legend, for the start of a session."""
    return [f"{spec.name:8s} {spec.description}" for spec in CHIMES.values()]
