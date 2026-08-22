"""Acoustic shot timing.

Why timing does not come from the camera: both cameras on this machine cap at
~30fps regardless of what we request, so a video-derived time is quantised to
33ms before the driver's variable pipeline delay is even counted. The training
plan moves par times in 0.1s steps, so a 33-100ms error is a large fraction of
the thing being measured.

So the microphone keeps time. The trick that makes it exact: the start beep is
played *and recorded on the same input stream* as the striker click. Draw time
is the difference between two events observed by one clock:

    draw_time = (click_sample - beep_sample) / sample_rate

Output latency, input latency and USB jitter all cancel, because they apply
equally to both events. This is how a real shot timer works — it is an acoustic
instrument, not a video one.

Detection:
* The beep is a known tone, found with a Goertzel filter at its own frequency.
* The striker click is a sharp broadband transient, found by high-pass filtering
  away voice and room rumble, then looking for a spectral-flux onset that rises
  far above a running median. Median/MAD is used rather than a fixed threshold
  so the detector adapts to room noise instead of needing a magic number.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from scipy import signal

SAMPLE_RATE = 48_000
BLOCK = 256  # 5.3ms of latency granularity
BEEP_FREQ = 2_400.0  # well above voice, below the click's main energy
BEEP_DUR = 0.30


@dataclass(frozen=True)
class Click:
    """A detected striker fall, timestamped on the audio sample clock."""

    sample: int
    strength: float

    def seconds(self, rate: int = SAMPLE_RATE) -> float:
        return self.sample / rate


@dataclass
class OnsetConfig:
    """Thresholds for click detection.

    ``ratio`` is the main sensitivity knob: high-frequency energy in this block
    must exceed the recent median by this factor. A ratio is used rather than an
    absolute level so the detector is scale-invariant — it works the same for a
    quiet room and a noisy one, and needs no per-room magic number. A striker
    fall is a 10-100x spike in HF energy; room noise wanders by well under 2x.

    Erring strict is deliberate: a false click silently corrupts a par time,
    while a missed click is obvious to the shooter and can simply be re-shot.
    """

    highpass_hz: float = 1_800.0
    ratio: float = 6.0
    # Absolute RMS floor. Deliberately conservative, because the ratio test
    # alone is not enough in a quiet room: measured here, the median
    # high-passed level was 2e-5 while ordinary transients (fan, keystroke)
    # reached 3e-3 — so `ratio * median` sits far below the room's own noise
    # and the floor is what actually decides. At 2e-4 this room produced 5
    # phantom clicks per 10s; at 5e-3, zero, while still sitting 10-100x below
    # a real striker fall. Erring high is the safe direction: a missed press is
    # obvious to the shooter, a phantom one silently corrupts a draw time.
    # `dryfire audiocal` replaces this with a value measured in your room.
    floor: float = 5e-3
    refractory_s: float = 0.15
    history_s: float = 1.5  # window for the running median
    warmup_blocks: int = 16  # blocks to observe before trusting the median


class ClickDetector:
    """Streaming onset detector over high-passed audio.

    Detects a click as a sharp *ratio* jump in high-frequency energy above the
    running median, rather than an absolute threshold or a raw energy
    difference. Both alternatives were tried and both fire on room noise, since
    a raw difference has no sense of scale.
    """

    def __init__(self, rate: int = SAMPLE_RATE, config: OnsetConfig | None = None):
        self.rate = rate
        self.cfg = config or OnsetConfig()
        self._sos = signal.butter(
            4, self.cfg.highpass_hz, btype="highpass", fs=rate, output="sos"
        )
        self._zi = signal.sosfilt_zi(self._sos) * 0.0
        self._history: list[float] = []
        self._hist_max = max(int(self.cfg.history_s * rate / BLOCK), 8)
        self._samples_seen = 0
        self._last_click = -(10**9)
        self._blocks_seen = 0
        self._muted_until = 0

    def push(self, block: np.ndarray) -> list[Click]:
        """Feed one block of mono float32. Returns clicks found in it."""
        filtered, self._zi = signal.sosfilt(self._sos, block, zi=self._zi)
        energy = float(np.sqrt(np.mean(filtered**2)) + 1e-12)

        muted = self._samples_seen < self._muted_until
        clicks: list[Click] = []
        if not muted and self._blocks_seen >= self.cfg.warmup_blocks:
            med = float(np.median(self._history)) + 1e-12
            threshold = max(self.cfg.ratio * med, self.cfg.floor)
            past_refractory = (
                self._samples_seen - self._last_click
                > self.cfg.refractory_s * self.rate
            )
            if energy > threshold and past_refractory:
                self._last_click = self._samples_seen
                clicks.append(
                    Click(sample=self._samples_seen, strength=energy / threshold)
                )

        # A click must not raise its own baseline, or a burst of fast reps
        # would progressively deafen the detector. Muted audio is excluded for
        # the same reason: a chime is not room tone and must not be learned
        # as though it were.
        if not clicks and not muted:
            self._history.append(energy)
            if len(self._history) > self._hist_max:
                self._history.pop(0)
        self._blocks_seen += 1
        self._samples_seen += len(block)
        return clicks

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    def mute_until(self, sample: int) -> None:
        """Suppress detections up to a sample index.

        Used so the app's own sounds — the start beep, and the hit chimes —
        cannot be detected as striker clicks. Muting is expressed on the sample
        clock rather than as a duration so it stays correct regardless of how
        far behind the audio callback is running.
        """
        self._muted_until = max(getattr(self, "_muted_until", 0), sample)


def goertzel_energy(x: np.ndarray, freq: float, rate: int = SAMPLE_RATE) -> float:
    """Energy at one frequency. Cheaper and sharper than an FFT bin for a
    single known tone."""
    n = len(x)
    if n == 0:
        return 0.0
    k = int(0.5 + n * freq / rate)
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)
    s0 = s1 = s2 = 0.0
    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2, s1 = s1, s0
    return float(s1 * s1 + s2 * s2 - coeff * s1 * s2) / n


def find_beep_onset(
    audio: np.ndarray,
    rate: int = SAMPLE_RATE,
    freq: float = BEEP_FREQ,
    win: int = 256,
    min_prominence: float = 20.0,
) -> int | None:
    """Locate the first sample where the beep tone appears in a recording.

    Returns a sample index, or None if the beep was never heard — meaning the
    microphone could not hear the speaker. That case *must* return None: a rep
    with no audible beep has no valid start time, and inventing one silently
    corrupts the log. An earlier version took the loudest window unconditionally
    and duly "found" a beep in pure room noise.

    Two independent conditions have to hold, so neither a loud broadband noise
    nor a quiet room can pass:

    1. *Prominence* — peak tone energy exceeds the median window by
       ``min_prominence``, i.e. there is a distinct event at all.
    2. *Tonality* — at the peak, energy at the beep frequency is a real fraction
       of the window's total energy, i.e. the event is a tone and not a clatter.
    """
    n_win = len(audio) // win
    if n_win < 8:
        return None
    tone = np.array(
        [
            goertzel_energy(audio[i * win : (i + 1) * win], freq, rate)
            for i in range(n_win)
        ]
    )
    total = np.array(
        [float(np.mean(audio[i * win : (i + 1) * win] ** 2)) for i in range(n_win)]
    )

    peak_i = int(np.argmax(tone))
    peak = float(tone[peak_i])
    median = float(np.median(tone)) + 1e-15
    if peak <= 0 or peak / median < min_prominence:
        return None
    # The beep is a pure tone, so it should dominate its own window's energy.
    if peak < 0.15 * float(total[peak_i]):
        return None

    # Rising edge: walk back from the peak to where it crosses 25% of it.
    threshold = 0.25 * peak
    i = peak_i
    while i > 0 and tone[i - 1] > threshold:
        i -= 1
    return int(i * win)


def make_beep(
    duration: float = BEEP_DUR,
    freq: float = BEEP_FREQ,
    rate: int = SAMPLE_RATE,
    amplitude: float = 0.6,
) -> np.ndarray:
    """A beep with a short raised-cosine ramp, so its onset is sharp enough to
    time against but does not click."""
    t = np.arange(int(duration * rate)) / rate
    tone = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
    ramp_n = int(0.004 * rate)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_n) / ramp_n))
    tone[:ramp_n] *= ramp
    tone[-ramp_n:] *= ramp[::-1]
    return tone.astype(np.float32)


@dataclass
class TimedRep:
    """One par-time repetition."""

    beep_sample: int | None
    clicks: list[Click] = field(default_factory=list)
    rate: int = SAMPLE_RATE

    @property
    def valid(self) -> bool:
        return self.beep_sample is not None and bool(self.clicks)

    @property
    def first_shot_time(self) -> float | None:
        """Seconds from the beep to the first click."""
        if not self.valid:
            return None
        return (self.clicks[0].sample - self.beep_sample) / self.rate

    @property
    def splits(self) -> list[float]:
        """Times between consecutive clicks."""
        return [
            (b.sample - a.sample) / self.rate
            for a, b in zip(self.clicks, self.clicks[1:])
        ]


class ShotTimer:
    """Full-duplex beep-and-listen timer.

    Runs one stream so the beep and the clicks share a sample clock. Everything
    is analysed post-hoc from the recorded buffer, which keeps the audio
    callback free of allocation and DSP.
    """

    def __init__(
        self,
        rate: int = SAMPLE_RATE,
        config: OnsetConfig | None = None,
        input_device: int | None = None,
        output_device: int | None = None,
    ) -> None:
        self.rate = rate
        self.cfg = config or OnsetConfig()
        self.input_device = input_device
        self.output_device = output_device

    def run_rep(self, listen_seconds: float = 6.0, delay: float = 0.0) -> TimedRep:
        """Play a beep, listen, and return the rep. Blocking."""
        beep = make_beep(rate=self.rate)
        total = int((listen_seconds + delay) * self.rate)
        out_buf = np.zeros(total, dtype=np.float32)
        start = int(delay * self.rate)
        out_buf[start : start + len(beep)] = beep

        recorded = np.zeros(total, dtype=np.float32)
        pos = {"i": 0, "o": 0}

        def callback(indata, outdata, frames, time_info, status):  # noqa: ANN001
            i, o = pos["i"], pos["o"]
            n = min(frames, total - i)
            if n > 0:
                recorded[i : i + n] = indata[:n, 0]
            pos["i"] = i + frames
            n = min(frames, total - o)
            outdata[:] = 0
            if n > 0:
                outdata[:n, 0] = out_buf[o : o + n]
            pos["o"] = o + frames

        with sd.Stream(
            samplerate=self.rate,
            blocksize=BLOCK,
            channels=1,
            dtype="float32",
            callback=callback,
            device=(self.input_device, self.output_device),
        ):
            deadline = time.monotonic() + listen_seconds + delay + 0.3
            while pos["i"] < total and time.monotonic() < deadline:
                time.sleep(0.01)

        return self.analyse(recorded)

    def analyse(self, recorded: np.ndarray) -> TimedRep:
        """Find the beep and the clicks in a recorded buffer."""
        beep_sample = find_beep_onset(recorded, self.rate)
        det = ClickDetector(self.rate, self.cfg)
        clicks: list[Click] = []
        # Do not let the beep itself count as a click.
        beep_end = (beep_sample + int(BEEP_DUR * self.rate) + int(0.05 * self.rate)) if beep_sample is not None else 0
        for i in range(0, len(recorded) - BLOCK, BLOCK):
            block = recorded[i : i + BLOCK]
            for c in det.push(block):
                if c.sample >= beep_end:
                    clicks.append(c)
        return TimedRep(beep_sample=beep_sample, clicks=clicks, rate=self.rate)


class LiveListener:
    """Continuous background click detection, for untimed drills.

    Block F is explicitly untimed for its first month ("this is the one block
    where the clock is the enemy"), but still needs a click to pair with each
    camera hit. This provides that without imposing a par.
    """

    def __init__(
        self,
        rate: int = SAMPLE_RATE,
        config: OnsetConfig | None = None,
        input_device: int | None = None,
    ) -> None:
        self.rate = rate
        self.cfg = config or OnsetConfig()
        self.input_device = input_device
        self._q: queue.Queue[Click] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._det = ClickDetector(rate, self.cfg)
        self._t0 = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        def callback(indata, frames, time_info, status):  # noqa: ANN001
            with self._lock:
                for c in self._det.push(indata[:, 0].copy()):
                    self._q.put(c)

        self._t0 = time.perf_counter()
        self._stream = sd.InputStream(
            samplerate=self.rate,
            blocksize=BLOCK,
            channels=1,
            dtype="float32",
            callback=callback,
            device=self.input_device,
        )
        self._stream.start()

    def poll(self) -> list[Click]:
        out: list[Click] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def mute_for(self, seconds: float) -> None:
        """Deafen the click detector for a while, starting now.

        Called before the app plays a sound of its own. Without this the
        speakers feed the microphone: a hit chime is detected as a strike,
        which scores a phantom shot, which plays another chime.
        """
        with self._lock:
            self._det.mute_until(self._det.samples_seen + int(seconds * self.rate))

    def wall_time(self, click: Click) -> float:
        """Convert a click's sample index to the perf_counter clock the camera
        uses, so audio and video events can be correlated."""
        return self._t0 + click.sample / self.rate

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# -- click-floor calibration --------------------------------------------------


def block_energies(
    audio: np.ndarray, rate: int = SAMPLE_RATE, highpass_hz: float = 1_800.0
) -> np.ndarray:
    """Per-block high-passed RMS — the quantity the click detector thresholds."""
    sos = signal.butter(4, highpass_hz, btype="highpass", fs=rate, output="sos")
    hf = signal.sosfilt(sos, audio)
    n = (len(hf) - BLOCK) // BLOCK
    return np.array(
        [float(np.sqrt(np.mean(hf[i * BLOCK : (i + 1) * BLOCK] ** 2))) for i in range(max(n, 0))]
    )


def peak_events(
    energies: np.ndarray, count: int, separation_blocks: int = 30
) -> list[float]:
    """The ``count`` loudest well-separated peaks.

    Used to measure striker clicks without a detector, which would otherwise be
    circular — we are calibrating the detector.
    """
    picked: list[float] = []
    taken: list[int] = []
    order = np.argsort(energies)[::-1]
    for idx in order:
        if len(picked) >= count:
            break
        if all(abs(int(idx) - t) >= separation_blocks for t in taken):
            taken.append(int(idx))
            picked.append(float(energies[idx]))
    return picked


@dataclass
class AudioCalibration:
    """Result of separating room noise from real striker clicks."""

    room_median: float
    room_max: float
    click_min: float
    click_median: float
    floor: float
    ratio: float
    separation: float  # click_min / room_max; >3 is comfortable
    ok: bool
    note: str = ""

    def config(self) -> OnsetConfig:
        return OnsetConfig(floor=self.floor, ratio=self.ratio)

    def lines(self) -> list[str]:
        out = [
            f"room noise  : median {self.room_median:.2e}, worst transient {self.room_max:.2e}",
            f"striker click: quietest {self.click_min:.2e}, typical {self.click_median:.2e}",
            f"separation   : {self.separation:.1f}x  "
            f"({'comfortable' if self.separation >= 3 else 'tight'})",
            f"chosen floor : {self.floor:.2e}",
        ]
        if self.note:
            out.append(self.note)
        return out


def calibrate_click_floor(
    room: np.ndarray,
    clicks_recording: np.ndarray,
    expected_clicks: int,
    rate: int = SAMPLE_RATE,
    ratio: float = 6.0,
) -> AudioCalibration:
    """Choose a detection floor that separates real clicks from room noise.

    Two-class calibration, the audio counterpart of ``detect.calibrate_scene``:
    measure the room with nothing happening, measure actual striker falls, and
    put the threshold between them — geometrically, so it sits at the same
    ratio below the quietest click as above the loudest room transient.

    Guessing a floor does not work. In one measured room the median high-passed
    level was 2e-5 while ordinary transients (a fan, a keystroke) reached 3e-3,
    a 170x spread. A fixed default is either deaf or constantly false-triggering
    depending on the room it lands in.
    """
    room_e = block_energies(room, rate)
    click_e = block_energies(clicks_recording, rate)
    if room_e.size == 0 or click_e.size == 0:
        raise ValueError("recordings too short to calibrate")

    room_median = float(np.median(room_e))
    room_max = float(room_e.max())
    peaks = peak_events(click_e, expected_clicks)
    if not peaks:
        raise ValueError("no clicks found in the second recording")

    click_min = float(min(peaks))
    click_median = float(np.median(peaks))
    separation = click_min / room_max if room_max > 0 else float("inf")

    if separation >= 1.5:
        floor = float(np.sqrt(room_max * click_min))  # geometric midpoint
        ok = True
        note = ""
    else:
        # Clicks are not louder than the room's own transients. Favour missing
        # a shot over inventing one: the shooter notices a missed rep, but a
        # phantom click silently corrupts a draw time.
        floor = room_max * 1.1
        ok = False
        note = (
            "Clicks are not clearly louder than room noise. The floor is set "
            "above the room, so some presses will be missed rather than noise "
            "being scored. Move the microphone closer to the pistol, or quieten "
            "the room (fans, HVAC, laptop under load)."
        )
    return AudioCalibration(
        room_median=room_median,
        room_max=room_max,
        click_min=click_min,
        click_median=click_median,
        floor=floor,
        ratio=ratio,
        separation=separation,
        ok=ok,
        note=note,
    )
