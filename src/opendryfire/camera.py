"""Camera capture with honest reporting of what the driver actually granted.

macOS/AVFoundation silently ignores most CAP_PROP writes. Rather than pretend a
setting took, every setter here reads the value back and reports the delta, so
calibration problems surface at startup instead of as mysterious missed shots.

For frame size the read-back is not enough, because it is itself a lie: the
backend echoes the size we asked for while sending a different one. Size is
therefore measured from a delivered frame — see ``Camera._configure``.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# AVFoundation on macOS, else let OpenCV choose.
_BACKEND = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

# How many times to re-apply the format before accepting what we are given.
# The re-application is opportunistic — the failure is intermittent and has not
# been reproduced on demand, so nothing here depends on it working. What the
# correctness depends on is measuring the result, below.
FORMAT_ATTEMPTS = 3
# Reads to wait for the first frame. A Continuity Camera takes up to a second
# to produce one, and read() returns None until it does.
FIRST_FRAME_TRIES = 40


@dataclass
class CameraInfo:
    """What a camera actually delivered, as opposed to what we asked for."""

    index: int
    width: int
    height: int
    reported_fps: float
    measured_fps: float | None = None
    backend: str = ""
    # Settings we tried to apply and whether the driver honored them. For width
    # and height "got" is measured from a delivered frame, not read back from
    # the property — see ``Camera._configure``.
    applied: dict[str, tuple[float, float]] = field(default_factory=dict)

    name: str = ""
    # What the driver's own property read-back claimed the size was, which is
    # not always what it sends.
    claimed: tuple[int, int] = (0, 0)
    # False when no frame arrived during setup, so width/height are the
    # driver's claim rather than a measurement.
    verified: bool = False

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def size_is_a_lie(self) -> bool:
        """True when the driver reports a size it is not actually sending."""
        return (
            self.verified
            and self.claimed != (0, 0)
            and self.claimed != self.frame_size
        )

    def summary(self) -> str:
        label = f"{self.name} " if self.name else ""
        measured = "" if self.verified else " (unverified — no frame arrived)"
        lines = [
            f"camera {self.index}: {label}{self.width}x{self.height}{measured} "
            f"@ {self.reported_fps:.1f}fps reported (backend={self.backend})"
        ]
        if self.size_is_a_lie():
            lines.append(
                f"  LIED the driver reports {self.claimed[0]}x{self.claimed[1]}; "
                f"the frames it sends are {self.width}x{self.height}"
            )
        if self.measured_fps is not None:
            lines.append(f"  measured: {self.measured_fps:.1f}fps")
        for name, (asked, got) in self.applied.items():
            mark = "ok " if abs(asked - got) < 1e-6 else "IGNORED"
            lines.append(f"  {mark} {name}: asked {asked:g}, got {got:g}")
        return "\n".join(lines)


class Camera:
    """A capture device that timestamps every frame on a monotonic clock.

    The timestamp is taken the instant ``read()`` returns, which includes the
    driver's pipeline delay. That delay is why timing does not come from here —
    see ``opendryfire.audio``. For hit *location* it is irrelevant.
    """

    def __init__(
        self,
        index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
    ) -> None:
        from .devices import CameraNotFound, describe, list_devices, resolve

        devices = list_devices()
        # A bare index is the caller naming the hardware directly, and it is
        # the escape hatch the mismatch error points people at — so only a name
        # that had to be looked up gets its lookup second-guessed below.
        self._named = not str(index).strip().isdigit()
        self._devices = devices
        try:
            index = resolve(index, devices)
        except CameraNotFound:
            raise

        self.index = index
        self.name = describe(index, devices)
        self._cap = cv2.VideoCapture(index, _BACKEND)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open camera {index} ({self.name}).\n"
                f"  - If this is an iPhone, Continuity Camera may have "
                f"disconnected; run `dryfire probe` to see what is available.\n"
                f"  - Otherwise it is usually permissions: System Settings > "
                f"Privacy & Security > Camera, and enable your terminal."
            )
        self.info = CameraInfo(
            index=index, width=0, height=0, reported_fps=0.0, name=self.name
        )
        self._configure(width, height, fps)
        self._verify_identity()

    def _verify_identity(self) -> None:
        """Check that the device that opened is the one the name resolved to.

        Name-to-index mapping is an inference about OpenCV's enumeration order
        (see ``devices.opencv_order``), and when it was wrong the only symptom
        was a log line confidently naming the wrong camera. The frames carry
        evidence: if the size being delivered is one that exactly one device in
        the list advertises, and that device is not this one, the mapping is
        disproved.
        """
        from .devices import CameraMismatch, identify

        if not self._named or not self.info.verified or not self._devices:
            return
        here = next((d for d in self._devices if d.index == self.index), None)
        if here is None or not here.formats or here.supports(self.info.frame_size):
            return
        actual = identify(self.info.frame_size, self._devices)
        if actual is None or actual.index == self.index:
            return
        self.release()
        raise CameraMismatch(
            f"asked for {here.name!r} at index {self.index}, but that device is "
            f"sending {self.info.width}x{self.info.height} — a format only "
            f"{actual.name!r} offers.\n"
            f"  The camera list and OpenCV's indexes disagree, which happens "
            f"when a Continuity Camera reconnects.\n"
            f"  Open it directly with:  --camera {actual.index}"
        )

    def _configure(self, width: int, height: int, fps: int) -> None:
        """Apply the format, then measure what the device is *actually* sending.

        The property read-back cannot be trusted for size. On AVFoundation it
        echoes the value we asked for even when the session stayed in the
        device's default format: a request for 1920x1440 read back as 1920x1440
        from ``cap.get`` while the arriving frames were 1920x1080, on roughly
        one open in eight. The two are not the same view — measured against the
        4:3 stream, the 16:9 one is scaled 0.9 with ~120 rows cut from the top
        and the bottom (DESIGN.md 16) — so a session that starts in the wrong
        one scores every hit through a homography built for the other.

        A delivered frame is therefore the measurement, and the property is
        recorded only as a claim, so the disagreement is visible instead of
        silent.
        """
        for _ in range(FORMAT_ATTEMPTS):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._request(cv2.CAP_PROP_FPS, fps, "fps")
            self.info.claimed = (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            self.info.reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            self.info.backend = self._cap.getBackendName()

            got = self._first_frame_size()
            if got is None:
                # Nothing arrived. Fall back to the claim rather than reporting
                # 0x0, and mark it unverified so callers can say so.
                self.info.width, self.info.height = self.info.claimed
                self.info.verified = False
                return
            self.info.verified = True
            self.info.width, self.info.height = got
            self.info.applied["width"] = (float(width), float(got[0]))
            self.info.applied["height"] = (float(height), float(got[1]))
            if got == (width, height):
                return

    def _first_frame_size(self) -> tuple[int, int] | None:
        """The size of the next frame the device actually hands over."""
        for _ in range(FIRST_FRAME_TRIES):
            r = self.read()
            if r is not None:
                h, w = r[0].shape[:2]
                return int(w), int(h)
        return None

    def _request(self, prop: int, value: float, name: str) -> None:
        self._cap.set(prop, value)
        got = float(self._cap.get(prop))
        self.info.applied[name] = (float(value), got)

    def try_manual_exposure(self, exposure: float = -7.0, gain: float = 0.0) -> bool:
        """Attempt to lock exposure down so only the laser is bright.

        Returns True only if the driver actually took the value. On the built-in
        MacBook camera this usually returns False — that is expected, and the
        fallback is a dim room plus the red-dominance test in ``detect``.
        """
        self._request(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure_off")
        self._request(cv2.CAP_PROP_EXPOSURE, exposure, "exposure")
        self._request(cv2.CAP_PROP_GAIN, gain, "gain")
        asked, got = self.info.applied["exposure"]
        return abs(asked - got) < 1e-6

    def read(self) -> tuple[np.ndarray, float] | None:
        """Return (frame, monotonic_timestamp) or None if the device dropped."""
        ok, frame = self._cap.read()
        t = time.perf_counter()
        if not ok or frame is None:
            return None
        return frame, t

    def measure_fps(self, n_frames: int = 120) -> float:
        """Measure the frame rate we actually get, which is often not the one
        the driver reports. Warms up for 10 frames first."""
        for _ in range(10):
            self.read()
        t0 = time.perf_counter()
        got = 0
        for _ in range(n_frames):
            if self.read() is not None:
                got += 1
        elapsed = time.perf_counter() - t0
        measured = got / elapsed if elapsed > 0 else 0.0
        self.info.measured_fps = measured
        return measured

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def enumerate_cameras(max_index: int = 4) -> list[CameraInfo]:
    """Probe device indices and report what opens, with names where available.

    macOS assigns indices by connection order, so the built-in camera is not
    reliably 0 once an iPhone is available over Continuity. Prefer selecting by
    name — see ``opendryfire.devices``.
    """
    from .devices import list_devices

    named = {d.index: d.name for d in list_devices()}
    found: list[CameraInfo] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _BACKEND)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            found.append(
                CameraInfo(
                    index=i,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    reported_fps=float(cap.get(cv2.CAP_PROP_FPS)),
                    backend=cap.getBackendName(),
                    name=named.get(i, ""),
                )
            )
        cap.release()
    return found
