"""Camera capture with honest reporting of what the driver actually granted.

macOS/AVFoundation silently ignores most CAP_PROP writes. Rather than pretend a
setting took, every setter here reads the value back and reports the delta, so
calibration problems surface at startup instead of as mysterious missed shots.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# AVFoundation on macOS, else let OpenCV choose.
_BACKEND = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


@dataclass
class CameraInfo:
    """What a camera actually delivered, as opposed to what we asked for."""

    index: int
    width: int
    height: int
    reported_fps: float
    measured_fps: float | None = None
    backend: str = ""
    # Settings we tried to apply and whether the driver honored them.
    applied: dict[str, tuple[float, float]] = field(default_factory=dict)

    name: str = ""

    def summary(self) -> str:
        label = f"{self.name} " if self.name else ""
        lines = [
            f"camera {self.index}: {label}{self.width}x{self.height} "
            f"@ {self.reported_fps:.1f}fps reported (backend={self.backend})"
        ]
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
        from .devices import CameraNotFound, describe, resolve

        try:
            index = resolve(index)
        except CameraNotFound:
            raise

        self.index = index
        self.name = describe(index)
        self._cap = cv2.VideoCapture(index, _BACKEND)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open camera {index} ({self.name}).\n"
                f"  - If this is an iPhone, Continuity Camera may have "
                f"disconnected; run `dryfire probe` to see what is available.\n"
                f"  - Otherwise it is usually permissions: System Settings > "
                f"Privacy & Security > Camera, and enable your terminal."
            )
        self.info = CameraInfo(index=index, width=0, height=0, reported_fps=0.0)
        self._request(cv2.CAP_PROP_FRAME_WIDTH, width, "width")
        self._request(cv2.CAP_PROP_FRAME_HEIGHT, height, "height")
        self._request(cv2.CAP_PROP_FPS, fps, "fps")
        self.info.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.info.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.info.reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.info.backend = self._cap.getBackendName()

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
