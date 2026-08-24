"""Camera discovery by name, and Continuity Camera preflight.

Selecting a camera by index is unreliable on macOS: indices are assigned by
enumeration order, and a Continuity Camera appears and disappears as the iPhone
connects, sleeps, or is picked up. ``--camera 1`` can silently become the
built-in webcam between one session and the next — and since the calibration is
saved per profile, a swapped camera scores every shot through a homography
built for a different lens.

So cameras are addressed by name. Mapping a name to the index OpenCV will use
is the whole problem, and the obvious answer is wrong: ``devicesWithMediaType:``
does **not** return devices in OpenCV's order. It was observed returning
``[MacBook, iPhone]`` one hour and ``[iPhone, MacBook]`` the next — it follows
connection order — while OpenCV's AVFoundation backend kept the built-in camera
at index 0 both times. Trusting that order sent ``--camera iphone`` to the
built-in webcam, named it "iPhone Camera" in the log, and scored every shot
through the wrong lens (DESIGN.md 18).

So the list is re-ordered the way OpenCV enumerates — built-in cameras first,
then the rest — and, because that is still an inference about someone else's
internals, ``camera.Camera`` verifies the identity of what it opened against
the formats the named device advertises, and refuses when they disprove it.

Three fallbacks, in order:
  1. pyobjc AVFoundation (authoritative: real names, formats, and effect state)
  2. ``system_profiler SPCameraDataType`` (names only, no formats to verify)
  3. bare indices (no names; a plain integer still works)
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass

try:  # pragma: no cover - platform dependent
    import AVFoundation as _AV

    HAVE_AVF = True
except Exception:  # pragma: no cover
    _AV = None
    HAVE_AVF = False

# Device types OpenCV's AVFoundation backend enumerates. Desk View cameras are
# deliberately excluded: AVFoundation lists them, OpenCV does not, and including
# them would shift every index by one.
_OPENCV_VISIBLE_TYPES = (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternal",
    "AVCaptureDeviceTypeContinuityCamera",
)


@dataclass(frozen=True)
class VideoDevice:
    index: int
    name: str
    model: str = ""
    uid: str = ""
    # Frame sizes the device advertises. Carried so that what a capture
    # actually delivers can be checked against the device it is supposed to be
    # — the only evidence available that the index mapping was right.
    formats: tuple[tuple[int, int], ...] = ()
    builtin: bool = False

    @property
    def is_continuity(self) -> bool:
        """True for an iPhone/iPad acting as a webcam, which can vanish."""
        n = self.name.lower()
        return "iphone" in n or "ipad" in n

    def supports(self, size: tuple[int, int]) -> bool:
        return tuple(size) in self.formats

    def __str__(self) -> str:
        tag = "  (Continuity — may disconnect)" if self.is_continuity else ""
        return f"[{self.index}] {self.name}{tag}"


def list_devices() -> list[VideoDevice]:
    """Video devices, indexed the way OpenCV indexes them."""
    if HAVE_AVF:
        try:
            return _list_avfoundation()
        except Exception:  # pragma: no cover - defensive
            pass
    if platform.system() == "Darwin" and shutil.which("system_profiler"):
        try:
            return _list_system_profiler()
        except Exception:  # pragma: no cover
            pass
    return []


def _list_avfoundation() -> list[VideoDevice]:  # pragma: no cover - needs hardware
    import CoreMedia as _CM

    devices = []
    for d in _AV.AVCaptureDevice.devicesWithMediaType_(_AV.AVMediaTypeVideo):
        sizes = set()
        try:
            for f in d.formats():
                dm = _CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
                sizes.add((int(dm.width), int(dm.height)))
        except Exception:  # pragma: no cover - defensive
            pass
        devices.append(
            VideoDevice(
                index=0,  # assigned by opencv_order below
                name=str(d.localizedName()),
                model=str(d.modelID()),
                uid=str(d.uniqueID()),
                formats=tuple(sorted(sizes)),
                builtin=str(d.deviceType()) == "AVCaptureDeviceTypeBuiltInWideAngleCamera",
            )
        )
    return opencv_order(devices)


def opencv_order(devices: list[VideoDevice]) -> list[VideoDevice]:
    """Re-index a device list the way OpenCV's AVFoundation backend does.

    Built-in cameras first, then everything else in the order the platform gave
    them. ``devicesWithMediaType:`` orders by connection instead, so a
    Continuity Camera that reconnects moves to the front of *that* list while
    OpenCV keeps the built-in webcam at index 0.
    """
    ordered = sorted(range(len(devices)), key=lambda i: (not devices[i].builtin, i))
    return [
        VideoDevice(
            index=new_index,
            name=devices[old].name,
            model=devices[old].model,
            uid=devices[old].uid,
            formats=devices[old].formats,
            builtin=devices[old].builtin,
        )
        for new_index, old in enumerate(ordered)
    ]


def _list_system_profiler() -> list[VideoDevice]:  # pragma: no cover - needs hardware
    raw = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        capture_output=True, text=True, timeout=15,
    ).stdout
    items = json.loads(raw).get("SPCameraDataType", [])
    return opencv_order([
        VideoDevice(
            index=i,
            name=str(it.get("_name", f"camera {i}")),
            model=str(it.get("spcamera_model-id", "")),
            uid=str(it.get("spcamera_unique-id", "")),
            builtin="facetime" in str(it.get("_name", "")).lower()
            or "macbook" in str(it.get("_name", "")).lower(),
        )
        for i, it in enumerate(items)
    ])


class CameraNotFound(RuntimeError):
    """Raised when a named camera is not currently connected."""


class CameraMismatch(RuntimeError):
    """Raised when the device that opened is provably not the one named."""


def identify(size: tuple[int, int], devices: list[VideoDevice]) -> VideoDevice | None:
    """The one device that advertises ``size``, if exactly one does.

    Used to check an opened capture against the name it was resolved from. Only
    a unique match is evidence — two cameras that both offer 1280x720 say
    nothing about which one is sending it.
    """
    matches = [d for d in devices if d.supports(size)]
    return matches[0] if len(matches) == 1 else None


def resolve(spec: str | int, devices: list[VideoDevice] | None = None) -> int:
    """Turn a name, a name fragment, or an index into a device index.

    Matching is case-insensitive substring, so ``iphone`` finds
    "iPhone Camera". A bare integer is passed through unchanged, so existing
    index-based usage keeps working.
    """
    if isinstance(spec, int):
        return spec
    text = str(spec).strip()
    if text.isdigit():
        return int(text)

    devices = list_devices() if devices is None else devices
    if not devices:
        raise CameraNotFound(
            f"cannot resolve camera {text!r}: no device list available. "
            f"Pass a numeric index instead."
        )

    needle = text.lower()
    matches = [d for d in devices if needle in d.name.lower()]
    if len(matches) == 1:
        return matches[0].index
    if not matches:
        known = "\n  ".join(str(d) for d in devices)
        hint = ""
        if needle in ("iphone", "ipad", "continuity"):
            hint = (
                "\n\nContinuity Camera appears only when the iPhone is nearby, "
                "unlocked at least once, with Wi-Fi and Bluetooth on, and not "
                "already in use by another app. Picking the phone up can drop it."
            )
        raise CameraNotFound(f"no camera matching {text!r}. Available:\n  {known}{hint}")
    names = ", ".join(d.name for d in matches)
    raise CameraNotFound(f"{text!r} is ambiguous — matches {names}. Be more specific.")


def describe(index: int, devices: list[VideoDevice] | None = None) -> str:
    devices = list_devices() if devices is None else devices
    for d in devices:
        if d.index == index:
            return d.name
    return f"camera {index}"


# -- Continuity Camera preflight ---------------------------------------------


@dataclass
class EffectState:
    """Whether any auto-framing or beautification effect is active.

    Center Stage is the dangerous one: it crops and pans the frame to follow
    people, so the mapping from pixels to target inches changes underneath a
    saved calibration. Nothing in the image looks wrong — the scores are just
    quietly incorrect from then on. Portrait blur and Studio Light are less
    destructive but still alter the laser dot's appearance.
    """

    center_stage: bool = False
    portrait: bool = False
    studio_light: bool = False
    controllable: bool = False
    checked: bool = False

    @property
    def any_active(self) -> bool:
        return self.center_stage or self.portrait or self.studio_light

    def warnings(self) -> list[str]:
        out = []
        if not self.checked:
            out.append(
                "Could not check video effects. If this is an iPhone, open "
                "Control Center > Video Effects and make sure Center Stage is OFF."
            )
            return out
        if self.center_stage:
            out.append(
                "CENTER STAGE IS ON. It pans and crops to follow people, which "
                "silently invalidates the calibration — turn it off in Control "
                "Center > Video Effects before calibrating."
            )
        if self.portrait:
            out.append("Portrait blur is on; it will soften the laser dot. Turn it off.")
        if self.studio_light:
            out.append("Studio Light is on; it alters exposure. Turn it off.")
        return out


def effect_state() -> EffectState:
    """Read the current video-effect state, if the platform allows."""
    if not HAVE_AVF:
        return EffectState()
    try:  # pragma: no cover - needs hardware
        return EffectState(
            center_stage=bool(_AV.AVCaptureDevice.isCenterStageEnabled()),
            portrait=any(
                bool(d.isPortraitEffectActive())
                for d in _AV.AVCaptureDevice.devicesWithMediaType_(_AV.AVMediaTypeVideo)
            ),
            studio_light=any(
                bool(d.isStudioLightActive())
                for d in _AV.AVCaptureDevice.devicesWithMediaType_(_AV.AVMediaTypeVideo)
            ),
            controllable=True,
            checked=True,
        )
    except Exception:  # pragma: no cover
        return EffectState()


def disable_center_stage() -> bool:
    """Take app-level control of Center Stage and switch it off.

    Scoped to this process, so it does not change the setting for video calls.
    Returns True if it is now off. The user can still re-enable it from Control
    Center mid-session, which is why runtime drift detection stays in place as
    the backstop.
    """
    if not HAVE_AVF:
        return False
    try:  # pragma: no cover - needs hardware
        _AV.AVCaptureDevice.setCenterStageControlMode_(1)  # 1 = App
        _AV.AVCaptureDevice.setCenterStageEnabled_(False)
        return not bool(_AV.AVCaptureDevice.isCenterStageEnabled())
    except Exception:
        return False


def preflight(index: int, devices: list[VideoDevice] | None = None) -> list[str]:
    """Checks worth running before a session. Returns messages to show."""
    devices = list_devices() if devices is None else devices
    msgs: list[str] = []
    dev = next((d for d in devices if d.index == index), None)

    if dev is not None and dev.is_continuity:
        if disable_center_stage():
            msgs.append("Center Stage disabled for this session.")
        msgs.extend(effect_state().warnings())
        msgs.append(
            "Continuity Camera: keep the iPhone still and plugged in. If it "
            "sleeps or is picked up the feed drops and the session ends."
        )
    else:
        state = effect_state()
        if state.any_active:
            msgs.extend(state.warnings())
    return msgs
