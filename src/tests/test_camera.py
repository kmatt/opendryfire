"""Camera setup tests.

The bug being guarded against: the capture backend does not always give you the
format you asked for, and its property read-back does not tell you. On this
machine a request for 1920x1440 read back as 1920x1440 from ``cap.get`` while
the frames arriving were 1920x1080 — roughly one open in eight. The 16:9 stream
is not the 4:3 one at another size: it is scaled 0.9 with ~120 rows cut off the
top and bottom, so a session started in the wrong one scores every hit through
a homography built for the other, and nothing looks wrong (DESIGN.md 16).

So size is measured from a delivered frame here, and the property is recorded
only as a claim.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from opendryfire import camera as camera_mod
from opendryfire import devices as devices_mod
from opendryfire.camera import Camera


class FakeCapture:
    """A capture device that echoes requested sizes while sending its own.

    ``delivers`` is the list of formats it really produces, one per configure
    attempt; the last is reused once exhausted. ``frames_before_first`` models a
    Continuity Camera taking a moment to wake up.
    """

    def __init__(self, delivers, frames_before_first=0, opened=True):
        self.delivers = list(delivers)
        self.frames_before_first = frames_before_first
        self._opened = opened
        self._props: dict[int, float] = {}
        self.attempt = -1
        self.reads = 0
        self.sets: list[tuple[int, float]] = []

    # -- what OpenCV exposes
    def isOpened(self):  # noqa: N802
        return self._opened

    def set(self, prop, value):  # noqa: A003
        self.sets.append((prop, float(value)))
        # The lie under test: the property remembers the request whatever the
        # session actually negotiated.
        self._props[prop] = float(value)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.attempt += 1
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0  # the driver's honest refusal of 60
        return self._props.get(prop, 0.0)

    def getBackendName(self):  # noqa: N802
        return "FAKE"

    def read(self):
        self.reads += 1
        if self.reads <= self.frames_before_first:
            return False, None
        i = min(max(self.attempt, 0), len(self.delivers) - 1)
        w, h = self.delivers[i]
        return True, np.zeros((h, w, 3), np.uint8)

    def release(self):
        self._opened = False


@pytest.fixture
def fake_device(monkeypatch):
    """No hardware, and no device enumeration on the way to it."""
    monkeypatch.setattr(devices_mod, "resolve", lambda spec, devices=None: 1)
    monkeypatch.setattr(devices_mod, "describe", lambda i, devices=None: "iPhone Camera")

    def install(cap):
        monkeypatch.setattr(camera_mod.cv2, "VideoCapture", lambda *a, **k: cap)
        return cap

    return install


def test_size_comes_from_the_frames_not_from_the_property(fake_device):
    cap = fake_device(FakeCapture([(1920, 1080)]))
    cam = Camera(1, 1920, 1440, 60)

    assert cam.info.frame_size == (1920, 1080)
    assert cam.info.claimed == (1920, 1440)
    assert cam.info.size_is_a_lie()
    assert cam.info.applied["height"] == (1440.0, 1080.0)
    assert "IGNORED height" in cam.info.summary()
    assert "LIED" in cam.info.summary()
    cap.release()


def test_a_format_that_takes_on_a_later_attempt_is_the_one_reported(fake_device):
    fake_device(FakeCapture([(1920, 1080), (1920, 1440)]))
    cam = Camera(1, 1920, 1440, 60)

    assert cam.info.frame_size == (1920, 1440)
    assert not cam.info.size_is_a_lie()
    assert "IGNORED height" not in cam.info.summary()


def test_it_gives_up_after_a_bounded_number_of_attempts(fake_device):
    cap = fake_device(FakeCapture([(1920, 1080)] * 99))
    cam = Camera(1, 1920, 1440, 60)

    width_sets = [v for prop, v in cap.sets if prop == cv2.CAP_PROP_FRAME_WIDTH]
    assert len(width_sets) == camera_mod.FORMAT_ATTEMPTS
    assert cam.info.frame_size == (1920, 1080)


def test_it_waits_for_a_camera_that_is_slow_to_send_its_first_frame(fake_device):
    fake_device(FakeCapture([(1920, 1440)], frames_before_first=25))
    cam = Camera(1, 1920, 1440, 60)

    assert cam.info.frame_size == (1920, 1440)
    assert cam.info.verified


def test_a_camera_that_never_sends_a_frame_is_reported_as_unverified(fake_device):
    fake_device(FakeCapture([(1920, 1440)], frames_before_first=10_000))
    cam = Camera(1, 1920, 1440, 60)

    # Falling back to the claim beats reporting 0x0, but it must not be
    # mistaken for a measurement: nothing was measured.
    assert not cam.info.verified
    assert cam.info.frame_size == (1920, 1440)
    assert "unverified" in cam.info.summary()
    assert not cam.info.size_is_a_lie()


def test_an_honoured_request_says_nothing_alarming(fake_device):
    fake_device(FakeCapture([(1920, 1440)]))
    cam = Camera(1, 1920, 1440, 60)

    summary = cam.info.summary()
    assert cam.info.frame_size == (1920, 1440)
    assert "LIED" not in summary
    assert "IGNORED width" not in summary
    assert "IGNORED height" not in summary
    # fps is a separate, honest refusal and should still be reported as one.
    assert "IGNORED fps" in summary


# -- identity -----------------------------------------------------------------


def test_it_refuses_a_device_whose_frames_prove_it_is_a_different_camera(monkeypatch):
    """The failure this exists for: the name resolved to an index, the index
    opened the wrong camera, and the only symptom was a log line naming the
    camera the user asked for (DESIGN.md 18)."""
    from opendryfire.devices import CameraMismatch, VideoDevice

    devices = [
        VideoDevice(0, "MacBook Pro Camera", builtin=True,
                    formats=((1552, 1552), (1760, 1328))),
        VideoDevice(1, "iPhone Camera", formats=((1280, 720), (1920, 1440))),
    ]
    monkeypatch.setattr(devices_mod, "list_devices", lambda: devices)
    monkeypatch.setattr(devices_mod, "resolve", lambda spec, d=None: 0)
    monkeypatch.setattr(devices_mod, "describe",
                        lambda i, d=None: "MacBook Pro Camera")
    # index 0 hands over 1920x1440, which only the iPhone offers
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture",
                        lambda *a, **k: FakeCapture([(1920, 1440)]))

    with pytest.raises(CameraMismatch) as exc:
        Camera("macbook", 1920, 1440, 60)
    assert "iPhone Camera" in str(exc.value)
    assert "--camera 1" in str(exc.value)


def test_a_bare_index_is_taken_at_its_word(monkeypatch):
    """It is the escape hatch the mismatch error points people at, so it must
    not itself be refused."""
    from opendryfire.devices import VideoDevice

    devices = [
        VideoDevice(0, "MacBook Pro Camera", builtin=True, formats=((1552, 1552),)),
        VideoDevice(1, "iPhone Camera", formats=((1920, 1440),)),
    ]
    monkeypatch.setattr(devices_mod, "list_devices", lambda: devices)
    monkeypatch.setattr(devices_mod, "describe", lambda i, d=None: "MacBook Pro Camera")
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture",
                        lambda *a, **k: FakeCapture([(1920, 1440)]))

    cam = Camera(0, 1920, 1440, 60)  # no exception
    assert cam.info.frame_size == (1920, 1440)


def test_a_format_both_cameras_offer_is_not_treated_as_proof(monkeypatch):
    from opendryfire.devices import VideoDevice

    devices = [
        VideoDevice(0, "MacBook Pro Camera", builtin=True,
                    formats=((1280, 720), (1552, 1552))),
        VideoDevice(1, "iPhone Camera", formats=((1280, 720), (1920, 1440))),
    ]
    monkeypatch.setattr(devices_mod, "list_devices", lambda: devices)
    monkeypatch.setattr(devices_mod, "resolve", lambda spec, d=None: 0)
    monkeypatch.setattr(devices_mod, "describe", lambda i, d=None: "MacBook Pro Camera")
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture",
                        lambda *a, **k: FakeCapture([(1280, 720)]))

    cam = Camera("macbook", 1280, 720, 60)  # ambiguous, so no accusation
    assert cam.info.frame_size == (1280, 720)
