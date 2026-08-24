"""Camera selection tests.

The bug being guarded against: Continuity Camera changes the index it occupies
as the iPhone connects and disconnects, so a saved ``--camera 1`` can silently
become the built-in webcam. The calibration is per-profile, not per-camera, so
the wrong lens gets scored through the right homography and every number is
quietly wrong. Selection is therefore by name.
"""

from __future__ import annotations

import pytest

from opendryfire.devices import (
    CameraNotFound,
    EffectState,
    VideoDevice,
    resolve,
)

DEVICES = [
    VideoDevice(0, "MacBook Pro Camera", "MacBook Pro Camera", "aaa"),
    VideoDevice(1, "iPhone Camera", "iPhone18,3", "bbb"),
]


def test_resolves_a_name_fragment_case_insensitively():
    assert resolve("iphone", DEVICES) == 1
    assert resolve("iPhone", DEVICES) == 1
    assert resolve("IPHONE", DEVICES) == 1
    assert resolve("MacBook", DEVICES) == 0


def test_passes_a_bare_index_through():
    """Existing index-based usage must keep working."""
    assert resolve(1, DEVICES) == 1
    assert resolve("0", DEVICES) == 0


def test_missing_camera_names_what_is_available():
    with pytest.raises(CameraNotFound) as exc:
        resolve("logitech", DEVICES)
    msg = str(exc.value)
    assert "MacBook Pro Camera" in msg
    assert "iPhone Camera" in msg


def test_missing_iphone_explains_continuity_rather_than_just_failing():
    """The common real-world case: the phone went to sleep."""
    with pytest.raises(CameraNotFound) as exc:
        resolve("iphone", [DEVICES[0]])
    assert "Continuity" in str(exc.value)


def test_ambiguous_name_is_refused_not_guessed():
    devices = [
        VideoDevice(0, "iPhone Camera"),
        VideoDevice(1, "iPhone Desk View Camera"),
    ]
    with pytest.raises(CameraNotFound) as exc:
        resolve("iphone", devices)
    assert "ambiguous" in str(exc.value).lower()


def test_continuity_devices_are_flagged():
    assert DEVICES[1].is_continuity
    assert not DEVICES[0].is_continuity
    assert VideoDevice(2, "iPad Camera").is_continuity


def test_empty_device_list_gives_an_actionable_error():
    with pytest.raises(CameraNotFound) as exc:
        resolve("iphone", [])
    assert "numeric index" in str(exc.value)


# -- effect preflight ---------------------------------------------------------


def test_center_stage_warning_explains_the_actual_consequence():
    warnings = EffectState(center_stage=True, checked=True).warnings()
    text = " ".join(warnings)
    assert "CENTER STAGE" in text
    assert "calibration" in text
    assert "Control Center" in text


def test_no_active_effects_produces_no_warnings():
    assert EffectState(checked=True).warnings() == []


def test_unchecked_state_still_advises_rather_than_staying_silent():
    """If the effect state cannot be read, say so — a silent pass would imply
    it was checked and found clean."""
    warnings = EffectState(checked=False).warnings()
    assert len(warnings) == 1
    assert "Center Stage" in warnings[0]


def test_any_active_reports_each_effect():
    state = EffectState(portrait=True, studio_light=True, checked=True)
    assert state.any_active
    text = " ".join(state.warnings()).lower()
    assert "portrait" in text
    assert "studio light" in text


# -- index order --------------------------------------------------------------
#
# ``devicesWithMediaType:`` orders by connection, so a Continuity Camera that
# reconnects jumps to the front of it while OpenCV keeps the built-in camera at
# index 0. Trusting the platform order sent `--camera iphone` to the built-in
# webcam and labelled it "iPhone Camera" in the log (DESIGN.md 18).

from opendryfire.devices import identify, opencv_order  # noqa: E402

MACBOOK = VideoDevice(0, "MacBook Pro Camera", "MacBook Pro Camera", "aaa",
                      formats=((640, 480), (1552, 1552), (1760, 1328), (1920, 1080)),
                      builtin=True)
IPHONE = VideoDevice(1, "iPhone Camera", "iPhone18,3", "bbb",
                     formats=((640, 480), (1280, 720), (1920, 1080), (1920, 1440)))


def test_a_reconnected_iphone_does_not_take_index_zero():
    """The platform lists it first; OpenCV does not."""
    platform_order = [
        VideoDevice(0, IPHONE.name, IPHONE.model, IPHONE.uid, IPHONE.formats, False),
        VideoDevice(1, MACBOOK.name, MACBOOK.model, MACBOOK.uid, MACBOOK.formats, True),
    ]
    ordered = opencv_order(platform_order)

    assert [d.name for d in ordered] == ["MacBook Pro Camera", "iPhone Camera"]
    assert [d.index for d in ordered] == [0, 1]
    assert resolve("iphone", ordered) == 1


def test_the_order_is_unchanged_when_the_platform_already_agrees():
    ordered = opencv_order([MACBOOK, IPHONE])
    assert [d.index for d in ordered] == [0, 1]
    assert [d.name for d in ordered] == ["MacBook Pro Camera", "iPhone Camera"]


def test_external_only_setups_keep_their_order():
    a = VideoDevice(0, "Camera A", formats=((640, 480),))
    b = VideoDevice(1, "Camera B", formats=((1280, 720),))
    assert [d.name for d in opencv_order([a, b])] == ["Camera A", "Camera B"]


def test_a_uniquely_offered_format_identifies_which_device_is_sending():
    devices = [MACBOOK, IPHONE]
    assert identify((1552, 1552), devices).name == "MacBook Pro Camera"
    assert identify((1920, 1440), devices).name == "iPhone Camera"


def test_a_format_both_cameras_offer_identifies_nothing():
    """Only a unique match is evidence — this is what stops the check crying
    wolf on a size every webcam supports."""
    assert identify((1920, 1080), [MACBOOK, IPHONE]) is None
    assert identify((3000, 2000), [MACBOOK, IPHONE]) is None
