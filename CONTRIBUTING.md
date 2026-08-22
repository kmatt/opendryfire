# Contributing to opendryfire

Thanks for looking. A few things that will save us both time.

## Before you file an issue

**Run `dryfire probe` and paste the output.** Camera behaviour varies wildly
between machines — frame rates, whether manual exposure works, what a driver
silently ignores — and the answer is usually in there.

**Then run the calibration commands.** Most "it doesn't detect my laser" reports
are a room problem, not a code problem, and the tools will tell you which:

```bash
uv run dryfire autocal    # measures your room, reports what fights the laser
uv run dryfire audiocal   # measures room noise vs your striker click
```

If `autocal` says it capped the margin, something in frame is nearly as red as a
laser — it will give you the coordinates. If `audiocal` says clicks aren't
clearly louder than the room, that is a microphone placement problem. Both
report the actual numbers; include them.

## Development

```bash
uv sync
uv run pytest src/tests -q
```

The importable package is `opendryfire`; the installed command is `dryfire`.

The suite needs no camera, microphone, or cartridge. Everything runs on
synthetic frames and synthesised audio, deliberately modelling the awkward cases
rather than the easy ones. If you can't reproduce a bug in a test, that's the
first thing worth fixing.

## Two rules that matter more than style

**1. Measure, don't guess.**

Every threshold in this codebase came from a measurement, and
[`docs/DESIGN.md`](docs/DESIGN.md) records which one — including the approaches
that were tried and abandoned, and the numbers that killed them. If you change a
constant, say what you measured. "This felt better" is not enough, because the
person after you cannot tell whether your number was principled or arbitrary.

Several defaults look strange until you read why: the red margin is high because
warm incandescent surfaces top out around 48 redness while a 635nm laser lands
far above; the audio floor is high because in one measured room the median
high-passed level was 2e-5 while ordinary transients hit 3e-3.

**2. Fail loudly, never plausibly.**

The worst bug this project can have is a number that looks reasonable and is
wrong, because it corrupts a training log silently and for months. Several
places refuse to guess on purpose:

- `find_beep_onset` returns `None` when no beep is audible, discarding the rep.
  An earlier version took the loudest window unconditionally and duly "found" a
  beep in pure noise.
- `build_homography` reprojects the points it was built from and rejects the fit
  if they don't land. `cv2.findHomography` does *not* reliably return `None` for
  collinear input.
- `calibrate_scene` caps the detection margin rather than raising it past the
  point where the laser itself would stop registering, and says so.

If your change makes something *silently* degrade rather than stop, it's the
wrong shape. Prefer an error the user can act on.

## Pull requests

- Add a test. If it's a bug fix, the test should fail before your change.
- Name tests for the behaviour, not the function:
  `test_a_dot_that_jumps_far_mid_pulse_is_still_one_shot`, not `test_push_2`.
- Update `docs/DESIGN.md` if you changed *why* something works, not just how.
- Keep the diff focused. Unrelated cleanups in the same PR make review slower.

## Things that would genuinely help

- **Linux and Windows support.** Detection, geometry, scoring and audio are
  portable; camera enumeration and naming in `devices.py` are macOS-specific and
  currently degrade to plain device indices elsewhere. Nobody has run this on
  either platform.
- **Cartridge compatibility data.** Development used a red 635nm unit with a
  ~100ms pulse. Other makers, green lasers, and short-pulse cartridges are
  unexplored. If you own one, `dryfire spike` output plus your `autocal` numbers
  is genuinely useful even if you write no code — the goal is for this to work
  with whatever cartridge you already have.
- **A real 60fps capture path.** OpenCV's AVFoundation backend delivers 30fps
  whatever you request, even where the hardware advertises 60. This costs
  nothing today because timing is acoustic, but more frames per pulse would
  improve drift measurement.
- **Target definitions.** Only USPSA Metric and NRA B-8 are built in.

## Safety

Any change that touches the drill flow should keep the safety framing intact.
This software scores dry fire; it does not make dry fire safe, and it should
never imply otherwise. Nothing here verifies that a firearm is unloaded, and no
feature should suggest it does.

## Licence

By contributing you agree that your contributions are licensed under
Apache-2.0, the same terms as the project.
