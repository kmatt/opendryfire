# opendryfire

[![tests](https://github.com/aysark/opendryfire/actions/workflows/tests.yml/badge.svg)](https://github.com/aysark/opendryfire/actions/workflows/tests.yml)

Scores laser dry fire practice from a webcam, and times it from the microphone.

Point a camera at a target, put a laser training cartridge in your (unloaded)
pistol, and it records where each shot landed and how long the draw took. Works
with **scaled targets** — a 1/5-scale target at 9 feet reports groups in
15-yard-equivalent inches, so home numbers are comparable to range numbers.

Works with any firing-pin-activated laser cartridge — no proprietary hardware,
no subscription, no vendor app. Developed on macOS; see [Hardware](#hardware).

> **Safety.** This scores dry fire; it does not make dry fire safe. All live
> ammunition leaves the room — not the bench, the room. Verify the chamber
> visually *and* physically. Point at something that would actually stop a
> round: a concrete wall, a loaded bookcase, an exterior wall with nothing
> behind it. Most negligent discharges happen during the reload *after*
> practice, not during it. Nothing in this program checks any of that.

## Why this exists

There is one long-standing open-source project in this space,
[ShootOFF](https://github.com/phrack/ShootOFF) (Java, GPL-3.0), whose last
substantive commit was in 2017. This is a smaller tool doing two things it
doesn't:

**Timing comes from the microphone, not the video.** Consumer webcams cap around
30fps regardless of what you request, so a video-derived shot time is quantised
to 33ms before the driver's variable pipeline delay is even counted. Instead,
the start beep is played *and recorded on the same input stream* as the striker
click:

    draw_time = (click_sample − beep_sample) / 48000

Output latency, input latency and USB jitter all cancel, because they apply
equally to both events. That is what a shot timer actually is — an acoustic
instrument.

**It understands scaled targets.** Scale and stand-off distance determine a
simulated distance, so a 0.6″ group on a 1/5-scale target at 9 ft is reported as
a 3.00″ group at 15 yards. The simulated distance is derived, never typed in, so
the two cannot drift apart.

Every shot needs **both** a camera flash and a microphone click within 150ms. A
click with no flash is a slide rack; a flash with no click is a reflection and
is discarded. That is what makes it usable in a room with red light in it.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/aysark/opendryfire && cd opendryfire
uv sync
```

The package is `opendryfire`; the command it installs is `dryfire`, because you
type it a lot.

macOS will prompt for camera and microphone access for your terminal on first
run; both are required.

## Quick start

```bash
uv run dryfire probe                   # what cameras and mics exist
uv run dryfire use-camera iphone       # pick one by name (optional)
uv run dryfire plan                    # what blocks and setups are configured
uv run dryfire calibrate --setup f-15  # click the four A-zone corners
uv run dryfire autocal                 # measure the room, set thresholds
uv run dryfire spike                   # confirm it sees your laser
uv run dryfire audiocal                # separate clicks from room noise
uv run dryfire run F15                 # go
uv run dryfire report                  # what previous sessions said
```

Order matters: calibrate **before** autocal, so thresholds can be measured
against the target plane rather than the whole room.

### Camera placement

Two to four feet from the target, off to one side — **not** downrange, and never
in the muzzle line. Perspective is removed mathematically, so an oblique view
costs nothing, and being close means a small target fills real pixels.

Keep bright red things out of frame. A charger LED or power-strip indicator is
nearly as red as a 635nm laser; `autocal` will find one, tell you where it is,
and refuse to raise thresholds past the point where the laser itself would stop
registering.

## Plans

What you run is a TOML file, not code. `dryfire plan` shows the active one.

```toml
[[setups]]
key = "f-15"
target = "uspsa"           # or "b8"
scale_denominator = 5      # 1/5 scale
stand_distance_ft = 9.0    # → 15 yd equivalent, derived

[[blocks]]
key = "F15"
name = "Distance Refinement"
setup = "f-15"
timed = false
reps = 10
```

Copy the bundled example, edit it, then pass `--plan my-plan.toml` or save it as
`~/.dryfire/plan.toml`. The bundled plan is one person's 26-week pistol
programme, included as a worked example rather than a recommendation.

Blocks can be **timed** (with an optional par ladder that drops the par once you
hit a set number of clean reps) or **untimed**. An untimed block refuses to
accept a par unless you pass `--timer`, because at distance a clock hides
exactly the error you are trying to find.

## Targets

Two built in, both defined in real inches at full scale:

- **USPSA/IPSC Metric** — A-zone 150 × 280 mm (5.906″ × 11.024″), exact
- **NRA B-8** — X/10/9/8/7/6/5 rings at published diameters

Calibration works off the target's **own** A-zone, clicked in a magnified view.
That matters: a homography maps one *plane* to another, so reference points must
be coplanar with the scoring surface. Markers on a board behind a standing
target sit in a different plane and skew every hit — about 0.3″ at typical
geometry, which is a quarter of a 1/5-scale A-zone and invisible in the output.

ArUco markers are supported for genuinely flat setups (paper target and markers
printed on one sheet, taped to a wall) via `dryfire targets --markers`.

## Sound

Each shot plays an outcome chime, so you can watch the front sight instead of a
screen. Rising = scoring zone, flat = on target outside it, falling = off
target, quiet double blip = a flash with no matching click, discarded.
`dryfire sounds` previews them; `--no-sound` silences them.

The chimes are built to be inaudible *to the shot detector* — the microphone is
listening for striker clicks, and a sharp ping is acoustically a click. Tones
stay well below the detector's high-pass corner, use a soft 12ms attack, and
mute the detector while playing.

## Output

Sessions are written as JSON to `~/.dryfire/sessions/`. Markdown export is off
by default; enable it with:

```bash
uv run dryfire vault ~/path/to/notes
```

That writes a note per session — group size at simulated distance, scoring-zone
hit count, and group shape classified as round / low-left / strung vertical /
off-centre, each mapped to a different fix. Adding `--tracker` to a run also
appends one line to a tracker note, inside a marker block it owns; it never
rewrites a line you wrote.

## Hardware

**Cartridge.** Any firing-pin-activated laser cartridge, from any maker. The
only property that matters is **pulse length**: around 100ms gives 3 frames at
30fps, which is plenty. Some cartridges emit a ~1ms flash and a camera will miss
most of those — a hardware limit no software fixes, so check before you buy.

Red 635-650nm is what the detector is tuned for. Green would need the colour
test in `detect.py` retargeted; that is a small change nobody has made yet.

**Camera.** Anything OpenCV can open. On macOS, cameras are selected by *name*
rather than index, because a Continuity Camera occupies an index only while the
iPhone is connected — a saved index silently becomes the built-in webcam when
the phone sleeps. If you use an iPhone, **turn Center Stage off**: it pans and
crops to follow people, which invalidates the calibration. It is disabled per session automatically,
with a warning if it is on.

**Microphone.** Any. `audiocal` measures your room and places the detection
threshold between room noise and real striker clicks. If the two overlap it says
so rather than picking a threshold inside the overlap.

## Known limits

- **Multi-shot strings are not true splits.** A striker-fired pistol is dead
  until the slide is cycled, so a laser-scored string is one shot per rack. Rack
  time is measured and reported separately. That is physics, not a bug.
- **Only the break is visible.** The laser is off while you aim, so a single
  shot cannot tell you how far you dipped. Every camera-based system shares
  this. Reading group *shape* over 10 shots is the way around it.
- **Manual exposure is often unavailable.** macOS/AVFoundation ignores the
  request on built-in and Continuity cameras. Compensated with red-dominance
  testing and a target-plane mask. A UVC webcam with real exposure control, or
  an IP camera with its own settings page, removes the constraint — RTSP
  buffering latency is normally disqualifying but does not matter here, since
  timing comes from audio.
- **macOS-first.** Camera naming, Center Stage handling and device enumeration
  are macOS-specific and degrade to plain device indices elsewhere. CI runs the
  full test suite on Linux, so the portable code — detection, geometry, scoring,
  audio — is known to work there. What has *not* happened on Linux or Windows is
  a real session against real hardware, which is where camera quirks live.

## Development

```bash
uv run pytest src/tests -q     # 149 tests, no hardware required
```

The suite runs on synthetic frames and synthesised audio, and deliberately
models the awkward cases rather than the easy ones: a laser dot whose core is
blown to white with red only in its halo, a 100ms pulse spanning several frames,
a dropped frame mid-pulse, a cream target against warm red-dominant wood, and a
striker click as a broadband burst over room tone.

`docs/DESIGN.md` records why each threshold is what it is, including the
approaches that were tried and failed and the measurements that killed them.
Worth reading before changing a constant.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Two rules carry most of the weight:
measure rather than guess when changing a constant, and fail loudly rather than
plausibly — a wrong number that looks reasonable corrupts a training log for
months without anyone noticing.

Bug reports are much more useful with `dryfire probe`, `autocal` and `audiocal`
output attached; most detection problems are a room, not a bug, and those
commands say which.

## License

Apache-2.0, which requires attribution: the copyright notice, the licence, and
the contents of `NOTICE` travel with the code and with any derivative work.
