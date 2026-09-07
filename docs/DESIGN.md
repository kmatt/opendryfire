# opendryfire — design notes

Why each part is the way it is, including the measurements behind the constants
and the things that were tried and rejected. Written so a threshold can be
changed later on evidence rather than vibes.

## 1. Timing comes from audio, not video

Measured on this machine (`dryfire probe`):

| Camera | Requested | Measured |
|---|---|---|
| MacBook Pro (idx 0) | 1280x720 @ 60 | **29.7 fps** |
| iPhone 17 Continuity (idx 1) | 1280x720 @ 60 | **30.0 fps** |

Both cap at 30fps whatever is requested. That is 33ms of quantisation, plus a
variable AVFoundation pipeline delay, against a plan whose par ladder moves in
0.1s steps.

So the start beep is played and recorded on the same input stream as the striker
click, and the draw time is the difference between two events on one clock. Both
events carry identical latency, so it cancels exactly. This is what a real shot
timer is — an acoustic instrument.

A consequence worth stating: video timestamps are never used for timing
anywhere, only for pairing within a ±150ms window.

## 2. Detection thresholds are measured, not guessed

The room is warm-lit with cherry wood shelving and a bamboo table — a
red-dominant background, i.e. the worst realistic case for a red laser. Sweeping
a real capture:

| brightness ≥ | red margin ≥ | false blobs |
|---|---|---|
| 230 | 12 | 852 |
| 240 | 25 | 411 |
| 248 | 40 | 50 |
| **248** | **60** | **0** |

The separating insight: incandescent light is broadband, so warm surfaces top
out around 48 "redness" (mean R minus max of mean G/B). A 635nm laser is
monochromatic and lands far above that. Brightness alone does **not** separate
them — the bamboo table has specular highlights reaching R=238.

Defaults are therefore 248 / 55. `dryfire autocal` re-derives both against the
actual room, because these are specific to one camera and one set of lamps.

### Red dominance is tested on the halo, not the peak

A <5mW dot at close range saturates its own core to white, so the brightest
pixel is often (255,255,255) with no red dominance at all. ShootOFF's published
algorithm tests colour at the peak pixel, which fails on a close bright dot. The
component is dilated by 5px first and dominance measured over that ring.

## 3. Homography reference points must be coplanar

The one place a plausible-looking setup silently produces wrong numbers.

A homography maps one *plane* to another. With the camera ~1m away at 20° off
axis, reference markers sitting 20mm behind the target face displace a hit by
about 7mm — 0.28", roughly a quarter of a 1/5-scale A-zone. Invisible in the
output; corrupts a phase of training.

So the default reference is the target's **own engraved A-zone**, which cannot
be out of plane with itself. ArUco is supported but only for genuinely flat
setups (paper target and markers on one sheet, taped to a wall), and
`dryfire targets --markers` prints that warning next to the file it writes.

### Auto-detecting the silhouette was tried and rejected

Segmenting the cream PLA target by brightness and saturation looked obvious.
Measured:

| Surface | Saturation | Value |
|---|---|---|
| PLA target | 77 | 147 |
| Bamboo table | **74** | 99 |
| Cherry wood | 160 | 61 |

The table is within 3 points of the target's saturation, so the segmentation
bleeds into it and the "target" contour swallows the desk. Dropped in favour of
four explicit clicks with a magnifier.

## 4. Sensor fusion is what makes it trustworthy

| Camera | Mic | Verdict |
|---|---|---|
| flash | click | scored shot |
| — | click | slide rack, or a miss off the target |
| flash | — | **rejected** — not a shot |

A stray reflection has to arrive within 150ms of a striker fall to be believed,
and ambient light does not coordinate itself with a trigger finger. This is why
the false-positive problem is tractable in a red-lit room at all, and why
thresholds only need to be good rather than perfect.

`--no-audio-gate` disables it for rooms too noisy for the click detector, at the
cost of that protection.

## 5. Click detection uses an energy *ratio*, not a difference

First implementation used spectral flux against a median/MAD threshold. It fired
constantly on room noise, because a raw energy difference has no sense of scale.

Replaced with: high-pass at 1.8kHz, then require this block's RMS to exceed the
running median by 6×. A striker fall is a 10-100× spike in HF energy; room noise
wanders by well under 2×. Being a ratio, it is scale-invariant — the same
setting works in a quiet room and a noisy one with no per-room tuning.

Clicks are excluded from their own baseline history, or a burst of fast reps
would progressively deafen the detector.

## 6. Failing loud beats failing plausibly

A silently mis-scaled target corrupts every number derived from it, for as long
as nobody notices. The same failure mode exists throughout this code, so several
places refuse to guess:

- `find_beep_onset` returns `None` when no beep is audible, and the rep is
  discarded. An earlier version took the loudest window unconditionally and duly
  "found" a beep in pure noise — a test now pins this.
- `build_homography` reprojects the points it was just built from and rejects
  the fit if they do not land. `cv2.findHomography` does *not* reliably return
  `None` for collinear input; it returns a matrix that maps none of them
  correctly.
- Calibration scale is checked against the block's target scale before a session
  starts, so a 1/3 calibration cannot be used to score a 1/5 block.
- Camera property writes are read back and reported as `IGNORED` when the driver
  discards them, rather than assumed to have worked.

## 7. Things that are limits, not bugs

**Multi-shot strings cannot produce a true split.** A striker-fired pistol is dead until
the slide is cycled. The rack is timed and subtracted, which is repeatable and
useful, but it is not a live-fire split and blocks that use it should say so in
their own warning text.

**Only the break is observable.** The laser is off during aiming, so a single
shot cannot report pre-break wobble. Every camera-based system shares this. The
plan already routes around it by reading group shape over 10 shots. A true
pre-break trace needs a SIRT-class pistol with a continuous take-up laser —
different hardware.

**Manual exposure is unavailable.** `CAP_PROP_EXPOSURE` writes return -1 on both
cameras; AVFoundation via OpenCV does not expose it. Compensated with red
dominance and the target-plane mask. A UVC webcam with manual exposure, or the
Reolink's own web UI, would remove this constraint — the Reolink's RTSP
buffering latency would normally be disqualifying, but since timing comes from
audio it does not matter here.

## 8. Test strategy

154 tests, no hardware. Synthetic frames model the awkward physics rather than
the easy case:

- a dot whose core is blown to white with red only in the halo
- a 100ms pulse spanning 3 frames at 30fps and 6 at 60fps
- a dropped frame mid-pulse
- a cream PLA target rendered against warm cherry wood and a bamboo table
  (a red-dominant background, i.e. the hard case for a red laser)

and synthesised audio models a striker click as a broadband burst with a 1.5ms
exponential decay over pink-ish room tone.

The tests that matter most are the round-trips in `test_integration.py`: a
rendered dot at a known target position, scored back into inches. They are the
only thing that catches the class of bug where every unit passes and the system
still reports a low-left group as high-right.

## 9. Cameras are addressed by name, not index

macOS assigns capture indices by enumeration order, and a Continuity Camera
enters and leaves that list as the iPhone connects, sleeps, or is picked up. A
saved `--camera 1` therefore silently becomes the built-in webcam — and since
calibration is stored per *profile*, not per camera, the wrong lens gets scored
through the right homography. Every number stays plausible and is wrong.

`AVCaptureDevice.devicesWithMediaType:` returns the same list, in the same
order, that OpenCV's AVFoundation backend indexes into. Verified on this
machine: both report exactly `[MacBook Pro Camera, iPhone Camera]`. Note that
AVFoundation's *discovery session* API additionally returns Desk View cameras,
which OpenCV does not enumerate — using that list would shift every index by
one, so the legacy API is the correct one to match against.

Fallbacks: pyobjc → `system_profiler SPCameraDataType` → bare indices.

### Center Stage is a correctness bug, not a cosmetic one

Continuity Camera's Center Stage crops and pans the frame to follow people. Under
a saved homography that silently changes the pixels-to-inches mapping mid
session. It can be read and disabled programmatically:

```python
AVCaptureDevice.setCenterStageControlMode_(1)  # 1 = App
AVCaptureDevice.setCenterStageEnabled_(False)
```

Control mode `App` scopes this to the process, so video calls keep their
setting. `preflight()` does this for Continuity devices and warns if Portrait
or Studio Light are also active. Control Center can still override it, which is
why the runtime drift detector remains the backstop.

### Frame rate is not negotiable through OpenCV

The iPhone's formats advertise 60fps at 1920x1080 and 1920x1440, but OpenCV's
AVFoundation backend delivers 30fps whatever is requested — measured across
five resolution/rate combinations. Since timing is acoustic this costs nothing;
30fps still yields 3 frames per 100ms pulse. Capturing through AVFoundation
directly would recover 60fps but is a capture-layer rewrite for no gain on the
headline metric.

Continuity also renegotiates resolution between sessions — the same request
returned 1920x1440 once and 1920x1080 later. `CameraInfo` always reports what
was actually granted rather than what was asked for.

## 10. Threshold calibration must be capped

Second room, second lighting setup, and a new failure mode: a wireless charger's
red LED sits at redness 158, against warm ambient surfaces at ~48. Uncapped,
`calibrate_scene` chose a margin of **201** to clear it.

That number is worse than useless. A margin that high would reject the laser
along with the LED — trading a visible false-positive problem for invisible
blindness, where a missed hit is indistinguishable from not having fired.

So the margin is capped, and when the cap binds the report says so, gives the
offender's bounding box, and recommends fixes in order: remove the red light
from view, or calibrate the target plane first so the mask excludes everything
off-target.

The figures in the paragraph above (LED at 158, ambient at ~48, dot at 120-200)
are all *per-pixel* redness, and the original cap of 120 was set in those units.
That was wrong, and §10a is what it cost.

This also reordered the documented workflow. It used to be
spike → autocal → calibrate; it is now **calibrate → autocal → spike**, because
autocal is much better informed once it can restrict itself to the target plane.

Testing dominance over a *dilated halo* rather than the peak pixel does most of
the work by itself: small point sources like an LED are rejected anyway, because
dilating a 6x6 blob by 5px pulls in enough dark surroundings to drag its mean
redness below threshold. That is the primary mechanism, not a side effect — see
§10a, where treating it as incidental is exactly what went wrong.

## 10a. The margin must be measured in the units it is tested in

Reported from a real session: `autocal` ran, then `spike` detected nothing at
all, while the calibration report said *"Clean: nothing in this room reads as a
laser."*

`calibrate_scene` derived the margin from the maximum **per-pixel** redness in
the scene. `find_blobs` tests the **mean redness of a blob's dilated halo**.
Those are different quantities, and the halo mean is far lower, because dilating
a small bright thing pulls in the darker background around it. Measured on the
frames from that room:

| | per-pixel max | halo mean |
|---|---|---|
| bamboo target stand | 78 | 42 |

So a margin chosen on the left-hand column was enforced against the right-hand
one. Autocal picked **98.8** from an object whose halo mean was 42 — and the
margin the detector then enforced was more than twice the number the offending
object actually presents to it.

Two things made it worse:

- The candidate screen ran at `r >= 200` while the chosen `bright_threshold` was
  250. The redness-78 pixels on the stand sit at r = 200-248 and the detector
  never examines them, so the margin was raised to clear an object the
  brightness threshold had already excluded. Among pixels actually at r >= 250,
  the worst redness was 52.
- The cap did not bind, because 98.8 < 120 — and 120 was itself a per-pixel
  number, so it could not have bound in the right place anyway.

**The margin now comes out of `find_blobs` itself.** Calibration builds a probe
detector at the chosen threshold with the margin disabled, and takes the worst
`Blob.redness` it reports. One implementation, one definition of "redness"; the
two cannot drift apart again.

The cap is restated in halo-mean units, `MAX_USABLE_MARGIN = 55`:

| | halo-mean redness |
|---|---|
| synthetic reference dot | 66 |
| small red LED on a charger | 39 |
| bright warm bamboo surface | 0 (excluded by `max_area`) |
| **real 635nm dot** | **not yet measured — see §15** |

55 is therefore chosen as "never require more than the shipped default", which
§2 measured at 248/55 with zero false blobs and working detection. That is a
conservative bound, not a measurement of the laser. The number that would
justify it directly — the halo-mean redness of a real dot — has never been
captured, because the diagnostic built to capture it was saving the wrong
frame. Tighten this constant once §15's crops exist.

### Counting false blobs cannot show a calibration is good

The deeper problem was the self-check. It counted false positives after
choosing thresholds, and a detector that sees *nothing* has none — so the
strictest possible config scored perfectly and printed "Clean".

`sees_reference_dot` now stamps a modelled dot into a real captured frame and
puts it through the real `find_blobs`. If the config cannot see it, the report
leads with `BLIND` and says not to shoot against that calibration. The dot is
added to the frame rather than drawn over it, so a dot stamped on an
over-exposed background comes out correspondingly harder to detect — which is
the honest answer, because a room that bright really does hide the laser.

## 11. Hit chimes must be inaudible to the shot detector

Audible feedback lets you keep your eyes on the front sight rather than a
screen. But the microphone is listening for striker clicks, and **a sharp
ping is acoustically a click** — a broadband transient. Played naively, a hit
chime scores a phantom shot, which plays another chime.

Two independent defences:

1. **Spectral.** Chime fundamentals stay well below the detector's 1.8kHz
   high-pass. This was set by measurement, not theory — a 4th-order Butterworth
   is not a brick wall. Measured peak block HF energy as a ratio against a quiet
   room floor, where detection fires at 6:

   | tone | ratio | outcome |
   |---|---|---|
   | 880 Hz | 7.1 | **detected** — too close to the corner |
   | 587 Hz | 1.5 | safe |
   | 440 Hz | 0.5 | safe |

   The ceiling is 700Hz. The first attempt used 1318Hz and failed.

2. **Envelope.** The attack, not the tone, creates the broadband energy. Every
   chime uses a 12ms raised-cosine attack. A test asserts the *rectangular*
   version of the same tone IS detected, so the ramp is demonstrably doing work.

3. **Muting.** The detector is muted across the chime plus a tail, before
   playback starts. In timed mode there is no live listener, so instead the next
   rep waits for the chime to finish — otherwise the tail lands inside the next
   recording and that draw time comes out near zero.

## 12. Click detection needed the same treatment as the camera

Testing the chimes acoustically produced a result that had nothing to do with
chimes: **5 false clicks in 3 seconds of silence**. The chime figures (3, then 1
with muting) were *below* the ambient rate. The detector itself was the problem.

Measuring the actual room:

| statistic | high-passed RMS |
|---|---|
| median | 2.0e-5 |
| p99 | 3.3e-4 |
| worst transient | 3.0e-3 |

A 150x spread between median and worst. That breaks the assumption behind the
ratio test: `max(ratio * median, floor)` resolves to `max(1.2e-4, floor)`, so
the **floor decides**, and raising the ratio from 6 to 50 changed the false
count from 6 to 4. The synthetic tests missed this because they modelled room
tone as clean pink noise, which has almost nothing above 1.8kHz.

Measured against the real recording:

| floor | false clicks / 10s |
|---|---|
| 2e-4 (old default) | 5 |
| 4.6e-3 (room max x1.5) | **0** |

So `dryfire audiocal` does a two-class calibration — record the room, record
actual dry fire, place the floor at the geometric midpoint — exactly mirroring
what `autocal` does for the camera. When the two classes overlap it reports the
overlap instead of picking a threshold inside it, and biases the floor *above*
the room, because a missed press is visible to the shooter while a phantom one
silently corrupts a draw time.

The default floor moved from 2e-4 to 5e-3 for the same reason: still 10-100x
below a real striker fall, but above the ordinary transients of a quiet room.

Worth noting that sensor fusion already contained the blast radius — a click
with no flash is treated as a slide rack, not a shot — so phantom clicks were
never going to invent hits in Block F. They would have corrupted *timing* in
Block B, where the first click after the beep is the draw time.

## 13. Distance must not decide where a shot ends

Reported from a real session: one press scored as two hits. Cause was in shot
assembly — a blob further than `link_radius_px` (40px) from the previous frame's
blob was treated as a *new* shot, closing the open one and opening another in
the same call. The refractory window could not stop it, because that check had
already passed earlier in the same frame.

The model was simply wrong about the physics. A laser cartridge emits exactly
one pulse per trigger press, and a striker-fired pistol cannot fire again until
the slide is cycled. Two shots are therefore *seconds* apart, never 33ms. A
muzzle that dips or swings during the ~100ms the laser is lit moves the dot a
long way, and that is one shot with drift, not two shots.

Shot boundaries are now decided by **time**: any blob within `max_pulse_s`
(350ms — generous against a ~100ms pulse, far below a slide rack) extends the
open shot however far it moved. `link_radius_px` survives only to pick *which*
blob to follow when several are visible. The movement is still recorded, as
`Shot.drift`, because it is real information about follow-through — it just is
not a second hit.

## 14. Sound must not imply knowledge the program does not have

The same session: shots landing on the wall, off the target entirely, played the
rising "hit" chime. That was by design and the design was wrong.

`spike` runs *before* calibration, so there is no homography and nothing knows a
hit from a miss — but it was playing the A-zone chime for every detection. A
rewarding sound for a shot on the wall is worse than no sound, because it
asserts something false.

There is now a neutral `seen` tone for that case, and `spike` prints a line
saying the tone means "laser seen", not "hit". During a real `run` the geometry
exists, so hits and misses already sounded different; only the pre-calibration
path was misleading.

Incidentally the report confirmed something useful: the laser detects cleanly
against a dark maroon wall, not just the cream PLA target. The wall measures
R=77 with redness 48 — far below the 248 brightness threshold, so it cannot
trigger detection itself while still showing the dot.

## 15. A diagnostic that cannot show its evidence is worse than none

`dryfire spike` prints "Saving a crop of every detection so false positives are
visible" and then saves, for every detection, a picture with no laser in it.

`LaserDetector.push` returns a Shot when the pulse *ends* — `gap_frames + 1`
frames after the dot was last seen, so at 30fps the caller is holding a frame
from ~165ms after the break. `cmd_spike` cropped that frame. Every crop the tool
has ever written shows the target at the right coordinates with the laser
already dark.

The cost was not just a useless file. The crops were the only physical record of
what detection was doing, so they got read as though they showed dots — and the
bright target surface inside them was measured and reported as laser redness.
A diagnostic that produces confident-looking wrong evidence is worse than one
that produces none, because the wrong evidence gets used.

`LaserDetector.open_shot` now exposes the pulse in progress, and `spike` takes
its crop on the frame where the shot *opens*. The position is identical — the
hit point is fixed at the break and does not move as the pulse runs — so only
the pixels change, from "target, no laser" to "target, laser lit".

This is the same shape of failure as §10a's self-check: both were verification
steps that could not fail, and both were trusted because they existed.

## 16. The camera's report of its own format is not evidence

Symptom: `dryfire spike` framed the target differently from `dryfire calibrate`
— the top of the silhouette was cut — while both commands request the same
1920x1440 and both printed it.

Both were telling the truth about what they *asked* for. Measured, with the
iPhone Continuity Camera on OpenCV 5.0 / AVFoundation:

| request | `cap.get` reports | frames actually delivered |
|---|---|---|
| 1920x1440 | 1920x1440 | 1920x1440 (usual) |
| 1920x1440 | 1920x1440 | **1920x1080** (3 opens in ~25) |
| none | 1920x1080 | 1920x1080 (the device default) |

The property read-back echoes the request whether or not the session took it,
so `CameraInfo` was reporting a number that had never been observed. Two
candidate causes were tested and are **not** it: Center Stage (toggled on,
captured, toggled off, captured — 1920x1440 both times) and the order of the
`set`/`get` calls (12 further opens, 6 interleaved and 6 batched, all clean).
The trigger is still unknown; it appears to be a race in format negotiation.

The two formats are not the same view at two sizes. Frames captured back to
back in each and aligned by normalized cross-correlation (0.998) give

    x_16:9 = 0.90 · x_4:3 + 96        y_16:9 = 0.90 · y_4:3 − 108

— the 16:9 stream is the 4:3 one scaled 0.9, cutting ~120 rows off the top and
the bottom and showing ~107px more on each side. That is the cut-off head.

Through the calibration actually saved on this machine, scoring 16:9 pixels as
if they were 4:3 moves the A-zone centre by 0.90", the top-left corner by
0.76" and the bottom-right by 1.10" — on a printed A-zone 1.18" wide by 2.20"
tall, or 3.8–5.5" at the 15yd equivalent. It reads as an ordinary high group.
Nothing in the run would have looked wrong.

So: size is now measured from a delivered frame (`Camera._configure`), the
property is recorded only as a claim and reported as `LIED` when the two
disagree, and `frame_size_problem` compares the live stream against the
`image_size` the calibration recorded — fatal in `run` and in a masked
`autocal`, a warning in `spike`, which scores nothing.

The re-apply loop in `_configure` is opportunistic. The failure has not been
reproduced on demand, so nothing depends on the retry working; what the
correctness rests on is measuring the result and refusing to score through the
wrong one. This is §6 again: the read-back existed, looked like verification,
and could not fail.

## 16a. Refusing to score through the wrong format is not enough

Reported as a calibration that would not take: `run` refused with

```
this calibration was built on 1920x1080 frames but the camera is delivering
1920x1440 ...
```

on a profile that had been calibrated the same evening. The shooter's reading
was that the calibration was stale. It was the opposite — it was the *newest*
file on disk, and it was wrong.

`~/.dryfire/calibration/b-draw.json` had been written twice: once at 21:59
recording `image_size [1920, 1440]`, and again at 23:05 recording
`[1920, 1080]`. The 23:05 run is one of the ~1-in-8 opens from §16 that lands
in the device's default format. `_format_warnings` did say so —

```
  ! asked for 1920x1440; this device is sending 1920x1080
```

— and then `cmd_calibrate` opened the click window and saved anyway, over the
good file.

§16 made the wrong format impossible to *score* through. It left it possible to
*record*. Those are not the same hole, and the second one is worse in a
specific way: the artefact it produces is indistinguishable from a good one
after the fact. The corners really were clicked in that view, so the
reprojection error is perfect; the `image_size` field is honest about a format
nobody chose; and every later command sees a valid calibration disagreeing with
a live stream and reports the *stream* as the problem. The one moment the
evidence exists is while the camera is still open, and that is the moment the
old code spent on a warning line among several.

`calibration_format_problem` refuses there instead, *before* the corners are
clicked — placing four points is a minute of the shooter's time and all of it
is void. Asking for the delivered size explicitly (`--width 1920 --height
1080`) is allowed and is the escape hatch, on the same principle as a bare
`--camera 1` in §18: a format someone named is a choice, a format that arrived
on its own is an accident.

The general shape, and it is §6 and §18 again: a check that fires everywhere
the bad value is *read* still leaves the bad value being *written*, and a
persisted artefact outlives the context that would have explained it.

## 17. `max_area` was a guess wearing a measurement's clothes

Symptom: hits drawn several inches from where the laser visibly landed, a 2.6"
x 2.7" group on a printed target 3.62" wide, six of ten shots lasting a single
frame, and four showing 128–153px of "drift" inside one 33ms pulse.

The overlay draws each shot at `tf.to_image(x, y)` — the exact pixel detection
returned — so the geometry could not be adding the offset. Detection was.

Measured from the spike crops of that session: the laser's above-threshold
footprint was **at least 9,556 px**, and every one of those regions ran past the
edge of its 120x120 crop, so the true figure is larger. `max_area` was 4000.
`find_blobs` dropped the region for being too big, silently, and then scored
whatever fragment of its edge survived — the engraved A-zone borders and the
zone letters are dark enough to cut a splash into pieces, and a piece is small
enough to pass. Reproduced on the real scene with the real code:

| glow radius | region over threshold | reported |
|---|---|---|
| 45px | 2,987 | correct, 0.03" off |
| 85px on the A-zone | 14,577 | main region rejected, 193px scrap |
| 85px over the 'C' | 9,677 | 121px fragment **62px away** → 0.30" off |
| 85px on the A-zone edge | 11,810 | nothing at all |

Two constants were fighting. `autocal` lowers `bright_threshold` to catch the
dot — 248 → 200 in this room — and a lower threshold makes the above-threshold
region *larger*, which pushes it past the area ceiling. Measured on those same
crops: at 200 the median footprint is 3,946px, at the cap; at 248 it is 21px.
Nothing checked the two against each other.

And the check that should have caught it could not. `sees_reference_dot` stamps
a dot of `REFERENCE_DOT_HALO_PX = 9` — about 254px — and asks whether the config
finds it. It always found it. The dot being thrown away was forty times bigger.
That is the third time this shape of bug has appeared here (§10a, §15): a
verification that cannot fail, trusted because it exists.

The fix is to stop guessing the size:

- `find_blobs` applies the size ceiling *after* the redness test, and records
  what it rejected in `oversized`. A red-dominant region too big to score is
  the laser over-exposed, not a lamp, and `spike` and `run` now say so.
- `spike` measures the dot — footprint, peak, redness, per flash — and writes
  `~/.dryfire/dot.json`. It is the measurement nobody had ever taken.
- `calibrate_scene` sizes `max_area` from that measurement and from the biggest
  red region in the room, placing the ceiling between them, and says which.
  With no measurement it keeps the default and reports it as an assumption.
- `sees_reference_dot` takes the measured radius, so the probe is the dot this
  room actually produces and the check can fail.

Redness moved to a windowed computation (`_halo_redness`) so every candidate can
be measured before the size gate without paying for a full-frame dilation per
component. The value is identical — pinned by a test, because §10a was exactly
what happens when two implementations of "redness" disagree.

Distance matters to all of this, which is worth stating plainly since it is not
obvious: the homography does not care where the shooter stands, but the dot's
brightness follows the inverse square, so shooting closer makes the splash
bigger and pushes it further past whatever ceiling is in force.

## 18. The platform's camera list is not OpenCV's camera list

`devices.py` was built on one claim: that `devicesWithMediaType:` returns video
devices "in the same list, in the same order, that OpenCV's AVFoundation backend
indexes into — verified against both on this machine". It was verified, and it
was true that day. It is not a property of either library.

Measured about an hour apart, same machine, same two cameras:

```
09:xx  AVFoundation: [0] MacBook Pro Camera   [1] iPhone Camera
       OpenCV:       [0] MacBook Pro Camera   [1] iPhone Camera     agree

23:xx  AVFoundation: [0] iPhone Camera        [1] MacBook Pro Camera
       OpenCV:       [0] MacBook Pro Camera   [1] iPhone Camera     disagree
```

`devicesWithMediaType:` follows connection order, so a Continuity Camera that
reconnects moves to the front of it. OpenCV kept the built-in camera at index 0
in both. Identified by format fingerprint, not by trust: only the MacBook offers
1552x1552 and 1760x1328, only the iPhone offers 1920x1440, so opening an index
and asking what it grants says which device it is.

The consequence was the worst available: `dryfire calibrate --camera iphone`
resolved "iPhone Camera" to index 0, opened the **built-in webcam**, and printed
`camera: iPhone Camera (index 0)` — the log confirming the wrong thing. This is
precisely the failure the module's own docstring says name-addressing exists to
prevent, and it survived because the mapping was never checked against anything.

Two changes, because one of them is still an inference:

- `opencv_order` re-indexes the list the way OpenCV enumerates: built-in
  cameras first, then the rest in platform order. Consistent with both
  observations above, and with a backend that discovers by device type.
- `Camera._verify_identity` does not trust that. After opening, it compares the
  size actually being delivered against the formats the named device
  advertises; if some *other* device is the unique one offering that size, the
  mapping is disproved and the open fails with the index that would have
  worked. A size several cameras offer proves nothing and is not treated as
  evidence. A bare `--camera 1` skips the check: it is the escape hatch the
  error message points at, so it cannot be allowed to refuse.

The general lesson is the one from §6, applied to somebody else's ordering
guarantee: an assumption that can be checked cheaply against physical evidence
should be, because the day it stops holding it will not announce itself.

## 19. The camera has to be watched while the rep happens

Reported as "it doesn't work": every rep of `dryfire run B` printed

```
  ready...   click heard but no laser seen — off-target, or the cartridge did not fire
```

on shots that were on target. The microphone heard the striker every time, so
audio, pairing, and the cartridge were all fine. What was missing was the
camera, and the reason was structural rather than a threshold.

`_run_timed` was shaped like this:

```python
while time.perf_counter() < stop:      # 1.2s of reads before the beep
    ...
rep_result = timer.run_rep(listen_seconds=args.listen)   # BLOCKS ~6s
t_cam_end = time.perf_counter() + 0.35                   # 0.35s of reads after
```

`run_rep` opens the full-duplex stream, plays the beep and sleeps until the
recording is full. It is the whole rep. Nothing calls `cam.read()` for its
entire duration — so the camera was read for 1.2s *before* the beep and 0.35s
*after* the listen window closed, and never during the 6.3s in between. The
draw lands at beep + 1.5-2.5s, squarely inside the blind window. The only shot
this arrangement could ever see was one fired before the buzzer or six seconds
late.

The tempting reply is that the frames must still be queued somewhere. They are
not. Measured on the reference Continuity Camera, opening the stream, idling
6.0s with no reads, then draining as fast as possible for 0.35s:

```
frames recovered in the 0.35s drain: 11
per-read ms: 34 33 32 33 36 30 37 34 32 32 35
```

Eleven frames, each taking a full frame period to arrive. AVFoundation
discards late frames rather than queueing them, so the ~180 frames that elapsed
during the gap are gone, and the drain is simply a live read of the room *after*
the laser went out. No drain length could have recovered them; and even a FIFO
deep enough to hold six seconds could not have been consumed in 0.35s.

The fix is to run the audio rep on its own thread and keep reading the camera
on the main one, which is what `_run_untimed` already did with `LiveListener`.
Timing is untouched: it still comes from the single full-duplex stream inside
`run_rep`, and the camera still only ever answers *where*. Measured on the same
hardware, with a real `ShotTimer` rep running:

```
frames read during the 2.0s rep: 71  (29.8/s)
beep heard: True
```

The camera's full native rate, for the whole rep, with the beep still detected —
so hammering `cap.read()` on the main thread does not starve the CoreAudio
callback of the GIL. Detection costs 4.1ms per 1920x1440 frame including the
scoring mask, a 245fps ceiling, so the reads are not the limit either.

Two smaller things follow:

- The message was a lie of omission, and an expensive one: it named two causes
  ("off-target, or the cartridge did not fire") that are both the shooter's
  fault, when the actual cause was that nothing was looking. It now reports how
  many frames were examined and at what rate, so a blind loop can never again
  present itself as a miss.
- This shipped because nothing tested the run loop at all. `test_integration`
  covers pixels->inches->score, and every stage passed while the system as a
  whole could not see. `test_run_loop.py` drives `_run_timed` with a fake
  camera and a fake timer and asserts the camera is read *during* the listen
  window — the one fact no unit test was in a position to notice.

## 20. A discarded rep is a missing measurement, not a failed one

Reported twice in one evening as "did I advance?" — with the answer both times
being no, on blocks that looked like they should have:

```
B | 10 reps | best 0.89s, avg 2.17s | 8/10 clean
  9/9 in the A-zone (100%).
```

`clean_to_advance = 8` of `of_reps = 10`, and the summary says 8 of 10. It
still did not drop. Replaying the ladder against the saved session shows why:

```
  4    -    -     False  False   3/3     <- discarded, no beep
...
ladder saw 9 reps (window needs 10); final par 2.50s; drops: none
```

`ParLadder.record` returns False unless the window is full, and only *valid*
reps are recorded — a rep with no beep has no start time, so there is nothing
to judge and it is correctly not judged. But the block ran `for i in
range(args.reps)`, so a discarded rep consumed one of the ten. The window
could then never fill, and **a single discard made the block unable to advance
at all, whatever was shot in it.**

The rate is not negligible. Across every session on disk:

```
total reps 103, discarded for no beep 10  =  10%
```

all ten of them in timed blocks — 10 of 60, **17%**. (The untimed F-blocks
never play a beep, so they cannot contribute; the comparison is void, not
reassuring.) At 17%, a ten-rep block reaches the end with a full window about
16% of the time. The block was, in practice, almost never winnable.

The conflation is the bug. "Not clean" and "not measured" are different facts
and the loop treated them as one. So the block now runs until it has the valid
reps it asked for, re-running a discard rather than spending the slot on it.

Two bounds, because "run until valid" must not become unbounded on a dead
microphone: `MAX_ATTEMPT_FACTOR = 2` caps total attempts, and
`MAX_CONSECUTIVE_DISCARDS = 3` stops early. Both are deliberately loose — at
17%, ten valid reps need about twelve attempts and three straight discards is a
0.5% event, so either bound firing means something is broken rather than
unlucky, and both say so. A block that ends short announces itself, because the
consequence is otherwise *silence*: the ladder declines to evaluate a window it
never got, and an abstention is indistinguishable from shooting that was not
good enough. That is the whole failure being fixed, so it cannot be allowed to
recur one level up.

`Session.summary` was part of the same conflation. `8/10 clean` counted the
discard as a dirty rep, reading as "you missed the bar" when the rep was never
measured — and disagreeing with the ladder, which had judged nine. It now
reports over valid reps and names discards separately:

```
B | 10 reps | best 1.31s, avg 1.31s | 10/10 clean (2 discarded)
```

`ParLadder` itself is unchanged. Refusing to evaluate a short window is right;
it just needed the block to hand it a full one.

What this does **not** fix is the discard rate. Stream priming was eliminated
as the cause — input onset measured 42.7ms on all 20 opens, dead stable, so
the beep is always well inside the recording. The recorded beep level, though,
swings by a factor of 37 between runs (0.060, 0.011, 0.0016 RMS) and once
collapsed 37x mid-run at rep 8 and stayed down. `find_beep_onset` survives
because its prominence test is a ratio rather than a level, but it is running
with nothing like the margin that was there when its thresholds were set. That
is a lead, not a root cause, and 32 instrumented reps failed to reproduce an
actual miss.
