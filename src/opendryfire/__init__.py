"""opendryfire — webcam laser hit scoring for dry fire practice.

Hardware: any firing-pin-activated laser training cartridge plus a webcam.
Pulse length is the only property that matters — see ``detect`` for why.

Design principle: the microphone answers *when*, the camera answers *where*.
Shot timing is derived acoustically from a single audio stream that carries both
the app's own start beep and the striker click, so there is no camera pipeline
latency in the timing path. See docs/DESIGN.md.
"""

__version__ = "0.1.0"
