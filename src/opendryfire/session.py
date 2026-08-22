"""Shot records, sensor fusion, and session persistence.

The fusion rule is the heart of it: the camera says *where*, the microphone says
*when*, and each validates the other.

    click + flash within the window  -> a scored shot
    click, no flash                  -> a slide rack, or a hit off the target
    flash, no click                  -> not a shot; reject it

That last case is what makes the system trustworthy in a red-lit room. A stray
reflection has to arrive within ~150ms of a striker fall to be believed, and
ambient light does not coordinate itself with your trigger finger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .targets import GroupStats, ScaledTarget, group_stats

# A laser cartridge fires the instant the striker falls, so the flash and the
# click are simultaneous in the world. This window only has to absorb the
# camera's pipeline delay, which is tens of milliseconds.
PAIR_WINDOW_S = 0.15


@dataclass
class ShotRecord:
    """One scored shot, in printed-target inches."""

    index: int
    x: float
    y: float
    zone: str
    points: int
    t: float  # seconds since the drill's reference (beep, or first shot)
    split: float | None = None
    drift_px: float = 0.0
    frames: int = 0
    paired: bool = True  # whether a click confirmed it

    @property
    def sim_offset(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5


@dataclass
class Rep:
    """One repetition: a beep, a draw, and the shots that followed."""

    index: int
    shots: list[ShotRecord] = field(default_factory=list)
    par: float | None = None
    first_shot_time: float | None = None
    valid: bool = True
    note: str = ""

    @property
    def clean(self) -> bool:
        """The plan's gate: on time *and* in the primary zone.

        'The sight picture is the gate, not the beep' — so a fast rep that
        misses the A-zone is not clean, and neither is an accurate one that
        blew the par.
        """
        if not self.valid or not self.shots:
            return False
        on_time = (
            self.par is None
            or (self.first_shot_time is not None and self.first_shot_time <= self.par)
        )
        return on_time and self.shots[0].points > 0 and self.shots[0].zone in ("A", "X", "10", "9")


@dataclass
class Session:
    """A dry fire session: one block, one target setup, many reps."""

    block: str
    started: str
    setup_key: str
    target_name: str
    scale: float
    stand_distance_ft: float
    sim_distance_yd: float
    reps: list[Rep] = field(default_factory=list)
    par_history: list[float] = field(default_factory=list)
    notes: str = ""
    finished: str = ""

    @classmethod
    def begin(cls, block: str, setup_key: str, target: ScaledTarget) -> "Session":
        return cls(
            block=block,
            started=datetime.now().isoformat(timespec="seconds"),
            setup_key=setup_key,
            target_name=target.spec.name,
            scale=target.scale,
            stand_distance_ft=target.stand_distance_ft,
            sim_distance_yd=target.sim_distance_yd,
        )

    # -- derived numbers ------------------------------------------------------

    @property
    def all_shots(self) -> list[ShotRecord]:
        return [s for r in self.reps for s in r.shots]

    @property
    def clean_count(self) -> int:
        return sum(1 for r in self.reps if r.clean)

    @property
    def valid_reps(self) -> list[Rep]:
        return [r for r in self.reps if r.valid]

    def times(self) -> list[float]:
        return [
            r.first_shot_time
            for r in self.reps
            if r.valid and r.first_shot_time is not None
        ]

    def best_time(self) -> float | None:
        t = self.times()
        return min(t) if t else None

    def stats(self, target: ScaledTarget) -> GroupStats:
        return group_stats([(s.x, s.y) for s in self.all_shots], target)

    def summary(self) -> str:
        t = self.times()
        parts = [f"{self.block} | {len(self.reps)} reps"]
        if t:
            parts.append(f"best {min(t):.2f}s, avg {sum(t) / len(t):.2f}s")
        parts.append(f"{self.clean_count}/{len(self.reps)} clean")
        return " | ".join(parts)

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.finished = self.finished or datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "Session":
        d = json.loads(path.read_text())
        reps = [
            Rep(
                index=r["index"],
                shots=[ShotRecord(**s) for s in r.get("shots", [])],
                par=r.get("par"),
                first_shot_time=r.get("first_shot_time"),
                valid=r.get("valid", True),
                note=r.get("note", ""),
            )
            for r in d.pop("reps", [])
        ]
        return cls(reps=reps, **d)


def pair_events(
    flashes: list[tuple[float, float, float, float, int]],
    clicks: list[float],
    window: float = PAIR_WINDOW_S,
) -> list[tuple[tuple[float, float, float, float, int] | None, float | None]]:
    """Match camera flashes to microphone clicks on a shared wall clock.

    Each flash is ``(t, x, y, drift, frames)``; each click is a time. Returns
    (flash, click) pairs, with None on either side for unmatched events, so the
    caller can distinguish a real shot from a slide rack or a stray reflection.

    Greedy nearest-match in time order is sufficient here: real shots are
    separated by a slide rack, i.e. seconds apart, so there is never a crowded
    field of ambiguous candidates.
    """
    pairs: list[tuple[tuple[float, float, float, float, int] | None, float | None]] = []
    remaining = sorted(clicks)
    for flash in sorted(flashes, key=lambda f: f[0]):
        best_i, best_dt = None, window
        for i, c in enumerate(remaining):
            dt = abs(c - flash[0])
            if dt <= best_dt:
                best_i, best_dt = i, dt
        if best_i is None:
            pairs.append((flash, None))
        else:
            pairs.append((flash, remaining.pop(best_i)))
    for c in remaining:
        pairs.append((None, c))
    return pairs


def sessions_dir(root: Path | None = None) -> Path:
    base = root or Path.home() / ".dryfire"
    return base / "sessions"


def config_path(root: Path | None = None) -> Path:
    return (root or Path.home() / ".dryfire") / "config.json"


def calibration_path(root: Path | None = None, name: str = "default") -> Path:
    return (root or Path.home() / ".dryfire") / "calibration" / f"{name}.json"
