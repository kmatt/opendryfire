"""Drill rules: how a repetition is judged, and how a par ladder advances.

*What* to run lives in a plan file (see ``plans.py``). This module is the part
that is the same whoever wrote the plan — the rules that are easy to implement
"helpfully" and wrongly:

* A par ladder advances only on a clear majority of clean reps, and a rep is
  clean only if the hit landed in the scoring zone. A fast miss is not progress.
* An untimed block stays untimed unless the clock is explicitly asked for.
* A group is diagnosed by *shape*, which selects the fix, rather than by size
  alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .plans import BlockSpec, Plan, load_plan
from .session import Rep, Session
from .targets import ScaledTarget

__all__ = [
    "BlockSpec",
    "ParLadder",
    "Plan",
    "coaching_note",
    "evaluate_rep",
    "load_plan",
]


@dataclass
class ParLadder:
    """Par advancement for a timed block.

    Deliberately strict in one direction: it will not advance on 7 of 10, and
    it does not reward a fast rep that missed. The failure mode being guarded
    against is building speed on a bad trigger press, so the par only drops
    when the hits supported it.
    """

    par: float
    step: float = 0.1
    needed: int = 8
    window: int = 10
    floor: float = 0.8
    history: list[bool] = field(default_factory=list)
    advanced_at: list[float] = field(default_factory=list)

    def record(self, clean: bool) -> bool:
        """Record a rep. Returns True if the par just dropped."""
        self.history.append(clean)
        recent = self.history[-self.window :]
        if len(recent) < self.window:
            return False
        if sum(recent) >= self.needed:
            new_par = round(max(self.par - self.step, self.floor), 2)
            if new_par < self.par:
                self.advanced_at.append(self.par)
                self.par = new_par
                self.history.clear()  # earn the next drop from scratch
                return True
        return False

    @property
    def clean_in_window(self) -> int:
        return sum(self.history[-self.window :])

    def status(self) -> str:
        return (
            f"par {self.par:.2f}s | {self.clean_in_window}/"
            f"{min(len(self.history), self.window)} clean "
            f"(need {self.needed}/{self.window} to drop)"
        )


def evaluate_rep(
    rep: Rep,
    par: float | None,
    primary_zone: str = "A",
) -> tuple[bool, str]:
    """Judge a rep and say why, so the reason is visible rather than a verdict.

    Order matters: the hit is checked before the clock, because a miss inside
    par is a worse outcome than a hit outside it.
    """
    if not rep.valid:
        return False, "invalid — no beep heard, rep discarded"
    if not rep.shots:
        return False, "no shot detected"
    shot = rep.shots[0]
    if shot.zone != primary_zone:
        return False, f"hit {shot.zone}, not {primary_zone} — sight picture"
    if par is not None and rep.first_shot_time is not None:
        if rep.first_shot_time > par:
            return False, f"{rep.first_shot_time:.2f}s over {par:.2f}s par"
        return True, f"clean {rep.first_shot_time:.2f}s"
    return True, "clean"


def coaching_note(session: Session, target: ScaledTarget) -> list[str]:
    """Turn the numbers into a diagnosis and a suggested fix.

    Group *shape* selects the fix, not group size: the same 3-inch group means
    different things depending on whether it is round, strung, or displaced.
    """
    notes: list[str] = []
    shots = session.all_shots
    if len(shots) < 3:
        return ["Too few shots to diagnose. Run the full block."]

    stats = session.stats(target)
    notes.append(
        f"Group {stats.sim_extreme_spread:.2f}\" at "
        f"{target.sim_distance_yd:.0f} yd equivalent ({stats.shape})."
    )

    shape = stats.shape
    if "low-left" in shape:
        notes.append(
            "Low-left is trigger, not sights — wall drill and one-handed "
            "trigger isolation, not more distance work."
        )
    elif "off-center" in shape:
        notes.append(
            "Tight but off-centre is a zero problem, not a you problem — "
            "confirm zero at the range before chasing it in dry fire."
        )
    elif "strung vertical" in shape:
        notes.append(
            "Vertical stringing usually means an inconsistent grip or a "
            "changing sight picture between shots. Work the grip."
        )
    elif shape == "round":
        if stats.sim_extreme_spread > 3.0:
            notes.append(
                "Round but large is the wobble zone — the fix is volume and "
                "grip, not trigger work."
            )
        else:
            notes.append("Round and tight. This is what you want.")

    if stats.n:
        pct = 100 * stats.in_primary / stats.n
        zone = target.spec.primary_zone
        notes.append(f"{stats.in_primary}/{stats.n} in the {zone}-zone ({pct:.0f}%).")

    return notes
