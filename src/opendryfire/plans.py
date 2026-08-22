"""Loading training plans from TOML.

A *plan* is the part of this tool that is nobody else's business: which target
setups you use, which blocks you run on them, and what par times you are
chasing. Keeping it as data rather than code means the program can ship one
person's programme as a worked example without imposing it.

Resolution order for the active plan:

1. ``--plan <path>`` on the command line
2. ``~/.dryfire/plan.toml``
3. the bundled ``plans/lean-track.toml``
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .targets import TARGETS, ScaledTarget

BUNDLED_DIR = Path(__file__).parent / "plans"
DEFAULT_PLAN_NAME = "lean-track"


class PlanError(ValueError):
    """A plan file that cannot be used as written."""


@dataclass(frozen=True)
class BlockSpec:
    """One block of dry fire work."""

    key: str
    name: str
    setup_key: str
    timed: bool
    start_par: float | None = None
    target_par: float | None = None
    par_step: float = 0.1
    reps: int = 10
    minutes: int = 15
    ladder: bool = False
    clean_to_advance: int = 8
    of_reps: int = 10
    guidance: str = ""
    warning: str = ""
    _setups: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def target(self) -> ScaledTarget:
        return self._setups[self.setup_key]


@dataclass
class Plan:
    name: str
    description: str
    setups: dict[str, ScaledTarget]
    blocks: dict[str, BlockSpec]
    source: Path | None = None

    def block(self, key: str) -> BlockSpec:
        found = self.blocks.get(key) or self.blocks.get(key.upper())
        if found is None:
            raise PlanError(
                f"no block {key!r} in plan {self.name!r}. "
                f"Available: {', '.join(sorted(self.blocks))}"
            )
        return found

    def setup(self, key: str) -> ScaledTarget:
        # Accept either spelling, since block keys and setup keys look alike
        # and confusing them is the obvious mistake.
        found = self.setups.get(key) or self.setups.get(key.lower().replace("_", "-"))
        if found is None:
            raise PlanError(
                f"no setup {key!r} in plan {self.name!r}. "
                f"Available: {', '.join(sorted(self.setups))}"
            )
        return found

    def describe(self) -> list[str]:
        lines = [f"{self.name} — {self.description}", ""]
        lines.append("Setups")
        for key, s in self.setups.items():
            lines.append(f"  {key:10s} {s.describe()}")
        lines.append("")
        lines.append("Blocks")
        for key, b in self.blocks.items():
            clock = "timed" if b.timed else "untimed"
            par = f", par {b.start_par:g}s" if b.start_par else ""
            lines.append(f"  {key:10s} {b.name}  [{clock}{par}]")
        return lines


def _scale_from(raw: dict, where: str) -> float:
    """Accept either an explicit scale or a denominator, but not neither."""
    if "scale_denominator" in raw:
        denom = float(raw["scale_denominator"])
        if denom <= 0:
            raise PlanError(f"{where}: scale_denominator must be positive")
        return 1.0 / denom
    if "scale" in raw:
        scale = float(raw["scale"])
        if not 0 < scale <= 1:
            raise PlanError(f"{where}: scale must be between 0 and 1")
        return scale
    raise PlanError(f"{where}: needs either scale or scale_denominator")


def parse_plan(data: dict, source: Path | None = None) -> Plan:
    """Build a Plan from parsed TOML, validating as we go.

    Validation is strict and the errors name the offending key. A plan is
    edited by hand, so a typo is likely and a silent fallback to a default
    would be scored as though it were the setup you meant.
    """
    setups: dict[str, ScaledTarget] = {}
    for i, raw in enumerate(data.get("setups", [])):
        where = f"setups[{i}]"
        key = raw.get("key")
        if not key:
            raise PlanError(f"{where}: missing key")
        target_key = raw.get("target", "uspsa")
        if target_key not in TARGETS:
            raise PlanError(
                f"{where}: unknown target {target_key!r}. "
                f"Built-in: {', '.join(sorted(TARGETS))}"
            )
        distance = raw.get("stand_distance_ft")
        if distance is None:
            raise PlanError(f"{where}: missing stand_distance_ft")
        setups[key] = ScaledTarget(
            spec=TARGETS[target_key],
            scale=_scale_from(raw, where),
            stand_distance_ft=float(distance),
        )

    if not setups:
        raise PlanError("plan defines no setups")

    blocks: dict[str, BlockSpec] = {}
    for i, raw in enumerate(data.get("blocks", [])):
        where = f"blocks[{i}]"
        key = raw.get("key")
        if not key:
            raise PlanError(f"{where}: missing key")
        setup_key = raw.get("setup")
        if setup_key not in setups:
            raise PlanError(
                f"{where} ({key}): setup {setup_key!r} is not defined. "
                f"Available: {', '.join(sorted(setups))}"
            )
        timed = bool(raw.get("timed", False))
        start_par = raw.get("start_par")
        if timed and start_par is None:
            raise PlanError(f"{where} ({key}): timed blocks need a start_par")
        if raw.get("ladder") and not timed:
            raise PlanError(f"{where} ({key}): a par ladder needs timed = true")

        blocks[key] = BlockSpec(
            key=key,
            name=raw.get("name", key),
            setup_key=setup_key,
            timed=timed,
            start_par=float(start_par) if start_par is not None else None,
            target_par=float(raw["target_par"]) if "target_par" in raw else None,
            par_step=float(raw.get("par_step", 0.1)),
            reps=int(raw.get("reps", 10)),
            minutes=int(raw.get("minutes", 15)),
            ladder=bool(raw.get("ladder", False)),
            clean_to_advance=int(raw.get("clean_to_advance", 8)),
            of_reps=int(raw.get("of_reps", 10)),
            guidance=str(raw.get("guidance", "")).strip(),
            warning=str(raw.get("warning", "")).strip(),
            _setups=setups,
        )

    if not blocks:
        raise PlanError("plan defines no blocks")

    return Plan(
        name=str(data.get("name", "unnamed")),
        description=str(data.get("description", "")),
        setups=setups,
        blocks=blocks,
        source=source,
    )


def load_plan(path: str | Path | None = None, root: Path | None = None) -> Plan:
    """Load the active plan. See the module docstring for resolution order."""
    if path is not None:
        p = Path(path).expanduser()
        if not p.exists():
            raise PlanError(f"no plan file at {p}")
        return parse_plan(tomllib.loads(p.read_text()), p)

    user = (root or Path.home() / ".dryfire") / "plan.toml"
    if user.exists():
        return parse_plan(tomllib.loads(user.read_text()), user)

    bundled = BUNDLED_DIR / f"{DEFAULT_PLAN_NAME}.toml"
    return parse_plan(tomllib.loads(bundled.read_text()), bundled)


def bundled_plans() -> list[Path]:
    return sorted(BUNDLED_DIR.glob("*.toml"))
