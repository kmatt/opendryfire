"""Optional Markdown export of session results.

Off unless configured, because it writes into a personal notes folder.

Two modes, and the distinction is deliberate:

* ``session_note`` creates a **new** note per session. Safe — it only ever adds
  a file.
* ``append_to_tracker`` edits a note you maintain by hand. It only ever appends
  inside a marker block it owns, never rewrites what is already there, and is
  behind a separate flag again.

Someone's training log may hold months of hand-kept data. Nothing here
overwrites a line a human wrote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .drills import coaching_note
from .session import Session
from .targets import ScaledTarget

MARKER_BEGIN = "<!-- dryfire:begin -->"
MARKER_END = "<!-- dryfire:end -->"

# Markdown export is opt-in and unconfigured by default: this writes into a
# personal notes folder, so it should never happen because a default pointed
# somewhere plausible. Set it with `dryfire vault set <path>`.
DEFAULT_TRACKER_NAME = "Dry fire tracker.md"
DEFAULT_SESSION_FOLDER = "Dry Fire Sessions"


@dataclass
class VaultPaths:
    """Where session notes are written, if anywhere.

    Written for Obsidian but not specific to it — the output is plain Markdown
    with YAML frontmatter and ``[[wikilinks]]``, which degrades to readable text
    in any editor.
    """

    root: Path
    session_folder: str = DEFAULT_SESSION_FOLDER
    tracker_name: str = DEFAULT_TRACKER_NAME
    plan_note: str = ""

    @property
    def tracker(self) -> Path:
        return self.root / self.tracker_name

    @property
    def sessions(self) -> Path:
        return self.root / self.session_folder

    def exists(self) -> bool:
        return self.root.is_dir()

    @classmethod
    def from_config(cls, cfg: dict) -> "VaultPaths | None":
        """Build from saved settings, or None if export is not configured."""
        root = cfg.get("vault")
        if not root:
            return None
        return cls(
            root=Path(root).expanduser(),
            session_folder=cfg.get("vault_sessions", DEFAULT_SESSION_FOLDER),
            tracker_name=cfg.get("vault_tracker", DEFAULT_TRACKER_NAME),
            plan_note=cfg.get("vault_plan_note", ""),
        )


def format_summary_line(session: Session, target: ScaledTarget) -> str:
    """A single line in the tracker's own idiom, ready to paste by hand."""
    stats = session.stats(target)
    bits = [f"**{session.block}**"]
    times = session.times()
    if times:
        bits.append(f"best {min(times):.2f}s")
        bits.append(f"{session.clean_count}/{len(session.reps)} clean")
    if stats.n:
        bits.append(
            f"{stats.sim_extreme_spread:.2f}\" @ {target.sim_distance_yd:.0f}yd eq"
        )
        bits.append(f"{stats.in_primary}/{stats.n} A")
        bits.append(stats.shape)
    return " · ".join(bits)


def session_note(
    session: Session,
    target: ScaledTarget,
    paths: VaultPaths,
    write: bool = True,
) -> tuple[Path, str]:
    """Render (and optionally write) a per-session note."""
    when = datetime.fromisoformat(session.started)
    stats = session.stats(target)
    times = session.times()
    slug = f"Dry Fire {when:%Y-%m-%d %H%M} {session.block}"
    path = paths.sessions / f"{slug}.md"

    lines = [
        "---",
        f'title: "{slug}"',
        "tags: [training/dryfire, area/marksmanship]",
        f"created: {when:%Y-%m-%d}",
        f"block: {session.block}",
        "---",
        "",
        f"Block **{session.block}** · {target.describe()}",
    ]
    if paths.plan_note:
        lines.append(f"Part of [[{paths.plan_note}]].")
    lines += [
        "",
        "## Numbers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Reps | {len(session.reps)} |",
        f"| Shots detected | {stats.n} |",
        f"| Clean | {session.clean_count}/{len(session.reps)} |",
    ]
    if times:
        lines += [
            f"| Best draw | {min(times):.2f}s |",
            f"| Average draw | {sum(times) / len(times):.2f}s |",
        ]
    if session.par_history:
        lines.append(f"| Par ladder | {' → '.join(f'{p:.2f}' for p in session.par_history)} |")
    if stats.n:
        lines += [
            f"| Group (printed) | {stats.extreme_spread:.2f}\" |",
            f"| **Group @ {target.sim_distance_yd:.0f}yd equivalent** | "
            f"**{stats.sim_extreme_spread:.2f}\"** |",
            f"| In {target.spec.primary_zone}-zone | {stats.in_primary}/{stats.n} |",
            f"| Group shape | {stats.shape} |",
        ]
    lines += ["", "## Read", ""]
    lines += [f"- {n}" for n in coaching_note(session, target)]

    if session.all_shots:
        lines += [
            "",
            "## Shots",
            "",
            "| # | x″ | y″ | Zone | Time |",
            "|---|---|---|---|---|",
        ]
        for s in session.all_shots:
            t = f"{s.t:.2f}s" if s.t else "—"
            lines.append(
                f"| {s.index + 1} | {s.x:+.2f} | {s.y:+.2f} | {s.zone} | {t} |"
            )

    lines += ["", f"Tracker line: `{format_summary_line(session, target)}`", ""]
    body = "\n".join(lines)

    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return path, body


def append_to_tracker(
    session: Session,
    target: ScaledTarget,
    paths: VaultPaths,
    write: bool = True,
) -> tuple[Path, str]:
    """Append one line to the weekly tracker, inside a block we own.

    Never rewrites the user's own lines: if the marker block is absent it is
    created at the end of the file, and if present the new line is inserted
    just before its closing marker.
    """
    path = paths.tracker
    existing = path.read_text() if path.exists() else ""
    stamp = date.today().strftime("%-m/%-d/%Y")
    line = f"- {stamp} — {format_summary_line(session, target)}"

    if MARKER_BEGIN in existing and MARKER_END in existing:
        updated = existing.replace(MARKER_END, f"{line}\n{MARKER_END}", 1)
    else:
        block = (
            f"\n\n{MARKER_BEGIN}\n"
            f"<!-- appended by dryfire; edit freely, it only ever adds lines -->\n"
            f"{line}\n{MARKER_END}\n"
        )
        updated = existing.rstrip() + block

    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated)
    return path, line


def current_week(plan_start: date, today: date | None = None) -> int:
    """Which week of a programme we are in, counting from its start date."""
    today = today or date.today()
    return max(1, ((today - plan_start).days // 7) + 1)
