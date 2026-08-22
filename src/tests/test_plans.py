"""Plan loading and validation.

A plan is hand-edited TOML, so typos are likely. Every error here should name
the offending key and refuse to run — a plan that silently falls back to a
default would score a session against a setup you did not mean, and the numbers
would look perfectly reasonable.
"""

from __future__ import annotations

import pytest

from opendryfire.plans import PlanError, bundled_plans, load_plan, parse_plan

MINIMAL = {
    "name": "T",
    "setups": [
        {"key": "a", "target": "uspsa", "scale_denominator": 5,
         "stand_distance_ft": 9.0}
    ],
    "blocks": [{"key": "X", "setup": "a", "timed": False}],
}


def test_the_bundled_plan_loads():
    plan = load_plan()
    assert plan.blocks
    assert plan.setups
    assert plan.source is not None


def test_bundled_plan_derives_the_documented_distances():
    """The bundled example claims 9 ft at 1/5 scale simulates 15 yards. If the
    arithmetic is wrong every group size in the output is wrong with it."""
    plan = load_plan()
    assert plan.setup("f-15").sim_distance_yd == pytest.approx(15.0, abs=0.01)
    assert plan.setup("f-25").sim_distance_yd == pytest.approx(25.0, abs=0.01)


def test_scale_can_be_given_as_a_denominator_or_a_fraction():
    by_denom = parse_plan(MINIMAL)
    frac = {**MINIMAL, "setups": [
        {"key": "a", "target": "uspsa", "scale": 0.2, "stand_distance_ft": 9.0}
    ]}
    assert by_denom.setup("a").scale == pytest.approx(parse_plan(frac).setup("a").scale)


def test_a_setup_with_no_scale_is_refused():
    bad = {**MINIMAL, "setups": [
        {"key": "a", "target": "uspsa", "stand_distance_ft": 9.0}
    ]}
    with pytest.raises(PlanError, match="scale"):
        parse_plan(bad)


def test_a_block_pointing_at_a_missing_setup_names_the_problem():
    bad = {**MINIMAL, "blocks": [{"key": "X", "setup": "typo", "timed": False}]}
    with pytest.raises(PlanError) as exc:
        parse_plan(bad)
    assert "typo" in str(exc.value)
    assert "Available" in str(exc.value)


def test_an_unknown_target_lists_the_built_in_ones():
    bad = {**MINIMAL, "setups": [
        {"key": "a", "target": "nonesuch", "scale_denominator": 5,
         "stand_distance_ft": 9.0}
    ]}
    with pytest.raises(PlanError) as exc:
        parse_plan(bad)
    assert "uspsa" in str(exc.value)


def test_a_timed_block_without_a_par_is_refused():
    bad = {**MINIMAL, "blocks": [{"key": "X", "setup": "a", "timed": True}]}
    with pytest.raises(PlanError, match="start_par"):
        parse_plan(bad)


def test_a_ladder_on_an_untimed_block_is_refused():
    """A par ladder with no par is meaningless; better to say so than to run."""
    bad = {**MINIMAL, "blocks": [
        {"key": "X", "setup": "a", "timed": False, "ladder": True}
    ]}
    with pytest.raises(PlanError, match="ladder"):
        parse_plan(bad)


def test_an_empty_plan_is_refused():
    with pytest.raises(PlanError, match="setups"):
        parse_plan({"name": "T"})
    with pytest.raises(PlanError, match="blocks"):
        parse_plan({**MINIMAL, "blocks": []})


def test_unknown_block_lists_what_is_available():
    plan = parse_plan(MINIMAL)
    with pytest.raises(PlanError) as exc:
        plan.block("nope")
    assert "X" in str(exc.value)


def test_block_lookup_is_case_insensitive():
    plan = parse_plan(MINIMAL)
    assert plan.block("x").key == "X"


def test_a_missing_plan_file_is_an_error_not_a_silent_default():
    with pytest.raises(PlanError, match="no plan file"):
        load_plan("/nonexistent/plan.toml")


def test_a_user_plan_overrides_the_bundled_one(tmp_path):
    (tmp_path / "plan.toml").write_text(
        'name = "Mine"\n'
        '[[setups]]\nkey = "s"\ntarget = "b8"\n'
        'scale_denominator = 2\nstand_distance_ft = 6.0\n'
        '[[blocks]]\nkey = "Z"\nsetup = "s"\ntimed = false\n'
    )
    plan = load_plan(root=tmp_path)
    assert plan.name == "Mine"
    assert "Z" in plan.blocks


def test_every_bundled_plan_parses():
    """Ship nothing that does not load."""
    assert bundled_plans()
    for path in bundled_plans():
        assert load_plan(path).blocks


def test_describe_mentions_whether_a_block_is_timed():
    text = "\n".join(load_plan().describe())
    assert "untimed" in text
    assert "timed" in text
