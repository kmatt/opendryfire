"""Drill rules, exercised against the bundled example plan.

These are the rules that are easy to implement 'helpfully' and wrongly — a
ladder that advances on 7 of 10, or a rep that counts as clean because it beat
the beep while missing the scoring zone. Both make a shooter worse, so they are
pinned here.
"""

from __future__ import annotations

import pytest

from opendryfire.drills import ParLadder, coaching_note, evaluate_rep
from opendryfire.plans import load_plan
from opendryfire.session import Rep, ShotRecord, Session, pair_events

PLAN = load_plan()
PLAN_SETUPS = PLAN.setups
BLOCKS = PLAN.blocks


def shot(x=0.0, y=0.0, zone="A", points=5, t=1.5, index=0) -> ShotRecord:
    return ShotRecord(index=index, x=x, y=y, zone=zone, points=points, t=t)


# -- par ladder ---------------------------------------------------------------


def test_ladder_drops_par_after_eight_of_ten_clean():
    lad = ParLadder(par=2.5)
    for _ in range(7):
        assert not lad.record(True)
    lad.record(False)
    lad.record(False)
    assert lad.record(True) is True  # 8 of 10
    assert lad.par == pytest.approx(2.4)


def test_ladder_does_not_drop_on_seven_of_ten():
    lad = ParLadder(par=2.5)
    for clean in [True] * 7 + [False] * 3:
        lad.record(clean)
    assert lad.par == pytest.approx(2.5)


def test_ladder_requires_a_fresh_window_after_advancing():
    """Otherwise one good run would cascade several drops at once, which is
    exactly 'chasing a par your sights can't support'."""
    lad = ParLadder(par=2.5)
    for _ in range(10):
        lad.record(True)
    assert lad.par == pytest.approx(2.4)
    for _ in range(9):
        assert not lad.record(True)
    assert lad.par == pytest.approx(2.4)
    assert lad.record(True)
    assert lad.par == pytest.approx(2.3)


def test_ladder_will_not_descend_below_its_floor():
    lad = ParLadder(par=0.85, floor=0.8)
    for _ in range(10):
        lad.record(True)
    assert lad.par == pytest.approx(0.8)
    for _ in range(10):
        lad.record(True)
    assert lad.par == pytest.approx(0.8)


# -- rep evaluation -----------------------------------------------------------


def test_a_fast_miss_is_not_clean():
    """'The sight picture is the gate, not the beep.'"""
    rep = Rep(index=0, shots=[shot(zone="C", points=3)], par=2.5, first_shot_time=1.2)
    ok, why = evaluate_rep(rep, 2.5)
    assert not ok
    assert "sight picture" in why


def test_an_accurate_but_slow_rep_is_not_clean_either():
    rep = Rep(index=0, shots=[shot()], par=2.5, first_shot_time=2.9)
    ok, why = evaluate_rep(rep, 2.5)
    assert not ok
    assert "par" in why


def test_a_fast_a_zone_hit_is_clean():
    rep = Rep(index=0, shots=[shot()], par=2.5, first_shot_time=2.1)
    ok, why = evaluate_rep(rep, 2.5)
    assert ok
    assert "2.1" in why


def test_a_rep_with_no_beep_is_discarded_not_scored():
    rep = Rep(index=0, shots=[shot()], par=2.5, first_shot_time=None, valid=False)
    ok, why = evaluate_rep(rep, 2.5)
    assert not ok
    assert "discarded" in why


def test_untimed_block_judges_on_the_hit_alone():
    rep = Rep(index=0, shots=[shot()], par=None, first_shot_time=None)
    ok, _ = evaluate_rep(rep, None)
    assert ok


# -- block definitions --------------------------------------------------------


def test_block_f_is_untimed_by_default():
    """The plan: 'This is the one block where the clock is the enemy.'"""
    assert BLOCKS["F15"].timed is False
    assert BLOCKS["F25"].timed is False
    assert BLOCKS["F15"].start_par is None


def test_block_b_starts_where_the_plan_says():
    b = BLOCKS["B"]
    assert b.start_par == pytest.approx(2.5)
    assert b.target_par == pytest.approx(1.6)
    assert b.clean_to_advance == 8


def test_blocks_map_to_the_right_simulated_distances():
    assert BLOCKS["F15"].target.sim_distance_yd == pytest.approx(15.0, abs=0.01)
    assert BLOCKS["F25"].target.sim_distance_yd == pytest.approx(25.0, abs=0.01)
    assert BLOCKS["B"].target.sim_distance_yd == pytest.approx(7.0, abs=0.01)


def test_transition_block_carries_the_slide_rack_caveat():
    assert "rack" in BLOCKS["D"].warning.lower()


# -- sensor fusion ------------------------------------------------------------


def test_a_flash_with_a_click_is_a_confirmed_shot():
    flashes = [(10.00, 5.0, 5.0, 0.0, 3)]
    pairs = pair_events(flashes, [10.02])
    assert len(pairs) == 1
    assert pairs[0][0] is not None and pairs[0][1] is not None


def test_a_flash_with_no_click_is_rejected():
    """A stray reflection in a red-lit room must not score."""
    pairs = pair_events([(10.0, 5.0, 5.0, 0.0, 3)], [])
    assert pairs[0][1] is None


def test_a_click_with_no_flash_is_a_rack_not_a_shot():
    pairs = pair_events([], [10.0])
    assert pairs[0][0] is None
    assert pairs[0][1] == 10.0


def test_events_too_far_apart_are_not_paired():
    pairs = pair_events([(10.0, 5.0, 5.0, 0.0, 3)], [10.9])
    flash_pair = [p for p in pairs if p[0] is not None][0]
    assert flash_pair[1] is None


def test_multiple_shots_pair_in_order():
    flashes = [(1.0, 0, 0, 0, 3), (4.0, 1, 1, 0, 3), (7.0, 2, 2, 0, 3)]
    pairs = pair_events(flashes, [7.03, 1.02, 4.01])
    matched = [(f[0], c) for f, c in pairs if f and c]
    assert len(matched) == 3
    for ft, ct in matched:
        assert abs(ft - ct) < 0.15


# -- coaching -----------------------------------------------------------------


def test_low_left_group_is_diagnosed_as_trigger_and_routed_to_block_a():
    target = PLAN_SETUPS["f-15"]
    s = Session.begin("F15", "f-15", target)
    s.reps = [
        Rep(index=i, shots=[shot(x=-0.45 + 0.02 * i, y=-0.5 + 0.02 * i, index=i)])
        for i in range(6)
    ]
    notes = " ".join(coaching_note(s, target)).lower()
    assert "trigger" in notes
    assert "not sights" in notes


def test_off_center_group_is_called_a_zero_problem():
    target = PLAN_SETUPS["f-15"]
    s = Session.begin("F15", "f-15", target)
    s.reps = [
        Rep(index=i, shots=[shot(x=0.75, y=0.62 + 0.01 * i, index=i)]) for i in range(6)
    ]
    notes = " ".join(coaching_note(s, target))
    assert "zero" in notes.lower()


def test_group_is_reported_at_simulated_distance_not_printed_size():
    """A 0.6 inch group on a 1/5 target is a 3 inch group at 15 yards, and the
    3 inch number is the one comparable to the range Distance Card."""
    target = PLAN_SETUPS["f-15"]
    s = Session.begin("F15", "f-15", target)
    s.reps = [
        Rep(index=0, shots=[shot(x=-0.3, y=0.0, index=0)]),
        Rep(index=1, shots=[shot(x=0.3, y=0.0, index=1)]),
        Rep(index=2, shots=[shot(x=0.0, y=0.1, index=2)]),
    ]
    stats = s.stats(target)
    assert stats.extreme_spread == pytest.approx(0.6, abs=0.01)
    assert stats.sim_extreme_spread == pytest.approx(3.0, abs=0.05)


def test_session_round_trips_through_disk(tmp_path):
    target = PLAN_SETUPS["b-draw"]
    s = Session.begin("B", "b-draw", target)
    s.reps = [Rep(index=0, shots=[shot()], par=2.5, first_shot_time=2.1)]
    p = s.save(tmp_path / "s.json")
    back = Session.load(p)
    assert back.block == "B"
    assert back.reps[0].shots[0].zone == "A"
    assert back.reps[0].first_shot_time == pytest.approx(2.1)
    assert back.sim_distance_yd == pytest.approx(7.0, abs=0.01)
