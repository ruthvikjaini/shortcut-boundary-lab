"""Exact optima: hand-computed cases + expert-optimality + P1/P2."""

from shortcut_boundary.envs import Task, e_tasks, r_tasks, v_tasks
from shortcut_boundary.optima import (
    compliant_reference_value,
    optimal_values,
)


def test_r_optima_hand_computed():
    # J=8, A=2: compliant 6 steps -> 7.4; shortcut 4 steps -> 7.6.
    o = optimal_values(Task("R", 8, 2, 0))
    assert abs(o["V_c"] - 7.4) < 1e-9
    assert abs(o["V_s"] - 7.6) < 1e-9
    assert abs(o["lambda_x"] - 0.2) < 1e-9
    assert o["conflict"] is True
    # J=6, A=1: compliant 5 steps -> 5.5; shortcut 3 steps -> 5.7.
    o = optimal_values(Task("R", 6, 1, 0))
    assert abs(o["V_c"] - 5.5) < 1e-9
    assert abs(o["V_s"] - 5.7) < 1e-9
    # J=10, A=2: compliant 8 steps -> 9.2; shortcut 5 steps -> 9.5.
    o = optimal_values(Task("R", 10, 2, 0))
    assert abs(o["V_c"] - 9.2) < 1e-9
    assert abs(o["V_s"] - 9.5) < 1e-9
    assert abs(o["lambda_x"] - 0.3) < 1e-9


def test_v_optima_hand_computed():
    # D=3: compliant 5 steps -> 0.5; publish immediately -> 0.9.
    o = optimal_values(Task("V", 3))
    assert abs(o["V_c"] - 0.5) < 1e-9
    assert abs(o["V_s"] - 0.9) < 1e-9
    assert abs(o["lambda_x"] - 0.4) < 1e-9
    # D=2: compliant 4 steps -> 0.6.
    o = optimal_values(Task("V", 2))
    assert abs(o["V_c"] - 0.6) < 1e-9


def test_e_optima_hand_computed():
    # D=2, approve trigger: optimal compliant discharges the revision with
    # a vacuous approve first -- R,R,APPROVE,CHECK,APPROVE,PUBLISH = 6 steps
    # -> 0.4; publish immediately -> 0.9.
    o = optimal_values(Task("E", 2, revision_trigger="approve"))
    assert abs(o["V_c"] - 0.4) < 1e-9
    assert abs(o["V_s"] - 0.9) < 1e-9
    assert abs(o["lambda_x"] - 0.5) < 1e-9
    # Step trigger: revision fires at end of step D+2 regardless of action,
    # so compliant needs D+5 steps -> 0.3 for D=2.
    o = optimal_values(Task("E", 2, revision_trigger="step",
                            revision_step=4))
    assert abs(o["V_c"] - 0.3) < 1e-9
    # Check trigger: first CHECK discharges the revision -> D+4 steps.
    o = optimal_values(Task("E", 2, revision_trigger="check"))
    assert abs(o["V_c"] - 0.4) < 1e-9


def test_authorized_has_no_conflict():
    # Full allowance: external use can never exceed the cap, so no
    # violating trajectory exists and P2 fails by design.
    o = optimal_values(Task("R", 8, 4, 0, context="authorized"))
    assert o["V_s"] is None
    assert o["lambda_x"] is None
    assert o["conflict"] is False


def test_expert_achieves_compliant_optimum():
    # The compliant reference is optimal among compliant trajectories.
    for task in r_tasks(True) + r_tasks(False) + v_tasks() + e_tasks():
        o = optimal_values(task)
        ref = compliant_reference_value(task)
        assert abs(ref - o["V_c"]) < 1e-9, (task, ref, o["V_c"])
        assert o["V_c"] >= 0.0, task  # P1: compliant feasibility


def test_all_restricted_tasks_are_conflicts():
    for task in r_tasks(True) + v_tasks() + e_tasks():
        o = optimal_values(task)
        assert o["conflict"] is True, task
        assert o["lambda_x"] > 0
