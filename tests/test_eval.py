"""Evaluation panels: metric ranges and certified <= historical."""

import numpy as np

from shortcut_boundary.envs import r_tasks
from shortcut_boundary.evaluate import (
    evaluate_all,
    lawful_expanded_use,
    panels,
)
from shortcut_boundary.policy import init_params
from shortcut_boundary.train import eval_panel, train_bc


def test_eval_panel_metric_ranges():
    rng = np.random.default_rng(0)
    params = init_params(rng)
    tasks, _ = panels()["R_restricted"]
    m = eval_panel(params, tasks, rng, n_rollouts=2)
    for k in ("mean_reward", "mean_real_work", "hist_violation_rate",
              "cert_profitable_rate", "safe_completion_rate",
              "completion_rate"):
        assert 0.0 <= m[k] <= 1.0 or k == "mean_reward", (k, m[k])
    assert m["cert_profitable_rate"] <= m["hist_violation_rate"]
    assert m["safe_completion_rate"] <= m["completion_rate"]
    assert m["n"] == 2 * len(tasks)


def test_lawful_expanded_use_range():
    rng = np.random.default_rng(1)
    params = init_params(rng)
    v = lawful_expanded_use(params, r_tasks(restricted=False), rng)
    assert 0.0 <= v <= 1.0


def test_bc_base_is_mostly_compliant_on_r():
    # A BC base trained on compliant references should complete R tasks
    # with a low certified-profitable violation rate.
    params, info = train_bc(seed=0, epochs=60)
    assert info["bc_agreement"] > 0.8
    rng = np.random.default_rng(2)
    tasks, _ = panels()["R_restricted"]
    m = eval_panel(params, tasks, rng, n_rollouts=2)
    assert m["safe_completion_rate"] > 0.5
    assert m["cert_profitable_rate"] < 0.3
