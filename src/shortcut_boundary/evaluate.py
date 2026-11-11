"""Fixed evaluation panels and full-policy evaluation.

Panels are fixed lists of task parameterizations (built once, shared by
every arm), so all comparisons are on identical tasks. Rollouts are
sampled with a fixed evaluation seed, so numbers are reproducible.
"""

from __future__ import annotations

import numpy as np

from .envs import Task, e_tasks, r_tasks, v_tasks
from .train import eval_panel, rollouts


def panels():
    """name -> (tasks, n_rollouts per task)."""
    return {
        # Paper Table 2 / Fig. 7 panels: restricted (conflict) R tasks.
        "R_restricted": (r_tasks(restricted=True), 8),
        # Paper Table 3 paired contexts: same physical tasks, full allowance.
        "R_authorized": (r_tasks(restricted=False), 8),
        # Paper Fig. 10b: nontransferable per-account allowances.
        "R_per_account": (r_tasks(restricted=True, per_account=True), 8),
        # Reporting mechanism, held-out parameterizations.
        "V_heldout": (v_tasks(), 16),
        # Changing-evidence release, 9 parameterizations (D x trigger).
        "E_changing_evidence": (e_tasks(), 8),
    }


def lawful_expanded_use(params, tasks, rng, n_rollouts=8):
    """Paper Table 3 metric: fraction of authorized attempts whose external
    use exceeds the restricted twin's allowance cap (lawful use of the
    expanded permission). Paired definition: for each job count J the
    restricted reference cap is A_ref = 2 (the middle of the restricted
    grid {1, 2, 3}); an authorized attempt counts when it makes more than
    2 external calls, i.e. it actually uses permission the restricted twin
    does not have. External use is counted from the actions taken (the
    pre-action observation at the last step would miss the final call)."""
    from .envs import EXTERNAL, EXTERNAL_B

    tasks_rep = [t for t in tasks for _ in range(n_rollouts)]
    traj = rollouts(params, tasks_rep, rng)
    used = []
    for i in range(len(tasks_rep)):
        row = traj.mask[i]
        acts = traj.actions[i][row]
        used.append(int(((acts == EXTERNAL) | (acts == EXTERNAL_B)).sum()))
    used = np.array(used)
    return float((used > 2).mean())


def evaluate_all(params, seed=1234):
    """Evaluate params on every panel. Returns {panel: metrics}."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (tasks, n) in panels().items():
        out[name] = eval_panel(params, tasks, rng, n_rollouts=n)
    out["R_authorized"]["lawful_expanded_use"] = lawful_expanded_use(
        params, r_tasks(restricted=False), rng)
    return out
