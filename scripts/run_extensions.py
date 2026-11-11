"""Extensions (ours, beyond the paper).

E1. Lambda ablation vs the exact threshold (paper Eq. 5): the paper states
    the compliant optimum is strictly preferred iff lambda > lambda_x, but
    does not ablate lambda. Here lambda in {0.05, 0.1, 0.2, 0.3, 0.6, 1.0}
    on R at 25% conflict, 64 updates, seed 0. The panel's lambda_x values
    are 0.1-0.4 (mean 0.225), so the grid straddles the threshold.
E2. Conflict-mixture threshold curve: task-only GRPO on R, 64 updates,
    seed 0, conflict share in {0, 0.05, 0.10, 0.25, 0.50, 1.0}. At what
    minority share does shortcut acquisition take off?
E3. Durability: after safety training (safety_R_25 seed 0), continue with
    the task-only reward for 64 updates. Does the shortcut return?
"""

from __future__ import annotations

import time

import numpy as np

from shortcut_boundary.evaluate import panels
from shortcut_boundary.optima import optimal_values
from shortcut_boundary.policy import load_params, save_params
from shortcut_boundary.train import eval_panel, train_grpo

from common import CKPT, RESULTS, write_json

EVAL_SEED = 1234


def main():
    out = {}
    base = load_params(str(CKPT / "base_seed0.npz"))
    r_tasks, _ = panels()["R_restricted"]
    lam_x = [optimal_values(t)["lambda_x"] for t in r_tasks]
    out["panel_lambda_x"] = {
        "values": [round(v, 3) for v in lam_x],
        "mean": round(sum(lam_x) / len(lam_x), 4),
        "min": round(min(lam_x), 3), "max": round(max(lam_x), 3)}

    # E1: lambda ablation.
    out["E1_lambda_ablation"] = {}
    for lam in (0.05, 0.1, 0.2, 0.3, 0.6, 1.0):
        t0 = time.time()
        params, _ = train_grpo(base, seed=0, steps=64, reward_mode="safety",
                               lam=lam, mechanisms=("R",), conflict_share=0.25,
                               eps=0.1)
        m = eval_panel(params, r_tasks, np.random.default_rng(EVAL_SEED),
                       n_rollouts=8)
        out["E1_lambda_ablation"][str(lam)] = m
        print(f"E1 lam={lam}: cert={m['cert_profitable_rate']:.3f} "
              f"safe={m['safe_completion_rate']:.3f} ({time.time()-t0:.0f}s)")

    # E2: mixture threshold curve.
    out["E2_mixture_curve"] = {}
    for share in (0.0, 0.05, 0.10, 0.25, 0.50, 1.0):
        t0 = time.time()
        params, _ = train_grpo(base, seed=0, steps=64, reward_mode="task",
                               mechanisms=("R",), conflict_share=share,
                               eps=0.1)
        m = eval_panel(params, r_tasks, np.random.default_rng(EVAL_SEED),
                       n_rollouts=8)
        out["E2_mixture_curve"][str(share)] = m
        print(f"E2 share={share}: cert={m['cert_profitable_rate']:.3f} "
              f"safe={m['safe_completion_rate']:.3f} ({time.time()-t0:.0f}s)")

    # E3: durability of safety under task-only continuation.
    t0 = time.time()
    parent = load_params(str(CKPT / "safety_R_25_seed0.npz"))
    params, res = train_grpo(parent, seed=0, steps=64, reward_mode="task",
                             mechanisms=("R",), conflict_share=0.25, eps=0.1)
    m = eval_panel(params, r_tasks, np.random.default_rng(EVAL_SEED),
                   n_rollouts=8)
    out["E3_durability"] = {"task_continuation_64": m,
                            "history": res["history"]}
    print(f"E3: after task-only continuation cert={m['cert_profitable_rate']:.3f} "
          f"safe={m['safe_completion_rate']:.3f} ({time.time()-t0:.0f}s)")
    save_params(params, str(CKPT / "durability_seed0.npz"))
    write_json(out, RESULTS / "extension.json")


if __name__ == "__main__":
    main()
