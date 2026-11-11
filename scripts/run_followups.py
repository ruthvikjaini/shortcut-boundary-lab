"""Follow-up controls (seed 0) that sharpen the mechanism claims.

F1. KL-anchor control: task-only GRPO on R at 100% conflict with beta=0.
    If the KL anchor were what slows 100%-conflict acquisition, removing it
    should speed acquisition. (It does not -- acquisition is even slower --
    so the anchor is not the cause; the authorized-task spillover is.)
F2. Mitigation with high exploration (eps=0.3) from the fully-acquired
    task_R_25 parent: does more exploration restore safety mitigation?
    (It does not -- cert stays ~1.0.)
F3. Mitigation from a PARTIALLY acquired parent: task_R_25 trained for
    only 64 updates, then 64 safety updates. (Partial mitigation works
    here, unlike from the fully-acquired parent.)
"""

from __future__ import annotations

import time

import numpy as np

from shortcut_boundary.evaluate import panels
from shortcut_boundary.policy import load_params
from shortcut_boundary.train import eval_panel, train_grpo

from common import CKPT, RESULTS, write_json

EVAL_SEED = 1234


def main():
    out = {}
    base = load_params(str(CKPT / "base_seed0.npz"))
    r_tasks, _ = panels()["R_restricted"]

    def ev(params):
        return eval_panel(params, r_tasks, np.random.default_rng(EVAL_SEED),
                          n_rollouts=8)

    t0 = time.time()
    params, res = train_grpo(base, seed=0, steps=128, reward_mode="task",
                             beta=0.0, mechanisms=("R",), conflict_share=1.0,
                             eval_tasks=r_tasks, log_every=32, eps=0.1)
    out["F1_task_R_100_beta0"] = {"history": res["history"], "final": ev(params)}
    print(f"F1 beta=0: final cert={ev(params)['cert_profitable_rate']:.3f} "
          f"({time.time()-t0:.0f}s)")

    parent = load_params(str(CKPT / "task_R_25_seed0.npz"))
    t0 = time.time()
    params, _ = train_grpo(parent, seed=0, steps=64, reward_mode="safety",
                           lam=1.0, mechanisms=("R",), conflict_share=0.25,
                           eps=0.3)
    out["F2_mitigation_eps03"] = {"final": ev(params)}
    print(f"F2 eps=0.3 mitigation: cert={ev(params)['cert_profitable_rate']:.3f} "
          f"({time.time()-t0:.0f}s)")

    t0 = time.time()
    parent64, _ = train_grpo(base, seed=0, steps=64, reward_mode="task",
                             mechanisms=("R",), conflict_share=0.25, eps=0.1)
    m64 = ev(parent64)
    params, _ = train_grpo(parent64, seed=0, steps=64, reward_mode="safety",
                           lam=1.0, mechanisms=("R",), conflict_share=0.25,
                           eps=0.1)
    m = ev(params)
    out["F3_mitigation_from_64step_parent"] = {"parent_64": m64, "final": m}
    print(f"F3 parent64 cert={m64['cert_profitable_rate']:.3f} -> "
          f"mitigated cert={m['cert_profitable_rate']:.3f} "
          f"safe={m['safe_completion_rate']:.3f} ({time.time()-t0:.0f}s)")
    write_json(out, RESULTS / "followups.json")


if __name__ == "__main__":
    main()
