"""Main grid: BC bases (3 seeds) + all training arms, with full evaluation.

Arms (128 updates unless noted; lr 3e-4, group 8, batch 24 tasks,
eps-greedy 0.1 exploration, KL beta 0.02):
  task_R_25 / task_R_50 / task_R_100 -- task-only GRPO on R at 25/50/100%
      conflict mixture (paper RQ1 / Table 2 + Fig. 7).
  safety_R_25 / safety_R_100 -- safety GRPO (lambda=1 binary history
      penalty, Eq. 4) on R (paper RQ2).
  task_V / safety_V -- V-only training, 100 updates (cross-mechanism source).
  task_RV_25 / safety_RV_25 -- R+V training (transfer source for E).

Each arm is evaluated at its final checkpoint on every panel
(evaluate_all, fixed eval seed 1234). History (R_restricted panel, or
V_heldout for V arms) is logged every 32 steps. Checkpoints are saved
under results/checkpoints/.
"""

from __future__ import annotations

import time

import numpy as np

from shortcut_boundary.evaluate import evaluate_all, panels
from shortcut_boundary.policy import load_params, save_params
from shortcut_boundary.train import train_bc, train_grpo

from common import CKPT, RESULTS, mean3, write_json

SEEDS = (0, 1, 2)
STEPS = 128
EVAL_SEED = 1234


def run_arm(base, seed, tag, **kw):
    t0 = time.time()
    params, res = train_grpo(base, seed=seed, steps=kw.pop("steps", STEPS),
                             eps=0.1, **kw)
    save_params(params, str(CKPT / f"{tag}_seed{seed}.npz"))
    final = evaluate_all(params, seed=EVAL_SEED)
    out = {"tag": tag, "seed": seed, "seconds": round(time.time() - t0, 1),
           "history": res["history"],
           "skipped_no_signal_steps": res["skipped_no_signal_steps"],
           "final": final}
    print(f"{tag} seed {seed}: {time.time() - t0:.0f}s "
          f"R_restr cert={final['R_restricted']['cert_profitable_rate']:.3f} "
          f"safe={final['R_restricted']['safe_completion_rate']:.3f}")
    return out


def main():
    CKPT.mkdir(parents=True, exist_ok=True)
    grid = {"arms": {}, "bases": {}}
    bases = {}
    for seed in SEEDS:
        base, info = train_bc(seed=seed)
        bases[seed] = base
        save_params(base, str(CKPT / f"base_seed{seed}.npz"))
        grid["bases"][seed] = info
        print(f"base seed {seed}: agreement={info['bc_agreement']:.3f}")

    r_tasks, _ = panels()["R_restricted"]
    v_tasks, _ = panels()["V_heldout"]

    arms = [
        ("task_R_25", dict(reward_mode="task", mechanisms=("R",),
                           conflict_share=0.25, eval_tasks=r_tasks)),
        ("task_R_50", dict(reward_mode="task", mechanisms=("R",),
                           conflict_share=0.50, eval_tasks=r_tasks)),
        ("task_R_100", dict(reward_mode="task", mechanisms=("R",),
                            conflict_share=1.0, eval_tasks=r_tasks)),
        ("safety_R_25", dict(reward_mode="safety", lam=1.0,
                             mechanisms=("R",), conflict_share=0.25,
                             eval_tasks=r_tasks)),
        ("safety_R_100", dict(reward_mode="safety", lam=1.0,
                              mechanisms=("R",), conflict_share=1.0,
                              eval_tasks=r_tasks)),
        ("task_V", dict(reward_mode="task", mechanisms=("V",),
                        conflict_share=1.0, eval_tasks=v_tasks, steps=100)),
        ("safety_V", dict(reward_mode="safety", lam=1.0, mechanisms=("V",),
                          conflict_share=1.0, eval_tasks=v_tasks, steps=100)),
        ("task_RV_25", dict(reward_mode="task", mechanisms=("R", "V"),
                            conflict_share=0.25, eval_tasks=r_tasks)),
        ("safety_RV_25", dict(reward_mode="safety", lam=1.0,
                              mechanisms=("R", "V"), conflict_share=0.25,
                              eval_tasks=r_tasks)),
    ]
    for tag, kw in arms:
        kw = dict(kw)
        kw["log_every"] = 32
        runs = [run_arm(bases[s], s, tag, **kw) for s in SEEDS]
        grid["arms"][tag] = runs

    # 3-seed means for the headline panels.
    summary = {}
    for tag, runs in grid["arms"].items():
        summary[tag] = {}
        for panel in runs[0]["final"]:
            summary[tag][panel] = {}
            for metric in runs[0]["final"][panel]:
                if metric == "n":
                    continue
                summary[tag][panel][metric] = mean3(
                    [r["final"][panel][metric] for r in runs])
    grid["summary_3seed"] = summary
    write_json(grid, RESULTS / "grid.json")


if __name__ == "__main__":
    main()
