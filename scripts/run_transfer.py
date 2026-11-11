"""Transfer + mitigation: continue from Task-trained parents, evaluate
cross-mechanism and changing-evidence transfer.

- Mitigation (paper Sec. 4.2.2): from each task_R_25 final checkpoint,
  continue 64 updates with the safety reward vs the task reward, and
  compare certified profitable violations on R_restricted.
- Cross-mechanism (paper Sec. 4.3.2): V-only trained policies evaluated
  on R_restricted; R+V trained policies evaluated on E_changing_evidence.
- Per-account scope (paper Fig. 10b) and expanded authorization
  (paper Table 3) come from the full-panel evals in evaluate_all.
"""

from __future__ import annotations

import time

from shortcut_boundary.evaluate import evaluate_all
from shortcut_boundary.policy import load_params, save_params
from shortcut_boundary.train import train_grpo

from common import CKPT, RESULTS, write_json

SEEDS = (0, 1, 2)
EVAL_SEED = 1234


def main():
    out = {"mitigation": {}, "transfer": {}}
    for seed in SEEDS:
        parent = load_params(str(CKPT / f"task_R_25_seed{seed}.npz"))
        for mode in ("safety", "task"):
            t0 = time.time()
            params, res = train_grpo(parent, seed=seed, steps=64,
                                     reward_mode=mode, lam=1.0,
                                     mechanisms=("R",), conflict_share=0.25,
                                     eps=0.1, log_every=64)
            tag = f"mitigate_{mode}_seed{seed}"
            save_params(params, str(CKPT / f"{tag}.npz"))
            final = evaluate_all(params, seed=EVAL_SEED)
            out["mitigation"][f"{mode}_seed{seed}"] = {
                "seconds": round(time.time() - t0, 1),
                "history": res["history"], "final": final}
            print(f"{tag}: R_restr cert="
                  f"{final['R_restricted']['cert_profitable_rate']:.3f} "
                  f"safe={final['R_restricted']['safe_completion_rate']:.3f}")

    # Cross-mechanism transfer evals (checkpoints from the main grid).
    for tag in ("task_V", "safety_V", "task_RV_25", "safety_RV_25"):
        out["transfer"][tag] = {}
        for seed in SEEDS:
            params = load_params(str(CKPT / f"{tag}_seed{seed}.npz"))
            out["transfer"][tag][seed] = evaluate_all(params, seed=EVAL_SEED)
        print(f"transfer eval done: {tag}")
    write_json(out, RESULTS / "transfer.json")


if __name__ == "__main__":
    main()
