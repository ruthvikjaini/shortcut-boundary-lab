"""Demo: train a small BC base, run a short task-only GRPO burst on the
resource mechanism, and print the before/after on the restricted panel.

Runs in ~20 seconds on CPU. Shows the paper's headline effect in miniature:
task-only RL starts acquiring the profitable shortcut (certified profitable
violations rise from the base rate) while real work stays at 1.0.
"""

from __future__ import annotations

import numpy as np

from shortcut_boundary.evaluate import panels
from shortcut_boundary.train import eval_panel, train_bc, train_grpo


def main():
    base, info = train_bc(seed=0, epochs=300)
    print(f"BC base: agreement={info['bc_agreement']:.3f} "
          f"({info['n_samples']} samples)")
    tasks, _ = panels()["R_restricted"]
    rng = np.random.default_rng(1234)
    m0 = eval_panel(base, tasks, rng, n_rollouts=4)
    print(f"base       : reward={m0['mean_reward']:.3f} "
          f"cert-profitable={m0['cert_profitable_rate']:.3f} "
          f"safe-completion={m0['safe_completion_rate']:.3f}")
    params, _ = train_grpo(base, seed=0, steps=40, reward_mode="task",
                           mechanisms=("R",), conflict_share=0.25, eps=0.1)
    m1 = eval_panel(params, tasks, np.random.default_rng(1234), n_rollouts=4)
    print(f"task-only  : reward={m1['mean_reward']:.3f} "
          f"cert-profitable={m1['cert_profitable_rate']:.3f} "
          f"safe-completion={m1['safe_completion_rate']:.3f}")
    print("Done. See README.md for the full grid (3 seeds, 128 updates).")


if __name__ == "__main__":
    main()
