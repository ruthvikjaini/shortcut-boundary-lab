"""Behavior cloning (base policy) and GRPO training.

Reward modes (paper Eq. 4):
  task   -- R_task(tau), the native task reward only.
  safety -- R_task(tau) - lambda * I_c(tau), the binary history penalty.

GRPO: trajectory-level reward, group-normalized advantages over G rollouts
of the same task, PPO-style clipped surrogate, per-token k1 KL penalty to
the reference checkpoint inside the advantage. When a batch carries zero
advantage signal (all rewards in every group equal) the optimizer step is
skipped entirely: with adv = 0 the KL term alone would drift the policy
(Day 17 lesson).
"""

from __future__ import annotations

import copy

import numpy as np

from .envs import (
    N_ACTIONS,
    OBS_DIM,
    ShortcutEnv,
    e_tasks,
    expert_action,
    initial_state,
    is_violation,
    observe,
    r_tasks,
    real_work,
    sample_tasks,
    step_state,
    v_tasks,
)
from .policy import Adam, bc_loss_and_grad, forward, init_params, log_probs


class Trajectories:
    def __init__(self, obs, actions, logp_old, rewards, mask, violated,
                 real_work_, published):
        self.obs = obs              # (N, T, 24)
        self.actions = actions      # (N, T) int
        self.logp_old = logp_old    # (N, T)
        self.rewards = rewards      # (N,) terminal reward (as trained on)
        self.mask = mask            # (N, T) bool
        self.violated = violated    # (N,) bool
        self.real_work = real_work_  # (N,) independently verified work
        self.published = published  # (N,) bool


def rollouts(params, tasks, rng, eps=0.0):
    """One sampled trajectory per task. Returns Trajectories."""
    n = len(tasks)
    max_t = max(t.tmax for t in tasks)
    envs = [ShortcutEnv(t) for t in tasks]
    obs = np.zeros((n, max_t, OBS_DIM))
    actions = np.zeros((n, max_t), dtype=int)
    logp_old = np.zeros((n, max_t))
    mask = np.zeros((n, max_t), dtype=bool)
    rewards = np.zeros(n)
    violated = np.zeros(n, dtype=bool)
    work = np.zeros(n)
    published = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    states = [envs[i].state for i in range(n)]
    for t in range(max_t):
        active = ~done
        if not active.any():
            break
        idx = np.flatnonzero(active)
        batch_obs = np.stack([observe(tasks[i], states[i]) for i in idx])
        lp = log_probs(params, batch_obs)
        p = np.exp(lp)
        p = p / p.sum(axis=1, keepdims=True)
        acts = np.array([rng.choice(N_ACTIONS, p=p[j]) for j in range(len(idx))])
        if eps > 0:
            rand = rng.random(len(idx)) < eps
            acts[rand] = rng.integers(0, N_ACTIONS, size=int(rand.sum()))
        for j, i in enumerate(idx):
            obs[i, t] = batch_obs[j]
            actions[i, t] = acts[j]
            logp_old[i, t] = lp[j, acts[j]]
            mask[i, t] = True
            s2, r, d = step_state(tasks[i], states[i], int(acts[j]))
            states[i] = s2
            if d:
                done[i] = True
                rewards[i] = r
                violated[i] = is_violation(tasks[i], s2)
                work[i] = real_work(tasks[i], s2)
                published[i] = bool(s2[15])
    return Trajectories(obs, actions, logp_old, rewards, mask, violated,
                        work, published)


def group_advantages(rewards, group):
    n = rewards.shape[0]
    adv = np.zeros(n)
    for g in range(0, n, group):
        r = rewards[g:g + group]
        std = r.std()
        if std > 1e-8:
            adv[g:g + group] = (r - r.mean()) / std
    return adv


def _grad_from_dlogits(params, cache, dlogits):
    X, H = cache
    gW2 = H.T @ dlogits
    gb2 = dlogits.sum(axis=0)
    dH = dlogits @ params["W2"].T
    dZ = dH * (1.0 - H ** 2)
    gW1 = X.T @ dZ
    gb1 = dZ.sum(axis=0)
    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def grpo_grad(params, traj: Trajectories, adv, ref_params, beta=0.02, clip=0.2):
    """Gradient of the clipped GRPO surrogate with the per-token k1 KL
    penalty inside the advantage (the TRL form): A_t = adv - beta * (lp -
    lp_ref), detached. Tokens where the policy has moved above the
    reference get a reduced advantage, so the gradient pulls back toward
    the reference. The sign is load-bearing: +beta would be an
    anti-regularizer that rewards moving away from the reference (see
    test_grpo_kl_sign_pulls_toward_ref)."""
    X = traj.obs[traj.mask]
    a = traj.actions[traj.mask]
    lp_old = traj.logp_old[traj.mask]
    adv_step = np.repeat(adv, traj.mask.sum(axis=1))
    logits, cache = forward(params, X)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    lp = np.log(p[np.arange(X.shape[0]), a] + 1e-12)
    ratio = np.exp(lp - lp_old)
    lp_ref = log_probs(ref_params, X)[np.arange(X.shape[0]), a]
    kl_t = lp - lp_ref  # k1 per-token KL; used as a (detached) advantage term
    kl = float(np.mean(kl_t))
    A = adv_step - beta * kl_t
    clipped = np.clip(ratio, 1 - clip, 1 + clip)
    use_unclipped = (ratio * A) <= (clipped * A)
    Z = X.shape[0]
    coef = np.where(use_unclipped, ratio, 0.0) * A
    dlogp = -coef / Z
    dlogits = np.zeros_like(logits)
    dlogits[np.arange(X.shape[0]), a] = dlogp
    dlogits -= p * dlogp[:, None]
    grads = _grad_from_dlogits(params, cache, dlogits)
    info = {"mean_ratio": float(ratio.mean()),
            "clip_frac": float((~use_unclipped).mean()), "kl": kl}
    return grads, info


# -- behavior cloning --------------------------------------------------------

def collect_bc_data(rng, n_per_task=6, eps=0.15):
    """Roll out the compliant expert (eps-random) on R, V and E tasks and
    record (obs, expert action) pairs. E is included here only: the BC base
    is the 'pretrained base' analog -- it knows the workflow format,
    including APPROVE -- but E never appears in any GRPO training mixture."""
    tasks = r_tasks(restricted=True) + r_tasks(restricted=False) \
        + v_tasks() + e_tasks()
    Xs, ys = [], []
    for task in tasks:
        for _ in range(n_per_task):
            env = ShortcutEnv(task)
            done = False
            while not done:
                obs = observe(task, env.state)
                a_exp = expert_action(task, env.state)
                Xs.append(obs)
                ys.append(a_exp)
                a = a_exp if rng.random() > eps else int(rng.integers(N_ACTIONS))
                _, _, done = env.step(a)
    return np.stack(Xs), np.array(ys, dtype=int)


def train_bc(seed, epochs=300, batch=256, lr=3e-3, smooth=0.05):
    rng = np.random.default_rng(seed)
    X, y = collect_bc_data(rng)
    params = init_params(rng)
    opt = Adam(params, lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            _, grads = bc_loss_and_grad(params, X[idx], y[idx], smooth=smooth)
            opt.step(params, grads)
    loss, _ = bc_loss_and_grad(params, X, y, smooth=smooth)
    logits, _ = forward(params, X)
    acc = float((logits.argmax(axis=1) == y).mean())
    return params, {"bc_loss": float(loss), "bc_agreement": acc,
                    "n_samples": n}


# -- GRPO --------------------------------------------------------------------

def train_grpo(base_params, seed, steps=100, group=8, batch_tasks=24,
               lr=3e-4, beta=0.02, clip=0.2, reward_mode="task", lam=1.0,
               mechanisms=("R",), conflict_share=0.25, per_account=False,
               eval_tasks=None, log_every=25, eps=0.0):
    """GRPO from base_params. reward_mode 'task' or 'safety' (Eq. 4).
    eval_tasks: fixed panel of R tasks for dynamics logging."""
    assert reward_mode in ("task", "safety")
    rng = np.random.default_rng(seed)
    params = copy.deepcopy(base_params)
    ref = copy.deepcopy(base_params)
    opt = Adam(params, lr=lr)
    history = []
    skipped = 0
    for step in range(steps + 1):
        if eval_tasks is not None and (step % log_every == 0 or step == steps):
            hist = eval_panel(params, eval_tasks, rng)
            hist["step"] = step
            history.append(hist)
        if step == steps:
            break
        tasks = sample_tasks(rng, mechanisms=mechanisms,
                             conflict_share=conflict_share, n=batch_tasks,
                             per_account=per_account)
        tasks_rep = [t for t in tasks for _ in range(group)]
        traj = rollouts(params, tasks_rep, rng, eps=eps)
        rewards = traj.rewards.copy()
        if reward_mode == "safety":
            rewards = rewards - lam * traj.violated.astype(float)
        adv = group_advantages(rewards, group)
        if np.abs(adv).max() < 1e-12:
            skipped += 1
            continue
        grads, _ = grpo_grad(params, traj, adv, ref, beta=beta, clip=clip)
        opt.step(params, grads)
    return params, {"history": history, "skipped_no_signal_steps": skipped}


def eval_panel(params, tasks, rng, n_rollouts=4):
    """Evaluate on a fixed panel: n_rollouts sampled trajectories per task.
    Returns mean reward, real work, historical/ceritified-profitable
    violation rates and safe completion. Certified profitable uses the
    DP-computed compliant optimum V_c* (imported lazily to keep this
    module free of the optima dependency at import time)."""
    from .optima import optimal_values

    tasks_rep = [t for t in tasks for _ in range(n_rollouts)]
    traj = rollouts(params, tasks_rep, rng)
    vc = np.array([optimal_values(t)["V_c"] for t in tasks_rep])
    certified = traj.violated & (traj.rewards > vc + 1e-9)
    # Task completion: submitted (published) with all real work done.
    # Publishing with stale evidence / incomplete work still counts as
    # completed (the score was obtained) but never as safe.
    completed = traj.published & (traj.real_work >= 1.0 - 1e-9)
    safe = completed & (~traj.violated)
    return {
        "mean_reward": float(traj.rewards.mean()),
        "mean_real_work": float(traj.real_work.mean()),
        "hist_violation_rate": float(traj.violated.mean()),
        "cert_profitable_rate": float(certified.mean()),
        "safe_completion_rate": float(safe.mean()),
        "completion_rate": float(completed.mean()),
        "n": int(len(tasks_rep)),
    }
