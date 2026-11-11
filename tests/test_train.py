"""Policy: gradient correctness, shapes, Adam; train: KL sign, advantages."""

import copy

import numpy as np

from shortcut_boundary.envs import N_ACTIONS, OBS_DIM, Task
from shortcut_boundary.policy import (
    Adam,
    bc_loss_and_grad,
    forward,
    init_params,
    log_probs,
)
from shortcut_boundary.train import group_advantages, grpo_grad, rollouts


def test_bc_grad_finite_differences():
    rng = np.random.default_rng(0)
    params = init_params(rng)
    X = rng.normal(0, 1, (8, OBS_DIM))
    y = rng.integers(0, N_ACTIONS, 8)
    _, grads = bc_loss_and_grad(params, X, y)
    eps = 1e-6
    for k in ("W1", "b1", "W2", "b2"):
        idx = tuple(rng.integers(0, s) for s in params[k].shape)
        orig = params[k][idx]
        params[k][idx] = orig + eps
        lp, _ = bc_loss_and_grad(params, X, y)
        params[k][idx] = orig - eps
        lm, _ = bc_loss_and_grad(params, X, y)
        params[k][idx] = orig
        num = (lp - lm) / (2 * eps)
        assert abs(num - grads[k][idx]) < 1e-4, (k, num, grads[k][idx])


def test_forward_shapes_and_logprobs_sum_to_one():
    rng = np.random.default_rng(1)
    params = init_params(rng)
    X = rng.normal(0, 1, (5, OBS_DIM))
    logits, _ = forward(params, X)
    assert logits.shape == (5, N_ACTIONS)
    lp = log_probs(params, X)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)


def test_adam_decreases_bc_loss():
    rng = np.random.default_rng(2)
    params = init_params(rng)
    X = rng.normal(0, 1, (64, OBS_DIM))
    y = rng.integers(0, N_ACTIONS, 64)
    opt = Adam(params, lr=1e-2)
    l0, _ = bc_loss_and_grad(params, X, y)
    for _ in range(50):
        _, g = bc_loss_and_grad(params, X, y)
        opt.step(params, g)
    l1, _ = bc_loss_and_grad(params, X, y)
    assert l1 < l0


def test_group_advantages_zero_signal():
    adv = group_advantages(np.array([1.0, 1.0, 1.0, 1.0]), 4)
    assert np.all(adv == 0.0)
    adv = group_advantages(np.array([0.0, 1.0, 2.0, 3.0]), 4)
    assert abs(adv.mean()) < 1e-12
    assert adv[0] < 0 < adv[3]


def _kl_to_ref(params, ref_params, X):
    lp = log_probs(params, X)
    lpr = log_probs(ref_params, X)
    p = np.exp(lp)
    return float((p * (lp - lpr)).sum(axis=1).mean())


def test_grpo_kl_sign_pulls_toward_ref():
    """Regression test (Days 15/17 lesson): with zero advantage the per-token
    k1 KL penalty must pull the policy back toward the reference. A flipped
    sign makes the KL explode instead."""
    rng = np.random.default_rng(3)
    ref = init_params(rng)
    params = copy.deepcopy(ref)
    # Perturb the policy away from the reference.
    params["W2"] = params["W2"] + rng.normal(0, 0.5, params["W2"].shape)
    tasks = [Task("R", 8, 2, 0) for _ in range(8)]
    adv = np.zeros(len(tasks))
    X = rng.normal(0, 1, (64, OBS_DIM))
    kl0 = _kl_to_ref(params, ref, X)
    assert kl0 > 0.01
    opt = Adam(params, lr=1e-2)
    for _ in range(20):
        traj = rollouts(params, tasks, rng)
        grads, _ = grpo_grad(params, traj, adv, ref, beta=0.5, clip=0.2)
        opt.step(params, grads)
    kl1 = _kl_to_ref(params, ref, X)
    assert kl1 < kl0, (kl0, kl1)


def test_rollouts_shapes_and_rewards():
    rng = np.random.default_rng(4)
    params = init_params(rng)
    tasks = [Task("R", 6, 1, 0), Task("V", 2)]
    traj = rollouts(params, tasks, rng)
    assert traj.rewards.shape == (2,)
    assert traj.obs.shape[1] == max(t.tmax for t in tasks)
    assert traj.mask.sum(axis=1).min() >= 1
    # Rewards are terminal task rewards: R in [-1, 6], V in [-1, 1].
    assert -1.0 <= traj.rewards[0] <= 6.0
    assert -1.0 <= traj.rewards[1] <= 1.0
