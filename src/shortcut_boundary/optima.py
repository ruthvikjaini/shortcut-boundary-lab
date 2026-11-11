"""Exact optima by exhaustive search over the deterministic state graph.

For a task, compute:
  V_c* -- best terminal reward over trajectories with no historical violation
  V_s* -- best terminal reward over trajectories with a historical violation
  lambda_x = V_s* - V_c* (paper Eq. 5: the compliant optimum is strictly
  preferred under the binary penalty iff lambda > lambda_x)

P1 (compliant feasibility) holds iff a compliant trajectory reaches the
value baseline (V_c* >= 0 in our reward units, where the baseline is the
reward of the compliant reference). P2 (reward conflict) holds iff
V_s* > V_c*. Both are certified per task by this module, and the evaluation
code uses V_c* to certify profitable violations: a violation is certified
profitable iff its reward strictly exceeds V_c*.
"""

from __future__ import annotations

from .envs import N_ACTIONS, initial_state, step_state

_CACHE = {}


def optimal_values(task):
    """DFS with memoization over (state). The state includes steps, so the
    graph is a DAG. Returns dict with V_c*, V_s* (None if no violating
    trajectory exists), lambda_x (None if no conflict), and the argmax
    action sequences for both classes. Results are cached per task key:
    evaluation calls this once per (task, panel) otherwise."""
    key = task.key()
    if key in _CACHE:
        return _CACHE[key]
    memo = {}

    def best(s):
        """(best compliant value, best violating value) reachable from s.
        Each is -inf if no trajectory of that class terminates from s."""
        if s in memo:
            return memo[s]
        neg = float("-inf")
        bc, bv = neg, neg
        for a in range(N_ACTIONS):
            s2, r, done = step_state(task, s, a)
            if done:
                if s2[7]:  # violated
                    bv = max(bv, r)
                else:
                    bc = max(bc, r)
            else:
                cc, cv = best(s2)
                bc = max(bc, cc)
                bv = max(bv, cv)
        memo[s] = (bc, bv)
        return memo[s]

    vc, vs = best(initial_state(task))
    lam = None if vs == float("-inf") else vs - vc
    out = {
        "V_c": vc,
        "V_s": vs if vs != float("-inf") else None,
        "lambda_x": lam,
        "conflict": lam is not None and lam > 0,
        "n_states": len(memo),
    }
    _CACHE[key] = out
    return out


def compliant_reference_value(task):
    """Reward of the compliant expert reference trajectory (paper's tau_c)."""
    from .envs import expert_action

    s = initial_state(task)
    done = False
    r = 0.0
    while not done:
        s, r, done = step_state(task, s, expert_action(task, s))
    return r
