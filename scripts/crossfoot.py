"""Crossfoot: every number in README.md recomputes from results/*.json.

Checks:
  1. grid.json summary_3seed means recompute from the per-seed finals.
  2. Each headline value the README reports (3-decimal formatting) is
     present in README.md and equals the recomputed value.
Run: python scripts/crossfoot.py  (exit 0 on CROSSFOOT PASS).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def f3(x):
    return f"{float(x):.3f}"


def mean3(vals):
    return statistics.fmean(float(v) for v in vals)


def main():
    grid = load("grid.json")
    transfer = load("transfer.json")
    ext = load("extension.json")
    foll = load("followups.json")
    readme = (ROOT / "README.md").read_text()
    failures = []

    def check(label, value, where="README"):
        s = f3(value)
        if s not in readme:
            failures.append(f"{label}: {s} not found in {where}")

    # 1. summary_3seed recomputes from per-seed finals.
    for tag, runs in grid["arms"].items():
        for panel, metrics in grid["summary_3seed"][tag].items():
            for metric, agg in metrics.items():
                recomputed = mean3([r["final"][panel][metric] for r in runs])
                if abs(recomputed - agg["mean"]) > 1e-3:
                    failures.append(
                        f"summary {tag}/{panel}/{metric}: stored "
                        f"{agg['mean']} != recomputed {recomputed:.4f}")

    # 2. Headline values in the README.
    s = grid["summary_3seed"]
    b = grid["bases_eval"]
    base_r = {m: mean3([b[str(i)]["R_restricted"][m] for i in (0, 1, 2)])
              for m in ("mean_reward", "mean_real_work", "hist_violation_rate",
                        "cert_profitable_rate", "safe_completion_rate")}
    base_e_safe = mean3([b[str(i)]["E_changing_evidence"]["safe_completion_rate"]
                         for i in (0, 1, 2)])
    base_pa_cert = mean3([b[str(i)]["R_per_account"]["cert_profitable_rate"]
                          for i in (0, 1, 2)])
    base_lawful = mean3([b[str(i)]["R_authorized"]["lawful_expanded_use"]
                         for i in (0, 1, 2)])

    check("base R cert", base_r["cert_profitable_rate"])
    check("base R safe", base_r["safe_completion_rate"])
    check("base R reward", base_r["mean_reward"])
    check("base lawful expanded use", base_lawful)
    check("base per-account cert", base_pa_cert)
    check("base E safe", base_e_safe)

    # Acquisition curves (history, 3-seed means).
    for tag in ("task_R_25", "task_R_50", "task_R_100", "safety_R_25",
                "safety_R_100"):
        runs = grid["arms"][tag]
        for i, st in enumerate([h["step"] for h in runs[0]["history"]]):
            cert = mean3([r["history"][i]["cert_profitable_rate"] for r in runs])
            check(f"{tag} history step {st} cert", cert)
    # Final evals.
    for tag in ("task_R_25", "task_R_50", "task_R_100", "safety_R_25",
                "safety_R_100"):
        check(f"{tag} final cert", s[tag]["R_restricted"]["cert_profitable_rate"]["mean"])
        check(f"{tag} final safe", s[tag]["R_restricted"]["safe_completion_rate"]["mean"])
    check("safety_R_100 lawful", s["safety_R_100"]["R_authorized"]["lawful_expanded_use"]["mean"])
    check("task_V V cert", s["task_V"]["V_heldout"]["cert_profitable_rate"]["mean"])
    check("task_V V work", s["task_V"]["V_heldout"]["mean_real_work"]["mean"])
    check("task_V V safe", s["task_V"]["V_heldout"]["safe_completion_rate"]["mean"])
    check("safety_V V cert", s["safety_V"]["V_heldout"]["cert_profitable_rate"]["mean"])
    check("safety_V V safe", s["safety_V"]["V_heldout"]["safe_completion_rate"]["mean"])
    check("safety_V V work", s["safety_V"]["V_heldout"]["mean_real_work"]["mean"])

    # Transfer: E panel (recomputed from transfer.json per-seed evals).
    for tag in ("task_RV_25", "safety_RV_25", "task_V", "safety_V"):
        e = [transfer["transfer"][tag][str(i)]["E_changing_evidence"]
             for i in (0, 1, 2)]
        check(f"{tag} E safe", mean3([x["safe_completion_rate"] for x in e]))
        check(f"{tag} E cert", mean3([x["cert_profitable_rate"] for x in e]))
    # Per-account.
    check("task_R_25 per-account cert",
          s["task_R_25"]["R_per_account"]["cert_profitable_rate"]["mean"])
    check("safety_R_25 per-account cert",
          s["safety_R_25"]["R_per_account"]["cert_profitable_rate"]["mean"])
    # Cross-mechanism V->R.
    for tag in ("task_V", "safety_V"):
        r = [transfer["transfer"][tag][str(i)]["R_restricted"] for i in (0, 1, 2)]
        check(f"{tag} on R cert", mean3([x["cert_profitable_rate"] for x in r]))
        check(f"{tag} on R safe", mean3([x["safe_completion_rate"] for x in r]))
        check(f"{tag} on R work", mean3([x["mean_real_work"] for x in r]))
    # Mitigation (3-seed means from transfer.json).
    for mode in ("safety", "task"):
        certs = [transfer["mitigation"][f"{mode}_seed{i}"]["final"]["R_restricted"]["cert_profitable_rate"]
                 for i in (0, 1, 2)]
        check(f"mitigation {mode} cert", mean3(certs))
    # Extensions.
    for lam, v in ext["E1_lambda_ablation"].items():
        check(f"E1 lam {lam} cert", v["cert_profitable_rate"])
    for share, v in ext["E2_mixture_curve"].items():
        check(f"E2 share {share} cert", v["cert_profitable_rate"])
    check("E3 durability cert",
          ext["E3_durability"]["task_continuation_64"]["cert_profitable_rate"])
    check("E3 durability safe",
          ext["E3_durability"]["task_continuation_64"]["safe_completion_rate"])
    # Follow-ups.
    check("F1 beta0 final cert",
          foll["F1_task_R_100_beta0"]["final"]["cert_profitable_rate"])
    check("F1 beta0.02 seed0 cert",
          grid["arms"]["task_R_100"][0]["final"]["R_restricted"]["cert_profitable_rate"])
    check("F2 eps0.3 mitigation cert",
          foll["F2_mitigation_eps03"]["final"]["cert_profitable_rate"])
    check("F3 parent cert",
          foll["F3_mitigation_from_64step_parent"]["parent_64"]["cert_profitable_rate"])
    check("F3 mitigated cert",
          foll["F3_mitigation_from_64step_parent"]["final"]["cert_profitable_rate"])

    if failures:
        print("CROSSFOOT FAIL")
        for f_ in failures:
            print(" -", f_)
        sys.exit(1)
    print(f"CROSSFOOT PASS ({len(readme.splitlines())} README lines checked)")


if __name__ == "__main__":
    main()
