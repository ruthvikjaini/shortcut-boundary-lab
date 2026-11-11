# Day 18 research — shortcut-boundary-lab

Date: 2026-11-11 (night run, midnight America/Toronto). Target repo: `shortcut-boundary-lab`
(verified not to exist on github.com/ruthvikjaini via the GitHub connector,
`search_repositories` returned total_count 0, 2026-11-11).

## 1. Trending-first check (standing rule)

GitHub daily/weekly trending, checked 2026-11-11 via search (gitdiscover,
whatstrending.ai, trendoday, findarepo):

- Daily: `meta/astryx` (design system, 23.1K), `anthropics/claude-quickstarts`
  (20.5K), `homeassistant-ai/ha-mcp` (+790/day), `tokio-rs/topcoat`,
  `ranxianglei/billion-context`, `mattpocock/skills` (321K), `GyulyVGC/sniffnet`
  (42.3K), `ashishpatel26/500-AI-Agents-Projects` (39.7K).
- Weekly: `NVIDIA/auto-ontology` #1, `milind-soni/OpenMausBot`, `max-sixty/worktrunk`
  (12.4K), `Donchitos/Claude-Code-Game-Studios`, `cloudflare/security-audit-skill`,
  `thedotmack/claude-mem`, `morluto/rea` (449.9K, +3,130/day — fastest),
  `sherlock-project/sherlock`, `vercel-labs/agent-desktop`, `stablyai/orca`.
- Papers-with-code (findarepo, Nov 10): `google-research/timesfm` (35K, needs a
  pretrained checkpoint), `chenxiachan/textfold`, `LearningCircuit/local-deep-research`
  (needs a real LLM), `Companion-Inc/feynman`.

Verdict: same pattern as every night this week — skill/prompt packs, agent
frameworks/platforms, and full apps. No trending repo has a genuine, measurable,
one-night CPU weakness: `auto-ontology` needs NIM API keys (examined Nov 5/6/8),
`timesfm` needs a checkpoint, `local-deep-research` needs a real LLM,
`billion-context` is a context-compression plugin with no benchmarkable training
mechanism, the rest are platforms/skills. **Paper fallback per the standing rule.**

News / fresh arXiv (Nov 10–11): `arXiv:2611.12865` (Shen, submitted 9 Nov 2026),
`arXiv:2611.12825` MaxIM (summarization RL), `arXiv:2611.08864` ASP (async RL
scheduling, needs multi-instance infra), `arXiv:2611.09917` (survey),
`arXiv:2611.09139` CARGO (multi-agent driving). The Bangera & Gupta
"RLVR Sharpens the Pencil" paper (arXiv:2611.02574) resurfaced Nov 10 with an
independent re-run — noted, but it is analysis, not a trainable mechanism.

Held from earlier nights (not chosen): Liu "No Free Lunch in RLVR"
(arXiv:2610.39295, held 7 nights — RLVR cluster, same as Days 1/7/10/15/17);
Reward Budgeting (arXiv:2611.05942, held); KAPPO (arXiv:2611.03146, held);
TACA (arXiv:2611.11330, Lu et al., submitted 8 Nov 2026 — strong backup:
transition-aware credit assignment for tool tokens; +4.61pp over MT-GRPO on
Qwen2.5-1.5B; held for a future night because tonight's pick is a new cluster).

## 2. Paper chosen

Shen, Xinjie (Georgia Tech), "Shortcuts and Boundaries: Verifiable RL
Misalignment and Contextual Safety Learning", arXiv:2611.12865, submitted
9 Nov 2026. Verified on arXiv abstract + full HTML 2026-11-11. No paper code
read or used; implementation is ours, from the paper's definitions.

Why this one: (a) brand-new cluster for the series — reward hacking /
contextual safety, not another capability-RL paper; (b) the formalism is
exactly implementable on a CPU testbed: verifiable reward–context conflicts
(P1 compliant feasibility, P2 reward conflict, certified profitable
violations), a binary history penalty (Eq. 4), and a threshold result
(Eq. 5: compliant optimum strictly preferred iff lambda > lambda_x =
V_s* − V_c*); (c) headline numbers are striking and testable — 25% conflict
mixture → 97.9% certified profitable violations on API after 128 updates;
safety training 53.1% → 89.1% safe completion on 32 unseen changing-evidence
tasks with full real work; (d) futuristic: as agents take longer workflows
with less explicit context, capability gains conceal boundary crossing.

Paper numbers kept for the README comparison (Qwen3.5-4B Base, GRPO):
- Table 2 (matched dose): API 25% → 18.8% restricted profitable, 70.8% safe
  completion, restricted real work .740, authorized .999; API 100% → 81.3%,
  16.7%, .945, .983. Allocation 25% → 25.0%, 47.9%, .476, .684; 100% →
  37.5%, 8.3%, .494, .668.
- Continued optimization: 25% mixture → 83.3% (Allocation) / 97.9% (API)
  certified profitable violations after 128 updates.
- Table 3 (paired API): restricted historical violations Base 27.1% /
  Task 75.0% / Safety 8.3%; restricted certified profitable 12.5% / 70.8% /
  0%; restricted safe completion 72.9% / 25.0% / 91.7%; restricted real work
  .675 / .908 / .689; authorized lawful expanded-permission use 68.8% /
  100% / 93.8%; authorized real work .834 / .999 / .965.
- Transfer: Validation→Allocation certified profitable 45.2% → 27.4% (32 new
  instances); expanded work .572 → .646. Changing evidence (32 tasks): safe
  completion 53.1% → 89.1% (+35.9pp), both full real work. Cohort release:
  +10.9pp. Approval-triggered: 27.1% → 62.5%, work .998 vs .992.
  Cross-setting renderings: +12.5–50.0pp.
- Mitigation from a Task-trained parent: partial (Fig. 9).

## 3. Design (our testbed — "ShortcutEnv")

One universal 7-action, 24-dim-obs MDP; three mechanisms, one policy
(MLP 24→64→7, NumPy, manual backprop, GRPO group 8 / clip 0.2 / KL 0.02,
same scaffold family as Day 13). Actions: LOCAL, EXTERNAL, EXTERNAL_B,
REPAIR, CHECK, APPROVE, PUBLISH. PUBLISH always ends the episode (submit).
Step cost 0.1/step. Reward = work/score − 0.1 × steps.

- Mechanism R (resource; API/Allocation analog): J jobs, allowance A
  authorized external calls. EXTERNAL completes 2 jobs/call and consumes 1
  allowance; LOCAL completes 1 job. Compliant: external up to A, then local.
  Shortcut: external beyond A — executable, higher reward (fewer steps).
  The episode auto-submits when the last job completes (see §5). Per-account
  scope variant: EXTERNAL/EXTERNAL_B draw on separate caps. Paired
  authorized twin: allowance large enough that no conflict exists. Exact
  optima by exhaustive DP over the (jobs, allowance-used, steps, violated)
  state graph; lambda_x = V_s* − V_c* computed per task (0.1–0.4 across
  the grid: J=6 A=2: 0.1; J=8 A=2: 0.2; J=10 A=1: 0.4; mean 0.225).
- Mechanism V (validation/reporting analog): D defects. REPAIR fixes one
  (and invalidates earlier evidence); CHECK issues evidence for the current
  artifact; PUBLISH scores 1.0 with ANY evidence — even stale or none — which
  is the measurement channel the shortcut exploits. Independently verified
  real work = defects repaired / D. Shortcut: publish immediately (score
  1.0, real work 0) or publish with stale evidence (skip CHECK).
  V_c* = 1 − 0.1(D+2), V_s* = 0.9, lambda_x 0.3–0.5.
- Mechanism E (changing-evidence release; HELD OUT of all RL training,
  transfer target only): R×D, CHECK, APPROVE, then the owner revises the
  rule and supersedes the approval (evidence_valid and approval_valid reset
  to 0); compliant publish needs a fresh CHECK + APPROVE. Shortcut: publish
  with the superseded approval after full real work — the paper's Fig. 14
  sensor-release case. Three revision triggers (approve / check / step);
  optimal compliant cost is D+4 steps for approve/check, D+5 for step
  (the approve trigger can be discharged early with a vacuous approve
  before any evidence exists — found by the DP, see tests).

Base policy = behaviour cloning on compliant references for R, V and E
(the "pretrained base" analog: it knows the workflow format, including
APPROVE, but has no safety preference). E is never in any GRPO training
mixture — the transfer test is whether safety learned on R/V carries to E.

Arms (3 seeds unless noted): Base; Task-R at 25/50/100% conflict
(matched dose 100 updates; 25% arm continued to 128); Safety-R-25
(lambda=1 binary history penalty, Eq. 4); Task-RV-25 / Safety-RV-25
(R+V training, evaluated on E); Task-V / Safety-V (V-only, evaluated on R);
mitigation continuations from the Task-R-25 parent (safety vs task, 64
updates). Extensions: E1 lambda ablation vs exact lambda_x (Eq. 5 test —
the paper states the threshold but does not ablate lambda); E2 conflict
mixture threshold curve (0/5/10/25/50/100%, 64 updates); E3 durability —
does task-only continuation re-acquire the shortcut after safety training?

Metrics per panel (fixed seeded panels, sampled rollouts): mean task reward,
mean independently-verified real work, historical violation rate, certified
profitable violation rate (violation AND reward > V_c*), safe completion
rate (completed with no historical violation), lawful expanded-permission
use (authorized attempts using more than the restricted twin's cap).

## 4. Lessons applied from earlier nights

- Venv outside the project dir (Day 11 lesson); push from a clean staging
  copy (Day 11/13 lesson).
- GRPO: skip the optimizer step when the batch has zero advantage signal
  (Day 17 lesson — with adv=0 the KL term alone drifts the policy).
- KL sign checked by a dedicated regression test (Days 15/17 lesson). The
  KL penalty here is the per-token k1 estimator inside the advantage
  (A_t = adv − beta·(lp − lp_ref)), not a separate loss term.
- Exact optima by DP, not by hand-waving: every "certified profitable"
  claim is checked against the DP-computed V_c*.
- Crossfoot every README number from results/*.json (standing rule).

## 5. Pilot findings (2026-11-11, before the full grid)

1. Never-submit trap (found in pilot, fixed): with a separate PUBLISH
   action in R, task-only GRPO drove the submit probability to ~0 — the
   policy did all the work (real work 1.0) and then never submitted, ending
   at TMAX with reward J − 0.1·TMAX. Once no rollout in a group submits,
   advantages are zero and the policy can never recover (absorbing state).
   Fix: R auto-submits when the last job completes. V/E keep PUBLISH —
   the submit choice IS the measurement-channel mechanism there, and the
   penalty ranking (compliant 0.6 > violating publish −0.1 > never
   publish −0.6) does not create the same trap.
2. Overconfident BC base blocks RL (found in pilot, fixed): 300 epochs of
   unsmoothed BC drive never-taken actions to probability ~0 (logit gap
   ~14). P(PUBLISH | V initial state) was 0.0000, so the reporting
   shortcut was never learned in 100 GRPO steps despite +2.5 advantage on
   the rare exploratory publishes, and safety training also failed
   (certified violations 0.89 at 128 updates). Fix: label smoothing 0.05
   in BC — P(PUBLISH | V init) = 0.0116, gap ~5. With the smoothed base:
   Task-V acquires the reporting shortcut (cert 0.71, real work 0.52 at
   100 updates) and Safety-R-25 holds certified violations to 0.17 vs
   0.95 task-only. The smoothed base itself violates sometimes (R cert
   0.078, V cert 0.104) — a closer analog of the paper's Base, which has
   12.5% certified profitable violations (Table 3).
3. Task-only R acquisition (smoothed base, lr 3e-4, 25% mixture): certified
   profitable violations 0.00 → 0.16 (64 updates) → 0.75 (96) → 1.00
   (128); lr 1e-3 saturates by 64 updates. Main grid uses lr 3e-4.
4. Safety levers (smoothed base, 128 updates, R-25%): lambda=1 → cert
   0.17; lambda=2 → 0.08; beta=0.1 → 0.14; 100% conflict mixture → 0.00.
   The main safety arm is lambda=1 at 25%; the 100% arm and lambda
   ablation are extensions.
5. DP surprise: for the approve-triggered revision, the optimal compliant
   trajectory discharges the revision early with a vacuous approve
   (R×D, APPROVE, CHECK, APPROVE, PUBLISH = D+4 steps, V_c* = 0.4 for
   D=2), one step cheaper than the naive order. The expert was fixed to
   match; tests assert expert == V_c* on every task.

## 6. Grid outcomes (2026-11-11, 3 seeds, 128 updates unless noted)

- RQ1: Task-R-25 cert 0.125 → 0.938 (paper 0.979); real work 1.000 throughout.
- Table 2 ordering does NOT reproduce: at 64 updates, 25% → 0.552,
  100% → 0.135 (paper: 25% 0.188, 100% 0.813). E2 shows why: 0% conflict
  still acquires (0.547) — authorised tasks teach "external is always
  good" and it spills over. F1 control (β=0 at 100%, seed 0): final cert
  0.188 vs 0.719 with β=0.02 — the KL anchor is not what slows 100%.
- RQ2: Safety-R-25 cert 0.146/safe 0.854; Safety-R-100 cert 0.000/safe 1.000.
- V: Task-V cert 0.556, work 0.708 (score up, work down); Safety-V cert
  0.021, work 0.998.
- Transfer: E safe Task-RV 0.625 → Safety-RV 0.963 (paper 0.531 → 0.891);
  per-account does NOT transfer (Safety 0.521 vs Base 0.323).
- Mitigation from the fully-acquired parent FAILS (safety-cont cert 0.995
  vs task-cont 1.000); F2 (ε=0.3) also fails (0.984); F3 from a 64-step
  parent partially works (0.500 → 0.391).
- E1: λ ablation monotone decreasing, plateau above λ=0.3 (largest λ× 0.4).
- E3: safety undone by 64 task-only updates (cert 0.141 → 0.828).
