# Frozen VLA Representation & Residual Control — Research Log

> **Status:** exploratory engineering study; **not** a validated improvement over a frozen VLA.
> **Scope:** π₀.₅ (ManiSkill SFT), RLinf, ManiSkill, geometry probing, GT-pose residual TD3, scripted control.
> **Record date:** 2026-11-24. Source: experiment outputs and code excerpts shared in the project conversation. Numerical claims should be cross-checked with the on-disk repository using `CODEX_AUDIT_PROMPT.md`.

## 1. Central question

**Do frozen VLA representations help reinforcement-learning adaptation under distribution shift?**

We split the question into two independent tests:

1. **Representation:** Is geometric information (object–gripper relative position) decodable from internal π₀.₅ features, including OOD environments?
2. **Control / positive control:** Can a residual controller improve frozen π₀.₅ behavior **even when given ground-truth pose**? Only after this control works should we compare VLA latent against generic vision/state inputs for RL.

This ordering matters. Failure of the GT-pose residual TD3 **does not refute** the representation hypothesis; it instead points to the RL algorithm, action parameterization, reward, data scale, controller semantics, or their interaction.

## 2. Evaluation setup

| Item | Configuration |
|---|---|
| Base policy | Frozen π₀.₅ ManiSkill SFT checkpoint |
| Simulator / framework | ManiSkill 3 / RLinf |
| Task | PutOnPlateInScene25Main-v3 |
| OOD (vision) | PutOnPlateInScene25VisionTexture05-v1 |
| OOD (execution) | PutOnPlateInScene25EEPose-v1 |
| Episode horizon | 80 environment steps |
| Action chunk | 5 actions / chunk; 16 chunks / episode |
| Typical evaluation | 20 parallel episodes, 1 rollout epoch |
| Fixed IDs | `EVAL_FIXED_RESET_IDS=true` |
| Primary metric | `success_once` with `success_at_end` and `return` |

**Confounding:** Main uses `obj_set=train`, while the selected OOD variants use `obj_set=test`; main-vs-OOD differences are not necessarily perturbation-only. Historical Frozen OOD evaluations used 50 episodes; later experiments mostly used 20. Comparisons are descriptive unless matching evaluation conditions are verified.

## 3. Summary of what was established

- Frozen policy performance drops between Main and the chosen OOD tasks (historical 50-episode `success_once`: Main **66%**, Vision **34%**, Execution **26%**).
- Internal π₀.₅ features contain decodable geometry: ID position error **2.90 cm** from π₀.₅ alone, **2.87 cm** with robot state, vs robot state alone **3.68 cm**. OOD probe with π₀.₅+state recorded Vision **4.52 cm**, Execution **4.86 cm** (state-only: **5.03** / **8.10 cm**).
- Residual TD3 was integrated and successfully trained online (completed 10-rollout run: **3,200 chunk transitions**). This **did not establish improved control performance**.
- The conservative TD3 actor became practically a constant `[+2,+2,+2] mm` vector: on 320 sampled chunk states, mean Euclidean distance to that constant was **0.000034 mm**, maximum **0.000699 mm**.
- Scripted GT relative-position correction affected outcomes: Vision 20 episodes, Frozen **30%**, GT scripted **15% (2 mm), 40% (5 mm), 45% (10 mm)** in `success_once`. These are exploratory one-run results; `success_at_end` did not improve consistently.
- **No VLA-latent-driven residual RL experiment was completed.** The RL policy used GT pose + qpos + VLA *output action* (21D), not the 2048D π₀.₅ internal representation.

## 4. Documentation index

| Document | Purpose |
|---|---|
| [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) | Full results tables, comparisons, sample size, numerical provenance |
| [`METHOD_AND_IMPLEMENTATION.md`](METHOD_AND_IMPLEMENTATION.md) | Architecture, state/action/reward/transition, engineering implementation |
| [`DIAGNOSTICS_AND_LESSONS.md`](DIAGNOSTICS_AND_LESSONS.md) | Hypotheses, failed attempts, actor saturation, lessons, limitations |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Directory map, evaluation invocation templates, audit checklist |
| [`CODEX_AUDIT_PROMPT.md`](CODEX_AUDIT_PROMPT.md) | Prompt for Codex to ground these notes in the real repository |

## 5. Narrative for a KIST internship application

The strongest verifiable story is **implementation and rigorous diagnosis**, not a claimed novel algorithm: linked a frozen VLA to a simulator, extracted features for geometry probes, implemented online residual TD3 and action interception, compared against non-learning controls, discovered and quantified output saturation, then tested a simple GT-based controller to isolate whether the action pathway could affect task success.

**Avoid claiming** (i) validated OOD improvement, (ii) that TD3 learned meaningful state-conditioned corrections, or (iii) that the VLA latent improved RL. The data do not support those conclusions.