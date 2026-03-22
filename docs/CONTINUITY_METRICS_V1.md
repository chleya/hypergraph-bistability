# Continuity Metrics v1

## Purpose

The project has moved beyond pure retrieval-recall validation.

The next evaluation layer should reflect the real product claim:

- the agent should continue work
- preserve blockers and critical issue context
- preserve decisions and next steps
- avoid restarting from scratch when continuity information is already available

This first version adds lightweight continuity metrics on top of the existing eval suite without requiring a new scenario annotation format.

## Metrics

### 1. `task_continuation`

Definition:

- fraction of evaluated turns where the system both:
  - retrieves at least one expected context item
  - and uses at least one expected response signal, if response signals were defined

Interpretation:

- measures whether the agent actually resumes the task instead of only retrieving context silently

### 2. `blocker_preservation`

Definition:

- among evaluated turns that include blocker-like expected signals, fraction where the matched retrievals or response signals preserve at least one blocker signal

Current blocker-like cues include:

- `bug`
- `failure`
- `rollback`
- `stale`
- `backoff`
- `timeout`
- `migration`
- `cache invalidation`
- `checkpoint`
- `error`
- `issue`
- `root cause`

Interpretation:

- proxy for whether the agent keeps the important obstacle or failure mode in working context

### 3. `decision_continuity`

Definition:

- among evaluated turns that include decision/plan-like expected signals, fraction where the matched retrievals or response signals preserve at least one decision-oriented signal

Current decision-like cues include:

- `plan`
- `next step`
- `rollback`
- `migration notes`
- `checklist`
- `decision`
- `inspect`
- `patch`
- `summary`

Interpretation:

- proxy for whether the agent continues the work structure, not only the raw issue memory

### 4. `repeated_work_avoidance_proxy`

Definition:

- fraction of evaluated turns where retrieval hit exists and the assistant response does not look like a restart/fallback prompt

Current restart cues include:

- `what aspect interests you most`
- `how can i help`
- `tell me more`
- `what would you like to explore`
- `what should we focus on`

Interpretation:

- proxy for whether the system avoids throwing the user back into re-explaining already-known context

## Scope

This v1 is intentionally heuristic.

It is designed to:

- be cheap to add
- expose continuity gaps earlier than recall alone
- prepare for later, stronger continuity metrics

It is not intended to be the final continuity evaluation framework.

## Recommended Use

- track continuity metrics next to recall/precision/response recall
- use them to identify where retrieval is correct but task continuation still feels weak
- treat low `repeated_work_avoidance_proxy` as a signal that response contracts or action policies need work

## Dedicated Regression Set

Continuity regression is now run against a dedicated scenario subset rather than the full `stress` tier.

Current regression set:

- `debugging_resume_with_preference`
- `blocker_resume_no_reexplaining`
- `task_continuity`
- `decision_resume_after_interruption`
- `artifact_chain_resume`
- `plan_resume_without_restart`
- `contradiction_link_resume`
- `conflict_unit_dominance`

Current command:

```powershell
python -m hypergraph_bistability.cli run-continuity-regression --output _continuity_regression_v3.json
```

Current snapshot:

- runtime
  - `task_continuation = 1.000`
  - `blocker_preservation = 1.000`
  - `decision_continuity = 1.000`
  - `repeated_work_avoidance_proxy = 1.000`

- baseline
  - `task_continuation = 0.000`
  - `blocker_preservation = 0.875`
  - `decision_continuity = 1.000`
  - `repeated_work_avoidance_proxy = 0.000`

Interpretation:

- the dedicated continuity regression now separates runtime from baseline on actual continuation behavior
- the remaining differentiator is no longer blocker or decision recall alone, but whether the system avoids restart-style behavior while preserving structure

## Default Runtime Gate

Continuity regression should now be treated as a default gate for runtime changes.

At minimum, changes to retrieval or response contracts should not regress:

- `task_continuation`
- `blocker_preservation`
- `repeated_work_avoidance_proxy`

Recommended workflow:

```powershell
python -m hypergraph_bistability.cli run-continuity-regression --output _continuity_regression_v3.json
```
