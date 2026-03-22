# Innovation Map v1

## Purpose

The project has reached a meaningful milestone:

- stable runtime
- typed artifact memory
- relation-aware retrieval
- real MiniMax-backed core and stress eval success
- first mechanism experiments for competition and associative expansion

The next question is not "can it run?" but "where should innovation effort go next?"

This document organizes the next-stage ideas into three layers:

1. near-term implementable
2. mid-term research-oriented
3. long-term high-risk / high-upside

Each item includes:

- core idea
- why it matters
- current code touchpoints
- whether it is primarily product-facing or research-facing

## Layer 1: Near-Term Implementable

These are the best next steps if the goal is to improve the current agent quickly while preserving a credible research arc.

### 1. Two-Stage Competitive Retrieval

Core idea:
- retrieve in two passes
- pass 1 selects salient focal nodes / hyperedges
- pass 2 expands around them and re-competes before final context selection

Why it matters:
- current results already show competition reduces noise
- current associative expansion helps continuity but introduces more irrelevant context
- a second competition pass is the natural next fix

Current code touchpoints:
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)
- [mechanism_runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/experiments/mechanism_runner.py)

Product vs research:
- both
- especially strong as a product-facing continuity/noise-reduction improvement

### 2. Hyperedge-Aware Expansion

Core idea:
- move beyond parent-artifact expansion
- expand through explicit hyperedge membership once `hyperedge_id` and `hyperedge_type` exist

Why it matters:
- current `parent_expansion` already improves response recall
- a true hyperedge-based expansion should recover work units instead of only parent-child chains

Current code touchpoints:
- [HYPEREDGE_TAXONOMY_V1.md](F:/hypergraph_bistability/docs/HYPEREDGE_TAXONOMY_V1.md)
- [write_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/write_policy.py)
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)

Product vs research:
- both

### 3. Hyperedge State

Core idea:
- give hyperedges explicit lifecycle state:
  - `active`
  - `paused`
  - `unresolved`
  - `resolved`
  - `superseded`

Why it matters:
- current graph stores structure but not work progress
- stateful hyperedges make continuity and task recovery much stronger

Current code touchpoints:
- future hyperedge schema in `turn_log`
- [session.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/session.py)
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)

Product vs research:
- primarily product-facing, with strong interpretability upside

### 4. Mode-Specific Expansion Trees

Core idea:
- keep graph/ hypergraph as storage
- run mode-specific expansion trees for retrieval execution

Example:
- debugging: issue -> evidence -> hypothesis -> plan
- coding: task -> issue -> patch plan -> validation
- planning: task -> constraint -> decision -> next step

Why it matters:
- current mode-specific contract already improves response behavior
- retrieval should also become mode-specific, not only answer formatting

Current code touchpoints:
- [context_assembler.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/context_assembler.py)
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [adaptive_controller.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/adaptive_controller.py)

Product vs research:
- both

### 5. Confidence / Uncertainty Tags

Core idea:
- every node or hyperedge can carry:
  - `verified`
  - `tentative`
  - `speculative`
  - `contradicted`

Why it matters:
- current memory treats most retrieved content as equally reusable
- debugging and research agents need confidence-aware retrieval

Current code touchpoints:
- [write_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/write_policy.py)
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- eval extensions in [runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/evals/runner.py)

Product vs research:
- both

## Layer 2: Mid-Term Research-Oriented

These ideas are likely to produce the most distinctive research story if the near-term system work succeeds.

### 6. Conflict Hyperedges

Core idea:
- represent incompatible hypotheses, decisions, or plans as first-class conflict structures

Why it matters:
- many real tasks contain unresolved contradictions
- conflict-aware memory is more realistic than simple accumulation

Current code touchpoints:
- [write_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/write_policy.py)
- [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)
- future hyperedge schema

Product vs research:
- primarily research-facing at first

Current evidence:

- initial contradiction-link experiments showed clear value over baseline
- after isolating a true single-edge-only comparison, conflict hyperedges also beat contradiction-link-only retrieval
- latest result file: [\_experiment_conflict_links_v4.json](F:/hypergraph_bistability/_experiment_conflict_links_v4.json)
- this moves `conflict hyperedges` from a speculative idea to an experimentally supported direction
- latest MiniMax run file: [\_experiment_conflict_links_llm.json](F:/hypergraph_bistability/_experiment_conflict_links_llm.json)
- that run confirms the same structural advantage on the real LLM path:
  - `conflict_hyperedge_aware` reaches recall `1.000`, precision proxy `1.000`, irrelevant context `0.000`, and response recall `1.000`
  - relative to single-edge-only contradiction handling, conflict hyperedges improve recall and precision while preserving response quality

### 7. Decision Residue

Core idea:
- preserve not only what was decided, but why

Store:
- selected plan
- rejected alternatives
- rationale
- constraints

Why it matters:
- current agents often resume decisions without remembering the reason
- decision continuity is more valuable than simple fact recall

Current code touchpoints:
- [HYPEREDGE_TAXONOMY_V1.md](F:/hypergraph_bistability/docs/HYPEREDGE_TAXONOMY_V1.md)
- [session.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/session.py)
- future `decision_hyperedge`

Product vs research:
- both

Current implementation status:

- decision residue now exists as a concrete structure in the runtime:
  - `decision`
  - `rationale`
  - `constraint`
  - `alternative`
- it is visible in `hypergraph_view` as `decision_residues`
- current experiment file: [\_experiment_decision_residue.json](F:/hypergraph_bistability/_experiment_decision_residue.json)

Current conclusion:

- strong vs baseline
- not better than the current best formal runtime path yet
- should be treated as a valuable structural layer that still needs harder validation cases

### 7b. Task Phase / Progress State

Core idea:
- represent where a task currently sits in its lifecycle

Current phase set:
- `analysis`
- `decision`
- `implementation`
- `verification`
- `closure`

Why it matters:
- continuity is not only about retrieving the right artifacts
- it is also about resuming at the right stage of work

Current code touchpoints:
- [write_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/write_policy.py)
- [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)

Product vs research:
- both

Current implementation status:

- `task_phase` is now attached to writes and retrieved items
- `hypergraph_view` exposes `member_task_phases` and `phase_summary`
- current experiment file: [\_experiment_phase_progress.json](F:/hypergraph_bistability/_experiment_phase_progress.json)

Current conclusion:

- strong vs baseline
- not better than the current best formal runtime path yet
- likely needs harder progress-state scenarios before it becomes an independently validated advantage

### 8. Continuity Metrics Beyond Recall

Core idea:
- evaluate whether the agent continues work, not only whether it mentions prior text

Candidate metrics:
- task continuation score
- blocker preservation rate
- repeated-work avoidance
- decision continuity
- unresolved issue carryover

Why it matters:
- this project should not stop at retrieval benchmarking
- continuity is the user-facing value proposition

Current code touchpoints:
- [runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/evals/runner.py)
- [EXPERIMENT_PLAN_V1.md](F:/hypergraph_bistability/docs/EXPERIMENT_PLAN_V1.md)

Product vs research:
- both, highly important

### 9. Hypergraph as External Working Memory

Core idea:
- the hypergraph is not just storage
- it acts as an external structured working memory for the LLM

Why it matters:
- this gives the project a stronger research identity than "memory-enhanced RAG"
- it directly matches the project's core intuition: competition -> focus -> associative expansion -> response

Current code touchpoints:
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)
- [context_assembler.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/context_assembler.py)
- [adaptive_controller.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/adaptive_controller.py)

Product vs research:
- primarily research-facing, with product implications

### 10. Competitive Hypergraph Retrieval Tree

Core idea:
- store memory in graph / hypergraph form
- execute retrieval as a competitive expansion tree

Interpretation:
- graph stores the structure
- tree is the per-query retrieval trace
- competition performs pruning and focus selection

Why it matters:
- this is a clearer cognitive model than "just graph retrieval"
- it may become the project's signature mechanism

Current code touchpoints:
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [HYPEREDGE_TAXONOMY_V1.md](F:/hypergraph_bistability/docs/HYPEREDGE_TAXONOMY_V1.md)
- experiment runners in [mechanism_runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/experiments/mechanism_runner.py)

Product vs research:
- strongly research-facing, but still implementable

## Layer 3: Long-Term High-Risk / High-Upside

These ideas are not the next implementation step, but they may become the most original part of the project if the foundations hold.

### 11. Phase-Transition Memory Control

Core idea:
- make mode changes and task focus shifts correspond to explicit dynamical regime changes

Why it matters:
- this is where the project's physics-based identity becomes unique
- it goes beyond prompt-level mode switching

Current code touchpoints:
- [adaptive_controller.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/adaptive_controller.py)
- [core](F:/hypergraph_bistability/src/hypergraph_bistability/core)

Product vs research:
- mostly research-facing

### 12. Local World Models per Task Cluster

Core idea:
- let each active task cluster behave like a partial world model

Contain:
- goal
- known facts
- unknowns
- candidate hypotheses
- active plan
- failed attempts

Why it matters:
- this moves the system from memory recovery toward real problem-state tracking

Current code touchpoints:
- future task hyperedges and hypergraph view
- [session.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/session.py)

Product vs research:
- both, but implementation-heavy

### 13. Self-Reorganization of Hyperedge Topology

Core idea:
- hyperedges should not be static
- repeated co-activation should merge, split, or reweight hyperedges over time

Why it matters:
- this would make the memory system adaptive rather than hand-shaped
- strong research value if it can be made stable

Current code touchpoints:
- future hyperedge maintenance module
- experiment layer rather than production path at first

Product vs research:
- high-risk research

### 14. Hyperedge Routing Signals for LLM Context Injection

Core idea:
- use hyperedge activation as an explicit routing signal for what enters the prompt

Why it matters:
- makes the hypergraph behave like a controller for external attention
- interesting bridge to current LLM/agent research

Current code touchpoints:
- [context_assembler.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/context_assembler.py)
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)

Product vs research:
- primarily research-facing

### 15. Learned Hyperedge Scoring

Core idea:
- replace some heuristic scoring with learned or adaptive scoring over node/hyperedge utility

Why it matters:
- current heuristics are effective, but likely not optimal
- a learned scorer could discover better retrieval priorities than hand-written rules

Current code touchpoints:
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- experiment layer in [mechanism_runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/experiments/mechanism_runner.py)

Product vs research:
- both, but should come after stronger structural baselines

## Recommended Priority

### Immediate next priorities

1. two-stage competitive retrieval
2. hyperedge-aware expansion
3. hyperedge state
4. confidence / uncertainty tags
5. conflict hyperedges
6. harder decision-progress and phase-progress scenarios

Reason:
- these directly address current experiment findings
- these improve product usefulness and preserve research direction
- conflict hyperedges now have cleaner empirical support than several other mid-term ideas
- decision residue and task phase are now structurally integrated, but they still need harder scenarios rather than more local heuristics

### Best research storyline

If the goal is a strong research narrative, the best chain is:

1. competition reduces irrelevant context
2. associative expansion improves continuity
3. mode-specific control improves response utilization
4. hyperedge-aware retrieval improves structured continuation
5. competitive retrieval tree explains the whole mechanism

### What to avoid

Avoid spending the next phase on:

- generic tool integrations only
- more prompt tweaking without retrieval changes
- large benchmark expansion before mechanism experiments mature
- making the hypergraph bigger without making it more semantically meaningful

## Bottom Line

The project should continue in the direction of:

- structured work-state memory
- competition-driven focus selection
- controlled associative expansion
- hyperedge-native continuity

That is the path most likely to produce both:

- a genuinely useful agent memory runtime
- a research story that is not reducible to "RAG with extra steps"
