# Hyperedge Taxonomy v1

## Goal

This project now has a working memory/runtime loop, typed artifacts, relation-aware retrieval, and stable MiniMax-backed evals.

The next step is to make the "hypergraph" claim operational:

- nodes represent durable work objects
- hyperedges represent meaningful work units
- activation and retrieval operate over hyperedge structure, not only over isolated text items

This document defines a first practical taxonomy for hyperedges and maps it onto the current codebase.

## Design Principle

The hypergraph should not replace the object layer.

Use a 3-layer model:

1. Object Layer
   - explicit nodes: `task`, `subtask`, `blocker`, `evidence`, `hypothesis`, `plan`, `decision`, `result`, `preference`, `constraint`
2. Hyperedge Layer
   - meaningful work-unit groupings over multiple nodes
3. Control Layer
   - activation, competition, mode-dependent retrieval, and working-memory selection

In short:

- object layer answers "what exists and how it is related"
- hyperedge layer answers "which set of objects belongs to the same work unit"
- control layer answers "what should be active now"

## Current State

The current implementation already contains the seed of this architecture:

- typed writes in [write_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/write_policy.py)
- typed retrieval in [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- relation wiring in [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)
- inspectable graph view in [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)
- eval graph snapshots in [runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/evals/runner.py)

But the current graph is still artifact-centric:

- node types are mostly `log`, `hypothesis`, `plan`
- edges are mostly parent-child artifact relations
- there is no first-class hyperedge entity yet

## Hyperedge Types

### 1. `task_hyperedge`

Purpose:
- represent an active unit of work

Typical member node types:
- `task`
- `subtask`
- `blocker`
- `plan`
- `owner`
- `deadline`
- `constraint`

Creation rule:
- create when a user explicitly states ongoing work, a checklist, or a resumed task
- attach subsequent `subtask`, `blocker`, and `plan` nodes to the same task hyperedge

Activation rule:
- activate strongly on resume cues such as:
  - "back to"
  - "continue"
  - "pick up"
  - "what matters most"

Decay rule:
- slow decay
- task hyperedges should remain available across interruptions

Mode retrieval behavior:
- `planning`: highest priority
- `coding`: high priority if linked to active code work
- `review`: medium priority
- `debugging`: medium priority unless task is incident-linked

### 2. `evidence_hyperedge`

Purpose:
- group observations that support or challenge a diagnosis

Typical member node types:
- `issue`
- `evidence`
- `log`
- `metric`
- `hypothesis`
- `component`

Creation rule:
- create when an issue/incident is paired with logs, error traces, or suspected causes
- add new logs and hypotheses to the same evidence hyperedge when `linked_task` matches

Activation rule:
- activate on debugging and investigation prompts
- activation should favor grouped recall of:
  - issue
  - evidence
  - root cause hypothesis

Decay rule:
- medium decay
- keep active while incident is unresolved

Mode retrieval behavior:
- `debugging`: highest priority
- `coding`: medium priority
- `planning`: low priority
- `review`: medium priority for postmortems

### 3. `decision_hyperedge`

Purpose:
- capture a concrete decision together with its reasoning context

Typical member node types:
- `decision`
- `alternative`
- `rationale`
- `constraint`
- `expected_impact`
- `result`

Creation rule:
- create when the agent or user commits to an approach, rejects alternatives, or records a tradeoff

Activation rule:
- activate on prompts that ask:
  - "why"
  - "what did we decide"
  - "what are the tradeoffs"
  - "should we still do this"

Decay rule:
- very slow decay
- decisions are durable anchors

Mode retrieval behavior:
- `planning`: high priority
- `review`: highest priority
- `coding`: medium priority
- `debugging`: medium if it affects diagnostic strategy

### 4. `change_hyperedge`

Purpose:
- represent a concrete implementation or remediation package

Typical member node types:
- `patch`
- `plan`
- `test_result`
- `bug`
- `rollback_plan`
- `deploy_note`

Creation rule:
- create when code changes or remediation steps are discussed together with expected validation or rollback

Activation rule:
- activate on:
  - "what should I inspect first"
  - "continue the fix"
  - "what changed"
  - "what is the rollback"

Decay rule:
- medium decay before merge/deploy
- slow decay after a production incident if rollback remains relevant

Mode retrieval behavior:
- `coding`: highest priority
- `debugging`: high priority
- `planning`: medium priority
- `review`: high priority for release review

## Node Taxonomy v1

Minimum first-class node types to support the hyperedges above:

- `task`
- `subtask`
- `blocker`
- `issue`
- `evidence`
- `log`
- `metric`
- `hypothesis`
- `plan`
- `decision`
- `result`
- `constraint`
- `preference`
- `patch`
- `test_result`
- `rollback_plan`
- `deploy_note`

## Hyperedge Metadata v1

Each hyperedge should eventually carry:

- `hyperedge_id`
- `hyperedge_type`
- `linked_task`
- `member_node_ids`
- `status`
  - `active`
  - `paused`
  - `resolved`
  - `superseded`
- `priority`
- `created_at_turn`
- `last_activated_turn`
- `mode_bias`

## Hyperedge State v1

The current implementation now supports a minimal derived hyperedge state model in `hypergraph_view`.

Initial states:

- `active`
  - the hyperedge appeared in recent writes or recent retrievals
- `paused`
  - the hyperedge exists but was not recently active
- `resolved`
  - one of the member nodes explicitly contains completion language such as `resolved`, `fixed`, `done`, `completed`, or `closed`
- `superseded`
  - one of the member nodes contains replacement language such as `superseded`, `replaced by`, `obsolete`, or `deprecated`
- `conflicted`
  - the hyperedge contains multiple distinct `hypothesis` nodes

This is intentionally a derived state model, not yet a persisted state machine.

Reason:

- it gives immediate visibility into task/incident lifecycle
- it avoids schema churn before the semantics are better validated
- it creates a clean base for future persisted hyperedge state

## Conflict Hyperedges

The implementation now supports a derived `conflict_hyperedge` layer on top of ordinary hyperedges.

Purpose:

- group mutually competing hypotheses inside the same task/evidence unit
- distinguish:
  - invalidated hypotheses
  - still-active hypotheses
  - the current dominant hypothesis

Current derivation rule:

- start from `hypothesis` nodes
- use `contradicts` edges as the primitive invalidation signal
- group by `linked_task + backing_hyperedge_id`
- derive:
  - `hypothesis_node_ids`
  - `contradicted_node_ids`
  - `active_hypothesis_node_ids`
  - `dominant_node_id`
  - `status`

Current statuses:

- `active_conflict`
- `resolved_conflict`
- `competing_hypotheses`

Design interpretation:

- `contradicts` is the local pairwise primitive
- `conflict_hyperedge` is the higher-order work unit built from those primitives

This distinction matters because a single contradiction edge can suppress one old explanation, but a conflict hyperedge can select the dominant remaining explanation among several still-active candidates.

Current runtime implication:

- conflict hyperedges are now not only an analysis view
- in debugging/root-cause retrieval, the runtime also gives extra priority to:
  - the dominant hypothesis inside a conflict unit
  - backing evidence in the same conflict unit

## Mapping to Current Code

### Current write path

Existing:
- `MemoryWriteDecision` already stores:
  - `artifact_type`
  - `artifact_id`
  - `linked_task`
  - `relation_type`
  - `parent_artifact_id`

Relevant file:
- [write_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/write_policy.py)

Next change:
- extend `MemoryWriteDecision` with:
  - `node_type`
  - `hyperedge_type`
  - `hyperedge_id`

### Current relation wiring

Existing:
- parent-child artifact relations are attached in:
  - [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)

Next change:
- replace one-hop parent inference with:
  - node linking
  - hyperedge membership assignment

### Current retrieval

Existing:
- retrieval is kind-aware and relation-aware
- parent artifact expansion already exists

Relevant file:
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)

Next change:
- add hyperedge-aware expansion:
  - retrieve one matching node
  - recover the whole hyperedge neighborhood
  - score by `hyperedge_type` and mode

### Current graph view

Existing:
- `get_artifact_graph()` builds nodes and edges from turn log

Relevant file:
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)

Next change:
- add:
  - `get_hypergraph_view()`
  - explicit `hyperedges`
  - node-to-hyperedge membership

### Current evals

Existing:
- graph snapshots and failure analysis are already in eval output

Relevant file:
- [runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/evals/runner.py)

Next change:
- add continuity metrics at hyperedge level:
  - hyperedge recall
  - task-chain recall
  - decision continuity
  - plan continuation fidelity

## Mode-Specific Retrieval Priorities

### `debugging`

Priority order:
1. `evidence_hyperedge`
2. `change_hyperedge`
3. `task_hyperedge`
4. `decision_hyperedge`

Expected answer shape:
- issue
- likely cause / hypothesis
- first checks
- next remediation step

### `coding`

Priority order:
1. `change_hyperedge`
2. `task_hyperedge`
3. `evidence_hyperedge`
4. `decision_hyperedge`

Expected answer shape:
- active issue
- diff-style summary
- first checks or next patch step
- validation / rollback note if relevant

### `planning`

Priority order:
1. `task_hyperedge`
2. `decision_hyperedge`
3. `change_hyperedge`
4. `evidence_hyperedge`

Expected answer shape:
- active task
- priorities
- constraints
- next steps

### `review`

Priority order:
1. `decision_hyperedge`
2. `change_hyperedge`
3. `task_hyperedge`
4. `evidence_hyperedge`

Expected answer shape:
- what changed
- why it changed
- risks
- unresolved follow-ups

## Minimal Implementation Order

### Batch 1

Goal:
- add schema without breaking existing runtime

Changes:
- add `node_type`, `hyperedge_type`, `hyperedge_id` to writes
- keep existing `artifact_type` and parent links for compatibility

### Batch 2

Goal:
- build first-class hyperedge view

Changes:
- add `get_hypergraph_view()`
- expose `nodes`, `edges`, `hyperedges`
- update CLI graph summary to optionally show hyperedges

### Batch 3

Goal:
- make retrieval hyperedge-aware

Changes:
- when a node matches, recover its hyperedge peers
- score peers by mode and hyperedge type

Status:
- implemented in experiment form as `hyperedge_expansion`
- validated against `parent_expansion` and `single_hit`
- current evidence supports making this the preferred retrieval direction for chain-heavy scenarios

### Batch 4

Goal:
- evaluate continuity at hyperedge level

Changes:
- add stress scenarios centered on:
  - task resume
  - evidence chain recovery
  - decision recall
  - change-package continuation

## Why This Matters

Without explicit hyperedge semantics, the system risks collapsing into:

- typed text retrieval
- pairwise artifact linking
- prompt-level mode control

With this taxonomy, the system can become:

- task-structured
- mode-aware
- continuity-oriented
- genuinely hypergraph-shaped rather than graph-shaped in name only

## Immediate Recommendation

The next concrete implementation step should be:

1. extend write schema with `node_type`, `hyperedge_type`, `hyperedge_id`
2. keep current artifact graph intact as compatibility layer
3. add a parallel `hypergraph_view`
4. then make retrieval expand through hyperedge membership

That is the smallest path from the current working agent runtime to a truly hypergraph-native memory system.

## Updated Recommendation

That path is now partially complete:

- Batch 1 schema exists
- `hypergraph_view` exists
- experimental hyperedge-aware retrieval exists

The next recommendation is no longer "add hyperedge-aware expansion".

It is:

1. promote the validated `hyperedge_expansion` strategy from experiment code toward runtime candidacy
2. use that structure as the base for a future hyperedge-aware two-stage competitive retrieval design
3. add richer hyperedge state and continuity metrics on top of the now-working structural retrieval layer
