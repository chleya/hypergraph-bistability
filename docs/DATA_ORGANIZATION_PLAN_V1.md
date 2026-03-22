# Data Organization Plan V1

## Goal

Turn the project from a strong memory/runtime prototype into a clear **agent-oriented data organization scheme**.

This plan is intentionally narrower than "build a full general agent". The immediate target is:

- define what kinds of data the agent should store
- define how those data types relate
- define how they are retrieved into a working set
- define what is reusable versus task-local

The design center is:

**help an agent preserve work continuity, recover structured context, and apply reusable procedures without mixing everything into flat memory items**

## Current Strengths

The current codebase already has several strong building blocks:

- episodic facts and artifacts
- task continuity and blocker preservation
- conflict-aware structures
- decision residues
- task phase tags
- confidence / contradiction / verification relations
- hyperedge-aware retrieval
- continuity and product regressions

Concretely, the following mechanisms are already established:

- `competition`
- `hyperedge_expansion`
- `confidence-aware retrieval`
- `conflict_hyperedges`
- `decision_residues`
- `task_phase`
- `stable_v1` runtime profile

These are enough to stop thinking in terms of "just memory" and start thinking in terms of **data organization for agent work**.

## Main Gap

The project still lacks a unified way to organize all data classes.

Right now, important structures exist, but they are spread across:

- `turn_log`
- writes
- retrieved items
- hypergraph views
- mechanism-specific heuristics

What is missing is a single explicit plan for:

1. what object classes exist
2. what lifecycle each class follows
3. what is reusable
4. what belongs in the current working set
5. how queries map onto those structures

## Target Layers

The recommended organization is five layers.

### 1. Event Layer

Raw events and observations.

Examples:

- user inputs
- assistant outputs
- tool outputs
- verification outcomes
- status updates

Purpose:

- preserve raw provenance
- support replay and auditability

### 2. Memory Object Layer

Typed agent-facing objects.

Initial object classes:

- `fact`
- `task`
- `blocker`
- `hypothesis`
- `plan`
- `decision`
- `constraint`
- `alternative`
- `verification`
- `procedure`
- `template`

Purpose:

- make data queryable by role, not just by text

### 3. Hyperedge Layer

Work-unit level grouping.

Current or planned hyperedge types:

- `evidence_hyperedge`
- `conflict_hyperedge`
- `decision_hyperedge`
- `change_hyperedge`
- `procedure_hyperedge`

Purpose:

- bind together items that should be recovered together
- support structured expansion instead of flat retrieval

### 4. State Layer

Lifecycle and trust annotations.

Examples:

- `active`
- `paused`
- `resolved`
- `superseded`
- `conflicted`
- `verified`
- `tentative`
- `speculative`
- `contradicted`
- `analysis`
- `decision`
- `implementation`
- `verification`
- `closure`

Purpose:

- tell the runtime what is still live, what is weak, and what should be prioritized

### 5. Working Set Layer

The current front-of-mind package the agent actually uses.

Target contents:

- current task
- active blocker
- dominant conflict
- active decisions
- relevant procedures
- next-step candidates
- handoff / closeout status

Purpose:

- isolate "everything stored" from "what the agent should reason over now"

## Missing Pieces

The main missing capabilities are:

### A. Unified schema

There is no single formal document or API that unifies:

- nodes
- relations
- hyperedges
- states
- procedures
- working set
- reusability

### B. Procedural memory

The system knows a lot about what happened, but still lacks a first-class place for:

- playbooks
- checklists
- handoff templates
- review templates
- closeout procedures

### C. Reusability

The runtime still mostly treats memory as task-local.

It needs an explicit reuse dimension:

- `instance_only`
- `task_local`
- `project_reusable`
- `cross_task_reusable`

### D. Working set abstraction

Retrieval is strong, but the project still lacks a single explicit `WorkingSet` object.

Without that, the runtime still behaves more like "retrieve many useful things" than "maintain an explicit work surface".

### E. Query API

There is no stable query layer yet for asking things like:

- what is the current task state
- what conflict is dominant
- what decision remains in force
- what procedure applies here
- what is ready to hand off

## Schema Direction

The first formal schema should introduce these core objects.

### Node

- `node_id`
- `node_type`
- `content`
- `linked_task`
- `confidence_tag`
- `task_phase`
- `reusability_class`

### Relation

- `relation_type`
- `src_node_id`
- `dst_node_id`

### Hyperedge

- `hyperedge_id`
- `hyperedge_type`
- `member_node_ids`
- `status`
- `linked_task`
- `reusability_class`

### Procedure

- `procedure_id`
- `procedure_type`
- `steps`
- `preconditions`
- `verification_rules`
- `closeout_rules`
- `applicable_modes`
- `reusability_class`

### WorkingSet

- `task_id`
- `active_nodes`
- `active_hyperedges`
- `active_procedures`
- `dominant_conflicts`
- `active_decisions`
- `next_step_candidates`

## Reusability Model

This project should explicitly separate:

### Episodic structures

Specific to one task instance.

Examples:

- a single incident hypothesis
- a single hotfix decision
- one release verification result

### Task-local reusable structures

Useful throughout one task chain.

Examples:

- a task-specific checklist
- recurring guardrails inside one release cycle

### Project-reusable structures

Useful across similar work in the same project.

Examples:

- release handoff checklist for this repo
- review summary conventions for this team

### Cross-task reusable structures

Portable procedures or templates.

Examples:

- generic incident closeout checklist
- generic diff-style review template

## Phase Plan

### Phase 1: Data Schema V1

Goal:

- formalize the existing object model

Deliverables:

- `DATA_SCHEMA_V1.md`
- unified definitions for node, relation, hyperedge, state, and reusability
- schema-aligned runtime view

### Phase 2: Procedural Memory V1

Goal:

- add reusable procedural structures

Initial procedural types:

- `debug_playbook`
- `release_handoff_checklist`
- `review_summary_template`
- `incident_closeout_checklist`

Deliverables:

- `procedure_hyperedge`
- procedure-aware write rules
- procedure-aware retrieval hooks
- procedure-aware product/long-task regression gates
- `procedure_continuity` as a practical regression metric

### Phase 3: Working Set / Workspace

Goal:

- derive an explicit front-of-mind work surface

Deliverables:

- `get_working_set()`
- working-set-oriented debug view
- working-set-oriented response assembly

Current implementation status:

- minimal working-set derivation is now implemented on top of the current `hypergraph_view`
- current query surface:
  - `get_working_set()`
  - `query_current_task_state()`
  - `query_dominant_conflict()`
  - `query_decision_residue()`
  - `query_applicable_procedures()`
  - `query_handoff_bundle()`

Current interpretation:

- this is not yet a full external backend API
- but it is now the first concrete bridge from internal memory/runtime structure toward an inspectable agent work surface

### Phase 4: Query/API Layer

Goal:

- expose the data organization scheme as a usable backend interface

Target queries:

- `query_current_task_state()`
- `query_dominant_conflict()`
- `query_decision_residue()`
- `query_applicable_procedures()`
- `query_handoff_bundle()`

## Recommended Priority

If only a small number of things are tackled next, prioritize:

1. `DATA_SCHEMA_V1`
2. `reusability_class`
3. `Procedural Memory V1`
4. `WorkingSet`
5. `query layer`

## Scope Discipline

The project should avoid jumping directly to a grand "new database" claim.

The evidence currently supports a narrower and stronger description:

**an agent-oriented structured memory/runtime that is moving toward a reusable work-state backend**

That positioning is justified by:

- strong continuity regressions
- strong product regressions
- validated conflict-aware and hyperedge-aware retrieval
- formalized runtime profile

It does **not yet** justify claiming:

- full general-purpose database semantics
- mature storage engine abstraction
- fully productized agent backend API

## Immediate Next Step

The next concrete implementation step should be:

**Procedural Memory V1**

Reason:

- it fills a real missing layer
- it complements existing fact/state/relation strength
- it is directly relevant to real task execution and handoff quality
- it can be added without destabilizing the validated `stable_v1` runtime

Current practical continuation of that step:

- keep `stable_v1` as the formal runtime profile
- expand procedure-aware scenarios inside the formal product regression gate
- expand procedure-aware scenarios inside the formal long-task regression gate
- keep conflict-heavy practical regression as a parallel line instead of forcing it into the procedure gate
- judge conflict-heavy continuation with `conflict_continuity`, not only `procedure_continuity`
- judge progress by practical regression separation, not by adding many new procedure types

## One-Sentence Summary

The project should evolve from "strong memory mechanisms" into a clear **agent data organization scheme** built around:

- typed objects
- hyperedge work units
- lifecycle state
- reusability
- procedural memory
- explicit working sets
