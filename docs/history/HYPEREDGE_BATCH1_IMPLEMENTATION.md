# Hyperedge Batch 1 Implementation

## Goal

Introduce the minimum schema needed for hyperedge-native memory without breaking the current artifact-graph runtime.

Batch 1 adds:

- `node_type`
- `hyperedge_type`
- `hyperedge_id`

to the write path, turn log, retrieval path, and graph inspection layer.

The compatibility rule is simple:

- keep current `artifact_type`, `artifact_id`, and parent relations intact
- add hyperedge metadata in parallel

## Why Now

Experiment findings so far:

- competition helps
- associative expansion helps
- naive two-stage re-competition is not enough

This means the next bottleneck is structural:

- we need richer grouping semantics before a stronger second-stage competition can work

## Batch 1 Scope

### In Scope

- extend write schema
- extend retrieval schema
- persist hyperedge metadata in `turn_log`
- expose `hypergraph_view`

### Out of Scope

- hyperedge-aware retrieval expansion
- hyperedge state machine
- learned scoring
- conflict hyperedges

## Schema Additions

### Write schema

Add to `MemoryWriteDecision`:

- `node_type`
- `hyperedge_type`
- `hyperedge_id`

### Retrieval schema

Add to `RetrievedMemory`:

- `node_type`
- `hyperedge_type`
- `hyperedge_id`

### Turn log

Persist the same fields in:

- `writes`
- `retrieved_detail`

## Mapping Rules v1

### Node type mapping

Map current kinds/artifacts into node types:

- `preference` -> `preference`
- `task` -> `task`
- `fact` -> `issue` if issue-like, else `fact`
- `log` -> `log`
- `hypothesis` -> `hypothesis`
- `plan` -> `plan`
- `response` -> `response`
- `context` -> `context`

### Hyperedge type mapping

Infer from content/kind:

- `task_hyperedge`
  - task/checklist/resume/planning content
- `evidence_hyperedge`
  - issue/log/hypothesis content
- `change_hyperedge`
  - plan/patch/remediation/coding-fix content
- `decision_hyperedge`
  - decision/choice/tradeoff/rationale content

### Hyperedge id mapping

Construct deterministic ids from:

- `hyperedge_type`
- `linked_task` when available
- normalized content fallback when no task is available

## Files To Change

### Required

- [write_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/write_policy.py)
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)

### Optional in Batch 1

- eval snapshots in [runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/evals/runner.py)

## New Inspection API

Add alongside `get_artifact_graph()`:

- `get_hypergraph_view()`

Output shape:

- `nodes`
- `edges`
- `hyperedges`

Where:

- `nodes` hold node metadata
- `edges` keep current parent-child artifact relations
- `hyperedges` group node ids under a shared `hyperedge_id`

## Success Criteria

Batch 1 is complete when:

1. new metadata is present in all writes
2. retrieval items carry the metadata through `retrieved_detail`
3. session save/load preserves the metadata
4. `get_hypergraph_view()` returns stable output
5. all current tests remain green

## Next Step After Batch 1

Once Batch 1 lands, the next meaningful retrieval upgrade is:

- hyperedge-aware expansion

That is the first point where two-stage competitive retrieval can become structurally meaningful rather than heuristic-only.

## Follow-On Result

Batch 1 by itself was not enough to improve Experiment 2.

The first `hyperedge_expansion` attempt under Batch 1 underperformed `parent_expansion` because:

- most `hyperedge_id` values were still effectively single-write ids
- hyperedges did not yet represent shared work units
- expansion through hyperedge membership therefore had little real structure to exploit

This led directly to Batch 2:

- make `hyperedge_id` task-anchored rather than content-anchored
- let related `log` and `hypothesis` nodes share an `evidence_hyperedge`
- let retrieval start from turn-log items that retain full hyperedge metadata

After that change, the hyperedge-aware Experiment 2 ablation became meaningfully better than `parent_expansion`, both locally and on real MiniMax evaluation.
