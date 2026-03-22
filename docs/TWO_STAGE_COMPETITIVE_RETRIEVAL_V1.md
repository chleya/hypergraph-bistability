# Two-Stage Competitive Retrieval v1

## Goal

Fix the main tradeoff observed in Experiment 2:

- associative expansion improves chain recall and response recall
- but it also increases irrelevant context

The intended solution is a two-stage retrieval process:

1. select focal memories through competition
2. expand around those memories
3. re-competitively prune the expanded set before final context injection

## Retrieval Flow

### Stage 1: Focal Competition

Input:
- query
- working-memory state
- turn log
- optional vector results

Output:
- a small set of focal items

Purpose:
- identify the dominant issue/task/preference/plan anchors before expansion

Current implementation analog:
- current `RetrievalPolicy.collect()` top-ranked items

### Stage 2: Associative Expansion

Input:
- focal items from Stage 1

Expansion sources:
- parent artifact links
- future hyperedge membership
- future task-cluster neighbors

Output:
- focal items + expanded candidates

Purpose:
- recover the local work chain around the focus

### Stage 3: Re-Competition

Input:
- focal items
- expanded candidates

Output:
- final context frontier

Purpose:
- keep continuity gains from expansion
- suppress expansion noise before the LLM sees the context

## Scoring Intuition

For re-competition, a candidate score should combine:

- original retrieval score
- whether it is a focal item
- whether it is directly connected to a focal item
- mode-specific bias
- artifact kind relevance
- expansion depth penalty

Simple v1 score:

`final = base + focal_bonus + relation_bonus + mode_bonus - depth_penalty`

## V1 Experimental Policy

The first experimental version should remain conservative:

- focal set size: 2
- expanded candidate limit: 6
- final frontier size: 3
- parent expansion only
- no hyperedge membership yet

This is enough to test whether second-pass pruning helps.

## Why This Matters

Experiment 1 showed:
- competition reduces irrelevant context

Experiment 2 showed:
- associative expansion improves continuity, but increases noise

Two-stage retrieval is the most direct synthesis of those findings.

## Current Code Path

The v1 implementation should live in the experiment layer first:

- [mechanism_runner.py](F:/hypergraph_bistability/src/hypergraph_bistability/experiments/mechanism_runner.py)

The base primitives already exist in:

- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)

Once validated, the same logic can move into:

- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)

## Success Criterion

Compared with simple parent expansion, two-stage retrieval should:

- preserve or improve response recall
- preserve most chain recall
- reduce irrelevant context rate

That is the minimum requirement before promoting it into the main runtime.
