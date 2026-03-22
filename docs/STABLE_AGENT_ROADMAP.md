# Stable Agent Roadmap

## Current Position

The project now has:

- a canonical package structure under `src/hypergraph_bistability/`
- a structured agent turn pipeline
- write and retrieval policies
- session persistence with turn logs
- a terminal chat demo
- separated experiment directories
- offline-safe embedding fallback with local in-memory vector storage fallback

This is now a stable alpha runtime. It is still not a fully practical agent.

## Validation Status

Latest validation after dependency installation:

- `python -m pip install -e ".[dev,llm,vector]"` completed
- `python -m pytest tests\test_agent_runtime.py tests\test_evals.py -q` completed with `136 passed`
- formal product regression is green:
  - [\_product_regression_v25.json](F:/hypergraph_bistability/_product_regression_v25.json)
- formal long-task regression is green:
  - [\_long_task_regression_v10.json](F:/hypergraph_bistability/_long_task_regression_v10.json)
- conflict practical regression is green:
  - [\_conflict_regression_v1.json](F:/hypergraph_bistability/_conflict_regression_v1.json)
  - [\_conflict_regression_v2.json](F:/hypergraph_bistability/_conflict_regression_v2.json)
  - [\_conflict_regression_v3.json](F:/hypergraph_bistability/_conflict_regression_v3.json)
- working-set/query layer added on top of the runtime:
  - current task state
  - dominant conflict
  - decision residue
  - applicable procedures
  - handoff bundle
- document-driven writing path added:
  - `python -m hypergraph_bistability.cli write-from-docs ...`
  - `--instruction-file` added for UTF-8-safe non-ASCII prompts on Windows
  - dedicated Windows MiniMax helper scripts added for payload / prompt request paths
  - real MiniMax English document-writing path now returns grounded output again
- real MiniMax-backed product and long-task regressions are both green:
  - [\_llm_product_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_product_regression_minimax_one_click.json)
  - [\_llm_long_task_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_long_task_regression_minimax_one_click.json)

Important stability notes:

- embedding-backed features now degrade locally when Hugging Face access is unavailable
- vector retrieval no longer collapses when `chromadb` is unavailable
- sandboxed test runs may need broader file permissions for tempfile-based save/load checks
- `stable_v1` remains the formal runtime identity
- procedure-heavy and conflict-heavy practical lines are now separated instead of being judged by one mixed metric
- `write-from-docs` now follows instruction language more explicitly
- the Windows MiniMax doc-writer path is partially restored: English grounded output is working again through the dedicated helper-script path
- Chinese document-writing is still not clean on the real MiniMax path; the remaining issue is response-side mojibake rather than local prompt-file handling

## Target

Reach a "stable practical agent" level with these properties:

- usable for multi-turn sessions without rapid context drift
- robust save/load and inspectable state
- better memory usefulness than a short-history baseline
- clean local developer workflow
- clear path to OpenAI-backed and embedding-backed operation
- graceful degradation when external services are missing

## Positioning

The project has moved past pure mechanism validation.

The current position is best described as:

- late research / early productization
- a strong alpha memory/runtime core
- not yet a production-ready agent infrastructure

The mainline should now be managed as a three-stage roadmap.

### Stage 1: Consolidate The Core

Focus:

- keep `stable_v1` as the formal runtime
- keep formal practical gates green
- keep robustness checks active
- improve inspectability without introducing a second system identity

### Stage 2: Make The Runtime Consumable

Focus:

- make working-set/query surfaces usable by CLI, response assembly, and other runtime consumers
- stabilize the query contract
- expose current task/conflict/procedure/handoff state through practical interfaces
- extend document-driven read/write paths that let the agent help produce grounded project artifacts from selected local docs

### Stage 3: Practical Beta

Focus:

- broaden workload coverage
- harden persistence and observability
- maintain alignment between local and real-LLM paths
- expose a cleaner integration surface

## Phase 1: Runtime Stability

### Goal

Make the agent loop reliable, inspectable, and deterministic enough for iterative product work.

### Status

Mostly complete.

### Completed

- `TurnProcessor`, `ContextAssembler`, and `SessionState` introduced
- structured turn logs persisted
- terminal chat demo available
- Windows CLI output stabilized to ASCII-safe rendering
- embedding path made offline-safe with deterministic local fallback

### Remaining

- add explicit runtime event records for controller transitions and persistence operations
- tighten CLI state inspection formatting
- reduce residual test/runtime warnings where practical

### Acceptance Gates

- terminal chat demo runs end-to-end on Windows
- full test suite remains green
- save/load round-trip preserves runtime artifacts

## Phase 2: Memory Quality

### Goal

Move from "interesting memory" to "useful memory".

### Deliverables

- refine retrieval weighting across:
  - working memory
  - turn-log retrieval
  - vector retrieval
  - preference/task/fact prioritization
- add summarization and promotion rules
- add decay or demotion rules for stale content
- make write policy less eager for low-value assistant output
- bound context assembly by token budget and content type

### Acceptance Gates

- retrieval returns relevant preference/task state in scripted evals
- noisy or repeated assistant outputs are not over-written into memory
- context assembly remains bounded and readable

## Phase 3: Evaluation Harness

### Goal

Measure whether this memory system is actually useful.

### Deliverables

- `evals/` or `tests/integration/` scenarios for:
  - preference recall
  - task continuity
  - context switching
  - session recovery
- baseline comparison modes:
  - recent-history only
  - recent-history + vector search
  - current memory runtime
- summary metrics:
  - recall usefulness
  - irrelevant recall rate
  - continuity success
  - token overhead

### Acceptance Gates

- memory runtime matches or beats the recent-history baseline on core scenarios
- failures are reproducible and diagnosable from turn logs

### Current State

This phase is no longer just "build the harness".

It now includes:

- continuity regression
- formal product regression
- formal long-task regression
- conflict practical regression
- robustness / sidecar checks

Current practical metrics in use:

- `task_continuation`
- `blocker_preservation`
- `decision_continuity`
- `procedure_continuity`
- `conflict_continuity`
- `repeated_work_avoidance_proxy`

## Phase 4: Real LLM Operation

### Goal

Make the agent work in practical OpenAI-backed runs rather than mostly mock mode.

### Deliverables

- tested OpenAI-backed chat path
- configurable model and token budget handling
- graceful fallback between:
  - no LLM
  - LLM only
  - LLM + embeddings
- clearer prompt contract for retrieved memory sections
- deterministic handling of provider/configuration errors

### Acceptance Gates

- can run a real multi-turn session with OpenAI credentials
- memory retrieval visibly affects later responses
- errors degrade cleanly without corrupting session state

## Phase 5: Practical Interfaces

### Goal

Make the system pleasant to use during development and demos.

### Deliverables

- improved terminal chat demo
- richer `/state`, `/save`, and inspection output
- simple local evaluation commands
- optional Streamlit integration updated to use the runtime pipeline
- concise docs for install, chat, save, reload, and eval

### Acceptance Gates

- a new user can install, chat, save, reload, and inspect state in minutes
- runtime artifacts shown in CLI match actual persisted state

## Immediate Implementation Batches

### Batch 1: Evidence

Build the evaluation harness first.

Tasks:

1. add `evals/` with scripted multi-turn scenarios
2. implement recent-history baseline runner
3. implement current runtime runner
4. produce comparable JSON results and summary metrics

Why first:

- this tells us whether the memory system is helping or just adding complexity

### Batch 2: Memory Policy Refinement

Use eval failures to improve:

1. preference/task/fact scoring
2. promotion and summarization rules
3. stale-memory demotion
4. context budget management

### Batch 3: Real LLM Path

After the eval harness is stable:

1. harden OpenAI-backed chat path
2. expose model/config selection in CLI
3. test error degradation and session integrity

## Next Step

The next concrete implementation step should be:

1. keep `stable_v1` and existing formal gates green
2. expand conflict-heavy practical scenarios as a parallel line
3. continue using robustness sidecars to check for local optimization
4. only promote new practical scenarios when they are structurally clean across local and real-LLM paths

That is the shortest path from "stable alpha runtime" to "evidence-backed practical agent infrastructure".
