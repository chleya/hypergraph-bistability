# Practical Agent Productization Plan

## Objective

Move the project from a research prototype into a practical agent system that can be used for:

- long-running assistants
- coding and planning agents
- multi-context personal or team copilots
- interpretable memory-aware chat systems

The target is not just "the code runs". The target is:

- predictable behavior
- stable memory management
- useful retrieval and context composition
- practical interfaces for integration
- measurable quality against baseline agent memory approaches

## Current State

The repository already has strong foundations:

- a controllable memory core with explicit state variables
- physics-based control through `lambda`, `mu`, and `lambda_c`
- an agent wrapper
- optional embeddings and LLM integrations
- visualization and tests for core pieces
- a stable `stable_v1` runtime under `src/hypergraph_bistability/`
- formal product and long-task practical regressions
- procedure-aware practical gates for handoff / review / closeout continuity
- a separate conflict-heavy practical regression line
- real MiniMax-backed product and long-task regression runs

But it is not yet a practical production-grade agent system.

## Current Product Frame

The current project identity is narrower and stronger than a generic "agent platform".

It is now best described as:

- an `agent working-memory infrastructure`
- a structured memory/runtime backend for agent continuity
- a system that is being validated through practical regression, not only mechanism experiments

The current mainline is:

- keep `stable_v1` as the formal runtime
- keep `Procedural Memory v1` as the active mainline capability push
- judge progress through practical regression separation on:
  - product
  - long-task
  - conflict-heavy continuation

Current validated practical lines are:

- formal product regression
- formal long-task regression
- conflict-heavy practical regression
- robustness and sidecar suites for anti-local-optimization checks

Current infrastructure progress also now includes:

- a minimal working-set / query layer on top of the current runtime state
- queryable access to:
  - current task state
  - dominant conflict
  - decision residue
  - applicable procedures
  - handoff bundle state
- a minimal document-driven read/write entrypoint:
  - `write-from-docs`
  - selected local docs in, grounded artifact draft out
  - `--instruction-file` for UTF-8-safe Windows prompting
  - explicit language-following for document-grounded writing

Current practical limitation on that path:

- the Windows PowerShell MiniMax path now has dedicated helper scripts for payload-driven and prompt-driven requests, plus clearer stderr surfacing
- real MiniMax validation is partially closed:
  - English document-grounded writing now works again through the dedicated prompt-script path
  - Chinese document-grounded writing still returns mojibake on the real path, so response decoding/transport compatibility is not fully solved yet

## Main Gaps Between Current State and a Practical Agent

### 1. Memory Is Interesting but Not Yet Operationally Strong

Current strengths:

- explicit group/layer memory structure
- interpretable activation matrix
- controllable focus versus distributed state

Current weaknesses:

- slot assignment is still simplistic
- retrieval quality is not strongly evaluated
- memory writing policy is too naive for real multi-turn agent use
- no robust summarization, compression, or promotion pipeline
- no clear distinction between working memory, episodic memory, and durable knowledge

### 2. Agent Loop Is Too Thin

The current `HypergraphAgent` is more of a demonstration wrapper than a practical agent runtime.

Missing pieces:

- tool execution orchestration
- planner / executor loop
- message routing policies
- state update policy after tool calls
- robust context assembly before LLM calls
- guardrails for token budgets and stale memory

### 3. Reliability and Persistence Are Not Ready

Missing or incomplete:

- session lifecycle handling
- checkpointing and recovery
- structured persistence format for conversations, memory state, and embeddings
- versioned schema for saved agent states
- deterministic regression checks for agent behavior

### 4. Evaluation Is Not Product-Oriented Yet

Current evaluation mostly supports theory and simulation.

Missing product evaluation:

- task success rate
- relevance of recalled memories
- harmful or stale memory reuse rate
- mode-switch quality
- latency and token-cost profiles
- comparison against simple baselines such as recent-history + vector search

### 5. Integration Surface Is Too Narrow

For practical use, the project needs:

- a clean chat API
- a tool-calling interface
- streaming support
- external memory store options
- framework adapters
- service or local app deployment path

## What a Practical Agent Version Should Look Like

## Recommended Product Shape

The practical version should have four layers:

### Layer 1: Memory Kernel

Responsibilities:

- maintain the activation matrix
- control focus and collapse
- expose state transitions and diagnostics
- support deterministic update policies

Keep from current project:

- `AgentMemory`
- `AgentMemoryEnhanced`
- `lambda_c` logic
- mode switching and `set_n_high`

Needs to be added:

- stronger typed config objects
- clear policy hooks for write, decay, promote, merge, evict
- schema-stable persistence

### Layer 2: Memory Management Pipeline

Responsibilities:

- decide what gets written
- route content to groups/layers
- summarize old episodes
- promote important content to durable store
- retrieve relevant items for response generation

Needed components:

- write policy
- retrieval policy
- summarizer/compressor
- salience scoring
- recency and confidence weighting

This layer is currently the biggest missing piece.

### Layer 3: Agent Runtime

Responsibilities:

- manage the message loop
- select tools
- invoke tools
- update memory after tool results
- assemble final context for the LLM

Needed features:

- structured turn lifecycle
- tool result ingestion
- explicit system/user/tool memory channels
- retry and error handling
- context budget management

### Layer 4: Interfaces

Responsibilities:

- CLI
- Python API
- web or Streamlit UI
- framework adapters
- optional service deployment

## Recommended Near-Term Product Direction

Do not try to make this a fully general autonomous agent first.

The most realistic path is:

### Product Target 1: Memory-Aware Chat Agent

Capabilities:

- multi-turn chat
- context grouping
- interpretable memory dashboard
- controllable focus modes
- retrieval from short-term and durable memory

This target is achievable with the current foundation.

### Product Target 2: Coding / Planning Copilot Memory Backend

Capabilities:

- remember user preferences
- remember project constraints
- remember recent plans and decisions
- recover context after session gaps
- switch between exploratory and focused modes

This is a strong practical use case because the memory structure maps well to coding workflows.

## Priority Improvements

## Priority 0: Define the Agent Contract

Before adding features, define what one agent turn does.

A turn should have a fixed pipeline:

1. ingest user input
2. detect intent / conflict / topic / salience
3. retrieve relevant memories
4. update working memory state
5. call LLM with bounded context
6. process tool calls if needed
7. write distilled turn artifacts back to memory
8. persist state

Without this contract, new features will keep accumulating inconsistently.

## Priority 1: Build a Real Memory Pipeline

Add explicit modules:

- `memory/policies/write_policy.py`
- `memory/policies/retrieval_policy.py`
- `memory/policies/decay_policy.py`
- `memory/policies/promotion_policy.py`
- `memory/summarizer.py`

Key behavior to implement:

- decide whether incoming content should be ignored, stored raw, or summarized
- distinguish preferences, tasks, plans, facts, and transient chat content
- prevent flooding the memory matrix with low-value content
- combine symbolic slot structure with vector retrieval

Practical rule:

not every utterance should be written directly into active memory

## Priority 2: Improve Memory Typing

The practical system needs explicit memory classes:

- working memory
- episodic memory
- semantic memory
- preference memory
- task memory

Recommended implementation:

- use the activation matrix for working memory selection
- use a structured store for episodic summaries
- use embeddings or a document store for semantic recall
- use a stable JSON schema for preferences and long-lived facts

## Priority 3: Make Retrieval Competitive With Simpler Baselines

Today the project has novel control dynamics, but practical usefulness depends on retrieval quality.

Add retrieval scoring that combines:

- semantic similarity
- group/layer compatibility
- recency
- importance
- confidence
- current mode

Then compare against:

- last-N messages baseline
- pure vector search baseline
- summary + vector hybrid baseline

If the practical system cannot beat or at least match those baselines, the memory novelty will not matter.

## Priority 4: Strengthen HypergraphAgent Into a Runtime

Refactor `HypergraphAgent` into a clearer runtime structure:

- `AgentSession`
- `AgentRuntime`
- `TurnProcessor`
- `ContextAssembler`
- `MemoryCoordinator`

This is important because the current single-class design mixes:

- control logic
- memory mutation
- LLM calls
- persistence
- fallback mock behavior

That is acceptable for a demo, but not for a practical system.

## Priority 5: Persistence and Recovery

Add versioned state persistence:

- `agent_state.json`
- `memory_state.json`
- `conversation_log.jsonl`
- optional vector store directory

Requirements:

- resume after restart
- migrate old state schemas
- inspect saved states manually
- partial recovery if vector store is unavailable

## Priority 6: Evaluation Harness for Real Agent Tasks

Create task-oriented benchmarks in `tests/integration` or `evals/`.

Suggested scenarios:

- preference retention across 30 turns
- task-plan continuity across interruptions
- coding assistant remembering repository constraints
- context switching between personal and technical topics
- recovery after ambiguous or conflicting requests

Metrics:

- memory recall precision
- memory recall usefulness
- irrelevant recall rate
- task completion rate
- token usage
- latency

## Priority 7: User-Facing Interfaces

For practical adoption, provide:

- a stable Python API
- a terminal chat demo
- a Streamlit or lightweight web panel
- optional framework adapters for LangChain or similar systems

The UI should show:

- active groups/layers
- retrieved memory items
- mode and `lambda/lambda_c`
- what was written this turn
- what was ignored or summarized

That transparency is one of this project's strongest differentiators.

## Suggested Technical Roadmap

## Three-Stage Mainline Roadmap

The project is no longer mainly in "mechanism discovery".

It is now in the transition from:

- validated memory/runtime mechanisms

to:

- practical agent working-memory infrastructure

The recommended mainline should therefore be split into three stages.

### Stage 1: Consolidate The Core

Objective:

- turn the current validated runtime into a stable, inspectable memory/runtime core

Mainline requirements:

- keep `stable_v1` as the single formal runtime
- keep formal `product`, `long-task`, and `conflict` practical lines green
- continue using robustness sidecars to catch local optimization
- keep new work constrained to structure, retrieval, response grounding, and inspectability

Concrete deliverables:

- stable formal gates
- stable real-LLM practical validation
- minimal working-set / query layer
- stronger state inspection and failure analysis

What not to do here:

- do not start a second runtime identity
- do not open broad new mechanism families without practical failure pressure
- do not build broad service abstractions before runtime consumers exist

### Stage 2: Make The Runtime Consumable

Objective:

- turn the runtime from an internally strong system into a consumable backend surface for agent work

Mainline requirements:

- connect the working-set / query layer to real runtime consumers
- make handoff, task-state, decision, procedure, and conflict state explicitly available to other runtime surfaces
- keep abstractions minimal and driven by actual use

Concrete deliverables:

- CLI/state inspection that uses query APIs
- response assembly that can consume working-set/query outputs where justified
- stable Python-facing query contract
- clearer runtime/debug surfaces for current task state and handoff state

What not to do here:

- do not overbuild a framework shell with no active consumers
- do not split into too many API layers before the minimal contract proves useful

### Stage 3: Practical Beta

Objective:

- prove that the system is not only locally elegant, but usable as a practical agent memory backend

Mainline requirements:

- broaden workload coverage
- harden persistence, degradation, and recovery
- keep local and real-LLM paths aligned
- expose a clearer integration surface

Concrete deliverables:

- broader practical scenario coverage
- stronger persistence / migration / observability
- better degradation and provider-failure handling
- service/API-facing integration surface

Success condition:

- the project can honestly be described as a usable agent working-memory backend, not only a validated runtime prototype

## Current Runtime Gates

The project now has three practical evaluation layers that matter operationally:

### 1. Formal Product Gate

Purpose:

- procedure-aware handoff / review / closeout continuation

Current status:

- runtime continuity is fully green locally
- real MiniMax-backed product regression is also fully green

### 2. Formal Long-Task Gate

Purpose:

- multi-stage incident / review / release replay with procedure carryover

Current status:

- runtime continuity is fully green locally
- real MiniMax-backed long-task regression is also fully green

### 3. Conflict Practical Line

Purpose:

- dominant hypothesis and surviving evidence continuity on conflict-heavy chains

Current status:

- now tracked independently with `conflict_continuity`
- kept parallel to, not mixed with, the procedure gate story
- now includes a harder incident-packet scenario that also checks ruled-out theory preservation

## Phase A: Usable Internal Alpha

Objective:

- make one practical memory-aware chat agent that is stable for local use

Deliverables:

- explicit turn pipeline
- structured memory write and retrieval policies
- stable persistence
- terminal and Python API demos
- evaluation on a small set of real conversation tasks

Success criteria:

- can hold useful multi-turn context across sessions
- can avoid writing every message blindly
- can retrieve relevant user preferences and task state consistently

## Phase B: Practical Coding / Planning Agent

Objective:

- adapt the memory system to coding workflows

Deliverables:

- task memory and plan memory
- decision log memory
- repo-specific constraint memory
- stronger focused/exploratory switching during coding tasks

Success criteria:

- remembers project constraints over long sessions
- preserves decisions across tool use
- can recover from context drift better than a short-history baseline

## Phase C: External Integrations and Service Layer

Objective:

- make the system easy to integrate

Deliverables:

- documented Python SDK
- service API or local server mode
- structured callbacks for tool use and memory updates
- optional adapters for external frameworks

## Concrete Module Additions Recommended Next

Add these modules under `src/hypergraph_bistability/`:

### `memory/policies/`

- `write_policy.py`
- `retrieval_policy.py`
- `promotion_policy.py`
- `decay_policy.py`

### `memory/stores/`

- `working_store.py`
- `episodic_store.py`
- `preference_store.py`

### `agent/runtime/`

- `session.py`
- `turn_processor.py`
- `context_assembler.py`
- `tool_router.py`

### `evals/`

- `chat_memory_eval.py`
- `preference_recall_eval.py`
- `context_switch_eval.py`
- `coding_agent_memory_eval.py`

## Highest-Priority Implementation Batch

If the goal is "practical agent" rather than more research, the next implementation batch should be:

1. keep the formal `stable_v1` runtime and current practical gates green
2. expand conflict-heavy practical regression without polluting procedure metrics
3. keep robustness sidecars alive as anti-local-optimization checks
4. extend real MiniMax-backed practical regression where new gates become important
5. only promote new scenarios when they are structurally clean and stable across paths

This is more important than adding new dynamic variants or generic mechanism layering without practical signal.

## What Not to Do Yet

Avoid these until the practical loop exists:

- adding many more experiment scripts
- adding more dynamic variants without product use cases
- optimizing the UI before memory policies improve
- deep framework integration before the agent contract is stable

## Success Definition

The project reaches "practical agent" level when:

- the memory system improves real agent behavior on concrete tasks
- the agent can survive multi-turn and multi-session use
- retrieval is relevant and not noisy
- users can inspect why the agent recalled what it did
- the codebase has a clear product surface separate from research material

Current near-term success condition:

- `stable_v1` remains the canonical runtime
- procedure-aware product and long-task gates stay fully green
- conflict-heavy practical regression remains independently measurable with `conflict_continuity`
- real MiniMax-backed practical regressions continue to match the local story rather than collapse outside deterministic runs
