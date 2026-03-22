# Refactor Plan

## Goal

Turn the repository into a two-layer project:

- a stable package for reusable memory, control, agent, and visualization code
- a clearly separated research area for experiments, figures, paper outputs, and archived prototypes

This plan assumes incremental migration. Existing tests and legacy imports should keep working until each area is fully moved.

## Target Repository Shape

```text
src/
  hypergraph_bistability/
    agent/
    control/
    core/
    integrations/
    memory/
    visualization/
    cli.py

experiments/
  theory/
  verification/
  control/
  reconstruction/
  figures/
  archived/

tests/
  unit/
  integration/
  regression/

docs/
paper/
results/
figures/
examples/
```

## Design Rules

1. `src/hypergraph_bistability/` is the only public package namespace.
2. Research scripts must not define the public API.
3. Every experiment should have one owner script, one result directory, and one short summary.
4. Historical variants such as `_v2`, `_v3`, `_final`, `_success`, `_aggressive` should move out of the main source path unless they are still active.
5. Numerical claims used in the paper should be covered by regression tests.

## Current State

The repository currently contains:

- a new package facade under `src/hypergraph_bistability/`
- legacy implementation modules in `src/agent`, `src/core`, `src/multi_stability`
- many root-level experimental scripts in `src/`
- paper, figures, and result artifacts mixed with active code

The first package pass is complete, but implementation still lives mostly in legacy modules.

## Migration Phases

### Phase 1: Stabilize the Public Package

Objective:
- make the new package namespace the default way to import the project

Tasks:
- keep re-export wrappers in `src/hypergraph_bistability/`
- move stable implementations from `src/agent` and `src/core` into package-local modules
- leave compatibility shims behind in legacy locations
- add examples that only use package imports

Exit criteria:
- README, examples, and tests use `hypergraph_bistability.*`
- no new code depends on direct `src/agent` or `src/core` imports

### Phase 2: Separate Experiments from Product Code

Objective:
- remove research scripts from the main source root

Tasks:
- create `experiments/theory`, `experiments/verification`, `experiments/control`, `experiments/reconstruction`
- move active experiment scripts there with minimal import rewrites
- move stale variants into `experiments/archived`
- keep a small compatibility layer only where needed

Candidate moves:
- `Q1_*`, `Q2_*`, `Q3_*`, `Q4_*`, `verify_*`, `verification_*` -> `experiments/verification/`
- `experiment_*`, `control_*`, `PPO_*`, `q_learning_control.py` -> `experiments/control/`
- `generate_*fig*`, `final_figures.py`, `bridge_fm.py` -> `experiments/figures/`
- `tristability_*`, `multistable_*`, repeated exploratory scripts -> `experiments/archived/` unless still active

Exit criteria:
- `src/` root is no longer used as an experiment dumping ground
- active experiments are grouped by topic and easy to find

### Phase 3: Normalize Results and Figure Pipelines

Objective:
- make experiment outputs reproducible and navigable

Tasks:
- standardize output directories under `results/<experiment_name>/`
- standardize figure directories under `figures/<experiment_name>/`
- define one config block or CLI interface per experiment
- make summary markdown files point to exact result JSON files and figure paths

Exit criteria:
- each active experiment can be rerun from one command
- paper figures can be traced back to a script and a result file

### Phase 4: Test and Regression Coverage

Objective:
- convert key claims into protected behavior

Tasks:
- split tests into `unit`, `integration`, `regression`
- add regression tests for `lambda_c`, mode presets, `set_n_high`, and selected phase behavior
- add integration tests for `HypergraphAgent`, embeddings fallback behavior, and persistence

Exit criteria:
- package behavior is protected during migration
- theory-critical values are asserted in tests

### Phase 5: Remove Transitional Debt

Objective:
- finish the move and reduce duplicate implementations

Tasks:
- delete unused wrappers once migration is complete
- remove stale scripts that have archived equivalents
- update paper references if script paths changed
- simplify package metadata and imports

Exit criteria:
- package implementation is canonical
- archived material is preserved but not mixed with active code

## Immediate Execution Batches

### Batch A: Canonicalize Core Implementations

Scope:
- move stable code from legacy `src/agent` and `src/core` into package-local modules

Files to prioritize:
- `src/agent/agent_memory.py`
- `src/agent/agent_memory_enhanced.py`
- `src/agent/hypergraph_agent.py`
- `src/agent/adaptive_controller.py`
- `src/agent/embedding_memory.py`
- `src/agent/llm_integration.py`
- `src/core/model.py`
- `src/core/dynamics.py`
- `src/core/noise.py`
- `src/core/potential.py`
- `src/hypergraph_control.py`

Deliverables:
- package-local real implementations
- legacy wrappers that forward to the new locations

### Batch B: Establish Experiment Directories

Scope:
- create the `experiments/` tree and move active scripts without changing behavior

Files to prioritize:
- `src/Q1_lambda_c_scaling.py`
- `src/Q1_N_att_scaling.py`
- `src/Q2_lambda_mu_phase.py`
- `src/Q3_noise_escape.py`
- `src/Q4_asymmetric_k.py`
- `src/verification_a.py`
- `src/verification_a_v2.py`
- `src/verification_b.py`
- `src/verify_layered_cascade.py`
- `src/verify_control_strategies.py`
- `src/verify_ppo_control.py`

Deliverables:
- new experiment locations
- readme or index mapping old names to new paths

### Batch C: Archive Script Sprawl

Scope:
- move repeated exploration scripts out of the active path

Examples:
- `tristability_*`
- `multistable_*`
- `noise_*`
- `quick_*`
- `test_*` scripts under `src/` that are not the official test suite

Deliverables:
- a curated active experiment set
- an archive section for historical prototypes

## Risk Controls

1. Keep existing tests green after each batch.
2. Do not move paper or result files and code in the same batch unless path references are updated together.
3. Leave compatibility wrappers when moving modules across namespaces.
4. Avoid changing numerical behavior while doing structural migration.

## Success Criteria

The refactor is successful when:

- new users can understand the repository from the top-level README
- application code imports only from `hypergraph_bistability.*`
- experiments are grouped and reproducible
- legacy script sprawl is no longer on the critical path
- key research claims are guarded by tests

## Recommended Next Step

Implement Batch A first:

- move the stable memory, agent, and control implementations into `src/hypergraph_bistability/`
- keep the legacy modules as compatibility shims
- update tests and examples to prefer package imports

This gives the project a real canonical codebase before any large experiment migration begins.
