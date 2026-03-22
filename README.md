# Hypergraph Bistability

Research and prototype code for hypergraph competitive dynamics, multistability, and a physics-grounded agent memory system.

## What This Repository Contains

The project has two layers:

- `hypergraph_bistability`: the package layer for reusable memory, control, agent, and core dynamics APIs
- `experiments`, `paper`, `results`, `figures`: research artifacts, verification scripts, and paper outputs

The main practical deliverable is a controllable memory module built around a `k x L` activation matrix with:

- bistable local dynamics
- inter-group competition `lambda`
- inter-layer coupling `mu`
- physics-grounded control through the critical coupling `lambda_c`

The current default agent path is now formalized as runtime profile `stable_v1`, which fixes the validated practical stack around:

- `hyperedge_expansion`
- conflict-aware retrieval
- confidence modifiers
- decision residue support
- continuity/product regression gates

## Recommended Entry Points

Use the new package namespace for application code:

```python
from hypergraph_bistability.memory import AgentMemory
from hypergraph_bistability.agent import HypergraphAgent
from hypergraph_bistability.control import compute_lambda_c
```

Legacy imports such as `from agent.agent_memory import AgentMemory` still work during the transition.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[llm]"
pip install -e ".[vector]"
pip install -e ".[dev]"
```

## Quick Start

### Memory Module

```python
from hypergraph_bistability.memory import AgentMemory

mem = AgentMemory(k=4, L=2, use_physics_control=True)
mem.group_labels = ["goals", "skills", "preferences", "context"]
mem.layer_labels = ["current", "history"]

mem.write("User prefers concise answers", group=0, layer=0)
result = mem.process_prompt("I want to learn coding but I'm also tired")

print(result["regime"])
print(mem.get_context_for_llm())
```

### Agent Demo

```python
from hypergraph_bistability.agent import HypergraphAgent

agent = HypergraphAgent(k=4, L=2, use_embeddings=False)
print(agent.chat("Help me debug a Python function"))
```

## CLI

After `pip install -e .`, these commands are available:

```bash
hypergraph-bistability demo-memory
hypergraph-bistability demo-agent
hypergraph-bistability chat-demo
hypergraph-bistability chat-demo --session ./demo_state.json
hypergraph-bistability run-evals --output ./evals.json
hypergraph-bistability run-evals --tier core --output ./evals-core.json
hypergraph-bistability run-evals --tier stress --output ./evals-stress.json
hypergraph-bistability run-llm-evals --output ./llm-evals.json
hypergraph-bistability run-llm-evals --tier core --output ./llm-evals-core.json
hypergraph-bistability run-product-regression --output ./product-regression.json
hypergraph-bistability run-long-task-regression --output ./long-task-regression.json
hypergraph-bistability run-llm-product-regression --output ./llm-product-regression.json
hypergraph-bistability run-llm-long-task-regression --output ./llm-long-task-regression.json
hypergraph-bistability run-mechanism-experiment --experiment procedure_memory --output ./procedure-memory.json
```

`run-llm-evals` now defaults to the current MiniMax evaluation profile:

- model: `MiniMax-M2.7`
- base URL: `https://api.minimaxi.com/anthropic`
- temperature: `0.01`

## Windows + MiniMax Recommended Path

If Python HTTPS requests are restricted on your machine, use the built-in PowerShell transport:

```powershell
python -m hypergraph_bistability.cli run-llm-evals --tier core --base-url https://api.minimaxi.com/anthropic --model MiniMax-M2.7 --force-powershell --output .\_llm_eval_results_core.json
```

For one-click local runs on Windows, use the provided scripts:

```powershell
& F:\hypergraph_bistability\scripts\run_minimax_eval_one_click.ps1
& F:\hypergraph_bistability\scripts\run_minimax_eval_stress_one_click.ps1
& F:\hypergraph_bistability\scripts\run_minimax_product_regression_one_click.ps1
& F:\hypergraph_bistability\scripts\run_minimax_long_task_regression_one_click.ps1
& F:\hypergraph_bistability\scripts\run_minimax_competition_one_click.ps1
& F:\hypergraph_bistability\scripts\run_minimax_associative_expansion_one_click.ps1
& F:\hypergraph_bistability\scripts\run_minimax_procedure_memory_one_click.ps1
```

These scripts:

- set the MiniMax key
- force the PowerShell transport
- write JSON output files for later inspection
- preserve `llm_debug` blocks with raw `thinking/text` metadata

## MiniMax Direct Terminal Test

For a direct local PowerShell smoke test that bypasses this agent's Python network limits, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\minimax_direct_test.ps1 -ApiKey "YOUR_KEY" -Transport anthropic
```

You can also test both compatibility paths:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\minimax_direct_test.ps1 -ApiKey "YOUR_KEY" -Transport both
```

This prints:

- parsed text output
- raw JSON response
- transport-specific errors if the request fails

## Tests

```bash
python -m pytest tests -q
```

## Streamlit Visualization

```bash
streamlit run src/agent/streamlit_app.py
```

## Package Layout

```text
src/hypergraph_bistability/
  agent/           Stable agent APIs
  control/         Critical coupling and control helpers
  core/            Reusable dynamics and potential utilities
  integrations/    Embeddings and LLM adapters
  memory/          Primary memory modules
  visualization/   Visualization entry points
  cli.py           Small built-in demos
```

## Current Research Layout

```text
src/
  agent/               Legacy implementation modules
  core/                Legacy core modules
  multi_stability/     Research modules used by tests and experiments
  *.py                 Experimental scripts and figure generators
paper/                 Paper drafts
results/               JSON outputs and research summaries
figures/               Generated figures
tests/                 Unit and integration tests
```

## Next Refactor Direction

The current package layer is a compatibility-focused first pass. The next structural step is to move research scripts out of `src/` into dedicated `experiments/` directories and make `src/hypergraph_bistability/` the canonical implementation location.
