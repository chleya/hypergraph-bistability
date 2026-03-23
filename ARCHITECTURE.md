# Architecture Documentation

## Project Overview

**hypergraph-bistability** is a research project exploring complex systems dynamics on hypergraphs, with applications to agent memory systems. The project combines:

1. **Theoretical Research**: Hypergraph competitive dynamics, bistability, multistability
2. **Agent System**: Memory-augmented AI agents with working-set context
3. **Evaluation**: Regression and practical scenario testing

## Directory Structure

```
hypergraph_bistability/
├── src/hypergraph_bistability/   # Main package (canonical)
│   ├── agent/                    # Agent runtime & control
│   ├── memory/                   # Memory systems & policies
│   ├── core/                     # Scientific core (hypergraph dynamics)
│   ├── control/                  # Critical threshold computation
│   ├── evals/                    # Evaluation runners
│   ├── integrations/             # External integrations
│   ├── visualization/            # Plotting & debugging
│   ├── experiments/              # Research experiments
│   └── cli.py                   # Command-line interface
│
├── src/agent/                    # Deprecated - use hypergraph_bistability.agent
├── src/core/                    # Deprecated - use hypergraph_bistability.core
│
├── experiments/                  # Verification & research scripts
├── evals/                       # Evaluation configs & scenarios
├── figures/                     # Generated plots
├── paper/                       # Academic paper drafts
├── results/                     # Experiment results (JSON)
├── scripts/                     # Operational scripts
└── tests/                       # Unit & integration tests
```

## Module Descriptions

### Core Layer (`hypergraph_bistability.core`)

Scientific core for hypergraph dynamics research:

| Module | Purpose |
|--------|---------|
| `model.py` | MultiGroupHypergraph, MultiLayerHypergraph data structures |
| `dynamics.py` | Growth, fusion, split, deletion rules |
| `potential.py` | Double-well, multi-well potential functions |
| `noise.py` | Gaussian noise, noise effect computation |

**Use Case**: Theoretical analysis, phase transition studies

### Control Layer (`hypergraph_bistability.control`)

Critical threshold computation:

| Function | Purpose |
|----------|---------|
| `compute_lambda_c` | Calculate critical coupling threshold |
| `get_all_lambda_c` | Get thresholds for multiple configurations |
| `power_law_approximation` | Power law fitting for scaling analysis |

**Use Case**: Determining stability boundaries in hypergraph systems

### Memory Layer (`hypergraph_bistability.memory`)

Memory management with policies:

| Component | Description |
|-----------|-------------|
| `AgentMemory` | Core memory with decay dynamics |
| `AgentMemoryEnhanced` | Extended memory with LLM integration |
| `DurableMemory` | Layered storage (Working → Episodic → Durable) |
| `UnifiedNode` | Unified memory + skills system |
| `IntegratedAgentMemory` | Combined hypergraph + unified node memory |

**Policies**:
- `WritePolicy`: Decides what to persist
- `RetrievalPolicy`: Hybrid embedding + keyword retrieval
- `DecayPolicy`: Memory strength decay over time
- `PromotionPolicy`: Working → Episodic promotion

### Agent Layer (`hypergraph_bistability.agent`)

Agent runtime and control:

| Component | Description |
|-----------|-------------|
| `HypergraphAgent` | Main agent with memory + control |
| `QueryLayer` | Working-set state queries |
| `AdaptiveController` | Cognitive mode adaptation |
| `SessionState` | Session management |
| `runtime/` | Turn processing, context assembly |

### Evaluation Layer (`hypergraph_bistability.evals`)

Scenario-based evaluation:

| Runner | Purpose |
|--------|---------|
| `run_product_regression` | Core product functionality |
| `run_conflict_regression` | Conflict handling |
| `run_continuity_regression` | Task continuity |
| `run_long_task_regression` | Long-running tasks |
| `run_practical_robustness_regression` | Robustness tests |

## API Layers

### Layer 1: High-Level Entry Points (Package Root)

```python
from hypergraph_bistability import (
    HypergraphAgent,      # Main agent
    AgentMemory,          # Core memory
    AgentMemoryEnhanced,  # Extended memory
    compute_lambda_c,     # Critical threshold
)
```

### Layer 2: Submodule APIs

```python
# Agent APIs
from hypergraph_bistability.agent import (
    HypergraphAgent,
    QueryLayer,
    WorkingSet,
    TaskState,
    ConflictInfo,
)

# Memory APIs
from hypergraph_bistability.memory import (
    DurableMemory,
    UnifiedNode,
    WritePolicy,
    RetrievalPolicy,
)

# Core APIs
from hypergraph_bistability.core import (
    MultiGroupHypergraph,
    HypergraphDynamics,
)

# Evaluation APIs
from hypergraph_bistability.evals import (
    run_eval_suite,
    DEFAULT_EVAL_SCENARIOS,
)
```

### Layer 3: Internal APIs

Internal modules not guaranteed stable:

```python
# Internal - may change
from hypergraph_bistability.agent.runtime import turn_processor
from hypergraph_bistability.memory.policies import write_policy
```

## Data Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│           HypergraphAgent                │
│  ┌─────────────────────────────────┐   │
│  │    TurnProcessor                │   │
│  │  1. Build context               │   │
│  │  2. Query working set           │   │
│  │  3. Assemble messages           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         IntegratedAgentMemory            │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Hypergraph   │  │ UnifiedNode  │    │
│  │ (structure)  │  │ (skills)     │    │
│  └──────────────┘  └──────────────┘    │
│  ┌──────────────────────────────────┐   │
│  │        Policy Layer               │   │
│  │  WritePolicy                     │   │
│  │  RetrievalPolicy                 │   │
│  │  DecayPolicy                     │   │
│  │  PromotionPolicy                 │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│          DurableMemoryStore              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Working │ │Episodic │ │ Durable │   │
│  │ (mem)   │ │ (SQLite)│ │ (SQLite)│   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘
```

## Stable API Reference

### Package Root (`hypergraph_bistability`)

| Export | Type | Description |
|--------|------|-------------|
| `HypergraphAgent` | Class | Main agent with memory and control |
| `AgentMemory` | Class | Core memory system |
| `AgentMemoryEnhanced` | Class | Extended memory with LLM |
| `compute_lambda_c` | Function | Critical threshold calculation |
| `get_all_lambda_c` | Function | Multiple thresholds |
| `power_law_approximation` | Function | Power law fitting |

### Agent Module (`hypergraph_bistability.agent`)

| Export | Type | Description |
|--------|------|-------------|
| `HypergraphAgent` | Class | Main agent |
| `QueryLayer` | Class | Working-set state queries |
| `WorkingSet` | Dataclass | Current task context |
| `TaskState` | Dataclass | Task status information |
| `ConflictInfo` | Dataclass | Active conflicts |
| `AdaptiveController` | Class | Cognitive mode control |
| `SessionState` | Class | Session management |

### Memory Module (`hypergraph_bistability.memory`)

| Export | Type | Description |
|--------|------|-------------|
| `AgentMemory` | Class | Core memory |
| `AgentMemoryEnhanced` | Class | Extended memory |
| `DurableMemory` | Class | Layered storage |
| `UnifiedNode` | Class | Memory + skills |
| `IntegratedAgentMemory` | Class | Combined system |
| `WritePolicy` | Class | Persistence decisions |
| `RetrievalPolicy` | Class | Retrieval strategy |
| `DecayPolicy` | Class | Memory decay |
| `PromotionPolicy` | Class | Layer promotion |

### Core Module (`hypergraph_bistability.core`)

| Export | Type | Description |
|--------|------|-------------|
| `MultiGroupHypergraph` | Class | Hypergraph data structure |
| `MultiLayerHypergraph` | Class | Multi-layer extension |
| `HypergraphDynamics` | Class | Dynamics simulation |
| `compute_order_parameter` | Function | Order parameter M |
| `double_well_potential` | Function | Potential V(x) |
| `find_fixed_points` | Function | Fixed point analysis |

### Evals Module (`hypergraph_bistability.evals`)

| Export | Type | Description |
|--------|------|-------------|
| `run_eval_suite` | Function | Run all evaluations |
| `run_product_regression` | Function | Product tests |
| `run_conflict_regression` | Function | Conflict tests |
| `DEFAULT_EVAL_SCENARIOS` | List | Standard scenarios |

## Deprecated Paths

Old import paths still work but show deprecation warnings:

| Deprecated | Use Instead |
|------------|-------------|
| `from agent import X` | `from hypergraph_bistability.agent import X` |
| `from core import X` | `from hypergraph_bistability.core import X` |

## CLI Usage

```bash
# Run agent
hypergraph-bistability run

# Run evaluation
hypergraph-bistability eval --suite product

# Query working set
hypergraph-bistability ws

# Demo memory
hypergraph-bistability demo-memory
```

## Version

Current version: `0.1.0` (unstable)

---

*Last updated: 2026-03-23*
