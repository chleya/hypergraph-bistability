# Hypergraph Bistability

研究容量约束竞争动力学中的双稳态现象 / Studying bistability in capacity-constrained competitive dynamics.

## Core Discovery / 核心发现

The system reduces to a 1D stochastic dynamical system:

```
dM/dt = F(M) + η(t)
```

where M = |C_max| / |E| is the order parameter (maximum faction ratio).

### Bistable Structure / 双稳态结构

| Attractor | M Value | Stability |
|-----------|---------|-----------|
| M₁* | ≈ 0.45 | Deep well, stable |
| M₀ | ≈ 0.6 | Barrier, unstable |
| M₂* | ≈ 1.0 | Shallow well, weakly stable |

Key features:
- Asymmetric double-well potential
- Basin boundary ≈ 0.6
- Critical point K_c/N ≈ 0.35

---

## Agent Memory Module / 智能体记忆模块

An AI agent memory system based on hypergraph multistability theory with physics-grounded λ_c control.

### Quick Start / 快速开始

```python
from agent.agent_memory import AgentMemory

mem = AgentMemory(k=4, L=2, use_physics_control=True)
mem.group_labels = ["goals", "skills", "preferences", "context"]

# Write to memory
mem.write("User prefers concise answers", group=0, layer=0)

# Process prompt with physics-based λ control
result = mem.process_prompt("I want to learn coding but I'm also tired")
print(f"Conflict: {result['conflict_level']:.1%}")

# Switch modes
mem.switch_mode("exploratory")  # Light coupling
mem.switch_mode("focused")      # Strong selection
```

### Physics Control / 物理控制

Critical coupling formula / 临界耦合公式:
```
λ_c = 3h²l² / [h² + (k-1)l²]
```

| k | λ_c |
|---|-----|
| 2 | 0.167 |
| 3 | 0.068 |
| 4 | 0.044 |

Control via proximity ratio r = λ/λ_c:
- r < 0.5: Multi-attractor (independent groups)
- r > 0.85: Near-WTA collapse (synchronized)

### Mode Presets / 模式预设

| Mode | r | λ (k=4) | Use Case |
|------|---|---------|----------|
| neutral | 0.0 | 0.000 | Independent groups |
| exploratory | 0.3 | 0.013 | Light coupling |
| focused | 0.85 | 0.037 | Strong selection |
| sync | 0.5 | 0.022 | Layer sync |
| creative | 0.5 | 0.022 | Layer anti-sync |

## Installation / 安装

```bash
pip install -e .

# With LLM integration
pip install -e ".[llm]"
```

## Running Tests / 运行测试

```bash
# All pytest tests
python -m pytest tests/ -v

# LLM integration tests
python src/agent/llm_integration_test.py

# Dialogue scenarios
python src/agent/test_dialogue_scenarios.py

# Physics demo
python -c "from agent.agent_memory import physics_demo; physics_demo()"
```

## Project Structure / 项目结构

```
src/
├── agent/
│   ├── agent_memory.py          # Core memory module
│   ├── agent_memory_enhanced.py # LLM-enhanced version
│   ├── llm_integration.py       # LangChain tools
│   └── test_dialogue_scenarios.py
├── core/
│   ├── model.py, dynamics.py, noise.py, potential.py
│   └── memory_types.py
└── hypergraph_control.py        # λ_c computation

paper/
├── paper_final.md               # Main paper
└── Results_Agent_Memory.md      # Performance indicators
```

## File Overview / 文件概览

- `figures/` - Key figures / 关键图表
- `paper/` - Paper drafts / 论文草稿
- `src/` - Source code / 源代码
- `tests/` - Test suite / 测试套件

---

*Created: 2026-03-17*
*Updated: 2026-03-20*
