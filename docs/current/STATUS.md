# Hypergraph-Bistability 项目现状文档

> Last updated: 2026-03-23
> 状态: ✅ Current

## 1. 项目概述

### 定位

- **研究层**: 超图竞争动力学 + 复杂性重建理论
- **产品层**: Agent 工作记忆基础设施

### 核心理论贡献

| 理论 | 状态 | 描述 |
|------|------|------|
| 双稳态 | ✅ | k_c ≈ 2.35 处相变，双吸引子 M₁*≈0.45, M₂*≈1.0 |
| 阶跃-响应循环 | ✅ | 差异响应机制，复杂度守恒 |
| 原地重建 | ✅ | 96.4% 核心结构为新生成，非输运 |
| 多稳态扩展 | ✅ | N_att = 2^{k×L}, λ_c ≈ 0.70/k² |

### 产品验收

| 指标 | 得分 |
|------|------|
| task_continuation | 1.000 |
| blocker_preservation | 1.000 |
| decision_continuity | 1.000 |
| procedure_continuity | 1.000 |
| conflict_continuity | 1.000 |
| repeated_work_avoidance | 1.000 |

---

## 2. 代码架构

### 目录结构

```
src/hypergraph_bistability/
├── agent/                 # Agent 运行时
│   ├── hypergraph_agent.py    # 主 Agent
│   ├── runtime_profile.py    # stable_v1 配置
│   ├── query.py              # Query Layer
│   ├── runtime/              # 运行时组件
│   │   ├── turn_processor.py
│   │   └── context_assembler.py
│   └── cli.py               # CLI 入口
├── memory/                 # 记忆系统
│   ├── unified_node.py      # 统一节点
│   ├── integrated_memory.py # 集成内存
│   ├── durable_memory.py   # 持久层
│   └── policies/           # 策略模块
│       ├── write_policy.py
│       ├── retrieval_policy.py
│       ├── decay_policy.py
│       └── promotion_policy.py
├── evals/                  # 评估系统
│   ├── metrics.py
│   ├── scenarios.py
│   ├── evaluator.py
│   └── baselines.py
└── core/                   # 核心物理层

experiments/               # 研究实验
research/                  # 研究脚本
verification/              # 验证实验
control/                   # 控制实验
theory/                    # 理论实验

docs/
├── current/               # 当前有效文档
├── specs/                 # 规范文档
├── history/               # 历史文档
└── index.md               # 文档索引
```

### 稳定 API (6 个导出)

```python
from hypergraph_bistability import (
    HypergraphAgent,      # 主 Agent 类
    AgentMemory,          # 基础内存
    AgentMemoryEnhanced,  # 增强内存
    compute_lambda_c,     # 临界 lambda 计算
    get_all_lambda_c,     # 获取所有临界点
    power_law_approximation  # 幂律近似
)
```

### 废弃路径 (兼容层)

```python
# 旧路径仍然可用，但会显示 DeprecationWarning
from agent import X       # → from hypergraph_bistability.agent import X
from core import Y       # → from hypergraph_bistability.core import Y
```

---

## 3. 已实现功能

### Memory Pipeline

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| Write Policy | ✅ | memory/policies/write_policy.py |
| Retrieval Policy | ✅ | memory/policies/retrieval_policy.py |
| Decay Policy | ✅ | memory/policies/decay_policy.py |
| Promotion Policy | ✅ | memory/policies/promotion_policy.py |
| Working Memory | ✅ | memory/unified_node.py (Layer 0) |
| Episodic Memory | ✅ | memory/durable_memory.py (Layer 1) |
| Durable Memory | ✅ | memory/durable_memory.py (Layer 2) |

### Query Layer

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| WorkingSet | ✅ | agent/query.py |
| TaskState | ✅ | agent/query.py |
| ConflictInfo | ✅ | agent/query.py |
| DecisionResidue | ✅ | agent/query.py |
| ProcedureInfo | ✅ | agent/query.py |
| HandoffBundle | ✅ | agent/query.py |
| Working Set Context Injection | ✅ | turn_processor.py |

### Runtime

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| stable_v1 | ✅ | agent/runtime_profile.py |
| CLI | ✅ | cli.py |
| Write-from-docs | ✅ | cli.py |
| MiniMax 集成 | ✅ | scripts/ |
| Query API CLI (/ws) | ✅ | cli.py |

### Evaluation

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| Continuity Metrics | ✅ | evals/metrics.py |
| Practical Regression | ✅ | evals/scenarios.py |
| Product Regression | ✅ | evals/runner.py |
| Long-Task Regression | ✅ | evals/runner.py |
| Conflict Regression | ✅ | evals/runner.py |

---

## 4. 未实现功能

### 高优先级

| 功能 | 说明 |
|------|------|
| Summarizer/Compressor | 记忆压缩 |
| Tool Execution | Tool 调用编排 |
| Planner/Executor Loop | 规划-执行循环 |
| Message Routing | 消息路由策略 |

### 中优先级

| 功能 | 说明 |
|------|------|
| Checkpointing | 高级检查点 |
| Vector Store 外置 | 外部向量存储 |
| Streaming | 流式响应 |
| Web UI | Streamlit 界面 |

### 低优先级

| 功能 | 说明 |
|------|------|
| Framework Adapters | LangChain 适配器 |
| Service Deployment | 服务部署 |

---

## 5. 控制参数

### 核心参数

| 参数 | 描述 | 典型值 |
|------|------|--------|
| λ (lambda) | 耦合强度 | 0.0 - 1.0 |
| λ_c | 临界耦合 | ≈ 0.70/k² |
| k | 超图阶数 | ≥ 2 |
| L | 层数 | 1 - 3 |

### 控制命令

```bash
# 运行 Agent
python -m hypergraph_bistability.cli run-agent

# 写文档
python -m hypergraph_bistability.cli write-from-docs --doc-path <path>

# 运行回归测试
python -m hypergraph_bistability.cli run-product-regression
python -m hypergraph_bistability.cli run-continuity-regression

# 查询状态
python -m hypergraph_bistability.cli /ws
```

---

## 6. 相关文档

| 文档 | 位置 | 状态 |
|------|------|------|
| ARCHITECTURE.md | 根目录 | ✅ |
| REFACTOR_PLAN.md | docs/current/ | ✅ |
| STABLE_AGENT_ROADMAP.md | docs/current/ | ✅ |
| PHASE_SUMMARY_2026Q1.md | docs/current/ | ✅ |
| DATA_ORGANIZATION_PLAN_V1.md | docs/current/ | ✅ |
| HYPEREDGE_TAXONOMY_V1.md | docs/specs/ | ✅ |
| CONTINUITY_METRICS_V1.md | docs/specs/ | ✅ |

---

## 7. 下一步建议

### 短期 (1-2 周)

1. 完成 Retrieval 基线对比测试
2. 加强 Session 恢复机制
3. 添加更多 stress 测试场景

### 中期 (1 个月)

1. 实现 Summarizer 模块
2. 添加基本 Tool Execution 支持
3. 完善持久化 schema 版本管理

### 长期 (3 个月)

1. 添加 Web UI
2. 添加 Streaming 支持
3. 完善 Service 部署

---

*本文件是项目现状的唯一真相来源。详细信息请参考 docs/current/ 和 docs/specs/。*
