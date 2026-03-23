# Hypergraph-Bistability 项目现状文档

> Last updated: 2026-03-23
> 状态: ✅ Current

## 0. 收敛任务清单 (2026-03-23)

| # | 任务 | 状态 | 说明 |
|---|------|------|------|
| 1 | 项目身份收窄为一句话 | ⏳ | 定义唯一主线描述 |
| 2 | 唯一真相层 | ✅ | src/hypergraph_bistability/ 已确认 |
| 3 | 评测主线固定 | ✅ | 3 条正式门禁已定义 |
| 4 | 文档状态管理 | ✅ | current/specs/history 已建立 |
| 5 | 根目录产物归位 | ✅ | JSON/DB 已在 results/artifacts |
| 6 | Memory 抽象统一 | ✅ | 4 层职责已明确 |
| 7 | Query Layer 正式接口 | ✅ | 已标记为正式产品接口 |
| 8 | 大文件拆分 | ⚠️ hypergraph_agent.py (2356行) 已识别拆分边界 |

**已识别分组**:
- LLM Transport (~900行): _call_llm_via_* 方法
- Query Layer (~450行): query_* 方法 → 可整合到 agent/query.py
- Core Agent (~200行): chat, process_turn, generate_response
- Persistence (~150行): save, load, get_session_state
- Document (~150行): write_from_documents

**风险评估**: 高风险拆分，建议后续处理
| 9 | 研究 vs 产品分开表述 | ✅ | 已区分三层表述 |
| 10 | MiniMax/Windows 封装 | ✅ integrations/ 模块已存在 |

**现状**:
- integrations/llm.py: LLM 代理适配
- integrations/embeddings.py: 向量嵌入适配
- 需补充: PowerShell fallback, 编码处理, 错误诊断

---

## 1. 项目概述

### 定位 (一句话)

> **这是一个以 stable_v1 为主线的 agent working-memory runtime，研究代码是配套验证层。**

- **研究层**: 超图竞争动力学 + 复杂性重建理论
- **产品层**: Agent 工作记忆基础设施

### 核心理论贡献 (Validated Mechanism)

| 理论 | 状态 | 描述 |
|------|------|------|
| 双稳态 | ✅ | k_c ≈ 2.35 处相变，双吸引子 M₁*≈0.45, M₂*≈1.0 |
| 阶跃-响应循环 | ✅ | 差异响应机制，复杂度守恒 |
| 原地重建 | ✅ | 96.4% 核心结构为新生成，非输运 |
| 多稳态扩展 | ✅ | N_att = 2^{k×L}, λ_c ≈ 0.70/k² |

### Runtime 能力 (Runtime-Enabled Capability)

- Query Layer 工作集查询
- Working Set Context Injection
- Continuity Metrics 评估

### 产品行为 (Production-Ready Behavior)

- stable_v1 runtime
- CLI 接口
- Product/Long-task/Conflict 回归测试

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

### Memory Pipeline (4 层抽象)

| 抽象层 | 职责 | 代码位置 |
|--------|------|----------|
| **AgentMemory** | 在线运行态，LLM 集成 | memory/agent_memory.py, memory/llm_memory.py |
| **DurableMemory** | 持久化记忆对象层 | memory/durable_memory.py |
| **UnifiedNode** | 统一知识/技能节点 | memory/unified_node.py |
| **IntegratedAgentMemory** | 组合入口 | memory/integrated_memory.py |

#### Policies (策略层)

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| Write Policy | ✅ | memory/policies/write_policy.py |
| Retrieval Policy | ✅ | memory/policies/retrieval_policy.py |
| Decay Policy | ✅ | memory/policies/decay_policy.py |
| Promotion Policy | ✅ | memory/policies/promotion_policy.py |

### Query Layer (正式接口)

Query Layer 是项目的"产品接口"，作为稳定 API 经营。

| 接口 | 职责 | 代码位置 |
|------|------|----------|
| **WorkingSet** | 当前工作集快照 | agent/query.py |
| **TaskState** | 任务状态 | agent/query.py |
| **ConflictInfo** | 冲突信息 | agent/query.py |
| **DecisionResidue** | 决策残留 | agent/query.py |
| **ProcedureInfo** | 过程信息 | agent/query.py |
| **HandoffBundle** | 交接包 | agent/query.py |
| **QueryLayer** | 统一查询入口 | agent/query.py |
| Working Set Context Injection | 自动注入到 prompt | turn_processor.py |

#### CLI 接口

```bash
# 查询工作集状态
python -m hypergraph_bistability.cli /ws
```

### Runtime

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| stable_v1 | ✅ | agent/runtime_profile.py |
| CLI | ✅ | cli.py |
| Write-from-docs | ✅ | cli.py |
| MiniMax 集成 | ✅ | scripts/ |
| Query API CLI (/ws) | ✅ | cli.py |

### Evaluation (评测主线)

#### 正式门禁 (3 条)

| 回归测试 | CLI 命令 | 状态 |
|----------|----------|------|
| Product Regression | `run-product-regression` | ✅ |
| Long-Task Regression | `run-long-task-regression` | ✅ |
| Conflict Practical Regression | `run-conflict-regression` | ✅ |

#### Sidecar (降级)

以下不再是正式门禁，仅供实验验证：

| 回归测试 | CLI 命令 | 说明 |
|----------|----------|------|
| Continuity Regression | `run-continuity-regression` | 已合并到 product |
| Practical Sidecar | `run-practical-sidecar-regression` | 实验性质 |
| Conflict Sidecar | `run-conflict-sidecar-regression` | 实验性质 |
| Practical Robustness | `run-practical-robustness-regression` | 实验性质 |
| Mechanism Experiment | `run-mechanism-experiment` | 纯研究 |

#### 运行命令

```bash
# 正式门禁
python -m hypergraph_bistability.cli run-product-regression
python -m hypergraph_bistability.cli run-long-task-regression
python -m hypergraph_bistability.cli run-conflict-regression
```

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
