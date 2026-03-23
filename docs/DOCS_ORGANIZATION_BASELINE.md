# Docs 整理基线 / Docs Organization Baseline

> 基于 2026-03-23 代码状态整理

---

## 文档分层总览

| 状态 | 数量 | 文档 |
|------|------|------|
| ✅ Current | 4 | REFACTOR_PLAN, STABLE_AGENT_ROADMAP, DATA_ORGANIZATION_PLAN_V1, PHASE_SUMMARY_2026Q1 |
| ⚠️ Partial | 8 | AGENT_PRODUCT_PLAN, EXPERIMENT_PLAN_V1, HYPEREDGE_TAXONOMY_V1, HYPEREDGE_BATCH1_IMPLEMENTATION, CONTINUITY_METRICS_V1, TWO_STAGE_COMPETITIVE_RETRIEVAL_V1, INNOVATION_MAP_V1 |
| 📦 Archive | 2 | EXPERIMENT_INDEX, TASK_MiniMax_Agent_Implementation |

---

## 1. Current Valid - 仍可作为主线依据

### REFACTOR_PLAN.md
- **状态**: ✅ Current
- **描述**: 项目重构计划
- **对应代码**: 
  - `src/hypergraph_bistability/` - 规范包层
  - `src/agent/` - deprecated 兼容层
  - `src/core/` - deprecated 兼容层
- **落地程度**: 完全落地
- **建议**: 保留，作为架构指引

### STABLE_AGENT_ROADMAP.md
- **状态**: ✅ Current
- **描述**: Agent 运行时路线图
- **对应代码**:
  - `src/hypergraph_bistability/agent/runtime_profile.py` - stable_v1
  - `src/hypergraph_bistability/agent/query.py` - query layer
  - `src/hypergraph_bistability/evals/` - evaluation runners
- **落地程度**: 完全落地
- **建议**: 保留，作为 runtime 主线文档

### DATA_ORGANIZATION_PLAN_V1.md
- **状态**: ✅ Current
- **描述**: 数据组织方案（Event/Object/Hyperedge/State/Working Set）
- **对应代码**:
  - `src/hypergraph_bistability/memory/unified_node.py` - typed nodes
  - `src/hypergraph_bistability/agent/query.py` - working set
- **落地程度**: 部分落地（分层思想已实现，统一 schema 收口未完成）
- **建议**: 保留，更新统一 schema 部分

### PHASE_SUMMARY_2026Q1.md
- **状态**: ✅ Current
- **描述**: 2026 Q1 阶段总结
- **对应代码**: 多个模块
- **落地程度**: 验证完成
- **建议**: 保留，作为阶段里程碑

---

## 2. Partially Outdated - 部分过时

### AGENT_PRODUCT_PLAN.md
- **状态**: ⚠️ Partial
- **描述**: Agent 产品化计划
- **对应代码**:
  - ✅ working-set/query layer 已实现
  - ✅ write-from-docs 已实现
  - ⏳ planner/executor loop 部分实现
  - ❌ tool orchestration 不完整
- **建议**: 更新并保留，作为产品方向参考

### EXPERIMENT_PLAN_V1.md
- **状态**: ⚠️ Partial
- **描述**: 实验计划
- **对应代码**: `experiments/` 目录下各类实验
- **建议**: 归档或重写为实验记录

### HYPEREDGE_TAXONOMY_V1.md
- **状态**: ⚠️ Partial
- **描述**: 超图分类学
- **对应代码**:
  - `src/hypergraph_bistability/memory/unified_node.py`
  - `src/hypergraph_bistability/memory/policies/`
- **建议**: 更新实现细节后保留

### HYPEREDGE_BATCH1_IMPLEMENTATION.md
- **状态**: ⚠️ Partial
- **描述**: 批量实现规格
- **对应代码**: 已执行的部分实现
- **建议**: 标注完成状态后归档

### CONTINUITY_METRICS_V1.md
- **状态**: ⚠️ Partial
- **描述**: 连续性指标定义
- **对应代码**: `src/hypergraph_bistability/evals/metrics.py`
- **建议**: 与实际打分口径核对后更新

### TWO_STAGE_COMPETITIVE_RETRIEVAL_V1.md
- **状态**: ⚠️ Partial
- **描述**: 两阶段竞争检索（探索性）
- **对应代码**: 非默认 runtime
- **建议**: 保留作为探索分支参考

### INNOVATION_MAP_V1.md
- **状态**: ⚠️ Partial
- **描述**: 创新地图（研究方向库）
- **对应代码**: N/A
- **建议**: 保留作为研究方向库，不作为现状文档

---

## 3. Archive - 基本归档

### EXPERIMENT_INDEX.md
- **状态**: 📦 Archive
- **描述**: 实验索引
- **问题**: 与现状冲突（声称脚本已迁出但 src/ 仍保留）
- **建议**: 归档或删除

### TASK_MiniMax_Agent_Implementation.md
- **状态**: 📦 Archive
- **描述**: 一次性任务书
- **问题**: 文件编码有问题，内容可信度下降
- **建议**: 归档或删除

---

## 代码模块对照表

| 代码模块 | 对应文档 | 落地状态 |
|----------|----------|----------|
| `hypergraph_bistability/agent/runtime_profile.py` | STABLE_AGENT_ROADMAP | ✅ |
| `hypergraph_bistability/agent/query.py` | STABLE_AGENT_ROADMAP, DATA_ORGANIZATION_PLAN | ✅ |
| `hypergraph_bistability/agent/hypergraph_agent.py` | STABLE_AGENT_ROADMAP | ✅ |
| `hypergraph_bistability/agent/cli.py` | AGENT_PRODUCT_PLAN | ✅ |
| `hypergraph_bistability/memory/unified_node.py` | DATA_ORGANIZATION_PLAN, HYPEREDGE_TAXONOMY | ✅ |
| `hypergraph_bistability/memory/integrated_memory.py` | DATA_ORGANIZATION_PLAN | ✅ |
| `hypergraph_bistability/memory/durable_memory.py` | DATA_ORGANIZATION_PLAN | ✅ |
| `hypergraph_bistability/evals/` | CONTINUITY_METRICS | ✅ |
| `experiments/research/` | EXPERIMENT_PLAN_V1 | ✅ |

---

## 建议操作

1. **立即执行**:
   - 删除或归档 EXPERIMENT_INDEX.md
   - 删除或归档 TASK_MiniMax_Agent_Implementation.md

2. **短期计划**:
   - 更新 AGENT_PRODUCT_PLAN.md 标注已实现项
   - 核对 CONTINUITY_METRICS_V1.md 与 evals/metrics.py 打分口径

3. **长期计划**:
   - 将 REFACTOR_PLAN + STABLE_AGENT_ROADMAP + PHASE_SUMMARY 收敛为一份现状文档
   - 统一 schema 收口（DATA_ORGANIZATION_PLAN_V1）

---

*Last updated: 2026-03-23*
