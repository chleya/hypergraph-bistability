# Docs Index

> 最后更新: 2026-03-23

## 目录结构

```
docs/
├── current/          # 当前有效文档
├── specs/            # 规范文档
├── history/          # 历史/归档文档
└── index.md          # 本索引
```

---

## current/ - 当前有效文档

| 文档 | 状态 | 落地程度 | 代码对应 |
|------|------|----------|----------|
| [REFACTOR_PLAN.md](current/REFACTOR_PLAN.md) | Current | 70% | src/hypergraph_bistability |
| [STABLE_AGENT_ROADMAP.md](current/STABLE_AGENT_ROADMAP.md) | Current | 75% | agent/runtime_profile.py, query.py |
| [PHASE_SUMMARY_2026Q1.md](current/PHASE_SUMMARY_2026Q1.md) | Current | 80% | 多模块 |
| [DATA_ORGANIZATION_PLAN_V1.md](current/DATA_ORGANIZATION_PLAN_V1.md) | Current | 60% | unified_node, durable_memory, query |

---

## specs/ - 规范文档

| 文档 | 状态 | 描述 |
|------|------|------|
| [HYPEREDGE_TAXONOMY_V1.md](specs/HYPEREDGE_TAXONOMY_V1.md) | Spec | 超图分类学规范 |
| [CONTINUITY_METRICS_V1.md](specs/CONTINUITY_METRICS_V1.md) | Spec | 连续性指标定义 |

---

## history/ - 历史/归档文档

| 文档 | 状态 | 说明 |
|------|------|------|
| AGENT_PRODUCT_PLAN.md | Archived | 产品化计划，已部分过时 |
| EXPERIMENT_INDEX.md | Archived | 与现状冲突 |
| EXPERIMENT_PLAN_V1.md | Archived | 实验计划，已过时 |
| HYPEREDGE_BATCH1_IMPLEMENTATION.md | Archived | 实施记录 |
| INNOVATION_MAP_V1.md | Archived | 研究方向库 |
| TASK_MiniMax_Agent_Implementation.md | Archived | 一次性任务书，有编码问题 |
| TWO_STAGE_COMPETITIVE_RETRIEVAL_V1.md | Archived | 探索性支线 |

---

## 后续计划

1. **短期**: 更新 AGENT_PRODUCT_PLAN, 核对 CONTINUITY_METRICS
2. **长期**: 收敛为一份"当前真相"现状文档
