# Practical Agent Productization Plan

> Last updated: 2026-03-23
> Status: ⚠️ Partial - 部分实现

## Objective

Move the project from a research prototype into a practical agent system.

## Current State (2026-03-23)

### ✅ 已实现

| 优先级 | 项目 | 状态 | 对应代码 |
|--------|------|------|----------|
| P0 | Agent Contract 定义 | ✅ | hypergraph_agent.py turn pipeline |
| P1 | Memory Pipeline - Write Policy | ✅ | memory/policies/write_policy.py |
| P1 | Memory Pipeline - Retrieval Policy | ✅ | memory/policies/retrieval_policy.py |
| P1 | Memory Pipeline - Decay Policy | ✅ | memory/policies/decay_policy.py |
| P1 | Memory Pipeline - Promotion Policy | ✅ | memory/policies/promotion_policy.py |
| P2 | Memory Typing (分层) | ✅ | unified_node.py (Working/Episodic/Durable) |
| P2 | Query Layer | ✅ | agent/query.py (WorkingSet, TaskState, etc.) |
| P3 | Stable Runtime (stable_v1) | ✅ | agent/runtime_profile.py |
| P3 | CLI Interface | ✅ | cli.py |
| P3 | Write-from-docs | ✅ | cli.py write-from-docs |
| P4 | Persistence (JSON) | ✅ | durable_memory.py |
| P4 | Working Set Context Injection | ✅ | turn_processor.py, context_assembler.py |
| P6 | Evals - Continuity Metrics | ✅ | evals/metrics.py |
| P6 | Evals - Practical Regression | ✅ | evals/scenarios.py |

### ⏳ 部分实现 / 进行中

| 优先级 | 项目 | 状态 | 说明 |
|--------|------|------|------|
| P3 | Retrieval 对比基线 | ⏳ | 已有 hybrid retrieval，需正式对比测试 |
| P4 | Session 恢复 | ⏳ | 基础 JSON persistence，需加强 |
| P4 | 版本化 Schema | ⏳ | 需完善 |
| P6 | Real LLM 对比测试 | ⏳ | 已有 MiniMax 路径，需系统化 |

### ❌ 未实现

| 优先级 | 项目 | 状态 |
|--------|------|------|
| P2 | Summarizer/Compressor | ❌ |
| P3 | Tool Execution Orchestration | ❌ |
| P3 | Planner/Executor Loop | ❌ |
| P3 | Message Routing Policies | ❌ |
| P3 | State Update Policy after Tool Calls | ❌ |
| P4 | Checkpointing (高级) | ❌ |
| P4 | Vector Store 外置选项 | ❌ |
| P5 | Structured Chat API | ❌ |
| P5 | Streaming Support | ❌ |
| P5 | Framework Adapters | ❌ |
| P5 | Web/Streamlit UI | ❌ |
| P7 | Service Deployment | ❌ |

---

## 验收标准

当前 Product Regression 结果 (2026-03-23):

| 指标 | 得分 |
|------|------|
| task_continuation | 1.000 |
| blocker_preservation | 1.000 |
| decision_continuity | 1.000 |
| procedure_continuity | 1.000 |
| conflict_continuity | 1.000 |
| repeated_work_avoidance | 1.000 |

---

## 建议下一步

1. **短期**: 
   - 完成 Retrieval 基线对比测试
   - 加强 Session 恢复机制

2. **中期**:
   - 添加 Summarizer 模块
   - 实现 Tool Execution 基本支持

3. **长期**:
   - 添加 Web UI
   - 添加 Streaming 支持
   - 完善 Service 部署

---

*本文件已移动到 history/，保留作为历史参考。最新状态见 docs/index.md*
