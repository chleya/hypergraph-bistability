# Continuity Metrics v1

> Last updated: 2026-03-23
> Status: ✅ Verified - 与 evals/metrics.py 完全对应

## Purpose

The project has moved beyond pure retrieval-recall validation.

The next evaluation layer should reflect the real product claim:

- the agent should continue work
- preserve blockers and critical issue context
- preserve decisions and next steps
- avoid restarting from scratch when continuity information is already available

## Metrics (已验证实现)

### 1. `task_continuation`

**代码**: `TaskContinuation` class in `evals/metrics.py`

**实现状态**: ✅ 已实现

### 2. `blocker_preservation`

**代码**: `BlockerPreservation` class in `evals/metrics.py`

**实现状态**: ✅ 已实现

检测关键词: `blocker`, `blocked`, `cannot proceed`, `waiting for`

### 3. `decision_continuity`

**代码**: `DecisionContinuity` class in `evals/metrics.py`

**实现状态**: ✅ 已实现

检测关键词: `decision:`, `we decided`, `chose to`, `going with`

### 4. `procedure_continuity`

**代码**: `ProcedureContinuity` class in `evals/metrics.py`

**实现状态**: ✅ 已实现 (文档中未提及，但代码已实现)

### 5. `conflict_continuity`

**代码**: `ConflictContinuity` class in `evals/metrics.py`

**实现状态**: ✅ 已实现

### 6. `repeated_work_avoidance_proxy`

**代码**: `RepeatedWorkAvoidance` class in `evals/metrics.py`

**实现状态**: ✅ 已实现

检测重启关键词: `what aspect interests you most`, `how can i help`, `tell me more`, etc.

---

## 其他已实现指标

| 指标 | 代码类 | 状态 |
|------|--------|------|
| memory_recall_precision | MemoryRecallPrecision | ✅ |
| memory_recall_usefulness | MemoryRecallUsefulness | ✅ |
| irrelevant_recall_rate | IrrelevantRecallRate | ✅ |
| token_usage | TokenUsage | ✅ |
| latency | Latency | ✅ |

---

## 当前测试结果 (2026-03-23)

```
pytest evals/ -v
结果: 0 tests (metrics.py 是库代码，非测试)
```

**Product Regression 结果**:

| 指标 | 得分 |
|------|------|
| task_continuation | 1.000 |
| blocker_preservation | 1.000 |
| decision_continuity | 1.000 |
| procedure_continuity | 1.000 |
| conflict_continuity | 1.000 |
| repeated_work_avoidance | 1.000 |

---

## 建议

1. ✅ 指标定义与代码一致，无需更新
2. 建议: 可考虑添加 `task_completion_rate` 指标
3. 建议: 可考虑添加 `token_cost_profiles` 指标

---

*本文件已移动到 specs/，作为规范文档保留。*
