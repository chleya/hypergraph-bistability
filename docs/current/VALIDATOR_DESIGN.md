# Validator Design Note (Simplified)

## 定位

**Simplified Diagnostic Signal System** - 不是控制器，不是精细语义分类器。

Validator 的目的是简单检查回答是否提到任务上下文内容。

## 简化设计 (2026-03-23)

### 核心逻辑
- 检查 response 是否提到任何 handoff 内容
- 只返回两种状态：**has_continuity** / **no_continuity**

### 实现位置
- `src/hypergraph_bistability/agent/runtime/simple_validator.py`

### 不再追求
- ❌ 3 态分类 (fail/weak_pass/strong_pass)
- ❌ 复杂关键词匹配
- ❌ 短语精确匹配
- ❌ 人机一致

## 设计原则

1. **简单优先** - 宁可漏掉一些，不要误报太多
2. **可解释** - 匹配了什么类型 (task/blocker/next_step/decision)
3. **用于日志** - 主要用于事后日志分析，不是实时控制

## 理由

- 复杂的 3 态分类器难以调优
- 成熟方案很多（LLM 评估、简单启发式），不必重复造轮子
- 简化后更稳定、更易维护
