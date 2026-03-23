# API 稳定性修复清单

## 阶段 0：当前状态定义（不做任何代码改动）

- [ ] **明确 Stable 定义**
  - 只承诺 **import 稳定**
  - 不承诺行为、schema、持久化、CLI 稳定
  - v1.x 期间允许 breaking change

- [ ] **明确用户画像**
  - 当前：无外部用户
  - 决策：删除 src/agent, src/core 兼容层（无证据支持保留）

- [ ] **明确版本策略**
  - v1.0.0: 初始版本，允许 breaking change
  - v2.0.0: 开始承诺 import 稳定
  - 弃用窗口：1 个主版本号

---

## 阶段 1：收缩导出面（高优先级）

- [ ] **memory/__init__.py 收缩**
  - 只暴露：AgentMemory, AgentMemoryEnhanced, IntegratedAgentMemory
  - 删除：DurableMemory, UnifiedNodeManager, policies 直接暴露
  - 改为 `from .durable import DurableMemory` 显式访问

- [ ] **evals/__init__.py 收缩**
  - 只暴露：run_product_regression, run_long_task_regression, run_conflict_regression
  - 删除：scenario 常量、sidecar runner、实验型 runner
  - 改为 `from .scenarios import *` 显式访问

- [ ] **删除兼容层**
  - 删除：src/agent/__init__.py (deprecated wrapper)
  - 删除：src/core/__init__.py (deprecated wrapper)
  - 原因：无外部依赖证据

---

## 阶段 2：CLI 重构

- [ ] **CLI 拆分**
  - 创建：src/hypergraph_bistability/commands.py（内部可复用函数）
  - 简化：cli.py 只做参数解析和转发
  - 测试改为从 commands.py 导入，不从 cli.py

---

## 阶段 3：添加契约测试

- [ ] **import contract tests**
  - 验证所有 stable 导入路径可工作
  - 验证 deprecated 路径触发 warning

- [ ] **return-shape tests**
  - 验证 QueryLayer, WorkingSet 返回结构
  - 验证 run_*_regression 返回格式

- [ ] **deprecation tests**
  - 验证 deprecated 路径触发正确 warning

---

## 阶段 4：安装分发验证

- [ ] **验证 wheel 内容**
  - 运行 `python -m build` 生成 wheel
  - 检查旧路径是否仍能导入
  - 检查 src/agent, src/core 是否在发布包中

---

## 阶段 5：文档更新

- [ ] **更新 ARCHITECTURE.md**
  - 添加稳定性层级说明（Stable / Advanced / Internal）
  - 添加版本策略说明
  - 标注哪些对象可能 3 个月内重构

- [ ] **更新 STATUS.md**
  - 移除 "Stable" 标签，改为 "Candidate Stable (import only)"

- [ ] **更新 README.md**
  - 添加导入路径指南：什么时候用包根 vs 子模块

---

## 验收标准

- [ ] 85 个单元测试仍然通过
- [ ] 3 条回归门禁仍然通过
- [ ] memory/__init__.py <= 5 个导出
- [ ] evals/__init__.py <= 5 个导出
- [ ] 无 import * 在稳定层
- [ ] CLI 逻辑在 commands.py 中可独立测试
