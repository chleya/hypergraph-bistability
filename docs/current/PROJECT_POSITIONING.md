# Project Positioning

## Project定位

一个面向个人使用的、以**任务续接**为核心的 agent runtime。

不是通用 agent 平台，也不是超图理论研究产品。

## 核心问题

只有一个问题：

> **任务中断以后，能不能继续干活？**

## 主场景（4个）

1. **长对话助手** - 记住之前聊了什么，继续深入
2. **coding / planning copilot** - 记住在改什么代码、做什么计划
3. **文档写作 agent** - 记住在写什么主题、已经写了什么
4. **研究记录 / handoff 工具** - 记住研究进度、能交接给下次

## 开发原则

每次加新结构前，只问5件事：

1. **当前任务是什么？**
2. **下一步是什么？**
3. **卡在哪里？**
4. **中断后怎么恢复？**
5. **回答有没有继续围绕这4件事？**

如果一个新结构不能明显提升这5件事，就先不要加。

---

## 保留（产品骨架）

直接有用，是主线：

- `current_linked_task` - 当前任务锚点
- `save/load` - 会话持久化
- `handoff snapshot` - 中断恢复数据
- `task switch` 逻辑 - 任务切换处理
- `query_handoff_bundle()` - 查询交接数据
- 基础 working-set / query 能力
- continuity 相关真实回归测试
- `turn_log` - 对话历史
- 最小的 handoff diagnostics
- `stable_v1` 作为当前主线 runtime

---

## 降级（内部实验）

不是主线，但保留作为内部探索：

- 复杂 validator（三态/四态分类）
- UnifiedNode* / DurableMemory* / IntegratedAgentMemory
- 复杂 policy 层的对外地位
- 超图相关的产品解释层
- 大部分"memory manager"抽象

处理方式：不删光，不作为主线心智，不继续优先投入

---

## 删除 / 停止投入

继续做会拖回研究/框架开发：

- 继续精调 validator 规则追求人机一致
- 把 validator 当自动控制器
- 为 schema 完整性继续扩字段
- 把超图结构往产品 contract 硬绑定
- 新增更多中间 manager/layer/abstraction
- 为中间结构写大量"自证式"测试
- 往通用 agent 平台方向扩

---

## 下一步

不是继续设计，而是**用现有骨架跑真实任务**。

连续跑真实任务，观察哪里丢 task / next step / blocker，只在真实反复出问题的地方补结构。

---

*Last updated: 2026-03-23*
