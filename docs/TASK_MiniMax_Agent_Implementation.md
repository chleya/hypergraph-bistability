# Agent 实际调用 MiniMax 模型测试 - 任务书

**创建日期**: 2026-03-22
**项目**: hypergraph-bistability
**目标**: 完成最基础的 Agent 实际调用 MiniMax 模型进行测试

---

## 1. 当前状态

### ✅ 已完成
- 单元测试: 85 个测试全部通过 (mock 模式)
- CLI 接口: 支持 `--base-url`, `--model`, `--force-powershell` 参数
- MiniMax 集成: 支持 Anthropic 兼容 API (minimaxi.com)
- 测试脚本: 12 个 one-click 运行脚本

### ⚠️ 待完善
- 实际 LLM 调用的端到端测试
- LLM 响应质量评估
- 错误处理和重试机制
- 实际场景回归测试

---

## 2. 任务目标

### 核心目标
实现一个**可运行的、稳定的、可测试的** Agent，实际调用 MiniMax 模型进行对话和记忆交互。

### 成功标准
1. Agent 能够成功调用 MiniMax API 并获得响应
2. 基础对话流程完整: 用户输入 → 检索 → 生成响应 → 写入记忆
3. 至少有 3 个实际场景的回归测试通过
4. 错误处理机制完善 (API 失败、超时、限流等)

---

## 3. 详细任务分解

### 任务 1: 环境配置 (预计 30 分钟)
- [ ] 1.1 确认 API Key 配置正确
- [ ] 1.2 验证网络连接到 MiniMax API
- [ ] 1.3 测试基础 HTTP 调用是否成功

### 任务 2: 基础调用测试 (预计 1 小时)
- [ ] 2.1 运行现有 one-click 脚本验证端到端
- [ ] 2.2 检查 CLI `run-llm-*` 命令是否正常工作
- [ ] 2.3 验证响应格式解析正确

### 任务 3: 核心功能验证 (预计 2 小时)
- [ ] 3.1 对话流程: 单轮对话测试
- [ ] 3.2 对话流程: 多轮对话测试 (3-5 轮)
- [ ] 3.3 记忆功能: 验证信息被正确存储和检索
- [ ] 3.4 超图构建: 验证 hyperedge 正确生成

### 任务 4: 回归测试 (预计 2 小时)
- [ ] 4.1 产品回归测试 (product regression)
- [ ] 4.2 长任务回归测试 (long-task regression)
- [ ] 4.3 冲突检测回归测试 (conflict regression)

### 任务 5: 错误处理完善 (预计 1 小时)
- [ ] 5.1 API 超时处理
- [ ] 5.2 API 限流处理 (429)
- [ ] 5.3 API 错误响应处理 (4xx, 5xx)
- [ ] 5.4 Mock 降级机制

### 任务 6: 文档和测试报告 (预计 30 分钟)
- [ ] 6.1 更新 README.md
- [ ] 6.2 记录测试结果
- [ ] 6.3 编写使用说明

---

## 4. 执行命令

### 4.1 快速测试命令
```powershell
# 设置 API Key
$env:ANTHROPIC_API_KEY = "your-api-key"

# 运行产品回归测试
.\scripts\run_minimax_product_regression_one_click.ps1

# 运行评估测试
.\scripts\run_minimax_eval_one_click.ps1
```

### 4.2 单元测试命令
```powershell
# 运行所有单元测试
python -m pytest tests/test_agent_runtime.py -v

# 运行核心动力学测试
python -m pytest tests/test_core.py -v
```

---

## 5. 验收检查点

### Checkpoint 1: 环境就绪 (T+30min)
- [ ] API Key 配置完成
- [ ] 网络连接正常
- [ ] 基础 HTTP 调用成功

### Checkpoint 2: 端到端可用 (T+1.5h)
- [ ] CLI 命令正常运行
- [ ] 获得有效的 LLM 响应
- [ ] 响应格式解析正确

### Checkpoint 3: 功能完整 (T+3.5h)
- [ ] 单轮对话成功
- [ ] 多轮对话成功
- [ ] 记忆存储和检索成功

### Checkpoint 4: 回归通过 (T+5.5h)
- [ ] 产品回归测试通过
- [ ] 长任务回归测试通过
- [ ] 冲突回归测试通过

### Checkpoint 5: 交付 (T+6h)
- [ ] 错误处理完善
- [ ] 文档更新完成
- [ ] 测试报告生成

---

## 6. 已知问题

### 问题 1: API 连接
- 症状: 连接超时或被拒绝
- 解决: 检查 `--base-url` 参数，使用正确的端点

### 问题 2: 响应格式错误
- 症状: JSON 解析失败
- 解决: 检查 PowerShell 脚本的 UTF-8 编码处理

### 问题 3: Mock 降级失败
- 症状: API 失败后无响应
- 解决: 确认 `--force-powershell` 参数或环境变量配置

---

## 7. 联系人和资源

- GitHub: https://github.com/chleya/hypergraph-bistability
- 文档: `docs/`
- 测试: `tests/test_agent_runtime.py`
- CLI: `src/hypergraph_bistability/cli.py`

---

**签名**: ____________
**日期**: ____________
