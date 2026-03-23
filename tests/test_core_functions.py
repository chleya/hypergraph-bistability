#!/usr/bin/env python
"""Comprehensive test for core agent functions."""

import os
os.environ['ANTHROPIC_API_KEY'] = 'sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY'

from hypergraph_bistability.agent import HypergraphAgent

agent = HypergraphAgent(
    k=3, L=2, use_embeddings=False,
    llm_base_url='https://api.minimaxi.com/anthropic',
    llm_model='MiniMax-M2.7',
    llm_force_powershell_transport=True
)

print("=" * 60)
print("核心功能验证测试")
print("=" * 60)

# Test 1: Memory Storage
print("\n[1] 记忆存储测试")
result1 = agent.process_turn("请记住我的项目代号是 Phoenix。")
print(f"    输入: 请记住我的项目代号是 Phoenix。")
print(f"    写入数: {len(result1.writes)}")
print(f"    状态: {'OK' if len(result1.writes) > 0 else 'FAIL'}")

# Test 2: Memory Retrieval
print("\n[2] 记忆检索测试")
result2 = agent.process_turn("我的项目代号是什么？")
print(f"    输入: 我的项目代号是什么？")
print(f"    输出: {result2.assistant_response[:80]}...")
print(f"    检索到: {len(result2.retrieved_items)} 条")
if "Phoenix" in result2.assistant_response or "phoenix" in result2.assistant_response.lower():
    print(f"    状态: OK - 成功检索到项目代号")
else:
    print(f"    状态: 检查响应内容...")

# Test 3: Decision Memory
print("\n[3] 决策记忆测试")
result3 = agent.process_turn("我决定使用 Python 来开发这个项目。")
print(f"    输入: 我决定使用 Python 来开发这个项目。")
print(f"    写入数: {len(result3.writes)}")

result4 = agent.process_turn("我之前决定用什么语言？")
print(f"    输入: 我之前决定用什么语言？")
print(f"    输出: {result4.assistant_response[:80]}...")

# Test 4: Hyperedge Structure
print("\n[4] 超图结构测试")
state = agent.get_memory_state()
print(f"    对话轮数: {state.get('conversation_turns', 0)}")

# Check hypergraph view
if hasattr(agent, 'memory') and hasattr(agent.memory, 'get_hypergraph_view'):
    hg_view = agent.memory.get_hypergraph_view()
    print(f"    超图视图: {type(hg_view).__name__}")
    if hasattr(hg_view, 'hyperedges'):
        print(f"    超边数: {len(hg_view.hyperedges)}")
        print(f"    状态: OK")
    else:
        print(f"    状态: 无 hyperedges 属性")
else:
    print(f"    记忆类型: {type(agent.memory).__name__}")

# Test 5: Multi-turn continuity
print("\n[5] 多轮连续性测试")
conversations = [
    "我喜欢蓝色。",
    "我讨厌红色。",
    "我喜欢什么颜色？",
    "我讨厌什么颜色？"
]

for i, msg in enumerate(conversations):
    resp = agent.chat(msg)
    print(f"    轮{i+1}: {msg[:20]}... → {resp[:30]}...")

print("\n" + "=" * 60)
print("核心功能验证完成")
print("=" * 60)
