#!/usr/bin/env python
"""Test Chinese conversation with MiniMax."""

import os
os.environ['ANTHROPIC_API_KEY'] = 'sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY'

from hypergraph_bistability.agent import HypergraphAgent

agent = HypergraphAgent(
    k=3, L=2, use_embeddings=False,
    llm_base_url='https://api.minimaxi.com/anthropic',
    llm_model='MiniMax-M2.7',
    llm_force_powershell_transport=True
)

print("=" * 50)
print("中文对话测试")
print("=" * 50)

# Turn 1: 告诉 AI 名字
print("\n--- 第1轮 ---")
r1 = agent.chat("我叫张三，是一名软件工程师。")
print(f"用户: 我叫张三，是一名软件工程师。")
print(f"AI: {r1[:100]}...")

# Turn 2: 问 AI 我的名字
print("\n--- 第2轮 ---")
r2 = agent.chat("我叫什么名字？")
print(f"用户: 我叫什么名字？")
print(f"AI: {r2[:100]}...")

# Turn 3: 问职业
print("\n--- 第3轮 ---")
r3 = agent.chat("我的职业是什么？")
print(f"用户: 我的职业是什么？")
print(f"AI: {r3[:100]}...")

# Check memory state
state = agent.get_memory_state()
print(f"\n--- 记忆状态 ---")
print(f"对话轮数: {state.get('conversation_turns')}")
print(f"内容条目: {len(state.get('content_map', {}))}")

print("\n" + "=" * 50)
print("中文对话测试完成!")
print("=" * 50)
