"""
A/B Test: Working-set JSON context impact on MiniMax model behavior.

This test verifies whether the working-set JSON context in the prompt
actually influences the model's responses.
"""
import json
import os
from hypergraph_bistability.agent import HypergraphAgent
from hypergraph_bistability.agent.runtime import TurnProcessor, ContextAssembler

# Set up API key from environment or hardcoded
API_KEY = os.environ.get("MINIMAX_API_KEY", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJtaW5pbWF4Iiwicm9sZSI6ImFzc2lnbm1lbnQiLCJpYXQiOjE3NDIwNzIwMDAsImV4cCI6MTc0MjA3NTYwMH0.t9VV0BvLZQ0VzLq4Fp1JdJF2lSFT2I2d0Z9zLq3FqJvKzSj8Y3vQhGZ5QZ6hQz8qK9YvXzQhGZ5QZ6hQz8qK9YvXzQ")

def get_minimax_response(prompt: str) -> str:
    """Call MiniMax API to get response."""
    import requests
    
    url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "MiniMax-M2.1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    result = response.json()
    
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    return f"ERROR: {result}"

def test_working_set_impact():
    """Test if working-set JSON context changes model behavior."""
    
    # Scenario: User asks about current task
    user_input = "我之前让你做什么？"
    
    # Test 1: WITHOUT working-set context
    prompt_without = f"""You are an AI assistant. User asks: {user_input}

Answer briefly based on the conversation context."""
    
    # Test 2: WITH working-set context (JSON)
    ws_context = json.dumps({
        "task": {
            "name": "修复 API 错误",
            "status": "in_progress",
            "phase": "debugging",
            "blockers": 1,
            "decisions": 2,
            "procedures": 3
        },
        "handoff_ready": False
    }, ensure_ascii=False, indent=2)
    
    prompt_with = f"""You are an AI assistant. User asks: {user_input}

```json
{ws_context}
```

Based on the working-set JSON above, answer the user's question."""
    
    print("=" * 60)
    print("A/B Test: Working-set JSON Impact")
    print("=" * 60)
    
    print("\n[Test 1] WITHOUT working-set context:")
    print(f"Question: {user_input}")
    response_without = get_minimax_response(prompt_without)
    print(f"Response: {response_without[:200]}...")
    
    print("\n[Test 2] WITH working-set context:")
    print(f"Working-set JSON: {ws_context[:100]}...")
    response_with = get_minimax_response(prompt_with)
    print(f"Response: {response_with[:200]}...")
    
    # Analysis
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    
    # Check if response_with mentions task-related info
    keywords = ["修复", "API", "错误", "task", "bug", "fix"]
    has_task_info = any(kw in response_with for kw in keywords)
    
    print(f"Response WITH working-set mentions task: {has_task_info}")
    
    if has_task_info:
        print("\n✅ SUCCESS: Working-set JSON context IS influencing the model!")
    else:
        print("\n❌ FAIL: Working-set JSON context is NOT being used by the model")
    
    return has_task_info

def test_blocker_awareness():
    """Test if model is aware of blockers when present."""
    
    user_input = "有什么阻碍吗？"
    
    # WITH blocker info
    ws_with_blocker = json.dumps({
        "task": {
            "name": "完成报告",
            "status": "in_progress",
            "blockers": 2,
            "decisions": 1
        },
        "handoff_ready": False
    }, ensure_ascii=False)
    
    prompt_with_blocker = f"""User asks: {user_input}

```json
{ws_with_blocker}
```

Answer the user based on the JSON context above."""
    
    # WITHOUT blocker info (empty task)
    ws_empty = json.dumps({}, ensure_ascii=False)
    
    prompt_empty = f"""User asks: {user_input}

```json
{ws_empty}
```

Answer the user based on the JSON context above."""
    
    print("\n" + "=" * 60)
    print("Test: Blocker Awareness")
    print("=" * 60)
    
    print("\n[With Blocker] JSON: blockers=2")
    response_blocker = get_minimax_response(prompt_with_blocker)
    print(f"Response: {response_blocker[:200]}...")
    
    print("\n[Empty JSON]")
    response_empty = get_minimax_response(prompt_empty)
    print(f"Response: {response_empty[:200]}...")
    
    # Check if response mentions blockers
    blocker_keywords = ["阻碍", "blocker", "问题", "困难", "卡在"]
    mentions_blocker = any(kw in response_blocker for kw in blocker_keywords)
    mentions_nothing = any(kw in response_empty for kw in ["没有", "没有阻碍", "无", "nothing", "没有阻塞"])
    
    print(f"\nWith blocker mentions blocker: {mentions_blocker}")
    print(f"Empty mentions nothing: {mentions_nothing}")
    
    if mentions_blocker and mentions_nothing:
        print("\n✅ SUCCESS: Model correctly distinguishes blockers!")
        return True
    else:
        print("\n❌ Model not responding correctly to blockers")
        return False

if __name__ == "__main__":
    print("MiniMax A/B Test for Working-set JSON Context")
    print("=" * 60)
    
    result1 = test_working_set_impact()
    result2 = test_blocker_awareness()
    
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"Working-set Impact Test: {'PASS' if result1 else 'FAIL'}")
    print(f"Blocker Awareness Test: {'PASS' if result2 else 'FAIL'}")
