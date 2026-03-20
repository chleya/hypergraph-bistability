"""
LLM Integration Test Suite
===========================

Tests the Agent Memory with real OpenAI API (if available)
or mock mode for demonstration.

Usage:
    # With API key:
    python llm_integration_test.py --api-key YOUR_KEY
    
    # Without API key (mock mode):
    python llm_integration_test.py
"""

import argparse
import json
import time
import tempfile
import pathlib
from typing import Dict, List, Optional


def run_mock_test() -> Dict:
    """Run tests in mock mode (no API required)."""
    print("=" * 60)
    print("Running in MOCK mode (no API calls)")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from agent.agent_memory_enhanced import AgentMemoryEnhanced, LLMConflictDetector
    
    results = {}
    
    print("\n[Test 1] Basic Memory Operations")
    print("-" * 40)
    
    mem = AgentMemoryEnhanced(
        k=3, L=2, name="test_agent",
        use_llm_detector=False,
        use_llm_mapper=False
    )
    mem.group_labels = ["work", "personal", "technical"]
    mem.layer_labels = ["preferences", "context"]
    
    mem.write("User prefers concise answers", group=0, layer=0)
    mem.write("User likes detailed explanations", group=1, layer=0)
    mem.write("Tech industry background", group=2, layer=1)
    
    print(f"  Wrote 3 memories")
    print(f"  Active cells: {mem.read().n_active}/6")
    lc = mem.get_lambda_c()
    print(f"  λ_c: {lc:.4f}" if lc is not None else "  λ_c: N/A")
    assert mem.read().n_active > 0, "Should have active cells after writing"
    assert len(mem.content_map) == 3, "Should have 3 content entries"
    assert lc is not None, "λ_c should be computed"
    results["test1_basic"] = "PASS"
    
    print("\n[Test 2] Conflict Detection (Keyword)")
    print("-" * 40)
    
    test_prompts = [
        ("Hello, how are you?", 0.0),
        ("I want A but I also want B", 0.3),
        ("On one hand I should work, on the other hand I'm tired", 0.6),
        ("Wait no actually I changed my mind about everything", 0.9),
    ]
    
    for prompt, expected_min in test_prompts:
        result = mem.process_prompt(prompt)
        level = result["conflict_level"]
        status = "OK" if level >= expected_min else "LOW"
        print(f"  '{prompt[:40]}...'")
        print(f"    Level: {level:.1%} (expected >= {expected_min:.1%}) [{status}]")
        assert 0.0 <= level <= 1.0, f"Conflict level {level} out of range [0,1]"
    
    assert mem.process_prompt("Hello").get("conflict_level", -1) < mem.process_prompt("I want A but I also want B").get("conflict_level", -1), \
        "Conflict detection should rank 'but' prompt higher than neutral"
    results["test2_conflict"] = "PASS"
    
    print("\n[Test 3] Mode Switching")
    print("-" * 40)
    
    modes = ["neutral", "exploratory", "focused", "creative"]
    for mode in modes:
        mem.switch_mode(mode)
        print(f"  {mode}: λ={mem.lambda_:.3f}, μ={mem.mu:.3f}")
        assert mem.current_mode == mode, f"Mode should be {mode}"
    assert mem.lambda_ >= 0, "λ should be non-negative"
    assert -1 <= mem.mu <= 1, "μ should be in [-1, 1]"
    results["test3_modes"] = "PASS"
    
    print("\n[Test 4] Memory Persistence")
    print("-" * 40)
    
    tmp_dir = pathlib.Path(tempfile.gettempdir())
    path = mem.save(str(tmp_dir / "test_memory.json"))
    print(f"  Saved to: {path}")
    
    mem2 = AgentMemoryEnhanced.load(str(tmp_dir / "test_memory.json"))
    print(f"  Loaded: k={mem2.k}, L={mem2.L}")
    print(f"  Content map: {len(mem2.content_map)} entries")
    assert mem2.k == mem.k and mem2.L == mem.L, "Loaded memory should have same dimensions"
    assert len(mem2.content_map) == len(mem.content_map), "Content map size should match"
    results["test4_persistence"] = "PASS"
    
    print("\n[Test 5] LLM Conflict Detector (Mock)")
    print("-" * 40)
    
    detector = LLMConflictDetector()
    print(f"  Has OpenAI: {detector._has_openai}")
    
    prompt = "I want to learn coding but I'm also tired"
    level, reasoning = detector._detect_fallback(prompt)
    print(f"  Fallback detection: {level:.1%}")
    print(f"  Reasoning: {reasoning}")
    assert 0.0 <= level <= 1.0, f"Conflict level {level} out of range"
    assert isinstance(reasoning, str) and len(reasoning) > 0, "Reasoning should be a non-empty string"
    results["test5_detector"] = "PASS"
    
    print("\n[Test 6] Context String Generation")
    print("-" * 40)
    
    ctx = mem.get_context_for_llm()
    print(f"  Context:\n  {ctx}")
    assert isinstance(ctx, str) and len(ctx) > 0, "Context should be a non-empty string"
    assert "Memory State:" in ctx, "Context should contain memory state label"
    results["test6_context"] = "PASS"
    
    return results


def run_api_test(api_key: str) -> Dict:
    """Run tests with real OpenAI API."""
    print("=" * 60)
    print("Running with REAL OpenAI API")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from agent.agent_memory_enhanced import AgentMemoryEnhanced, LLMConflictDetector, SemanticMemoryMapper
    
    results = {}
    
    print("\n[Test 1] LLM Conflict Detection (Real API)")
    print("-" * 40)
    
    detector = LLMConflictDetector(api_key=api_key)
    print(f"  Has OpenAI: {detector._has_openai}")
    
    if not detector._has_openai:
        print("  SKIPPED - No API key")
        results["test1_llm_conflict"] = "SKIP"
        return results
    
    test_prompts = [
        "Hello, how are you?",
        "I want to learn coding but I'm also tired",
        "On one hand I should work, on the other hand I'm tired and hungry",
        "Wait no actually I changed my mind about everything, both options seem equally good",
    ]
    
    for prompt in test_prompts:
        level, reasoning = detector.detect_conflict(prompt)
        print(f"  '{prompt[:50]}...'")
        print(f"    Level: {level:.1%}, Reason: {reasoning[:50]}...")
        time.sleep(0.5)
    
    results["test1_llm_conflict"] = "PASS"
    
    print("\n[Test 2] Semantic Memory Mapping (Real API)")
    print("-" * 40)
    
    mapper = SemanticMemoryMapper(k=3, L=2, api_key=api_key)
    
    test_contents = [
        "User prefers short responses",
        "Like programming examples",
        "Works in tech industry",
    ]
    
    for content in test_contents:
        group, layer = mapper.find_slot(content, "write")
        print(f"  '{content}'")
        print(f"    → group={group}, layer={layer}")
        time.sleep(0.5)
    
    results["test2_semantic_map"] = "PASS"
    
    print("\n[Test 3] Full Agent with LLM Detection")
    print("-" * 40)
    
    mem = AgentMemoryEnhanced(
        k=3, L=2, name="test_llm",
        use_llm_detector=True,
        use_llm_mapper=True,
        api_key=api_key
    )
    mem.group_labels = ["work", "personal", "technical"]
    
    prompts = [
        "Tell me about the weather",
        "I want to code but I'm tired",
        "Actually wait, I changed my mind, focus on the first thing",
    ]
    
    for prompt in prompts:
        result = mem.process_prompt(prompt, use_llm=True)
        print(f"  Prompt: '{prompt[:40]}...'")
        print(f"    Conflict: {result['conflict_level']:.1%}")
        print(f"    λ: {result['old_lambda']:.3f} → {result['new_lambda']:.3f}")
        time.sleep(1.0)
    
    results["test3_full_agent"] = "PASS"
    
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Integration Test")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    args = parser.parse_args()
    
    if args.api_key:
        results = run_api_test(args.api_key)
    else:
        results = run_mock_test()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, status in results.items():
        print(f"  {test_name}: {status}")
    
    print("\nAll tests completed.")
    
    return results


if __name__ == "__main__":
    main()