"""
Experiment: Durable Memory Layer Verification
=============================================

Goal: Verify the durable memory layer works correctly

Run: python experiments/verification/experiment_durable_memory.py
"""

import sys
import os
os.chdir("F:/hypergraph_bistability")
sys.path.insert(0, "src")

from hypergraph_bistability.memory.durable_memory import (
    DurableMemoryManager,
    DurableMemoryStore,
)
from hypergraph_bistability.memory.policies import (
    create_decay_policy,
    create_promotion_policy,
)


def test_basic_crud():
    """Test basic Create/Read/Update/Delete operations."""
    print("\n=== Test 1: Basic CRUD ===")
    
    store = DurableMemoryStore(":memory:")
    
    mem_id = store.add(
        content="user preference: simple code style",
        kind="preference",
        importance=0.9,
        layer=0,
        session_id="test_session"
    )
    print(f"[OK] Created memory id={mem_id}")
    
    mem = store.get(mem_id)
    assert mem is not None
    assert mem.content == "user preference: simple code style"
    print(f"[OK] Read memory: {mem.content}")
    
    store.update_importance(mem_id, 0.95)
    mem = store.get(mem_id)
    assert mem.importance == 0.95
    print(f"[OK] Updated importance to {mem.importance}")
    
    result = store.delete(mem_id)
    assert result is True
    mem = store.get(mem_id)
    assert mem is None
    print("[OK] Deleted memory")
    
    store.close()
    print("PASSED: Test 1\n")


def test_layer_management():
    """Test layer promotion and demotion."""
    print("\n=== Test 2: Layer Management ===")
    
    store = DurableMemoryStore(":memory:")
    
    mem_id = store.add(
        content="important decision",
        kind="decision",
        importance=0.8,
        layer=0,
        session_id="test"
    )
    
    store.promote(mem_id, 1)
    mem = store.get(mem_id)
    assert mem.layer == 1
    print(f"[OK] Promoted to layer {mem.layer}")
    
    store.promote(mem_id, 2)
    mem = store.get(mem_id)
    assert mem.layer == 2
    print(f"[OK] Promoted to layer {mem.layer}")
    
    store.demote(mem_id, 1)
    mem = store.get(mem_id)
    assert mem.layer == 1
    print(f"[OK] Demoted to layer {mem.layer}")
    
    store.close()
    print("PASSED: Test 2\n")


def test_search():
    """Test content search."""
    print("\n=== Test 3: Search ===")
    
    store = DurableMemoryStore(":memory:")
    
    memories = [
        ("user preference: likes Chinese", "preference", 0.9, 0),
        ("user preference: likes Python", "preference", 0.8, 0),
        ("current task: write docs", "task", 0.7, 0),
        ("code review done", "task", 0.5, 1),
        ("decision: need to refactor agent", "decision", 0.85, 1),
    ]
    
    for content, kind, importance, layer in memories:
        store.add(content, kind, importance, layer, session_id="test")
    
    results = store.search("preference")
    assert len(results) >= 1
    print(f"[OK] Search 'preference': found {len(results)} results")
    
    results = store.search("task", kind="task")
    assert len(results) >= 1
    print(f"[OK] Search kind='task': found {len(results)} results")
    
    store.close()
    print("PASSED: Test 3\n")


def test_statistics():
    """Test statistics gathering."""
    print("\n=== Test 4: Statistics ===")
    
    store = DurableMemoryStore(":memory:")
    
    for i in range(10):
        store.add(
            content=f"memory {i}",
            kind=["preference", "task", "decision"][i % 3],
            importance=0.5 + (i * 0.05),
            layer=i % 3,
            session_id="test"
        )
    
    stats = store.get_statistics()
    print(f"  Total: {stats['total']}")
    print(f"  By layer: {stats['by_layer']}")
    print(f"  By kind: {stats['by_kind']}")
    
    assert stats['total'] == 10
    print("PASSED: Test 4\n")


def test_manager_integration():
    """Test DurableMemoryManager integration."""
    print("\n=== Test 5: Manager Integration ===")
    
    manager = DurableMemoryManager(":memory:", working_capacity=3)
    
    for i in range(5):
        manager.add_memory(
            content=f"test memory {i}",
            kind="task",
            importance=0.5 + (i * 0.1),
            session_id="test"
        )
    
    status = manager.get_status()
    print(f"  After adding 5: {status}")
    
    results = manager.retrieve("test", limit=3)
    print(f"  Retrieved {len(results)} memories")
    
    promo_policy = create_promotion_policy()
    promo_stats = manager.run_promotion_cycle(promo_policy)
    print(f"  Promotion stats: {promo_stats}")
    
    decay_policy = create_decay_policy()
    decay_stats = manager.run_decay_cycle(decay_policy)
    print(f"  Decay stats: {decay_stats}")
    
    status = manager.get_status()
    print(f"  Final status: {status}")
    
    manager.close()
    print("PASSED: Test 5\n")


def test_session_isolation():
    """Test session-based memory isolation."""
    print("\n=== Test 6: Session Isolation ===")
    
    store = DurableMemoryStore(":memory:")
    
    store.add("session A memory", "task", 0.8, layer=0, session_id="session_a")
    store.add("session B memory", "task", 0.8, layer=0, session_id="session_b")
    store.add("session A another", "preference", 0.7, layer=0, session_id="session_a")
    
    session_a_mems = store.get_by_session("session_a")
    session_b_mems = store.get_by_session("session_b")
    
    assert len(session_a_mems) == 2
    assert len(session_b_mems) == 1
    print(f"  Session A: {len(session_a_mems)} memories")
    print(f"  Session B: {len(session_b_mems)} memories")
    
    store.close()
    print("PASSED: Test 6\n")


def test_decay_policy():
    """Test decay policy evaluation."""
    print("\n=== Test 7: Decay Policy ===")
    
    import time
    
    policy = create_decay_policy()
    
    old_memory = {
        "content": "old memory",
        "kind": "task",
        "importance": 0.3,
        "created_at": time.time() - 30 * 24 * 3600,
        "last_accessed": time.time() - 30 * 24 * 3600,
        "access_count": 1,
        "layer": 2,
    }
    
    recent_memory = {
        "content": "new memory",
        "kind": "preference",
        "importance": 0.9,
        "created_at": time.time() - 1 * 24 * 3600,
        "last_accessed": time.time(),
        "access_count": 10,
        "layer": 1,
    }
    
    old_result = policy.evaluate("old_mem", old_memory)
    recent_result = policy.evaluate("recent_mem", recent_memory)
    
    print(f"  Old memory action: {old_result.action}")
    print(f"  Recent memory action: {recent_result.action}")
    
    assert old_result.action in ["demote", "remove"]
    assert recent_result.action == "keep"
    
    print("PASSED: Test 7\n")


def test_promotion_policy():
    """Test promotion policy evaluation."""
    print("\n=== Test 8: Promotion Policy ===")
    
    import time
    
    policy = create_promotion_policy()
    
    important_memory = {
        "content": "important user preference",
        "kind": "preference",
        "importance": 0.9,
        "created_at": time.time() - 2 * 24 * 3600,
        "last_accessed": time.time(),
        "access_count": 5,
        "layer": 0,
    }
    
    trivial_memory = {
        "content": "casual chat",
        "kind": "chat",
        "importance": 0.2,
        "created_at": time.time(),
        "last_accessed": time.time(),
        "access_count": 1,
        "layer": 0,
    }
    
    important_result = policy.evaluate("imp_mem", important_memory)
    trivial_result = policy.evaluate("triv_mem", trivial_memory)
    
    print(f"  Important: should_promote={important_result.should_promote}, new_layer={important_result.new_layer}")
    print(f"  Trivial: should_promote={trivial_result.should_promote}")
    
    assert important_result.should_promote is True
    assert trivial_result.should_promote is False
    
    print("PASSED: Test 8\n")


def main():
    print("=" * 60)
    print("Durable Memory Layer Verification Experiment")
    print("=" * 60)
    
    try:
        test_basic_crud()
        test_layer_management()
        test_search()
        test_statistics()
        test_manager_integration()
        test_session_isolation()
        test_decay_policy()
        test_promotion_policy()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
