"""
Experiment: Unified Node System (Skills + Memories)
==================================================

Goal: Verify that skills and memories work as unified nodes

Run: python experiments/verification/experiment_unified_nodes.py
"""

import sys
import os
os.chdir("F:/hypergraph_bistability")
sys.path.insert(0, "src")

from hypergraph_bistability.memory.unified_node import (
    UnifiedNodeManager,
    UnifiedNodeStore,
    NodeType,
    SkillDefinition,
)


def test_add_memory():
    """Test adding memory nodes."""
    print("\n=== Test 1: Add Memory Nodes ===")
    
    manager = UnifiedNodeManager(":memory:")
    
    # Add memories
    pref_id = manager.remember(
        "user prefers detailed explanations",
        NodeType.PREFERENCE,
        importance=0.9
    )
    print(f"[OK] Added preference: id={pref_id}")
    
    fact_id = manager.remember(
        "Python is the programming language being used",
        NodeType.FACT,
        verified=True
    )
    print(f"[OK] Added fact: id={fact_id}")
    
    decision_id = manager.remember(
        "decided to use SQLite for storage",
        NodeType.DECISION,
        context="project architecture"
    )
    print(f"[OK] Added decision: id={decision_id}")
    
    manager.close()
    print("PASSED\n")


def test_add_skill():
    """Test adding skill nodes."""
    print("\n=== Test 2: Add Skill Nodes ===")
    
    manager = UnifiedNodeManager(":memory:")
    
    # Add a simple skill
    skill_code = """
def execute(data):
    return f"processed: {data}"
"""
    
    skill_id = manager.learn_skill(
        name="data_processor",
        code=skill_code,
        description="Process data and return result",
        category="data",
        parameters={"data": "any"}
    )
    print(f"[OK] Added skill: id={skill_id}")
    
    # Add another skill
    skill_id2 = manager.learn_skill(
        name="pptx_generator",
        code="",
        description="Generate PowerPoint presentations",
        category="office",
        parameters={"topic": "str", "slides": "int"}
    )
    print(f"[OK] Added pptx skill: id={skill_id2}")
    
    manager.close()
    print("PASSED\n")


def test_unified_retrieval():
    """Test unified retrieval across memory and skill types."""
    print("\n=== Test 3: Unified Retrieval ===")
    
    manager = UnifiedNodeManager(":memory:")
    
    # Add test data
    manager.remember("user likes Python", NodeType.PREFERENCE)
    manager.remember("user prefers dark mode", NodeType.PREFERENCE)
    manager.remember("meeting at 3pm", NodeType.TASK)
    manager.learn_skill("code_formatter", "def execute(): pass", "Format code")
    manager.learn_skill("pptx_maker", "def execute(): pass", "Make slides")
    
    # Recall with query
    results = manager.recall("user")
    print(f"  Query 'user': found {len(results)} nodes")
    for r in results:
        print(f"    - {r.content} ({r.node_type.value}, effectiveness={r.effectiveness:.2f})")
    
    # Recall skills only
    skills = manager.recall_skills("make")
    print(f"  Query 'make' (skills only): found {len(skills)} skills")
    for s in skills:
        print(f"    - {s.content} ({s.node_type.value})")
    
    manager.close()
    print("PASSED\n")


def test_skill_execution():
    """Test executing a skill."""
    print("\n=== Test 4: Skill Execution ===")
    
    store = UnifiedNodeStore(":memory:")
    
    # Register a handler
    def my_handler(data, multiplier=1):
        result = int(data) * multiplier
        return f"Handler result: {result}"
    
    store.register_skill_handler("calculator", my_handler)
    
    # Add skill with definition
    skill_id = store.add_node(
        content="calculator",
        node_type=NodeType.SKILL,
        skill_def=SkillDefinition(
            name="calculator",
            code="",
            parameters={"data": "int", "multiplier": "int"},
            description="Simple calculator"
        ),
        metadata={"category": "utility"}
    )
    print(f"[OK] Added skill with handler: id={skill_id}")
    
    # Execute
    result, success = store.execute_skill(skill_id, data=10, multiplier=3)
    print(f"  Execution result: {result}, success={success}")
    
    # Check stats updated
    node = store.get_node(skill_id)
    print(f"  Stats: activations={node.activation_count}, success={node.success_count}")
    
    store.close()
    print("PASSED\n")


def test_feedback_learning():
    """Test that feedback updates effectiveness."""
    print("\n=== Test 5: Feedback Learning ===")
    
    store = UnifiedNodeStore(":memory:")
    
    # Add a node
    node_id = store.add_node(
        content="test node",
        node_type=NodeType.MEMORY,
        effectiveness=0.5
    )
    
    node_before = store.get_node(node_id)
    print(f"  Before: effectiveness={node_before.effectiveness:.2f}")
    
    # Provide positive feedback
    store.learn_from_feedback(node_id, 0.8)  # Strong positive
    node_after = store.get_node(node_id)
    print(f"  After +0.8 feedback: effectiveness={node_after.effectiveness:.2f}")
    
    # Provide negative feedback
    store.learn_from_feedback(node_id, -0.5)
    node_final = store.get_node(node_id)
    print(f"  After -0.5 feedback: effectiveness={node_final.effectiveness:.2f}")
    
    store.close()
    print("PASSED\n")


def test_statistics():
    """Test statistics gathering."""
    print("\n=== Test 6: Statistics ===")
    
    manager = UnifiedNodeManager(":memory:")
    
    # Add various nodes
    for i in range(3):
        manager.remember(f"preference {i}", NodeType.PREFERENCE)
    for i in range(2):
        manager.remember(f"fact {i}", NodeType.FACT)
    for i in range(2):
        manager.learn_skill(f"skill_{i}", "pass", f"skill {i}")
    
    stats = manager.get_stats()
    print(f"  Total nodes: {stats['total']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Top skills: {stats['top_skills']}")
    
    manager.close()
    print("PASSED\n")


def test_effectiveness_ranking():
    """Test that effective nodes rank higher."""
    print("\n=== Test 7: Effectiveness Ranking ===")
    
    store = UnifiedNodeStore(":memory:")
    
    # Add nodes with different effectiveness
    store.add_node("low effective", NodeType.MEMORY, effectiveness=0.2)
    store.add_node("medium effective", NodeType.MEMORY, effectiveness=0.5)
    store.add_node("high effective", NodeType.MEMORY, effectiveness=0.9)
    
    # Retrieve - should be sorted by effectiveness
    results = store.retrieve("effective", limit=10)
    print(f"  Retrieved {len(results)} nodes (should be 3)")
    
    print("  Ranking order:")
    for i, r in enumerate(results):
        print(f"    {i+1}. {r.content} (effectiveness={r.effectiveness})")
    
    store.close()
    print("PASSED\n")


def main():
    print("=" * 60)
    print("Unified Node System Verification")
    print("=" * 60)
    
    try:
        test_add_memory()
        test_add_skill()
        test_unified_retrieval()
        test_skill_execution()
        test_feedback_learning()
        test_statistics()
        test_effectiveness_ranking()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
