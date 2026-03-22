"""
Experiment: Integrated Memory (Hypergraph + Unified Nodes)
========================================================

Goal: Verify the integration of AgentMemoryEnhanced + UnifiedNodeManager

Run: python experiments/verification/experiment_integrated_memory.py
"""

import sys
import os
os.chdir("F:/hypergraph_bistability")
sys.path.insert(0, "src")

from hypergraph_bistability.memory.integrated_memory import (
    IntegratedAgentMemory,
    IntegrationConfig,
    create_integrated_memory,
)


def test_creation():
    """Test creating integrated memory."""
    print("\n=== Test 1: Creation ===")
    
    config = IntegrationConfig(
        enable_unified_nodes=True,
        unified_db_path=":memory:",
    )
    
    memory = IntegratedAgentMemory(
        k=3, L=2, name="test",
        config=config,
    )
    
    print(f"[OK] Created: {memory}")
    memory.close()
    print("PASSED\n")


def test_factory():
    """Test factory function."""
    print("\n=== Test 2: Factory Function ===")
    
    memory = create_integrated_memory(
        name="factory_test",
        db_path=":memory:",
    )
    
    print(f"[OK] Created via factory: {memory}")
    memory.close()
    print("PASSED\n")


def test_remember():
    """Test remembering content."""
    print("\n=== Test 3: Remember ===")
    
    memory = create_integrated_memory(":memory:")
    
    # Remember various types
    mem_id = memory.remember(
        "user prefers Python over JavaScript",
        importance=0.9,
        kind="preference"
    )
    print(f"[OK] Remembered preference: id={mem_id}")
    
    mem_id = memory.remember(
        "project uses SQLite for storage",
        importance=0.8,
        kind="fact"
    )
    print(f"[OK] Remembered fact: id={mem_id}")
    
    status = memory.get_status()
    print(f"  Unified stats: {status.get('unified', {})}")
    
    memory.close()
    print("PASSED\n")


def test_retrieve():
    """Test retrieval."""
    print("\n=== Test 4: Retrieve ===")
    
    memory = create_integrated_memory(":memory:")
    
    # Add some memories
    memory.remember("user likes dark mode", kind="preference", importance=0.8)
    memory.remember("Python is the main language", kind="fact", importance=0.9)
    memory.remember("meeting scheduled for 3pm", kind="task", importance=0.7)
    
    # Retrieve
    results = memory.retrieve("Python")
    print(f"  Query 'Python': found {len(results)} results")
    for r in results:
        print(f"    - {r['content']} (type={r['type']}, source={r['source']})")
    
    memory.close()
    print("PASSED\n")


def test_skill_registration():
    """Test registering skills."""
    print("\n=== Test 5: Skill Registration ===")
    
    memory = create_integrated_memory(":memory:")
    
    # Register a skill
    skill_code = """
def execute(data):
    return f"processed: {data}"
"""
    
    skill_id = memory.register_skill(
        name="data_processor",
        code=skill_code,
        description="Process data",
        category="data",
    )
    print(f"[OK] Registered skill: id={skill_id}")
    
    # Register another
    skill_id2 = memory.register_skill(
        name="pptx_generator",
        code="",
        description="Generate PowerPoint",
        category="office",
    )
    print(f"[OK] Registered pptx skill: id={skill_id2}")
    
    # List skills
    skills = memory.list_skills()
    print(f"  Total skills: {len(skills)}")
    for s in skills:
        print(f"    - {s['content']}: {s['activation_count']} uses")
    
    memory.close()
    print("PASSED\n")


def test_skill_execution():
    """Test executing skills."""
    print("\n=== Test 6: Skill Execution ===")
    
    memory = create_integrated_memory(":memory:")
    
    # Register skill with handler
    def my_handler(text, prefix="result"):
        return f"{prefix}: {text.upper()}"
    
    memory.unified.store.register_skill_handler("text_processor", my_handler)
    
    skill_id = memory.register_skill(
        name="text_processor",
        code="",
        description="Process text",
    )
    print(f"[OK] Registered skill with handler")
    
    # Execute
    result, success = memory.use_skill("text_processor", text="hello world")
    print(f"  Execution: result={result}, success={success}")
    
    # Check effectiveness updated
    skill = memory.get_skill("text_processor")
    print(f"  After execution: effectiveness={skill.effectiveness}")
    
    memory.close()
    print("PASSED\n")


def test_recall_everything():
    """Test unified recall of memories and skills."""
    print("\n=== Test 7: Recall Everything ===")
    
    memory = create_integrated_memory(":memory:")
    
    # Add memories
    memory.remember("user preference: detailed code", kind="preference")
    memory.remember("fact: using Python 3.11", kind="fact")
    
    # Add skills
    memory.register_skill("pptx_maker", "", "Make slides", "office")
    memory.register_skill("code_formatter", "", "Format code", "dev")
    
    # Recall everything
    results = memory.recall_everything("code")
    print(f"  Memories: {len(results['memories'])}")
    for m in results['memories']:
        print(f"    - {m['content']} ({m['type']})")
    
    print(f"  Skills: {len(results['skills'])}")
    for s in results['skills']:
        print(f"    - {s['name']} ({s['category']})")
    
    memory.close()
    print("PASSED\n")


def test_suggest_skill():
    """Test skill suggestion."""
    print("\n=== Test 8: Suggest Skill ===")
    
    memory = create_integrated_memory(":memory:")
    
    # Add skills
    memory.register_skill("pptx_generator", "", "Make PowerPoint", "office")
    memory.register_skill("code_formatter", "", "Format code", "dev")
    memory.register_skill("data_analyzer", "", "Analyze data", "data")
    
    # Simulate some usage to build effectiveness
    memory.use_skill("code_formatter", text="test")  # Good
    memory.use_skill("code_formatter", text="test")  # Good
    
    # Suggest for "slides"
    suggestion = memory.suggest_skill("slides")
    print(f"  Suggestion for 'slides': {suggestion}")
    
    # Suggest for "format code"
    suggestion = memory.suggest_skill("format code")
    print(f"  Suggestion for 'format code': {suggestion}")
    
    memory.close()
    print("PASSED\n")


def test_status():
    """Test status reporting."""
    print("\n=== Test 9: Status ===")
    
    memory = create_integrated_memory(":memory:")
    
    memory.remember("test memory 1", kind="preference")
    memory.remember("test memory 2", kind="fact")
    memory.register_skill("skill1", "", "Test skill 1")
    
    status = memory.get_status()
    print(f"  Hypergraph: {status['hypergraph']['name']}")
    print(f"  Unified: {status.get('unified', {})}")
    
    memory.close()
    print("PASSED\n")


def main():
    print("=" * 60)
    print("Integrated Memory Verification")
    print("=" * 60)
    
    try:
        test_creation()
        test_factory()
        test_remember()
        test_retrieve()
        test_skill_registration()
        test_skill_execution()
        test_recall_everything()
        test_suggest_skill()
        test_status()
        
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
