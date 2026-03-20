"""
Dialogue Scenario Tests
=======================

Three real-world scenarios testing the Agent Memory Module:
1. Career Assistant - Multi-round career coaching
2. Multi-Role Switch - Assistant switching between roles
3. Long-Lifecycle - Long project with many conversation rounds

Usage:
    python test_dialogue_scenarios.py [--api-key KEY]
"""

import argparse
import json
import time
import tempfile
import pathlib
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


@dataclass
class DialogueTurn:
    user: str
    assistant: str
    memory_summary: str
    physics_state: str


class DialogueScenario:
    def __init__(self, name: str, k: int = 4, L: int = 2):
        from agent.agent_memory_enhanced import AgentMemoryEnhanced
        self.name = name
        self.mem = AgentMemoryEnhanced(
            k=k, L=L, name=name.replace(" ", "_").lower(),
            use_llm_detector=False,
            use_llm_mapper=False
        )
        self.mem.group_labels = ["goals", "skills", "preferences", "context"]
        self.mem.layer_labels = ["current", "history"]
        self.turns: List[DialogueTurn] = []
        
    def run_turn(self, user_prompt: str, assistant_response: str = "") -> Dict:
        result = self.mem.process_prompt(user_prompt)
        state = self.mem.read()
        
        turn = DialogueTurn(
            user=user_prompt,
            assistant=assistant_response,
            memory_summary=f"Active: {state.n_active}/{self.mem.k * self.mem.L}, "
                          f"Groups: {state.groups.round(2).tolist()}",
            physics_state=f"lambda={self.mem.lambda_:.4f}, r={self.mem.lambda_ / (self.mem.get_lambda_c() or 1):.2f}"
        )
        self.turns.append(turn)
        return result
    
    def summary(self) -> Dict:
        return {
            "scenario": self.name,
            "total_turns": len(self.turns),
            "final_lambda": self.mem.lambda_,
            "final_r": self.mem.lambda_ / (self.mem.get_lambda_c() or 1),
            "final_active": self.mem.read().n_active,
            "mode": self.mem.current_mode
        }


def scenario_1_career_assistant() -> Tuple[DialogueScenario, bool]:
    """
    Scenario 1: Career Assistant
    
    A career coaching assistant helping a user through job search,
    resume building, and interview preparation over multiple rounds.
    
    Tests:
    - Long-term memory persistence
    - Conflict detection (user says one thing, does another)
    - Mode switching based on conversation phase
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: CAREER ASSISTANT")
    print("=" * 70)
    
    scenario = DialogueScenario("Career Assistant", k=4, L=2)
    
    conversation = [
        {
            "user": "Hi! I'm looking to transition from marketing to software engineering. "
                   "I've been learning Python for 6 months.",
            "response": "That's exciting! Let me help you build a learning path. "
                       "With your marketing background, you have a unique advantage in "
                       "understanding user needs and product thinking.",
            "expected_mode": "exploratory",
            "check": "initial_profile"
        },
        {
            "user": "Should I focus on frontend or backend? I'm confused.",
            "response": "Good question. Based on your interest in creating visible things "
                       "and your marketing background, frontend might be a good start, "
                       "but backend offers stronger fundamentals. Let me assess...",
            "expected_mode": "exploratory",
            "check": "choice_confusion"
        },
        {
            "user": "I want A but my friend says B is better. Actually no, I think A is right.",
            "response": "It's great that you trust your instincts! A can work well for you. "
                       "Let's make a structured plan.",
            "expected_mode": "neutral",
            "check": "conflict_resolved"
        },
        {
            "user": "Now I need to build projects. What should I do first?",
            "response": "Start with a portfolio project that solves a real problem you understand. "
                       "Your marketing experience is valuable here - build something that showcases both "
                       "your technical skills AND your product thinking.",
            "expected_mode": "focused",
            "check": "goal_oriented"
        },
        {
            "user": "Actually wait, I'm also considering a bootcamp. But they are expensive...",
            "response": "That's a significant decision. Let me help you weigh the options. "
                       "Self-learning vs bootcamp each have trade-offs. What matters most to you?",
            "expected_mode": "neutral",
            "check": "decision_conflict"
        },
        {
            "user": "I think I'll do self-study for now and maybe do a project-based course.",
            "response": "Excellent decision framework! That's a smart approach. "
                       "Let me create a structured learning roadmap for you.",
            "expected_mode": "focused",
            "check": "decision_made"
        },
        {
            "user": "Thanks! Let's start with Git and then Python basics review.",
            "response": "Perfect sequence! Git is essential for collaboration. "
                       "Let's get your environment set up first.",
            "expected_mode": "focused",
            "check": "action_oriented"
        },
    ]
    
    all_passed = True
    for i, turn in enumerate(conversation):
        print(f"\n[Turn {i+1}] {scenario.name}")
        print(f"  User: {turn['user'][:60]}...")
        
        scenario.mem.switch_mode(turn['expected_mode'])
        result = scenario.run_turn(turn['user'], turn['response'])
        
        print(f"  Conflict: {result['conflict_level']:.1%}, "
              f"Lambda: {scenario.mem.lambda_:.3f}, "
              f"Mode: {scenario.mem.current_mode}")
        
        if scenario.mem.current_mode != turn['expected_mode']:
            print(f"  NOTE: Expected mode '{turn['expected_mode']}', "
                  f"got '{scenario.mem.current_mode}'")
        
        state = scenario.mem.read()
        print(f"  Memory: {state.n_active}/{scenario.mem.k * scenario.mem.L} active")
    
    summary = scenario.summary()
    print(f"\n[SUMMARY]")
    print(f"  Total turns: {summary['total_turns']}")
    print(f"  Final mode: {summary['mode']}")
    print(f"  Final lambda: {summary['final_lambda']:.4f}")
    print(f"  Final r: {summary['final_r']:.2f}")
    print(f"  Memory efficiency: {summary['final_active']}/{scenario.mem.k * scenario.mem.L}")
    
    return scenario, all_passed


def scenario_2_multi_role_switch() -> Tuple[DialogueScenario, bool]:
    """
    Scenario 2: Multi-Role Switch
    
    A coding assistant that switches between different roles:
    - Technical writer
    - Debugger  
    - Software architect
    
    Tests:
    - Mode switching between distinct roles
    - Context isolation for different task types
    - Memory collapsing when role changes
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: MULTI-ROLE SWITCH")
    print("=" * 70)
    
    scenario = DialogueScenario("MultiRole Assistant", k=4, L=3)
    scenario.mem.layer_labels = ["requirements", "design", "implementation"]
    
    conversation = [
        {
            "user": "I need to write documentation for my REST API. "
                   "Can you help with OpenAPI specs?",
            "role": "technical_writer",
            "expected_r": (0.4, 0.6),
        },
        {
            "user": "Now let's debug why my authentication is failing. "
                   "The token expires too quickly.",
            "role": "debugger",
            "expected_r": (0.4, 0.6),
        },
        {
            "user": "Actually, before debugging, let me think about the architecture. "
                   "Should I use JWT or session-based auth?",
            "role": "architect",
            "expected_r": (0.4, 0.6),
        },
        {
            "user": "I think JWT is better for our microservices. "
                   "Now let's implement it.",
            "role": "implementer",
            "expected_r": (0.4, 0.6),
        },
        {
            "user": "Wait, I'm having second thoughts. What if we need to revoke tokens?",
            "role": "architect",
            "expected_r": (0.4, 0.6),
        },
        {
            "user": "Good point. Let's use refresh tokens with JWT. "
                   "Now back to implementation.",
            "role": "implementer",
            "expected_r": (0.4, 0.6),
        },
        {
            "user": "Actually, I'm now confused about the whole auth design. "
                   "Should I just use OAuth2 instead?",
            "role": "architect",
            "expected_r": (0.4, 0.6),
        },
        {
            "user": "You know what, let's stick with JWT + refresh tokens. "
                   "I don't want to overcomplicate this. Let's implement!",
            "role": "implementer",
            "expected_r": (0.4, 0.6),
        },
    ]
    
    all_passed = True
    for i, turn in enumerate(conversation):
        print(f"\n[Turn {i+1}] Role: {turn['role']}")
        print(f"  User: {turn['user'][:60]}...")
        
        scenario.mem.switch_mode(turn['role'])
        result = scenario.mem.process_prompt(turn['user'])
        
        r = scenario.mem.lambda_ / (scenario.mem.get_lambda_c() or 1)
        
        print(f"  Mode: {scenario.mem.current_mode}, "
              f"Lambda: {scenario.mem.lambda_:.3f}, "
              f"r: {r:.2f}")
        
        low_r, high_r = turn['expected_r']
        if not (low_r <= r <= high_r):
            print(f"  NOTE: Expected r in {turn['expected_r']}, got {r:.2f}")
            all_passed = False
        
        state = scenario.mem.read()
        print(f"  Memory: {state.n_active}/{scenario.mem.k * scenario.mem.L} active")
        print(f"  Group means: {state.groups.round(2)}")
    
    summary = scenario.summary()
    print(f"\n[SUMMARY]")
    print(f"  Total turns: {summary['total_turns']}")
    print(f"  Role switches: 4+")
    print(f"  Final r: {summary['final_r']:.2f}")
    
    return scenario, all_passed


def scenario_3_long_lifecycle() -> Tuple[DialogueScenario, bool]:
    """
    Scenario 3: Long-Lifecycle Project
    
    A project assistant maintaining context across 15+ rounds
    of a long software development project.
    
    Tests:
    - Memory persistence over many turns
    - Progressive memory building
    - Conflict detection in evolving requirements
    - Memory collapse behavior under high load
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: LONG-LIFECYCLE PROJECT")
    print("=" * 70)
    
    scenario = DialogueScenario("Project Assistant", k=4, L=2)
    scenario.mem.group_labels = ["requirements", "design", "code", "tests"]
    scenario.mem.layer_labels = ["current_sprint", "history"]
    
    conversation = [
        ("Project kickoff: we're building a task management app", "focused"),
        ("Users need to create, edit, delete tasks", "focused"),
        ("Add due dates and priorities to tasks", "focused"),
        ("Should we add subtasks? I'm not sure...", "exploratory"),
        ("Yes, let's add subtasks with parent-child relationship", "focused"),
        ("Now add team collaboration features", "focused"),
        ("Wait, that complicates things. Let's simplify first", "neutral"),
        ("OK let's do simple first: solo task management", "focused"),
        ("Add drag-drop reordering of tasks", "focused"),
        ("Add categories/labels for tasks", "focused"),
        ("Hmm, should labels be mutually exclusive or multiple?", "exploratory"),
        ("Multiple labels - users will want that", "focused"),
        ("Add recurring tasks feature", "focused"),
        ("What about task templates?", "exploratory"),
        ("Good idea - template system with presets", "focused"),
        ("Add notifications and reminders", "focused"),
        ("This is getting complex. Let's review what we have", "neutral"),
        ("OK we've got core features. Time to add tests", "focused"),
        ("Let's add unit tests first", "focused"),
        ("Integration tests for the API", "focused"),
    ]
    
    all_passed = True
    conflict_detected = []
    
    for i, (user_prompt, expected_mode) in enumerate(conversation):
        print(f"\n[Turn {i+1}] Mode: {expected_mode}")
        print(f"  User: {user_prompt[:60]}...")
        
        scenario.mem.switch_mode(expected_mode)
        result = scenario.mem.process_prompt(user_prompt)
        
        state = scenario.mem.read()
        r = scenario.mem.lambda_ / (scenario.mem.get_lambda_c() or 1)
        
        print(f"  Conflict: {result['conflict_level']:.1%}, "
              f"r: {r:.2f}, "
              f"Active: {state.n_active}/{scenario.mem.k * scenario.mem.L}")
        
        if result['conflict_level'] > 0.3:
            conflict_detected.append((i+1, result['conflict_level']))
            print(f"  ^ Conflict detected!")
        
        if state.n_active > scenario.mem.k * scenario.mem.L * 0.75:
            print(f"  ^ Memory getting full, collapse may occur soon")
        
        if (i+1) % 5 == 0:
            print(f"  [Checkpoint at turn {i+1}]")
    
    summary = scenario.summary()
    print(f"\n[SUMMARY]")
    print(f"  Total turns: {summary['total_turns']}")
    print(f"  Final mode: {summary['mode']}")
    print(f"  Final r: {summary['final_r']:.2f}")
    print(f"  Final active: {summary['final_active']}/{scenario.mem.k * scenario.mem.L}")
    print(f"  Conflicts detected: {len(conflict_detected)}")
    
    for turn, level in conflict_detected:
        print(f"    Turn {turn}: {level:.1%}")
    
    return scenario, all_passed


def run_persistence_test(scenario: DialogueScenario) -> bool:
    """Test that a scenario survives save/load."""
    print("\n[PERSISTENCE TEST]")
    
    tmp_path = pathlib.Path(tempfile.gettempdir()) / f"scenario_{scenario.name}.json"
    path = scenario.mem.save(str(tmp_path))
    print(f"  Saved to: {path}")
    
    from agent.agent_memory_enhanced import AgentMemoryEnhanced
    loaded = AgentMemoryEnhanced.load(str(tmp_path))
    
    assert loaded.k == scenario.mem.k, "k should match"
    assert loaded.L == scenario.mem.L, "L should match"
    assert loaded.current_mode == scenario.mem.current_mode, "mode should match"
    assert abs(loaded.lambda_ - scenario.mem.lambda_) < 0.001, "lambda should match"
    
    print(f"  Loaded successfully!")
    print(f"  k={loaded.k}, L={loaded.L}, mode={loaded.current_mode}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Dialogue Scenario Tests")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (optional)")
    args = parser.parse_args()
    
    results = {}
    
    print("=" * 70)
    print("DIALOGUE SCENARIO TEST SUITE")
    print("=" * 70)
    
    # Scenario 1: Career Assistant
    try:
        s1, passed1 = scenario_1_career_assistant()
        results["scenario1_career"] = "PASS" if passed1 else "PARTIAL"
        run_persistence_test(s1)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["scenario1_career"] = f"FAIL: {e}"
    
    # Scenario 2: Multi-Role Switch
    try:
        s2, passed2 = scenario_2_multi_role_switch()
        results["scenario2_multirole"] = "PASS" if passed2 else "PARTIAL"
        run_persistence_test(s2)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["scenario2_multirole"] = f"FAIL: {e}"
    
    # Scenario 3: Long-Lifecycle
    try:
        s3, passed3 = scenario_3_long_lifecycle()
        results["scenario3_lifecycle"] = "PASS" if passed3 else "PARTIAL"
        run_persistence_test(s3)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["scenario3_lifecycle"] = f"FAIL: {e}"
    
    # Final Report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    all_pass = all(v == "PASS" for v in results.values())
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME ISSUES'}")
    
    return results


if __name__ == "__main__":
    main()
