#!/usr/bin/env python
"""
Simplified 2-state handoff continuity smoke detector.

This is NOT:
- An auto controller
- A semantic judge  
- For auto regeneration

This IS:
- A continuity smoke detector
- For logging and human sampling
- For catching obvious failures

States:
- fail: obvious missing continuity signal
- non_fail: everything else
"""
from typing import Dict, Any, List, Set


# Generic template patterns that indicate no real continuity
GENERIC_PATTERNS = [
    "based on your input",
    "here's my recommendation",
    "let me help you",
    "i can assist",
    "as requested",
    "thank you for",
    "i'll look into",
    "i'll check",
    "let me review",
    "sounds good",
    "let me know if",
    "feel free to",
    "hope this helps",
    "happy to help",
    "what would you like",
    "how can i help",
    "is there anything",
    "do you want me to",
]


def check_continuity_smoke(
    response: str,
    handoff_bundle: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simple continuity smoke check - catches obvious failures.
    
    Returns:
        status: "fail" | "non_fail"
        has_task: bool
        has_substantive: bool  
        is_generic: bool
        matched_types: list of what was matched
    """
    response_lower = response.lower()
    
    # Check for generic template first
    is_generic = any(pattern in response_lower for pattern in GENERIC_PATTERNS)
    
    # Collect all handoff content
    all_content: List[tuple] = []
    
    # Task
    task = handoff_bundle.get("linked_task", "")
    if task:
        all_content.append(("task", task.lower()))
    
    # Blockers
    for blocker in handoff_bundle.get("blockers", []):
        content = blocker if isinstance(blocker, str) else blocker.get("content", "")
        if content:
            all_content.append(("blocker", content.lower()))
    
    # Next steps
    for step in handoff_bundle.get("next_steps", []):
        content = step if isinstance(step, str) else step.get("content", "")
        if content:
            all_content.append(("next_step", content.lower()))
    
    # Decisions
    for decision in handoff_bundle.get("active_decisions", []):
        content = decision if isinstance(decision, str) else decision.get("content", "")
        if content:
            all_content.append(("decision", content.lower()))
    
    # Check for matches
    has_task = False
    has_substantive = False
    matched_types: Set[str] = set()
    
    for item_type, content in all_content:
        words = content.split()
        significant_words = [w for w in words if len(w) > 3]
        
        for word in significant_words:
            if word in response_lower:
                matched_types.add(item_type)
                if item_type == "task":
                    has_task = True
                else:
                    has_substantive = True
                break
    
    # Determine status
    if is_generic:
        status = "fail"
        reason = "generic_template"
    elif has_task and has_substantive:
        status = "non_fail"
        reason = "task_plus_substantive"
    elif has_task:
        # Only task mentioned, no substantive - borderline
        status = "non_fail" 
        reason = "task_only"
    elif has_substantive:
        # Substantive without task - still has continuity
        status = "non_fail"
        reason = "substantive_only"
    else:
        status = "fail"
        reason = "no_reference"
    
    return {
        "status": status,
        "has_task": has_task,
        "has_substantive": has_substantive,
        "is_generic": is_generic,
        "matched_types": list(matched_types),
        "reason": reason
    }


def main():
    """Test the smoke detector."""
    test_cases = [
        # FAIL cases
        {
            "name": "Generic response",
            "handoff": {"linked_task": "Fix login", "blockers": ["OAuth"], "next_steps": [], "active_decisions": []},
            "response": "Based on your input, here's my recommendation. Let me help you with that.",
            "expected": "fail"
        },
        {
            "name": "No reference",
            "handoff": {"linked_task": "Fix login", "blockers": [], "next_steps": [], "active_decisions": []},
            "response": "Hello! How can I help you today?",
            "expected": "fail"
        },
        # NON_FAIL cases
        {
            "name": "Task + blocker",
            "handoff": {"linked_task": "Fix login", "blockers": ["OAuth expired"], "next_steps": [], "active_decisions": []},
            "response": "I'll fix the login issue. The OAuth token is expired.",
            "expected": "non_fail"
        },
        {
            "name": "Task only",
            "handoff": {"linked_task": "Fix login", "blockers": [], "next_steps": [], "active_decisions": []},
            "response": "I'll work on the login fix.",
            "expected": "non_fail"  # borderline but non_fail
        },
        {
            "name": "Next step only",
            "handoff": {"linked_task": "", "blockers": [], "next_steps": ["Add retry"], "active_decisions": []},
            "response": "I should add retry logic.",
            "expected": "non_fail"
        }
    ]
    
    print("Continuity Smoke Detector Test")
    print("="*50)
    
    all_passed = True
    for tc in test_cases:
        result = check_continuity_smoke(tc["response"], tc["handoff"])
        passed = result["status"] == tc["expected"]
        all_passed = all_passed and passed
        
        print(f"\n[{tc['name']}]")
        print(f"  Expected: {tc['expected']}")
        print(f"  Got: {result['status']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Matched: {result['matched_types']}")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    print("\n" + "="*50)
    print(f"All tests: {'PASSED' if all_passed else 'FAILED'}")


if __name__ == "__main__":
    main()