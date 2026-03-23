#!/usr/bin/env python
"""Simplified handoff validator - just checks for any reference."""
from typing import Dict, Any, List


def has_any_reference(response: str, handoff_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple check: does response mention any handoff content?
    
    This is a simple diagnostic signal, not a classifier.
    """
    response_lower = response.lower()
    
    # Collect all handoff content
    items = []
    
    task = handoff_bundle.get("linked_task", "")
    if task:
        items.append(("task", task))
    
    for blocker in handoff_bundle.get("blockers", []):
        content = blocker if isinstance(blocker, str) else blocker.get("content", "")
        if content:
            items.append(("blocker", content))
    
    for step in handoff_bundle.get("next_steps", []):
        content = step if isinstance(step, str) else step.get("content", "")
        if content:
            items.append(("next_step", content))
    
    for decision in handoff_bundle.get("active_decisions", []):
        content = decision if isinstance(decision, str) else decision.get("content", "")
        if content:
            items.append(("decision", content))
    
    # Check for any keyword match
    matched_types = []
    for item_type, content in items:
        content_lower = content.lower()
        # Simple keyword check - any significant word appears in response
        words = [w for w in content_lower.split() if len(w) > 3]
        for word in words:
            if word in response_lower:
                matched_types.append(item_type)
                break
    
    # Simple result
    has_match = len(matched_types) > 0
    
    return {
        "has_reference": has_match,
        "matched_types": list(set(matched_types)),
        "item_count": len(items),
        "status": "has_continuity" if has_match else "no_continuity"
    }


def main():
    """Test the simplified validator."""
    # Test cases
    test_cases = [
        {
            "name": "Has task reference",
            "handoff": {"linked_task": "Fix login bug", "blockers": [], "next_steps": [], "active_decisions": []},
            "response": "I'll fix the login bug now.",
            "expected": "has_continuity"
        },
        {
            "name": "No reference",
            "handoff": {"linked_task": "Fix login bug", "blockers": [], "next_steps": [], "active_decisions": []},
            "response": "Hello! How can I help you today?",
            "expected": "no_continuity"
        },
        {
            "name": "Has blocker reference",
            "handoff": {"linked_task": "", "blockers": ["OAuth token expired"], "next_steps": [], "active_decisions": []},
            "response": "The OAuth token seems to be expired.",
            "expected": "has_continuity"
        }
    ]
    
    print("Simplified Handoff Validator Test")
    print("="*50)
    
    all_passed = True
    for tc in test_cases:
        result = has_any_reference(tc["response"], tc["handoff"])
        passed = result["status"] == tc["expected"]
        all_passed = all_passed and passed
        
        print(f"\n[{tc['name']}]")
        print(f"  Expected: {tc['expected']}")
        print(f"  Got: {result['status']}")
        print(f"  Matched: {result['matched_types']}")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    print("\n" + "="*50)
    print(f"All tests: {'PASSED' if all_passed else 'FAILED'}")


if __name__ == "__main__":
    main()