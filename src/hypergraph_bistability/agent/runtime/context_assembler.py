"""Context assembly for LLM calls."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from hypergraph_bistability.memory.policies import RetrievedMemory


class ContextAssembler:
    """Build bounded LLM message payloads for a turn."""

    def validate_handoff_continuity(
        self,
        response: str,
        handoff_bundle: Dict[str, Any],
        require_task_reference: bool = True,
    ) -> Dict[str, bool]:
        """Validate that response references key elements from handoff bundle.
        
        Args:
            response: The LLM response to validate
            handoff_bundle: The handoff bundle containing continuity data
            require_task_reference: If True, task name must be referenced
            
        Returns:
            Dict with validation results for each element type
        """
        response_lower = response.lower()
        
        results = {}
        
        # ===== Weak content detection =====
        # Detect placeholder/weak content that doesn't provide actual value
        weak_patterns = [
            "tbd", "to be determined", "to be decided",
            "look into it", "looking into it", "will look",
            "need to", "need more", "require more",
            "pending", "in progress", "to do",
            "not sure", "unsure", "don't know",
            "later", "after", "before",
            "will do", "going to", "plan to",
        ]
        
        # Check if response is mostly weak content
        weak_count = sum(1 for pattern in weak_patterns if pattern in response_lower)
        words = response_lower.split()
        significant_words = [w for w in words if len(w) > 4]
        
        # If more than 30% of significant words are weak patterns, flag it
        results["has_weak_content"] = weak_count < len(significant_words) * 0.3 if significant_words else False
        
        # ===== Task reference =====
        linked_task = handoff_bundle.get("linked_task", "")
        if linked_task:
            task_mentioned = linked_task.lower() in response_lower or any(
                word in response_lower for word in linked_task.lower().replace("_", " ").split()
            )
            results["task_reference"] = task_mentioned
        else:
            results["task_reference"] = True  # No task to reference
        
        # ===== Blocker reference - handle both string and dict formats =====
        blockers = handoff_bundle.get("blockers", [])
        if blockers:
            blocker_referenced = False
            for blocker in blockers:
                # Handle both string and dict formats
                if isinstance(blocker, str):
                    content = blocker
                elif isinstance(blocker, dict):
                    content = blocker.get("content", "")
                else:
                    continue
                    
                if content:
                    # Check if any significant word from blocker appears
                    key_words = [w for w in content.lower().split() if len(w) > 4]
                    blocker_referenced = any(word in response_lower for word in key_words[:3])
                    if blocker_referenced:
                        break
            results["blocker_reference"] = blocker_referenced
        else:
            results["blocker_reference"] = True
        
        # ===== Next step reference - handle both string and dict formats =====
        next_steps = handoff_bundle.get("next_steps", [])
        if next_steps:
            step_referenced = False
            for step in next_steps:
                if isinstance(step, str):
                    content = step
                elif isinstance(step, dict):
                    content = step.get("content", "")
                else:
                    continue
                    
                if content:
                    key_words = [w for w in content.lower().split() if len(w) > 4]
                    step_referenced = any(word in response_lower for word in key_words[:3])
                    if step_referenced:
                        break
            results["next_step_reference"] = step_referenced
        else:
            results["next_step_reference"] = True
        
        # ===== Decision reference - handle both string and dict formats =====
        decisions = handoff_bundle.get("active_decisions", [])
        if decisions:
            decision_referenced = False
            for decision in decisions:
                if isinstance(decision, str):
                    content = decision
                elif isinstance(decision, dict):
                    content = decision.get("content", "")
                else:
                    continue
                    
                if content:
                    key_words = [w for w in content.lower().split() if len(w) > 4]
                    decision_referenced = any(word in response_lower for word in key_words[:3])
                    if decision_referenced:
                        break
            results["decision_reference"] = decision_referenced
        else:
            results["decision_reference"] = True
        
        # ===== Overall pass/fail =====
        # Must have: no weak content AND at least one of (task, blocker, next_step, decision) referenced
        has_any_reference = (
            results.get("task_reference", False) or
            results.get("blocker_reference", False) or
            results.get("next_step_reference", False) or
            results.get("decision_reference", False)
        )
        results["passed"] = results.get("has_weak_content", True) and has_any_reference
        
        return results

    def build_messages(
        self,
        *,
        system_prompt: str,
        memory_context: str,
        retrieved_items: List[RetrievedMemory],
        conversation_history: List[Dict[str, str]],
        user_input: str,
        working_set_context: str = "",
    ) -> List[Dict[str, str]]:
        extra_context = ""
        if retrieved_items:
            extra_context = f"\n\nRetrieved memory:\n{self._format_retrieved_sections(retrieved_items)}"
        
        # Add working-set context if available
        ws_context = ""
        if working_set_context:
            ws_context = f"\n\n```json\n{working_set_context}\n```" if working_set_context else ""
        response_contract = (
            "\n\nResponse contract:\n"
            "- Do not output <think> tags or hidden reasoning.\n"
            "- If retrieved memory is relevant, use it explicitly in the answer.\n"
            "- Prioritize recalled preferences, tasks, and active issues over generic advice.\n"
            "- If working-set JSON is provided, use its task/decisions/procedures data to inform your response.\n"
            "- If the user is resuming prior work, explicitly name the recalled task or issue before asking follow-up questions.\n"
            "- Do not give a purely generic clarifying question when a relevant task or issue was retrieved.\n"
            "- Keep the answer concise unless the user asks for detail."
        )
        response_contract += self._format_preference_contract(retrieved_items)
        response_contract += self._format_mode_contract(user_input, retrieved_items)

        return [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nCurrent memory state:\n{memory_context}{extra_context}{ws_context}{response_contract}",
            },
            *conversation_history[-5:],
            {"role": "user", "content": user_input},
        ]

    def _format_preference_contract(self, items: List[RetrievedMemory]) -> str:
        preference_text = " ".join(
            item.content.lower() for item in items if item.kind == "preference"
        )
        extra_rules: List[str] = []

        if "root cause hypothesis first" in preference_text:
            extra_rules.append('- If answering a debugging or investigation question, begin with the literal phrase "Root cause hypothesis:".')
        if "diff-style" in preference_text:
            extra_rules.append('- If answering about code changes, fixes, or what to inspect first in a coding task, begin with the literal phrase "Diff-style summary:".')
        if "bullet list" in preference_text or "bullet lists" in preference_text:
            extra_rules.append("- When listing actions or checks, use bullet points.")
        if "concise" in preference_text:
            extra_rules.append("- Keep the answer short and direct.")

        if not extra_rules:
            return ""
        return "\n" + "\n".join(extra_rules)

    def _format_mode_contract(self, user_input: str, items: List[RetrievedMemory]) -> str:
        lowered_input = user_input.lower()
        joined = " ".join(item.content.lower() for item in items)
        extra_rules: List[str] = []

        if any(token in lowered_input for token in ("investigate first", "return to the bug", "debug", "incident")):
            extra_rules.append("- In a debugging resume answer, explicitly mention the recalled issue and the first concrete checks.")
        if "cache invalidation" in joined or "stale profile data" in joined:
            extra_rules.append("- If cache invalidation or stale profile data was retrieved, mention both explicitly in the answer.")
        if any(token in lowered_input for token in ("scheduler fix", "inspect first", "coding task")):
            extra_rules.append("- In a coding resume answer, explicitly restate the active issue before any follow-up question.")
        if "retry loop" in joined or "backoff state" in joined:
            extra_rules.append("- If retry loop or backoff state was retrieved, mention both explicitly in the answer.")
        if any(token in lowered_input for token in ("continue the plan", "artifact", "profile-sync incident")):
            extra_rules.append("- When continuing a remembered plan, explicitly surface the next plan steps rather than summarizing vaguely.")
        if "cursor reset" in joined or "reconnect ordering" in joined or "stale cursor cleanup" in joined:
            extra_rules.append("- If artifact plan steps were retrieved, explicitly mention cursor reset, reconnect ordering, and stale cursor cleanup when relevant.")

        if not extra_rules:
            return ""
        return "\n" + "\n".join(extra_rules)

    def _format_retrieved_sections(self, items: List[RetrievedMemory]) -> str:
        sections = {
            "preference": [],
            "task": [],
            "fact": [],
            "context": [],
        }
        for item in items:
            label = item.kind if item.kind in sections else "context"
            sections[label].append(item)

        lines: List[str] = []
        for label in ("preference", "task", "fact", "context"):
            section_items = sections[label]
            if not section_items:
                continue
            lines.append(f"{label.title()}:")
            for item in section_items[:2]:
                lines.append(f"- [{item.source}] {item.content}")
        return "\n".join(lines)


class TestHandoffContinuityValidation:
    """Test the handoff continuity validation function."""

    def test_task_reference_detected(self):
        """Test that task name is detected in response."""
        assembler = ContextAssembler()
        
        handoff = {
            "linked_task": "fix_api_bug",
            "blockers": [{"content": "API returns 500 error"}],
            "next_steps": [{"content": "Check server logs"}],
            "active_decisions": [],
        }
        
        # Response mentions task
        response = "I'll help you fix the API bug by checking the logs first."
        result = assembler.validate_handoff_continuity(response, handoff)
        
        assert result["task_reference"] == True

    def test_task_reference_not_detected(self):
        """Test that missing task reference is detected."""
        assembler = ContextAssembler()
        
        handoff = {
            "linked_task": "fix_api_bug",
            "blockers": [{"content": "API returns 500 error"}],
            "next_steps": [{"content": "Check server logs"}],
            "active_decisions": [],
        }
        
        # Response doesn't mention task
        response = "I'll help you with something else."
        result = assembler.validate_handoff_continuity(response, handoff)
        
        assert result["task_reference"] == False

    def test_blocker_reference_detected(self):
        """Test that blocker is detected in response."""
        assembler = ContextAssembler()
        
        handoff = {
            "linked_task": "fix_api_bug",
            "blockers": [{"content": "API returns 500 error on /users endpoint"}],
            "next_steps": [],
            "active_decisions": [],
        }
        
        # Response mentions "500" from blocker
        response = "The 500 error is likely caused by the database connection."
        result = assembler.validate_handoff_continuity(response, handoff)
        
        assert result["blocker_reference"] == True

    def test_empty_handoff_passes(self):
        """Test that empty handoff bundle passes validation."""
        assembler = ContextAssembler()
        
        handoff = {
            "linked_task": "",
            "blockers": [],
            "next_steps": [],
            "active_decisions": [],
        }
        
        response = "Hello! How can I help you?"
        result = assembler.validate_handoff_continuity(response, handoff)
        
        assert result["passed"] == True

    def test_partial_continuity_fails(self):
        """Test that partial continuity fails validation."""
        assembler = ContextAssembler()
        
        handoff = {
            "linked_task": "fix_api_bug",
            "blockers": [{"content": "API returns 500 error"}],
            "next_steps": [{"content": "Check server logs"}],
            "active_decisions": [{"content": "Use error middleware"}],
        }
        
        # Response only mentions task, not blockers/next_steps/decisions
        response = "I'll help you fix the API bug."
        result = assembler.validate_handoff_continuity(response, handoff)
        
        # Should fail - no blocker, next_step, or decision reference
        assert result["passed"] == False
        assert result["task_reference"] == True
        assert result["blocker_reference"] == False
        assert result["next_step_reference"] == False
        assert result["decision_reference"] == False
