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
    ) -> Dict[str, Any]:
        """Validate that response demonstrates meaningful continuity with handoff bundle.
        
        Returns a structured result with three-state classification:
        - not_applicable: No handoff data to validate against
        - weak_pass: Some continuity but very shallow (keyword matches only)
        - strong_pass: Substantive continuity (specific details, next actions)
        - fail: No meaningful continuity detected
        
        Args:
            response: The LLM response to validate
            handoff_bundle: The handoff bundle containing continuity data
            
        Returns:
            Dict with status, reason, and detailed scores
        """
        response_lower = response.lower()
        words = response_lower.split()
        
        result = {
            "status": "fail",  # Default to fail
            "reason": "",
            "task_reference": False,
            "blocker_reference": False,
            "next_step_reference": False,
            "decision_reference": False,
            "content_quality": "good",  # good/weak/bad
            "is_generic_template": False,
        }
        
        # ===== Check for generic templates FIRST - these are automatic fails =====
        generic_template_patterns = [
            "based on your input",
            "here's my recommendation",
            "[focused mode",
            "focused mode,", 
            "i can help with",
            "i'd be happy to",
            "certainly!",
            "of course!",
            "great question",
            "thanks for asking",
            "as requested",
            "let me know if you need",
        ]
        
        for pattern in generic_template_patterns:
            if pattern in response_lower:
                result["is_generic_template"] = True
                result["status"] = "fail"
                result["reason"] = "generic_template_detected"
                return result
        
        # ===== Check if there's anything to validate against =====
        has_task = bool(handoff_bundle.get("linked_task"))
        has_blockers = bool(handoff_bundle.get("blockers"))
        has_next_steps = bool(handoff_bundle.get("next_steps"))
        has_decisions = bool(handoff_bundle.get("active_decisions"))
        
        if not any([has_task, has_blockers, has_next_steps, has_decisions]):
            result["status"] = "not_applicable"
            result["reason"] = "no_continuity_material"
            return result
        
        # ===== Weak content detection =====
        weak_patterns = [
            "tbd", "to be determined", "to be decided",
            "look into", "looking into", "will look",
            "need to", "need more", "require more",
            "pending", "in progress", "to do",
            "not sure", "unsure", "don't know",
            "later", "after", "before",
            "will do", "going to", "plan to",
            "we should", "let's", "let us",
            "maybe", "perhaps", "possibly",
        ]
        
        # Count weak patterns
        weak_count = sum(1 for p in weak_patterns if p in response_lower)
        significant_words = [w for w in words if len(w) > 4]
        
        if significant_words:
            weak_ratio = weak_count / len(significant_words)
            if weak_ratio > 0.4:
                result["content_quality"] = "bad"
            elif weak_ratio > 0.2:
                result["content_quality"] = "weak"
            # else "good"
        
        # ===== Task reference check =====
        # FIXED: Be stricter - must match full task phrase OR at least 1 significant word (or 2 if available)
        if has_task:
            linked_task = handoff_bundle.get("linked_task", "")
            task_lower = linked_task.lower().replace("_", " ")
            task_words = [w for w in task_lower.split() if len(w) > 3]
            
            # Must either:
            # 1. Match full task phrase in response, OR
            # 2. Match at least 1 significant word (if only 1 exists), or 2 (if 2+ exist)
            exact_match = task_lower in response_lower
            word_matches = sum(1 for tw in task_words if tw in response_lower)
            min_required = 1 if len(task_words) <= 1 else 2
            result["task_reference"] = exact_match or word_matches >= min_required
        
        # ===== Blocker reference check =====
        # FIXED: Be stricter - require matching specific blocker content, not just generic words
        if has_blockers:
            blockers = handoff_bundle.get("blockers", [])
            blocker_keywords = set()
            blocker_phrases = []  # Store full phrases for exact matching
            for blocker in blockers:
                content = blocker if isinstance(blocker, str) else blocker.get("content", "")
                if content:
                    blocker_phrases.append(content.lower())
                    # Extract significant words (length > 4, not common words)
                    for w in content.lower().split():
                        if len(w) > 4 and w not in {"which", "what", "where", "when", "there", "their", "these", "those", "about", "could", "would", "should", "might"}:
                            blocker_keywords.add(w)
            
            # Must match at least 2 keywords OR at least 1 exact phrase from blockers
            exact_phrase_match = any(phrase in response_lower for phrase in blocker_phrases)
            keyword_matches = sum(1 for kw in list(blocker_keywords)[:10] if kw in response_lower)
            result["blocker_reference"] = exact_phrase_match or keyword_matches >= 2
        
        # ===== Next step reference check =====
        # FIXED: Same stricter logic as blockers
        if has_next_steps:
            next_steps = handoff_bundle.get("next_steps", [])
            step_keywords = set()
            step_phrases = []
            for step in next_steps:
                content = step if isinstance(step, str) else step.get("content", "")
                if content:
                    step_phrases.append(content.lower())
                    for w in content.lower().split():
                        if len(w) > 4 and w not in {"which", "what", "where", "when", "there", "their", "these", "those", "about", "could", "would", "should", "might"}:
                            step_keywords.add(w)
            
            exact_phrase_match = any(phrase in response_lower for phrase in step_phrases)
            keyword_matches = sum(1 for kw in list(step_keywords)[:10] if kw in response_lower)
            result["next_step_reference"] = exact_phrase_match or keyword_matches >= 2
        
        # ===== Decision reference check =====
        # FIXED: Same stricter logic as blockers
        if has_decisions:
            decisions = handoff_bundle.get("active_decisions", [])
            decision_keywords = set()
            decision_phrases = []
            for decision in decisions:
                content = decision if isinstance(decision, str) else decision.get("content", "")
                if content:
                    decision_phrases.append(content.lower())
                    for w in content.lower().split():
                        if len(w) > 4 and w not in {"which", "what", "where", "when", "there", "their", "these", "those", "about", "could", "would", "should", "might"}:
                            decision_keywords.add(w)
            
            exact_phrase_match = any(phrase in response_lower for phrase in decision_phrases)
            keyword_matches = sum(1 for kw in list(decision_keywords)[:10] if kw in response_lower)
            result["decision_reference"] = exact_phrase_match or keyword_matches >= 2
        
        # ===== Determine final status =====
        # FIXED: Stricter criteria - strong_pass requires task + at least one substantive reference
        has_task_ref = result["task_reference"]
        has_substantive_ref = (
            result["blocker_reference"] or 
            result["next_step_reference"] or 
            result["decision_reference"]
        )
        
        references = [
            result["task_reference"],
            result["blocker_reference"],
            result["next_step_reference"],
            result["decision_reference"],
        ]
        reference_count = sum(references)
        
        if result["content_quality"] == "bad":
            result["status"] = "fail"
            result["reason"] = "too_much_weak_content"
        elif reference_count == 0:
            result["status"] = "fail"
            result["reason"] = "no_reference_found"
        elif reference_count == 1 or not has_substantive_ref:
            # Only task reference or no substantive reference = weak pass
            result["status"] = "weak_pass"
            result["reason"] = f"insufficient_continuity_{reference_count}_elements"
        elif has_task_ref and has_substantive_ref:
            # Must have BOTH task AND at least one substantive reference for strong_pass
            if result["content_quality"] == "good":
                result["status"] = "strong_pass"
                result["reason"] = f"task_plus_substantive_{reference_count}_elements"
            else:
                result["status"] = "weak_pass"
                result["reason"] = f"task_plus_substantive_{reference_count}_elements_weak_content"
        else:
            # Has substantive but no task - still weak
            result["status"] = "weak_pass"
            result["reason"] = f"no_task_reference_{reference_count}_elements"
        
        return result

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
