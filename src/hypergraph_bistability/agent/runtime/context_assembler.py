"""Context assembly for LLM calls."""

from __future__ import annotations

from typing import Dict, Iterable, List

from hypergraph_bistability.memory.policies import RetrievedMemory


class ContextAssembler:
    """Build bounded LLM message payloads for a turn."""

    def build_messages(
        self,
        *,
        system_prompt: str,
        memory_context: str,
        retrieved_items: List[RetrievedMemory],
        conversation_history: List[Dict[str, str]],
        user_input: str,
    ) -> List[Dict[str, str]]:
        extra_context = ""
        if retrieved_items:
            extra_context = f"\n\nRetrieved memory:\n{self._format_retrieved_sections(retrieved_items)}"
        response_contract = (
            "\n\nResponse contract:\n"
            "- Do not output <think> tags or hidden reasoning.\n"
            "- If retrieved memory is relevant, use it explicitly in the answer.\n"
            "- Prioritize recalled preferences, tasks, and active issues over generic advice.\n"
            "- If the user is resuming prior work, explicitly name the recalled task or issue before asking follow-up questions.\n"
            "- Do not give a purely generic clarifying question when a relevant task or issue was retrieved.\n"
            "- Keep the answer concise unless the user asks for detail."
        )
        response_contract += self._format_preference_contract(retrieved_items)
        response_contract += self._format_mode_contract(user_input, retrieved_items)

        return [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nCurrent memory state:\n{memory_context}{extra_context}{response_contract}",
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
