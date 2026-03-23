"""Heuristic write policy for practical agent turns."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryWriteDecision:
    """Decision produced by the write policy for a single text item."""

    should_write: bool
    group: int
    layer: int
    importance: float
    kind: str
    reason: str
    content: str
    activate: bool = True
    artifact_type: Optional[str] = None
    artifact_id: Optional[str] = None
    linked_task: Optional[str] = None
    relation_type: Optional[str] = None
    parent_artifact_id: Optional[str] = None
    node_type: Optional[str] = None
    hyperedge_type: Optional[str] = None
    hyperedge_id: Optional[str] = None
    confidence_tag: Optional[str] = None
    task_phase: Optional[str] = None
    procedure_type: Optional[str] = None
    reusability_class: Optional[str] = None


class WritePolicy:
    """Simple production-oriented write policy.

    The policy aims to reduce blind "write everything" behavior while keeping
    the current system deterministic and easy to inspect.
    """

    PREFERENCE_KEYWORDS = ("prefer", "preference", "like", "dislike", "always", "never")
    TASK_KEYWORDS = ("todo", "task", "deadline", "plan", "next", "need to", "must", "should")
    FACT_KEYWORDS = ("remember", "important", "fact", "context", "working on", "issue", "failure", "debugging", "problem", "step")
    LOG_KEYWORDS = ("log:", "traceback", "stack trace", "error:", "exception", "warn:", "warning:", "stderr", "stdout")
    HYPOTHESIS_KEYWORDS = ("hypothesis", "suspect", "likely cause", "probably", "might be caused", "root cause")
    PLAN_KEYWORDS = ("plan:", "fix plan", "next steps", "patch plan", "remediation plan")
    DECISION_KEYWORDS = ("decision:", "we decided", "decision was", "chosen approach", "we chose")
    RATIONALE_KEYWORDS = ("reason:", "rationale:", "driven by", "tradeoff", "trade-off")
    CONSTRAINT_KEYWORDS = ("constraint:", "must not", "avoid ", "cannot ", "can't ", "should not")
    ALTERNATIVE_KEYWORDS = ("alternative:", "rejected:", "instead of", "rather than", "rejected option")
    PROCEDURE_KEYWORDS = ("playbook", "checklist", "template", "procedure", "workflow", "runbook")
    DEBUG_PLAYBOOK_KEYWORDS = ("debug playbook", "debugging playbook", "incident playbook", "triage steps")
    RELEASE_HANDOFF_KEYWORDS = ("release handoff checklist", "release checklist", "rollout checklist", "handoff checklist")
    REVIEW_TEMPLATE_KEYWORDS = ("review template", "review summary template", "diff-style summary", "review format")
    INCIDENT_CLOSEOUT_KEYWORDS = ("incident closeout checklist", "closeout checklist", "incident handoff", "ready to close", "ready to hand off")
    VERIFIED_KEYWORDS = ("confirmed", "verified", "observed", "reproduced", "measured", "log:", "traceback", "error:", "exception", "stderr", "stdout")
    TENTATIVE_KEYWORDS = ("seems", "appears", "likely", "probably", "suspect", "tentative")
    SPECULATIVE_KEYWORDS = ("maybe", "might", "could be", "possibly", "guess", "speculative")
    CONTRADICTED_KEYWORDS = ("ruled out", "not the cause", "was wrong", "no longer", "contradicted", "disproved")
    HYPEREDGE_NOISE_TOKENS = {
        "log", "hypothesis", "plan", "remember", "also", "inspect", "verify", "patch",
        "after", "before", "then", "next", "first", "worker", "service", "job",
        "path", "state", "issue", "incident", "debugging", "fix", "root", "cause",
    }
    LOW_VALUE_ASSISTANT_PATTERNS = (
        "how can i help",
        "could you clarify",
        "what aspect interests you most",
        "let me think about this",
        "i hear you exploring",
    )
    ASSISTANT_PERSIST_PREFIXES = (
        "summary:",
        "plan:",
        "decision:",
        "commitment:",
        "note:",
        # Additional important output patterns
        "analysis:",
        "finding:",
        "conclusion:",
        "recommendation:",
        "observation:",
        "insight:",
        "action:",
        "result:",
        "implementation:",
        "code:",
        "fix:",
    )
    # Keywords that indicate important content (alternative to prefixes)
    ASSISTANT_IMPORTANT_KEYWORDS = (
        "important",
        "critical",
        "key insight",
        "main finding",
        "core issue",
        "root cause",
        "solution",
        "implementation",
        "refactor",
        "optimize",
        "bug fix",
    )
    LINKED_TASK_PATTERNS = (
        ("scheduler hotfix", ("hotfix", "scheduler dedupe", "retry policy", "rollback notes", "rollout window", "tonight's rollout")),
        ("worker scheduler", ("worker scheduler", "scheduler")),
        ("retry loop", ("retry loop", "scheduler")),
        ("deployment pipeline", ("deployment pipeline", "pipeline")),
        ("cache invalidation", ("cache invalidation", "cache")),
        ("user profile service", ("user profile service", "profile", "profile-sync", "sync worker", "redis reconnect", "stale cursor", "cursor reset")),
        ("production rollout", ("production rollout", "rollout")),
        ("billing service", ("billing service", "rollout")),
        ("onboarding checklist", ("onboarding checklist", "onboarding")),
        ("database migration", ("database migration", "pipeline")),
    )

    def decide(
        self,
        text: str,
        *,
        role: str,
        turn_index: int,
        k: int,
        L: int,
        embedding_mapper=None,
    ) -> Optional[MemoryWriteDecision]:
        """Return a write decision or `None` when the item should be ignored."""
        normalized = " ".join(text.strip().split())
        if not normalized:
            return None

        lowered = normalized.lower()
        importance = self._estimate_importance(lowered)
        if role == "assistant" and self._is_low_value_assistant_output(lowered):
            return None
        
        # For assistant outputs: use importance as primary filter, not just prefix
        # If importance >= 0.45, allow persistence even without prefix match
        # This makes the logic less brittle - importance-based, not prefix-based
        if role == "assistant" and not self._is_persistable_assistant_output(lowered):
            if importance < 0.45:  # Only skip if both prefix doesn't match AND importance is low
                return None
        if role == "user" and importance < 0.2 and len(normalized) < 12:
            return None

        kind = self._classify_kind(lowered, role)
        if embedding_mapper:
            group, layer = embedding_mapper.find_slot(normalized, action="write")
        else:
            group, layer = self._fallback_slot(kind, turn_index, k, L)

        node_type = self._infer_node_type(kind, lowered)
        artifact_type = node_type if node_type in {"log", "hypothesis", "plan", "decision", "rationale", "constraint", "alternative"} else None
        linked_task = self._infer_linked_task(lowered)
        artifact_id = self._build_artifact_id(artifact_type or kind, normalized, linked_task) if artifact_type else None
        hyperedge_type = self._infer_hyperedge_type(kind, lowered)
        hyperedge_id = self._build_hyperedge_id(hyperedge_type, normalized, linked_task) if hyperedge_type else None
        confidence_tag = self._infer_confidence_tag(kind, lowered)
        task_phase = self._infer_task_phase(node_type, lowered)
        procedure_type = self._infer_procedure_type(node_type, lowered)
        reusability_class = self._infer_reusability_class(node_type, lowered, linked_task)

        return MemoryWriteDecision(
            should_write=True,
            group=group,
            layer=layer,
            importance=importance,
            kind=kind,
            reason=f"{role}:{kind}",
            content=normalized,
            activate=True,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            linked_task=linked_task,
            node_type=node_type,
            hyperedge_type=hyperedge_type,
            hyperedge_id=hyperedge_id,
            confidence_tag=confidence_tag,
            task_phase=task_phase,
            procedure_type=procedure_type,
            reusability_class=reusability_class,
        )

    def _estimate_importance(self, lowered: str) -> float:
        score = 0.15
        score += 0.25 if any(token in lowered for token in self.PREFERENCE_KEYWORDS) else 0.0
        score += 0.25 if any(token in lowered for token in self.TASK_KEYWORDS) else 0.0
        score += 0.2 if any(token in lowered for token in self.FACT_KEYWORDS) else 0.0
        score += 0.15 if "?" in lowered else 0.0
        score += min(0.2, len(lowered.split()) / 80.0)
        return min(score, 1.0)

    def _classify_kind(self, lowered: str, role: str) -> str:
        if any(token in lowered for token in self.PREFERENCE_KEYWORDS):
            return "preference"
        if any(token in lowered for token in self.LOG_KEYWORDS):
            return "log"
        if any(token in lowered for token in self.HYPOTHESIS_KEYWORDS):
            return "hypothesis"
        if any(token in lowered for token in self.PLAN_KEYWORDS):
            return "plan"
        if any(token in lowered for token in self.PROCEDURE_KEYWORDS):
            return "procedure"
        if any(token in lowered for token in self.DECISION_KEYWORDS):
            return "fact"
        if any(token in lowered for token in self.TASK_KEYWORDS):
            return "task"
        if any(token in lowered for token in self.FACT_KEYWORDS):
            return "fact"
        return "response" if role == "assistant" else "context"

    def _is_low_value_assistant_output(self, lowered: str) -> bool:
        if lowered.endswith("?") and len(lowered.split()) < 18:
            return True
        return any(pattern in lowered for pattern in self.LOW_VALUE_ASSISTANT_PATTERNS)

    def _is_persistable_assistant_output(self, lowered: str) -> bool:
        return lowered.startswith(self.ASSISTANT_PERSIST_PREFIXES)

    def _fallback_slot(self, kind: str, turn_index: int, k: int, L: int) -> tuple[int, int]:
        if kind == "preference":
            return 0, 0
        if kind == "task":
            return min(1, k - 1), 0
        if kind in {"log", "hypothesis", "plan", "fact"}:
            return min(2, k - 1), min(1, L - 1)
        group = turn_index % k
        layer = 0 if L == 1 else min(1, L - 1)
        return group, layer

    def _infer_linked_task(self, lowered: str) -> Optional[str]:
        for task_name, tokens in self.LINKED_TASK_PATTERNS:
            if any(token in lowered for token in tokens):
                return task_name

        words = re.findall(r"[a-z0-9_]+", lowered)
        if not words:
            return None
        if "fix" in words or "debug" in words or "investigating" in words:
            return " ".join(words[: min(4, len(words))])
        return None

    def _build_artifact_id(self, kind: str, content: str, linked_task: Optional[str]) -> str:
        basis = f"{kind}|{linked_task or 'unlinked'}|{content.lower()}"
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
        return f"{kind}_{digest}"

    def _infer_node_type(self, kind: str, lowered: str) -> str:
        procedure_type = self._infer_procedure_type("", lowered)
        if procedure_type == "debug_playbook":
            return "playbook"
        if procedure_type in {"release_handoff_checklist", "incident_closeout_checklist", "generic_checklist"}:
            return "checklist"
        if procedure_type in {"review_summary_template", "generic_template"}:
            return "template"
        if procedure_type in {"generic_playbook", "generic_procedure"}:
            return "procedure"
        if any(token in lowered for token in self.DECISION_KEYWORDS):
            return "decision"
        if any(token in lowered for token in self.RATIONALE_KEYWORDS):
            return "rationale"
        if any(token in lowered for token in self.CONSTRAINT_KEYWORDS):
            return "constraint"
        if any(token in lowered for token in self.ALTERNATIVE_KEYWORDS):
            return "alternative"
        if kind == "preference":
            return "preference"
        if kind == "task":
            return "task"
        if kind == "log":
            return "log"
        if kind == "hypothesis":
            return "hypothesis"
        if kind == "plan":
            return "plan"
        if kind == "fact" and any(token in lowered for token in ("issue", "bug", "failure", "problem", "incident")):
            return "issue"
        if kind == "fact":
            return "fact"
        return kind

    def _infer_hyperedge_type(self, kind: str, lowered: str) -> Optional[str]:
        if kind == "preference":
            return None
        if any(token in lowered for token in ("decision", "tradeoff", "choose", "constraint", "rationale", "reason:", "alternative:", "rejected:")):
            return "decision_hyperedge"
        if kind == "procedure" or any(token in lowered for token in self.PROCEDURE_KEYWORDS):
            return "procedure_hyperedge"
        if kind in {"task"} or any(token in lowered for token in ("checklist", "planning", "plan the rollout", "resume work")):
            return "task_hyperedge"
        if kind in {"log", "hypothesis"} or any(token in lowered for token in ("bug", "incident", "failure", "error", "root cause")):
            return "evidence_hyperedge"
        if kind in {"plan"} or any(token in lowered for token in ("patch", "rollback", "remediation", "fix")):
            return "change_hyperedge"
        return None

    def _build_hyperedge_id(self, hyperedge_type: str, content: str, linked_task: Optional[str]) -> str:
        anchor = self._infer_hyperedge_anchor(content.lower(), linked_task)
        basis = f"{hyperedge_type}|{anchor}"
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
        return f"{hyperedge_type}_{digest}"

    def _infer_confidence_tag(self, kind: str, lowered: str) -> str:
        if any(token in lowered for token in self.CONTRADICTED_KEYWORDS):
            return "contradicted"
        if kind == "log":
            return "verified"
        if kind == "hypothesis":
            if any(token in lowered for token in self.CONTRADICTED_KEYWORDS):
                return "contradicted"
            if any(token in lowered for token in self.SPECULATIVE_KEYWORDS):
                return "speculative"
            return "tentative"
        if kind == "plan":
            return "tentative"
        if any(token in lowered for token in self.VERIFIED_KEYWORDS):
            return "verified"
        if any(token in lowered for token in self.SPECULATIVE_KEYWORDS):
            return "speculative"
        if any(token in lowered for token in self.TENTATIVE_KEYWORDS):
            return "tentative"
        if kind in {"task", "preference"}:
            return "verified"
        return "tentative"

    def _infer_hyperedge_anchor(self, lowered: str, linked_task: Optional[str]) -> str:
        if linked_task:
            return linked_task.lower()

        normalized = re.sub(r"[^a-z0-9\s_-]+", " ", lowered)
        normalized = normalized.replace("-", " ")
        tokens = [
            token
            for token in normalized.split()
            if len(token) > 2 and token not in self.HYPEREDGE_NOISE_TOKENS
        ]
        if not tokens:
            return "unlinked"
        return "_".join(tokens[:3])

    def _infer_task_phase(self, node_type: str, lowered: str) -> Optional[str]:
        if node_type in {"playbook", "checklist", "template", "procedure"}:
            if any(token in lowered for token in ("handoff", "closeout", "ready to close", "ready to hand off")):
                return "closure"
            if any(token in lowered for token in ("verify", "validation", "checklist")):
                return "verification"
            return "implementation"
        if node_type in {"log", "issue", "evidence", "hypothesis"}:
            return "analysis"
        if node_type in {"decision", "rationale", "constraint", "alternative"}:
            return "decision"
        if node_type in {"plan", "patch"}:
            return "implementation"
        if any(token in lowered for token in ("verify", "verified", "validation", "test result", "confirmed", "reproduced")):
            return "verification"
        if any(token in lowered for token in ("resolved", "fixed", "done", "completed", "closed")):
            return "closure"
        if node_type in {"task", "subtask", "blocker"}:
            return "analysis"
        return None

    def _infer_procedure_type(self, node_type: str, lowered: str) -> Optional[str]:
        if node_type not in {"playbook", "checklist", "template", "procedure"} and not any(
            token in lowered for token in self.PROCEDURE_KEYWORDS
        ):
            return None
        if any(token in lowered for token in self.DEBUG_PLAYBOOK_KEYWORDS):
            return "debug_playbook"
        if any(token in lowered for token in self.RELEASE_HANDOFF_KEYWORDS):
            return "release_handoff_checklist"
        if any(token in lowered for token in self.REVIEW_TEMPLATE_KEYWORDS):
            return "review_summary_template"
        if any(token in lowered for token in self.INCIDENT_CLOSEOUT_KEYWORDS):
            return "incident_closeout_checklist"
        if "template" in lowered:
            return "generic_template"
        if "checklist" in lowered:
            return "generic_checklist"
        if "playbook" in lowered or "runbook" in lowered:
            return "generic_playbook"
        if any(token in lowered for token in ("procedure", "workflow")):
            return "generic_procedure"
        if node_type == "template":
            return "generic_template"
        if node_type == "checklist":
            return "generic_checklist"
        if node_type == "playbook":
            return "generic_playbook"
        if node_type == "procedure":
            return "generic_procedure"
        return None

    def _infer_reusability_class(self, node_type: str, lowered: str, linked_task: Optional[str]) -> str:
        if node_type in {"playbook", "checklist", "template", "procedure"}:
            if any(token in lowered for token in ("review summary", "diff-style", "incident closeout", "debug playbook")):
                return "cross_task_reusable"
            if linked_task:
                return "project_reusable"
            return "cross_task_reusable"
        if node_type in {"preference", "constraint"}:
            return "task_local" if linked_task else "project_reusable"
        return "task_local" if linked_task else "instance_only"
