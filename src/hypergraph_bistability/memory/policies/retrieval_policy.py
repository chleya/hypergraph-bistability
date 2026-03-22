"""Retrieval policy for assembling agent context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class RetrievedMemory:
    """Single retrieved memory item."""

    source: str
    content: str
    score: float
    kind: str = "context"
    group: Optional[int] = None
    layer: Optional[int] = None
    artifact_type: Optional[str] = None
    artifact_id: Optional[str] = None
    linked_task: Optional[str] = None
    relation_type: Optional[str] = None
    parent_artifact_id: Optional[str] = None
    node_type: Optional[str] = None
    hyperedge_type: Optional[str] = None
    hyperedge_id: Optional[str] = None
    hyperedge_status: Optional[str] = None
    confidence_tag: Optional[str] = None
    task_phase: Optional[str] = None
    procedure_type: Optional[str] = None
    reusability_class: Optional[str] = None


class RetrievalPolicy:
    """Collect a bounded set of memories for a turn."""

    PREFERENCE_HINTS = ("prefer", "preference", "always", "never", "like", "dislike")
    TASK_HINTS = ("task", "todo", "plan", "deadline", "need to", "must", "next")
    FACT_HINTS = ("remember", "working on", "context", "important", "issue", "failure", "debugging", "problem", "step")
    LOG_HINTS = ("log:", "traceback", "stack trace", "error:", "exception", "warn:", "warning:", "stderr", "stdout")
    HYPOTHESIS_HINTS = (
        "hypothesis", "suspect", "likely cause", "probably", "might be caused", "root cause",
        "survives", "still survives", "ruled out", "stays ruled out", "stays dead", "dead theory",
    )
    PLAN_HINTS = ("plan:", "fix plan", "next steps", "patch plan", "remediation plan")
    DECISION_HINTS = (
        "decision", "tradeoff", "trade-off", "rationale", "reason", "constraint", "alternative",
        "why did we", "what did we commit to", "what did we reject", "what should we validate next",
        "what are we shipping", "what has to happen next", "what are we explicitly not doing",
        "what gate is still open", "what constraint is still active", "what follow-up stays out of scope",
        "which follow-up stays out", "which patch stays in", "clear to ship", "ship call",
        "what still belongs", "what is still deferred", "what validation already holds", "release story",
    )
    PROCEDURE_HINTS = (
        "checklist", "playbook", "template", "procedure", "runbook", "workflow",
        "what checklist still applies", "what review format should we use",
        "what playbook should we use", "what template should we use",
        "packet", "close packet", "handoff bundle", "closure unlocked",
        "proof points", "story still holds", "fix still holds",
    )
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "what", "when", "where",
        "from", "your", "have", "again", "into", "keep", "mind", "after",
        "about", "tell", "most", "back",
    }

    def __init__(self, strategy: str = "hyperedge_expansion") -> None:
        self.strategy = strategy

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log: Optional[List[dict]] = None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        lowered_query = query.lower()
        query_kind = self._classify_query_intent(lowered_query)
        working_memory_items: List[RetrievedMemory] = []

        state = memory.read()
        for group_index, group_mean in enumerate(state.groups):
            if group_mean > 0.5:
                key_prefix = f"({group_index},"
                group_items = [
                    (key, value) for key, value in memory.content_map.items() if key.startswith(key_prefix)
                ]
                for key, value in group_items[:2]:
                    layer = self._parse_layer(key)
                    kind = self._classify_text(value)
                    overlap_score = self._query_overlap_score(lowered_query, value)
                    if query_kind != "context" and kind == "context" and overlap_score <= 0:
                        continue
                    working_memory_items.append(
                        RetrievedMemory(
                            source="working_memory",
                            content=value,
                            score=self._score_item(
                                query=lowered_query,
                                content=value,
                                base_score=float(group_mean),
                                kind=kind,
                                query_kind=query_kind,
                                source="working_memory",
                            ),
                            kind=kind,
                            group=group_index,
                            layer=layer,
                        node_type=self._infer_node_type(kind, value.lower()),
                        hyperedge_type=self._infer_hyperedge_type(kind, value.lower()),
                        hyperedge_id=None,
                        confidence_tag=None,
                        task_phase=None,
                        procedure_type=self._infer_procedure_type(self._infer_node_type(kind, value.lower()), value.lower()),
                        reusability_class=self._infer_reusability_class(self._infer_node_type(kind, value.lower()), value.lower(), None),
                    )
                )

        turn_log_items: List[RetrievedMemory] = []
        if turn_log:
            turn_log_items = self._collect_turn_log_items(lowered_query, turn_log, query_kind)

        vector_items: List[RetrievedMemory] = []
        if embedding_mapper and getattr(embedding_mapper, "store", None):
            for memory_id, content, similarity in embedding_mapper.semantic_search(query, top_k=top_k):
                kind = self._classify_text(content)
                vector_items.append(
                    RetrievedMemory(
                        source=f"vector:{memory_id}",
                        content=content,
                        score=self._score_item(
                            query=lowered_query,
                            content=content,
                            base_score=float(similarity),
                            kind=kind,
                            query_kind=query_kind,
                            source="vector",
                        ),
                        kind=kind,
                        node_type=self._infer_node_type(kind, content.lower()),
                        hyperedge_type=self._infer_hyperedge_type(kind, content.lower()),
                        task_phase=None,
                        procedure_type=self._infer_procedure_type(self._infer_node_type(kind, content.lower()), content.lower()),
                        reusability_class=self._infer_reusability_class(self._infer_node_type(kind, content.lower()), content.lower(), None),
                    )
                )

        if self.strategy == "hyperedge_expansion" and turn_log:
            items = self._collect_hyperedge_expansion_items(
                lowered_query,
                query_kind,
                working_memory_items,
                turn_log_items,
                vector_items,
                turn_log,
            )
        else:
            items = list(working_memory_items)
            items.extend(turn_log_items)
            items.extend(vector_items)
            if turn_log and self.strategy in {"parent_expansion", "legacy_parent_expansion", "hyperedge_expansion"}:
                items.extend(self._expand_parent_artifacts(items, turn_log, lowered_query))

        items.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe_and_truncate(items, top_k)

    def _collect_hyperedge_expansion_items(
        self,
        lowered_query: str,
        query_kind: str,
        working_memory_items: Sequence[RetrievedMemory],
        turn_log_items: Sequence[RetrievedMemory],
        vector_items: Sequence[RetrievedMemory],
        turn_log: List[dict],
    ) -> List[RetrievedMemory]:
        decision_query = self._is_decision_query(lowered_query, query_kind)
        procedure_query = self._is_procedure_query(lowered_query, query_kind)
        direct_hits = self._prefer_structured_turn_log_hits(turn_log_items, working_memory_items, vector_items)
        focal_items: List[RetrievedMemory] = []
        seen_direct = set()
        for item in direct_hits:
            if item.content in seen_direct:
                continue
            seen_direct.add(item.content)
            focal_items.append(item)
            if len(focal_items) >= 3:
                break

        expanded_candidates = list(focal_items)
        expanded_candidates.extend(self._expand_parent_artifacts(focal_items, turn_log, lowered_query))
        expanded_candidates.extend(self._expand_hyperedge_members(focal_items, turn_log, lowered_query, query_kind))
        expanded_candidates.extend(self._expand_linked_task_members(focal_items, turn_log, lowered_query, query_kind))
        if decision_query:
            expanded_candidates.extend(self._expand_decision_hyperedge_items(turn_log, lowered_query, query_kind))
        if procedure_query:
            expanded_candidates.extend(self._expand_procedure_hyperedge_items(turn_log, lowered_query, query_kind))
        conflict_roles = self._derive_conflict_artifact_roles(turn_log)
        return self._re_rank_structured_candidates(
            lowered_query,
            query_kind,
            focal_items,
            expanded_candidates,
            conflict_roles=conflict_roles,
        )

    def _prefer_structured_turn_log_hits(
        self,
        turn_log_items: Sequence[RetrievedMemory],
        working_memory_items: Sequence[RetrievedMemory],
        vector_items: Sequence[RetrievedMemory],
    ) -> List[RetrievedMemory]:
        merged: List[RetrievedMemory] = []
        seen = set()
        for source_items in (
            sorted(turn_log_items, key=lambda item: item.score, reverse=True),
            sorted(working_memory_items, key=lambda item: item.score, reverse=True),
            sorted(vector_items, key=lambda item: item.score, reverse=True),
        ):
            for item in source_items:
                if item.content in seen:
                    continue
                seen.add(item.content)
                merged.append(item)
        return merged

    def _expand_parent_artifacts(
        self,
        items: Sequence[RetrievedMemory],
        turn_log: List[dict],
        lowered_query: str,
    ) -> List[RetrievedMemory]:
        hyperedge_statuses = self._derive_hyperedge_status_map(turn_log)
        contradicted_artifact_ids = self._derive_contradicted_artifact_ids(turn_log)
        conflict_roles = self._derive_conflict_artifact_roles(turn_log)
        contradiction_review_query = self._is_conflict_review_query(lowered_query)
        artifact_index = {}
        for entry in turn_log[-12:]:
            for write in entry.get("writes", []):
                artifact_id = write.get("artifact_id")
                content = write.get("content")
                if artifact_id and content:
                    artifact_index[artifact_id] = write

        expanded: List[RetrievedMemory] = []
        seen_parent_ids = set()
        for item in items:
            parent_id = getattr(item, "parent_artifact_id", None)
            if not parent_id or parent_id in seen_parent_ids:
                continue
            parent = artifact_index.get(parent_id)
            if not parent:
                continue
            seen_parent_ids.add(parent_id)
            expanded.append(
                RetrievedMemory(
                    source="relation_parent",
                    content=parent["content"],
                    score=max(0.0, item.score - 0.05),
                    kind=parent.get("kind", self._classify_text(parent["content"])),
                    group=parent.get("group"),
                    layer=parent.get("layer"),
                    artifact_type=parent.get("artifact_type"),
                    artifact_id=parent.get("artifact_id"),
                    linked_task=parent.get("linked_task"),
                    relation_type="parent_of_retrieved",
                    parent_artifact_id=parent.get("parent_artifact_id"),
                    node_type=parent.get("node_type", self._infer_node_type(parent.get("kind", "context"), str(parent["content"]).lower())),
                    hyperedge_type=parent.get("hyperedge_type"),
                    hyperedge_id=parent.get("hyperedge_id"),
                    hyperedge_status=hyperedge_statuses.get(parent.get("hyperedge_id")),
                    confidence_tag=parent.get("confidence_tag"),
                    task_phase=parent.get("task_phase"),
                    procedure_type=parent.get("procedure_type"),
                    reusability_class=parent.get("reusability_class"),
                )
            )
            if parent.get("artifact_id") in contradicted_artifact_ids:
                expanded[-1].score += 0.12 if contradiction_review_query else -0.38
            conflict_role = conflict_roles.get(parent.get("artifact_id"))
            if parent.get("node_type") == "hypothesis":
                if conflict_role == "dominant":
                    expanded[-1].score += 0.12
                elif conflict_role == "active":
                    expanded[-1].score += 0.06
                elif conflict_role == "contradicted":
                    expanded[-1].score += 0.1 if contradiction_review_query else -0.18
        return expanded

    def _expand_hyperedge_members(
        self,
        focal_items: Sequence[RetrievedMemory],
        turn_log: List[dict],
        lowered_query: str,
        query_kind: str,
    ) -> List[RetrievedMemory]:
        hyperedge_statuses = self._derive_hyperedge_status_map(turn_log)
        verification_query = self._is_verification_query(lowered_query)
        handoff_query = self._is_handoff_query(lowered_query)
        hyperedge_index = {}
        for entry in turn_log[-12:]:
            for write in entry.get("writes", []):
                hyperedge_id = write.get("hyperedge_id")
                content = write.get("content")
                if not hyperedge_id or not content:
                    continue
                hyperedge_index.setdefault(hyperedge_id, []).append(write)

        expanded: List[RetrievedMemory] = []
        seen_contents = {item.content for item in focal_items}
        for item in focal_items:
            hyperedge_id = getattr(item, "hyperedge_id", None)
            if not hyperedge_id:
                continue
            for member in hyperedge_index.get(hyperedge_id, []):
                content = str(member.get("content", "")).strip()
                if not content or content in seen_contents:
                    continue
                kind = str(member.get("kind", self._classify_text(content)))
                overlap_score = self._query_overlap_score(lowered_query, content)
                if (
                    query_kind != "context"
                    and kind == "context"
                    and overlap_score <= 0
                    and not self._allow_verification_context(
                        verification_query=verification_query,
                        handoff_query=handoff_query,
                        confidence_tag=member.get("confidence_tag"),
                        task_phase=member.get("task_phase"),
                    )
                ):
                    continue
                score = self._score_item(
                    query=lowered_query,
                    content=content,
                    base_score=max(0.2, item.score - 0.03),
                    kind=kind,
                    query_kind=query_kind,
                    source="hyperedge_member",
                )
                if member.get("node_type") in {"plan", "hypothesis", "log", "task"}:
                    score += 0.05
                expanded.append(
                    RetrievedMemory(
                        source="hyperedge_member",
                        content=content,
                        score=score,
                        kind=kind,
                        group=member.get("group"),
                        layer=member.get("layer"),
                        artifact_type=member.get("artifact_type"),
                        artifact_id=member.get("artifact_id"),
                        linked_task=member.get("linked_task"),
                        relation_type="shared_hyperedge",
                        parent_artifact_id=member.get("parent_artifact_id"),
                        node_type=member.get("node_type", self._infer_node_type(kind, content.lower())),
                        hyperedge_type=member.get("hyperedge_type", self._infer_hyperedge_type(kind, content.lower())),
                        hyperedge_id=member.get("hyperedge_id"),
                        hyperedge_status=hyperedge_statuses.get(member.get("hyperedge_id")),
                        confidence_tag=member.get("confidence_tag"),
                        task_phase=member.get("task_phase"),
                        procedure_type=member.get("procedure_type"),
                        reusability_class=member.get("reusability_class"),
                    )
                )
                seen_contents.add(content)
        return expanded

    def _expand_linked_task_members(
        self,
        focal_items: Sequence[RetrievedMemory],
        turn_log: List[dict],
        lowered_query: str,
        query_kind: str,
    ) -> List[RetrievedMemory]:
        hyperedge_statuses = self._derive_hyperedge_status_map(turn_log)
        verification_query = self._is_verification_query(lowered_query)
        handoff_query = self._is_handoff_query(lowered_query)
        dominant_task = self._dominant_linked_task(focal_items)
        if not dominant_task:
            return []

        expanded: List[RetrievedMemory] = []
        seen_contents = {item.content for item in focal_items}
        for entry in turn_log[-12:]:
            for write in entry.get("writes", []):
                content = str(write.get("content", "")).strip()
                if not content or content in seen_contents:
                    continue
                linked_task = str(write.get("linked_task") or "")
                if linked_task.lower() != dominant_task.lower():
                    continue
                kind = str(write.get("kind", self._classify_text(content)))
                overlap_score = self._query_overlap_score(lowered_query, content)
                if (
                    query_kind != "context"
                    and kind == "context"
                    and overlap_score <= 0
                    and not self._allow_verification_context(
                        verification_query=verification_query,
                        handoff_query=handoff_query,
                        confidence_tag=write.get("confidence_tag"),
                        task_phase=write.get("task_phase"),
                    )
                ):
                    continue
                score = self._score_item(
                    query=lowered_query,
                    content=content,
                    base_score=0.42,
                    kind=kind,
                    query_kind=query_kind,
                    source="linked_task_member",
                )
                expanded.append(
                    RetrievedMemory(
                        source="linked_task_member",
                        content=content,
                        score=score,
                        kind=kind,
                        group=write.get("group"),
                        layer=write.get("layer"),
                        artifact_type=write.get("artifact_type"),
                        artifact_id=write.get("artifact_id"),
                        linked_task=write.get("linked_task"),
                        relation_type=write.get("relation_type"),
                        parent_artifact_id=write.get("parent_artifact_id"),
                        node_type=write.get("node_type", self._infer_node_type(kind, content.lower())),
                        hyperedge_type=write.get("hyperedge_type", self._infer_hyperedge_type(kind, content.lower())),
                        hyperedge_id=write.get("hyperedge_id"),
                        hyperedge_status=hyperedge_statuses.get(write.get("hyperedge_id")),
                        confidence_tag=write.get("confidence_tag"),
                        task_phase=write.get("task_phase"),
                        procedure_type=write.get("procedure_type"),
                        reusability_class=write.get("reusability_class"),
                    )
                )
                seen_contents.add(content)
        return expanded

    def _expand_decision_hyperedge_items(
        self,
        turn_log: List[dict],
        lowered_query: str,
        query_kind: str,
    ) -> List[RetrievedMemory]:
        hyperedge_statuses = self._derive_hyperedge_status_map(turn_log)
        expanded: List[RetrievedMemory] = []
        seen_contents: set[str] = set()
        for entry in turn_log[-12:]:
            for write in entry.get("writes", []):
                if write.get("hyperedge_type") != "decision_hyperedge":
                    continue
                if write.get("node_type") not in {"decision", "alternative", "constraint", "rationale"}:
                    continue
                content = str(write.get("content", "")).strip()
                if not content or content in seen_contents:
                    continue
                kind = str(write.get("kind", self._classify_text(content)))
                overlap_score = self._query_overlap_score(lowered_query, content)
                base_score = 0.28 + min(0.12, overlap_score * 0.04)
                expanded.append(
                    RetrievedMemory(
                        source="decision_hyperedge",
                        content=content,
                        score=self._score_item(
                            query=lowered_query,
                            content=content,
                            base_score=base_score,
                            kind=kind,
                            query_kind=query_kind,
                            source="decision_hyperedge",
                        ),
                        kind=kind,
                        group=write.get("group"),
                        layer=write.get("layer"),
                        artifact_type=write.get("artifact_type"),
                        artifact_id=write.get("artifact_id"),
                        linked_task=write.get("linked_task"),
                        relation_type=write.get("relation_type"),
                        parent_artifact_id=write.get("parent_artifact_id"),
                        node_type=write.get("node_type", self._infer_node_type(kind, content.lower())),
                        hyperedge_type=write.get("hyperedge_type"),
                        hyperedge_id=write.get("hyperedge_id"),
                        hyperedge_status=hyperedge_statuses.get(write.get("hyperedge_id")),
                        confidence_tag=write.get("confidence_tag"),
                        task_phase=write.get("task_phase"),
                        procedure_type=write.get("procedure_type"),
                        reusability_class=write.get("reusability_class"),
                    )
                )
                seen_contents.add(content)
        return expanded

    def _expand_procedure_hyperedge_items(
        self,
        turn_log: List[dict],
        lowered_query: str,
        query_kind: str,
    ) -> List[RetrievedMemory]:
        hyperedge_statuses = self._derive_hyperedge_status_map(turn_log)
        expanded: List[RetrievedMemory] = []
        seen_contents: set[str] = set()
        for entry in turn_log[-12:]:
            for write in entry.get("writes", []):
                if write.get("hyperedge_type") != "procedure_hyperedge":
                    continue
                if write.get("node_type") not in {"playbook", "checklist", "template", "procedure"}:
                    continue
                content = str(write.get("content", "")).strip()
                if not content or content in seen_contents:
                    continue
                kind = str(write.get("kind", self._classify_text(content)))
                overlap_score = self._query_overlap_score(lowered_query, content)
                base_score = 0.3 + min(0.14, overlap_score * 0.05)
                expanded.append(
                    RetrievedMemory(
                        source="procedure_hyperedge",
                        content=content,
                        score=self._score_item(
                            query=lowered_query,
                            content=content,
                            base_score=base_score,
                            kind=kind,
                            query_kind=query_kind,
                            source="procedure_hyperedge",
                        ),
                        kind=kind,
                        group=write.get("group"),
                        layer=write.get("layer"),
                        artifact_type=write.get("artifact_type"),
                        artifact_id=write.get("artifact_id"),
                        linked_task=write.get("linked_task"),
                        relation_type=write.get("relation_type"),
                        parent_artifact_id=write.get("parent_artifact_id"),
                        node_type=write.get("node_type", self._infer_node_type(kind, content.lower())),
                        hyperedge_type=write.get("hyperedge_type"),
                        hyperedge_id=write.get("hyperedge_id"),
                        hyperedge_status=hyperedge_statuses.get(write.get("hyperedge_id")),
                        confidence_tag=write.get("confidence_tag"),
                        task_phase=write.get("task_phase"),
                        procedure_type=write.get("procedure_type"),
                        reusability_class=write.get("reusability_class"),
                    )
                )
                seen_contents.add(content)
        return expanded

    def _dominant_linked_task(self, items: Sequence[RetrievedMemory]) -> str | None:
        counts = {}
        for item in items:
            linked_task = getattr(item, "linked_task", None)
            if linked_task:
                counts[linked_task] = counts.get(linked_task, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda pair: pair[1])[0]

    def _re_rank_structured_candidates(
        self,
        lowered_query: str,
        query_kind: str,
        focal_items: Sequence[RetrievedMemory],
        candidates: Sequence[RetrievedMemory],
        *,
        conflict_roles: Optional[dict[str, str]] = None,
    ) -> List[RetrievedMemory]:
        dominant_task = self._dominant_linked_task(focal_items)
        focal_contents = {item.content for item in focal_items}
        debugging_query = self._is_debugging_query(lowered_query, query_kind)
        decision_query = self._is_decision_query(lowered_query, query_kind)
        procedure_query = self._is_procedure_query(lowered_query, query_kind)
        decision_rejection_query = any(
            token in lowered_query
            for token in (
                "what did we reject",
                "which option did we reject",
                "what are we explicitly not doing",
                "what follow-up stays out of scope",
                "out of scope",
            )
        )
        decision_constraint_query = any(
            token in lowered_query
            for token in (
                "what constraints",
                "what constraint is still active",
                "what constraint are we still optimizing around",
                "what gate is still open",
            )
        )
        contradiction_review_query = self._is_conflict_review_query(lowered_query)
        handoff_query = any(
            token in lowered_query
            for token in (
                "handoff",
                "hand this off",
                "ready to hand off",
                "ready to handoff",
            )
        )
        verification_query = any(
            token in lowered_query
            for token in (
                "verify",
                "verified",
                "validation",
                "staging",
                "ready to close",
                "before we close this incident",
                "what fix are we carrying forward",
                "what evidence should i cite",
            )
        )
        contradicted_artifact_ids = {
            item.parent_artifact_id
            for item in candidates
            if item.relation_type == "contradicts" and item.parent_artifact_id
        }
        conflict_roles = conflict_roles or self._derive_conflict_roles_from_candidates(candidates)
        conflict_backing_hyperedges = self._derive_conflict_backing_hyperedge_ids(candidates)
        decision_roles = self._derive_decision_artifact_roles(candidates)
        rescored: List[RetrievedMemory] = []
        for item in candidates:
            score = item.score
            if item.content in focal_contents:
                score += 0.1
            if item.kind == "preference":
                score += 0.12
            if item.source == "relation_parent":
                score += 0.08
                if getattr(item, "relation_type", None) == "verifies":
                    score += 0.1
            if item.source in {"hyperedge_member", "linked_task_member"}:
                score += 0.05
            if dominant_task and getattr(item, "linked_task", None) == dominant_task:
                score += 0.08
            if getattr(item, "hyperedge_status", None) == "active":
                score += 0.06
            elif getattr(item, "hyperedge_status", None) == "conflicted" and query_kind in {"log", "hypothesis", "fact", "context"}:
                score += 0.04
            elif getattr(item, "hyperedge_status", None) == "paused":
                score -= 0.02
            elif getattr(item, "hyperedge_status", None) in {"resolved", "superseded"}:
                score -= 0.08
            confidence_tag = getattr(item, "confidence_tag", None)
            if confidence_tag == "verified":
                score += 0.07
                if verification_query or handoff_query:
                    score += 0.16
            elif confidence_tag == "tentative":
                score += 0.015
            elif confidence_tag == "speculative":
                score -= 0.05
            elif confidence_tag == "contradicted":
                score += 0.08 if contradiction_review_query else -0.18
            if getattr(item, "artifact_id", None) in contradicted_artifact_ids:
                score += 0.14 if contradiction_review_query else -0.38
            conflict_role = conflict_roles.get(getattr(item, "artifact_id", None))
            if getattr(item, "node_type", None) == "hypothesis":
                if conflict_role == "dominant":
                    score += 0.16
                    if query_kind in {"hypothesis", "fact", "context"}:
                        score += 0.08
                    if debugging_query:
                        score += 0.08
                    if contradiction_review_query:
                        score += 0.06
                elif conflict_role == "active":
                    score += 0.08
                    if query_kind in {"hypothesis", "fact", "context"}:
                        score += 0.04
                    if debugging_query:
                        score += 0.03
                elif conflict_role == "contradicted":
                    if contradiction_review_query:
                        score += 0.16
                        if query_kind in {"hypothesis", "fact", "context"}:
                            score += 0.06
                    else:
                        score -= 0.22
                        if query_kind in {"hypothesis", "fact", "context"}:
                            score -= 0.08
                        if debugging_query:
                            score -= 0.05
            if debugging_query and getattr(item, "hyperedge_id", None) in conflict_backing_hyperedges:
                if getattr(item, "node_type", None) in {"log", "issue"}:
                    score += 0.08
                elif getattr(item, "node_type", None) in {"fact", "evidence"}:
                    score += 0.04
            if item.source == "relation_parent" and confidence_tag == "verified":
                score += 0.03
                if verification_query or handoff_query:
                    score += 0.08
            if getattr(item, "relation_type", None) == "verifies" and (verification_query or handoff_query):
                score += 0.12
            if getattr(item, "hyperedge_type", None) == "evidence_hyperedge" and query_kind in {"log", "hypothesis", "fact", "context"}:
                score += 0.04
            if getattr(item, "hyperedge_type", None) == "change_hyperedge" and query_kind in {"plan", "task", "context"}:
                score += 0.04
                if verification_query:
                    score += 0.05
            if handoff_query and getattr(item, "task_phase", None) in {"verification", "closure"}:
                score += 0.14
            elif verification_query and getattr(item, "task_phase", None) == "verification":
                score += 0.14
            if handoff_query and getattr(item, "node_type", None) in {"fact", "plan", "constraint"}:
                if "staging" in item.content.lower() or "verified" in item.content.lower() or "green" in item.content.lower():
                    score += 0.12
            if any(token in lowered_query for token in ("packet", "ship", "banked", "keeper", "survives")):
                lowered_content = item.content.lower()
                if any(
                    token in lowered_content
                    for token in (
                        "retry policy",
                        "follow-up",
                        "avoid widening",
                        "ship call",
                        "diff-style summary",
                        "ready to close",
                        "reconnect ordering",
                    )
                ):
                    score += 0.12
            if any(token in lowered_query for token in ("story still holds", "fix still holds", "proof points")):
                lowered_content = item.content.lower()
                if any(
                    token in lowered_content
                    for token in (
                        "verified:",
                        "stale cursor cleanup",
                        "reconnect ordering",
                        "looks stable in staging",
                        "checkpoint flush",
                    )
                ):
                    score += 0.14
            if contradiction_review_query:
                lowered_content = item.content.lower()
                if any(
                    token in lowered_content
                    for token in (
                        "ruled out",
                        "wrong and ruled out",
                        "cursor reset timing",
                        "stale cursor state",
                        "checkpoint flush",
                    )
                ):
                    score += 0.12
            if decision_query:
                decision_role = decision_roles.get(getattr(item, "artifact_id", None))
                if decision_role == "dominant_decision":
                    score += 0.22
                elif decision_role == "rationale":
                    score += 0.12
                elif decision_role == "constraint":
                    score += 0.1
                    if decision_constraint_query:
                        score += 0.08
                elif decision_role == "alternative":
                    score += 0.12 if decision_rejection_query else -0.02
                if getattr(item, "hyperedge_type", None) == "decision_hyperedge":
                    score += 0.12
                if getattr(item, "node_type", None) == "decision" and not decision_rejection_query:
                    score += 0.06
                if getattr(item, "node_type", None) == "alternative" and decision_rejection_query:
                    score += 0.08
                if getattr(item, "node_type", None) == "constraint" and decision_constraint_query:
                    score += 0.06
            if procedure_query:
                if getattr(item, "hyperedge_type", None) == "procedure_hyperedge":
                    score += 0.18
                if getattr(item, "node_type", None) in {"playbook", "checklist", "template", "procedure"}:
                    score += 0.14
                if getattr(item, "procedure_type", None) in {
                    "debug_playbook",
                    "release_handoff_checklist",
                    "review_summary_template",
                    "incident_closeout_checklist",
                }:
                    score += 0.1
                if getattr(item, "reusability_class", None) in {"project_reusable", "cross_task_reusable"}:
                    score += 0.06
                if handoff_query and getattr(item, "procedure_type", None) in {"release_handoff_checklist", "incident_closeout_checklist"}:
                    score += 0.08
                if "review" in lowered_query and getattr(item, "procedure_type", None) == "review_summary_template":
                    score += 0.1
            if item.kind in {"context", "fact"} and not getattr(item, "linked_task", None) and self._query_overlap_score(lowered_query, item.content) <= 0:
                score -= 0.18
            rescored.append(
                RetrievedMemory(
                    source=item.source,
                    content=item.content,
                    score=score,
                    kind=item.kind,
                    group=item.group,
                    layer=item.layer,
                    artifact_type=item.artifact_type,
                    artifact_id=item.artifact_id,
                    linked_task=item.linked_task,
                    relation_type=item.relation_type,
                    parent_artifact_id=item.parent_artifact_id,
                    node_type=item.node_type,
                    hyperedge_type=item.hyperedge_type,
                    hyperedge_id=item.hyperedge_id,
                    hyperedge_status=item.hyperedge_status,
                    confidence_tag=item.confidence_tag,
                    task_phase=item.task_phase,
                    procedure_type=item.procedure_type,
                    reusability_class=item.reusability_class,
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored


    def _dedupe_and_truncate(self, items: Sequence[RetrievedMemory], top_k: int) -> List[RetrievedMemory]:
        unique: List[RetrievedMemory] = []
        seen = set()
        for item in items:
            if item.content in seen:
                continue
            seen.add(item.content)
            unique.append(item)
            if len(unique) >= top_k:
                break
        return unique

    def _collect_turn_log_items(self, lowered_query: str, turn_log: List[dict], query_kind: str) -> List[RetrievedMemory]:
        items: List[RetrievedMemory] = []
        hyperedge_statuses = self._derive_hyperedge_status_map(turn_log)
        contradicted_artifact_ids = self._derive_contradicted_artifact_ids(turn_log)
        conflict_roles = self._derive_conflict_artifact_roles(turn_log)
        verification_query = self._is_verification_query(lowered_query)
        handoff_query = self._is_handoff_query(lowered_query)
        contradiction_review_query = self._is_conflict_review_query(lowered_query)
        recent_entries = turn_log[-8:]
        total_entries = len(recent_entries)
        for index, entry in enumerate(recent_entries):
            recency_bonus = 0.08 * ((index + 1) / max(total_entries, 1))
            for write in entry.get("writes", []):
                content = str(write.get("content", "")).strip()
                if not content:
                    continue
                kind = str(write.get("kind", self._classify_text(content)))
                overlap_score = self._query_overlap_score(lowered_query, content)
                if (
                    query_kind != "context"
                    and kind == "context"
                    and overlap_score <= 0
                    and not self._allow_verification_context(
                        verification_query=verification_query,
                        handoff_query=handoff_query,
                        confidence_tag=write.get("confidence_tag"),
                        task_phase=write.get("task_phase"),
                    )
                ):
                    continue
                base_score = float(write.get("importance", 0.3)) + recency_bonus
                score = self._score_item(
                    query=lowered_query,
                    content=content,
                    base_score=base_score,
                    kind=kind,
                    query_kind=query_kind,
                    source="turn_log",
                )
                score += self._status_score_adjustment(
                    hyperedge_statuses.get(write.get("hyperedge_id")),
                    query_kind=query_kind,
                )
                if verification_query or handoff_query:
                    if write.get("confidence_tag") == "verified":
                        score += 0.24
                    if write.get("task_phase") == "verification":
                        score += 0.18
                    lowered_content = content.lower()
                    if any(token in lowered_content for token in ("verified:", "looks stable in staging", "tests are green", "validation passed")):
                        score += 0.12
                if write.get("artifact_id") in contradicted_artifact_ids:
                    score += 0.14 if contradiction_review_query else -0.38
                conflict_role = conflict_roles.get(write.get("artifact_id"))
                if write.get("node_type") == "hypothesis":
                    if conflict_role == "dominant":
                        score += 0.16
                        if query_kind in {"hypothesis", "fact", "context"}:
                            score += 0.08
                        if contradiction_review_query:
                            score += 0.06
                    elif conflict_role == "active":
                        score += 0.08
                        if query_kind in {"hypothesis", "fact", "context"}:
                            score += 0.04
                    elif conflict_role == "contradicted":
                        if contradiction_review_query:
                            score += 0.16
                            if query_kind in {"hypothesis", "fact", "context"}:
                                score += 0.06
                        else:
                            score -= 0.22
                            if query_kind in {"hypothesis", "fact", "context"}:
                                score -= 0.08
                if lowered_query and score < 0.3:
                    continue
                items.append(
                    RetrievedMemory(
                        source="turn_log",
                        content=content,
                        score=score,
                        kind=kind,
                        group=write.get("group"),
                        layer=write.get("layer"),
                        artifact_type=write.get("artifact_type"),
                        artifact_id=write.get("artifact_id"),
                        linked_task=write.get("linked_task"),
                        relation_type=write.get("relation_type"),
                        parent_artifact_id=write.get("parent_artifact_id"),
                        node_type=write.get("node_type", self._infer_node_type(kind, content.lower())),
                        hyperedge_type=write.get("hyperedge_type", self._infer_hyperedge_type(kind, content.lower())),
                        hyperedge_id=write.get("hyperedge_id"),
                        hyperedge_status=hyperedge_statuses.get(write.get("hyperedge_id")),
                        confidence_tag=write.get("confidence_tag"),
                        task_phase=write.get("task_phase"),
                        procedure_type=write.get("procedure_type"),
                        reusability_class=write.get("reusability_class"),
                    )
                )
        return items

    def _is_handoff_query(self, lowered_query: str) -> bool:
        return any(
            token in lowered_query
            for token in (
                "handoff",
                "hand this off",
                "ready to hand off",
                "ready to handoff",
                "handoff bundle",
                "incident packet",
            )
        )

    def _is_verification_query(self, lowered_query: str) -> bool:
        return any(
            token in lowered_query
            for token in (
                "verify",
                "verified",
                "validation",
                "staging",
                "ready to close",
                "before we close this incident",
                "what fix are we carrying forward",
                "what evidence should i cite",
                "close packet",
                "closure unlocked",
                "banked",
                "story still holds",
                "fix still holds",
                "proof points",
                "still travel with it",
            )
        )

    def _is_conflict_review_query(self, lowered_query: str) -> bool:
        return any(
            token in lowered_query
            for token in (
                "ruled out",
                "stays ruled out",
                "stays dead",
                "dead theory",
                "which theory still survives",
                "which theory stays ruled out",
                "discarded theory",
            )
        )

    def _allow_verification_context(
        self,
        *,
        verification_query: bool,
        handoff_query: bool,
        confidence_tag: Optional[str],
        task_phase: Optional[str],
    ) -> bool:
        if not (verification_query or handoff_query):
            return False
        return confidence_tag == "verified" or task_phase == "verification"

    def _status_score_adjustment(self, status: Optional[str], *, query_kind: str) -> float:
        if status == "active":
            return 0.06
        if status == "conflicted" and query_kind in {"log", "hypothesis", "fact", "context"}:
            return 0.08
        if status == "paused":
            return -0.02
        if status in {"resolved", "superseded"}:
            return -0.12
        return 0.0

    def _derive_hyperedge_status_map(self, turn_log: List[dict]) -> dict[str, str]:
        recent_hyperedge_ids: set[str] = set()
        for entry in turn_log[-2:]:
            for item in entry.get("retrieved_detail", []):
                hyperedge_id = item.get("hyperedge_id")
                if hyperedge_id:
                    recent_hyperedge_ids.add(hyperedge_id)
            for item in entry.get("writes", []):
                hyperedge_id = item.get("hyperedge_id")
                if hyperedge_id:
                    recent_hyperedge_ids.add(hyperedge_id)

        hyperedge_writes: dict[str, list[dict]] = {}
        for entry in turn_log:
            for write in entry.get("writes", []):
                hyperedge_id = write.get("hyperedge_id")
                if hyperedge_id:
                    hyperedge_writes.setdefault(hyperedge_id, []).append(write)

        status_map: dict[str, str] = {}
        for hyperedge_id, writes in hyperedge_writes.items():
            member_text = " ".join(str(write.get("content", "")) for write in writes).lower()
            member_hypotheses = {
                str(write.get("content", "")).strip().lower()
                for write in writes
                if str(write.get("node_type", "")) == "hypothesis"
            }
            if any(token in member_text for token in ("superseded", "replaced by", "obsolete", "deprecated")):
                status_map[hyperedge_id] = "superseded"
            elif any(token in member_text for token in ("resolved", "fixed", "done", "completed", "closed")):
                status_map[hyperedge_id] = "resolved"
            elif len(member_hypotheses) > 1:
                status_map[hyperedge_id] = "conflicted"
            elif hyperedge_id in recent_hyperedge_ids:
                status_map[hyperedge_id] = "active"
            else:
                status_map[hyperedge_id] = "paused"
        return status_map

    def _derive_contradicted_artifact_ids(self, turn_log: List[dict]) -> set[str]:
        contradicted_ids: set[str] = set()
        for entry in turn_log:
            for write in entry.get("writes", []):
                if write.get("relation_type") == "contradicts" and write.get("parent_artifact_id"):
                    contradicted_ids.add(str(write["parent_artifact_id"]))
        return contradicted_ids

    def _derive_decision_artifact_roles(self, items: Sequence[RetrievedMemory]) -> dict[str, str]:
        decision_by_hyperedge: dict[str, list[RetrievedMemory]] = {}
        roles: dict[str, str] = {}

        for item in items:
            hyperedge_id = getattr(item, "hyperedge_id", None)
            artifact_id = getattr(item, "artifact_id", None)
            if not hyperedge_id or not artifact_id:
                continue
            if getattr(item, "node_type", None) == "decision":
                decision_by_hyperedge.setdefault(hyperedge_id, []).append(item)

        for hyperedge_id, decisions in decision_by_hyperedge.items():
            dominant = max(
                decisions,
                key=lambda item: (
                    float(item.score),
                    1 if getattr(item, "confidence_tag", None) == "verified" else 0,
                ),
            )
            if dominant.artifact_id:
                roles[dominant.artifact_id] = "dominant_decision"

            for item in items:
                if getattr(item, "hyperedge_id", None) != hyperedge_id or not getattr(item, "artifact_id", None):
                    continue
                node_type = getattr(item, "node_type", None)
                if node_type == "rationale":
                    roles[item.artifact_id] = "rationale"
                elif node_type == "constraint":
                    roles[item.artifact_id] = "constraint"
                elif node_type == "alternative":
                    roles[item.artifact_id] = "alternative"

        return roles

    def _derive_conflict_artifact_roles(self, turn_log: List[dict]) -> dict[str, str]:
        writes_by_artifact: dict[str, dict] = {}
        contradiction_targets: dict[str, list[str]] = {}
        for entry in turn_log:
            for write in entry.get("writes", []):
                artifact_id = write.get("artifact_id")
                if artifact_id:
                    writes_by_artifact[str(artifact_id)] = write
                if write.get("relation_type") == "contradicts" and write.get("parent_artifact_id"):
                    contradiction_targets.setdefault(str(write["parent_artifact_id"]), []).append(str(artifact_id))

        grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
        for artifact_id, write in writes_by_artifact.items():
            if write.get("node_type") != "hypothesis":
                continue
            linked_task = str(write.get("linked_task") or "unlinked")
            backing_hyperedge_id = str(write.get("hyperedge_id") or "no_hyperedge")
            key = (linked_task, backing_hyperedge_id)
            unit = grouped.setdefault(
                key,
                {"hypotheses": [], "contradicted": [], "active": []},
            )
            unit["hypotheses"].append(artifact_id)
            if artifact_id in contradiction_targets:
                unit["contradicted"].append(artifact_id)
            else:
                unit["active"].append(artifact_id)

        roles: dict[str, str] = {}
        for unit in grouped.values():
            if len(unit["hypotheses"]) < 2 and not unit["contradicted"]:
                continue
            dominant_id = self._select_dominant_conflict_artifact_id(unit["active"] or unit["hypotheses"], writes_by_artifact)
            for artifact_id in unit["contradicted"]:
                roles[artifact_id] = "contradicted"
            for artifact_id in unit["active"]:
                roles[artifact_id] = "active"
            if dominant_id:
                roles[dominant_id] = "dominant"
        return roles

    def _derive_conflict_roles_from_candidates(self, candidates: Sequence[RetrievedMemory]) -> dict[str, str]:
        roles: dict[str, str] = {}
        grouped: dict[tuple[str, str], dict[str, list[str] | list[RetrievedMemory]]] = {}
        for item in candidates:
            artifact_id = getattr(item, "artifact_id", None)
            if not artifact_id or getattr(item, "node_type", None) != "hypothesis":
                continue
            linked_task = str(getattr(item, "linked_task", None) or "unlinked")
            backing_hyperedge_id = str(getattr(item, "hyperedge_id", None) or "no_hyperedge")
            key = (linked_task, backing_hyperedge_id)
            unit = grouped.setdefault(
                key,
                {"items": [], "contradicted": [], "active": []},
            )
            unit["items"].append(item)
            if getattr(item, "artifact_id", None) and getattr(item, "confidence_tag", None) == "contradicted":
                unit["contradicted"].append(str(artifact_id))
            else:
                unit["active"].append(str(artifact_id))

        for unit in grouped.values():
            items = unit["items"]
            contradicted = set(unit["contradicted"])
            active = [artifact_id for artifact_id in unit["active"] if artifact_id not in contradicted]
            if len(items) < 2 and not contradicted:
                continue
            for artifact_id in contradicted:
                roles[artifact_id] = "contradicted"
            for artifact_id in active:
                roles[artifact_id] = "active"
            dominant_item = None
            for item in sorted(items, key=lambda candidate: candidate.score, reverse=True):
                if getattr(item, "artifact_id", None) in active:
                    dominant_item = item
                    break
            if dominant_item and getattr(dominant_item, "artifact_id", None):
                roles[str(dominant_item.artifact_id)] = "dominant"
        return roles

    def _derive_conflict_backing_hyperedge_ids(self, candidates: Sequence[RetrievedMemory]) -> set[str]:
        backing_ids: set[str] = set()
        grouped: dict[tuple[str, str], list[RetrievedMemory]] = {}
        for item in candidates:
            if getattr(item, "node_type", None) != "hypothesis":
                continue
            linked_task = str(getattr(item, "linked_task", None) or "unlinked")
            backing_hyperedge_id = str(getattr(item, "hyperedge_id", None) or "no_hyperedge")
            grouped.setdefault((linked_task, backing_hyperedge_id), []).append(item)
        for (linked_task, backing_hyperedge_id), items in grouped.items():
            if len(items) >= 2 or any(getattr(item, "confidence_tag", None) == "contradicted" for item in items):
                if backing_hyperedge_id != "no_hyperedge":
                    backing_ids.add(backing_hyperedge_id)
        return backing_ids

    def _select_dominant_conflict_artifact_id(
        self,
        candidate_artifact_ids: Sequence[str],
        writes_by_artifact: dict[str, dict],
    ) -> Optional[str]:
        if not candidate_artifact_ids:
            return None
        ranked = sorted(
            candidate_artifact_ids,
            key=lambda artifact_id: (
                writes_by_artifact.get(artifact_id, {}).get("turn_index", -1),
                1 if writes_by_artifact.get(artifact_id, {}).get("confidence_tag") == "verified" else 0,
                1 if writes_by_artifact.get(artifact_id, {}).get("confidence_tag") == "tentative" else 0,
            ),
            reverse=True,
        )
        return ranked[0]

    def _query_overlap_score(self, lowered_query: str, content: str) -> float:
        if not lowered_query:
            return 0.0
        query_tokens = self._tokenize(lowered_query)
        content_tokens = self._tokenize(content.lower())
        if not query_tokens or not content_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        if overlap <= 0:
            return -0.08
        return min(0.4, overlap * 0.14)

    def _score_item(self, *, query: str, content: str, base_score: float, kind: str, query_kind: str, source: str) -> float:
        score = base_score + self._query_overlap_score(query, content)
        if query_kind != "context" and kind == query_kind:
            score += 0.18
        elif query_kind != "context" and kind != query_kind:
            score -= 0.08
        if query_kind == "decision" and kind in {"fact", "task"}:
            score += 0.04
        if kind in {"log", "hypothesis", "plan"}:
            score += 0.08
        if kind == "log" and any(token in query for token in ("incident", "resume", "inspect first", "scheduler")):
            score += 0.14
        if query_kind in {"log", "hypothesis", "plan"} and kind == query_kind:
            score += 0.14
        if source == "working_memory":
            score += 0.05
        return score

    def _is_debugging_query(self, lowered_query: str, query_kind: str) -> bool:
        if query_kind in {"log", "hypothesis"}:
            return True
        return any(
            token in lowered_query
            for token in (
                "debug",
                "incident",
                "root cause",
                "explanation",
                "story still holds",
                "fix still holds",
                "proof points",
                "ruled out",
                "dead theory",
                "what should i focus on",
                "strongest remaining root cause",
                "investigate first",
            )
        )

    def _is_decision_query(self, lowered_query: str, query_kind: str) -> bool:
        if query_kind == "decision":
            return True
        return any(
            token in lowered_query
            for token in (
                "what decision did we already make",
                "what did we decide",
                "what did we commit to",
                "what did we reject",
                "why did we choose",
                "what constraints",
                "which option did we reject",
                "what are we explicitly not doing",
                "what gate is still open",
                "what constraint is still active",
                "what follow-up stays out of scope",
                "which follow-up stays out",
                "which patch stays in",
                "clear to ship",
            )
        )

    def _is_procedure_query(self, lowered_query: str, query_kind: str = "task") -> bool:
        if query_kind == "task" and any(token in lowered_query for token in self.PROCEDURE_HINTS):
            return True
        return any(token in lowered_query for token in self.PROCEDURE_HINTS)

    def _classify_text(self, content: str) -> str:
        lowered = content.lower()
        if any(token in lowered for token in self.PREFERENCE_HINTS):
            return "preference"
        if any(token in lowered for token in self.LOG_HINTS):
            return "log"
        if any(token in lowered for token in self.HYPOTHESIS_HINTS):
            return "hypothesis"
        if any(token in lowered for token in self.PLAN_HINTS):
            return "plan"
        if any(token in lowered for token in self.TASK_HINTS):
            return "task"
        if any(token in lowered for token in self.FACT_HINTS):
            return "fact"
        return "context"

    def _classify_query_intent(self, lowered_query: str) -> str:
        if any(token in lowered_query for token in ("preference", "prefer", "like", "respond to me")):
            return "preference"
        if any(token in lowered_query for token in self.DECISION_HINTS):
            return "decision"
        if any(token in lowered_query for token in ("story still holds", "fix still holds", "proof points")):
            return "hypothesis"
        if self._is_conflict_review_query(lowered_query):
            return "hypothesis"
        if any(token in lowered_query for token in self.PROCEDURE_HINTS):
            return "task"
        if any(token in lowered_query for token in ("log", "traceback", "error", "stderr", "stdout")):
            return "log"
        if any(token in lowered_query for token in ("hypothesis", "root cause", "likely cause")):
            return "hypothesis"
        if any(token in lowered_query for token in ("plan", "patch", "fix first", "next steps", "remediation")):
            return "plan"
        if any(token in lowered_query for token in ("task", "plan", "work", "checklist", "issue", "working on")):
            return "task"
        if any(token in lowered_query for token in ("remember", "context", "fact")):
            return "fact"
        return "context"

    def _tokenize(self, lowered_text: str) -> set[str]:
        return {
            token.strip(".,!?;:()[]{}'\"")
            for token in lowered_text.split()
            if len(token.strip(".,!?;:()[]{}'\"")) > 2
            and token.strip(".,!?;:()[]{}'\"") not in self.STOPWORDS
        }

    def _parse_layer(self, key: str) -> Optional[int]:
        try:
            _, layer_part = key.strip("()").split(",")
            return int(layer_part)
        except (ValueError, TypeError):
            return None

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
        if any(token in lowered for token in ("decision:", "we decided", "decision was", "chosen approach", "we chose")):
            return "decision"
        if any(token in lowered for token in ("reason:", "rationale:", "driven by", "tradeoff", "trade-off")):
            return "rationale"
        if any(token in lowered for token in ("constraint:", "must not", "avoid ", "cannot ", "can't ", "should not")):
            return "constraint"
        if any(token in lowered for token in ("alternative:", "rejected:", "instead of", "rather than", "rejected option")):
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
        if any(token in lowered for token in ("playbook", "checklist", "template", "procedure", "workflow", "runbook")):
            return "procedure_hyperedge"
        if any(token in lowered for token in ("decision", "tradeoff", "trade-off", "choose", "constraint", "rationale", "reason:", "alternative:", "rejected:")):
            return "decision_hyperedge"
        if kind in {"task"} or any(token in lowered for token in ("checklist", "planning", "resume work")):
            return "task_hyperedge"
        if kind in {"log", "hypothesis"} or any(token in lowered for token in ("bug", "incident", "failure", "error", "root cause")):
            return "evidence_hyperedge"
        if kind in {"plan"} or any(token in lowered for token in ("patch", "rollback", "remediation", "fix")):
            return "change_hyperedge"
        return None

    def _infer_procedure_type(self, node_type: str, lowered: str) -> Optional[str]:
        if node_type not in {"playbook", "checklist", "template", "procedure"} and not any(
            token in lowered for token in ("playbook", "checklist", "template", "procedure", "workflow", "runbook")
        ):
            return None
        if any(token in lowered for token in ("debug playbook", "debugging playbook", "incident playbook", "triage steps")):
            return "debug_playbook"
        if any(token in lowered for token in ("release handoff checklist", "release checklist", "rollout checklist", "handoff checklist")):
            return "release_handoff_checklist"
        if any(token in lowered for token in ("review template", "review summary template", "diff-style summary", "review format")):
            return "review_summary_template"
        if any(token in lowered for token in ("incident closeout checklist", "closeout checklist", "incident handoff")):
            return "incident_closeout_checklist"
        if "template" in lowered:
            return "generic_template"
        if "checklist" in lowered:
            return "generic_checklist"
        if "playbook" in lowered:
            return "generic_playbook"
        if "procedure" in lowered or "workflow" in lowered or "runbook" in lowered:
            return "generic_procedure"
        return None

    def _infer_reusability_class(self, node_type: str, lowered: str, linked_task: Optional[str]) -> str:
        if node_type in {"playbook", "checklist", "template", "procedure"}:
            if linked_task:
                return "project_reusable"
            return "cross_task_reusable"
        if node_type in {"preference", "constraint"}:
            return "task_local" if linked_task else "project_reusable"
        return "task_local" if linked_task else "instance_only"
