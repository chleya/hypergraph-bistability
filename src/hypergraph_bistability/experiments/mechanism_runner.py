"""Mechanism-level experiment runner for ablation studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from hypergraph_bistability.agent import HypergraphAgent
from hypergraph_bistability.evals.runner import run_eval_suite
from hypergraph_bistability.evals.runner import (
    EvalTurnRecord,
    _artifact_graph_snapshot,
    _cleanup_session_files,
    _hypergraph_view_snapshot,
    _match_expected,
    _run_baseline_suite,
    _safe_ratio,
    _scenario_payload,
    _seed_eval_turn,
    _suite_payload,
)
from hypergraph_bistability.evals.scenarios import DEFAULT_EVAL_SCENARIOS, EvalScenario
from hypergraph_bistability.evals.scenarios import EvalTurn
from hypergraph_bistability.memory.policies import RetrievalPolicy, RetrievedMemory


EXPERIMENT_SCENARIO_MAP = {
    "competition": {
        "scenarios": {
            "preference_recall",
            "task_continuity",
            "context_switching",
            "layered_preferences",
        },
        "description": "Competition vs direct retrieval on preference/task continuity scenarios.",
    },
    "associative_expansion": {
        "scenarios": {
            "debugging_resume_with_preference",
            "coding_agent_resume",
            "artifact_chain_resume",
            "artifact_relation_chain",
        },
        "description": "Associative expansion vs single-hit retrieval on chain-heavy scenarios.",
    },
    "mode_specific": {
        "scenarios": {
            "task_continuity",
            "debugging_resume_with_preference",
            "coding_agent_resume",
            "artifact_relation_chain",
        },
        "description": "Mode-specific response and retrieval policy vs generic policy.",
    },
    "hyperedge_state": {
        "scenarios": {
            "conflicted_hypothesis_debugging",
            "resolved_issue_deprioritization",
        },
        "description": "State-aware retrieval vs state-blind retrieval on conflicted and resolved hyperedges.",
    },
    "phase_progress": {
        "scenarios": {
            "phase_resume_implementation_over_analysis",
            "phase_resume_verification_over_implementation",
        },
        "description": "Phase-aware retrieval vs phase-blind retrieval on progress-state continuity scenarios.",
    },
    "confidence_tags": {
        "scenarios": {
            "verified_vs_speculative_debugging",
            "contradicted_hypothesis_filtering",
        },
        "description": "Confidence-aware retrieval vs confidence-blind retrieval on verified/speculative/contradicted evidence.",
    },
    "decision_residue": {
        "scenarios": {
            "decision_resume_after_interruption",
        },
        "description": "Decision-residue-aware retrieval vs decision-residue-blind retrieval on decision continuity scenarios.",
    },
    "procedure_memory": {
        "scenarios": {
            "procedure_release_handoff",
            "procedure_review_template",
            "procedure_incident_closeout",
        },
        "description": "Procedure-aware retrieval vs procedure-blind retrieval on checklist/template/playbook continuity scenarios.",
    },
    "uncertainty_conflict": {
        "scenarios": {
            "conflict_verified_vs_speculative",
            "conflict_contradicted_vs_verified_resume",
        },
        "description": "Conflict-aware retrieval with confidence tags vs conflict-only and confidence-blind variants.",
    },
    "conflict_links": {
        "scenarios": {
            "contradiction_link_filtering",
            "contradiction_link_resume",
        },
        "description": "Contradiction-link-aware retrieval vs contradiction-blind retrieval on invalidated hypotheses.",
    },
    "two_stage_conflict": {
        "scenarios": {
            "contradiction_link_filtering",
            "conflict_unit_dominance",
            "contradiction_link_resume",
            "conflict_pair_preservation_under_preference_noise",
        },
        "description": "Two-stage conflict-aware retrieval vs one-stage conflict-aware and single-edge contradiction retrieval.",
    },
}


def run_mechanism_experiment(
    name: str,
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    """Run a predefined experiment slice over the existing eval suite."""
    if name not in EXPERIMENT_SCENARIO_MAP:
        raise ValueError(f"Unknown experiment '{name}'. Expected one of: {sorted(EXPERIMENT_SCENARIO_MAP)}")

    selected = _select_scenarios(EXPERIMENT_SCENARIO_MAP[name]["scenarios"])
    if name == "competition":
        results = _run_competition_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "associative_expansion":
        results = _run_associative_expansion_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "hyperedge_state":
        selected = _state_experiment_scenarios()
        results = _run_hyperedge_state_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "phase_progress":
        selected = _phase_progress_experiment_scenarios()
        results = _run_phase_progress_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "confidence_tags":
        selected = _confidence_experiment_scenarios()
        results = _run_confidence_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "decision_residue":
        selected = _decision_residue_experiment_scenarios()
        results = _run_decision_residue_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "procedure_memory":
        selected = _procedure_memory_experiment_scenarios()
        results = _run_procedure_memory_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "uncertainty_conflict":
        selected = _uncertainty_conflict_experiment_scenarios()
        results = _run_uncertainty_conflict_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "conflict_links":
        selected = _conflict_link_experiment_scenarios()
        results = _run_conflict_link_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    elif name == "two_stage_conflict":
        selected = _two_stage_conflict_experiment_scenarios()
        results = _run_two_stage_conflict_experiment(
            selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    else:
        results = run_eval_suite(
            scenarios=selected,
            include_llm=include_llm,
            llm_model=llm_model or "MiniMax-M2.7",
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
    payload = {
        "experiment": name,
        "description": EXPERIMENT_SCENARIO_MAP[name]["description"],
        "scenario_names": [scenario.name for scenario in selected],
        "results": results,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _select_scenarios(names: Iterable[str]) -> Sequence[EvalScenario]:
    selected = []
    allowed = set(names)
    for scenario in DEFAULT_EVAL_SCENARIOS:
        if scenario.name in allowed:
            selected.append(scenario)
    return selected


class DirectRetrievalPolicy(RetrievalPolicy):
    """Ablation policy that bypasses working-memory competition gating."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        items: List[RetrievedMemory] = []
        lowered_query = query.lower()
        query_kind = self._classify_query_intent(lowered_query)
        state = memory.read()

        for key, value in memory.content_map.items():
            group = self._parse_group(key)
            layer = self._parse_layer(key)
            group_score = 0.35
            if group is not None and group < len(state.groups):
                group_score = max(group_score, float(state.groups[group]))
            kind = self._classify_text(value)
            items.append(
                RetrievedMemory(
                    source="memory_pool",
                    content=value,
                    score=self._score_item(
                        query=lowered_query,
                        content=value,
                        base_score=group_score,
                        kind=kind,
                        query_kind=query_kind,
                        source="memory_pool",
                    ),
                    kind=kind,
                    group=group,
                    layer=layer,
                )
            )

        if turn_log:
            items.extend(self._collect_turn_log_items(lowered_query, turn_log, query_kind))
            items.extend(self._expand_parent_artifacts(items, turn_log, lowered_query))

        if embedding_mapper and getattr(embedding_mapper, "store", None):
            for memory_id, content, similarity in embedding_mapper.semantic_search(query, top_k=top_k):
                kind = self._classify_text(content)
                items.append(
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
                    )
                )

        items.sort(key=lambda item: item.score, reverse=True)
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

    def _parse_group(self, key: str) -> int | None:
        try:
            group_part, _ = key.strip("()").split(",")
            return int(group_part)
        except (ValueError, TypeError):
            return None


class SingleHitRetrievalPolicy(RetrievalPolicy):
    """Ablation policy that removes associative expansion and keeps only the top direct hit."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        base_items = super().collect(
            query,
            memory=memory,
            turn_log=None,
            embedding_mapper=embedding_mapper,
            top_k=top_k,
        )
        direct_hits = list(base_items)
        if turn_log:
            lowered_query = query.lower()
            query_kind = self._classify_query_intent(lowered_query)
            direct_hits.extend(self._collect_turn_log_items(lowered_query, turn_log, query_kind))
        direct_hits.sort(key=lambda item: item.score, reverse=True)
        unique: List[RetrievedMemory] = []
        seen = set()
        for item in direct_hits:
            if item.content in seen:
                continue
            seen.add(item.content)
            unique.append(item)
            if len(unique) >= 1:
                break
        return unique


class DecisionResidueBlindRetrievalPolicy(RetrievalPolicy):
    """Ablation policy that disables decision-residue-aware boosts for decision queries."""

    def _is_decision_query(self, lowered_query: str, query_kind: str) -> bool:
        return False


class ProcedureAwareRetrievalPolicy(RetrievalPolicy):
    """Experiment-only policy that explicitly promotes reusable procedures on procedure queries."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        items = super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=top_k,
        )
        if not turn_log:
            return items
        lowered_query = query.lower()
        if not self._is_procedure_query(lowered_query):
            return items

        hyperedge_statuses = self._derive_hyperedge_status_map(turn_log)
        procedure_candidates: List[RetrievedMemory] = []
        seen = {item.content for item in items}
        for entry in turn_log[-12:]:
            for write in entry.get("writes", []):
                if write.get("hyperedge_type") != "procedure_hyperedge":
                    continue
                if write.get("node_type") not in {"playbook", "checklist", "template", "procedure"}:
                    continue
                content = str(write.get("content", "")).strip()
                if not content or content in seen:
                    continue
                kind = str(write.get("kind", self._classify_text(content)))
                procedure_candidates.append(
                    RetrievedMemory(
                        source="procedure_hyperedge",
                        content=content,
                        score=self._score_item(
                            query=lowered_query,
                            content=content,
                            base_score=0.44,
                            kind=kind,
                            query_kind="task",
                            source="procedure_hyperedge",
                        ) + 0.12,
                        kind=kind,
                        group=write.get("group"),
                        layer=write.get("layer"),
                        artifact_type=write.get("artifact_type"),
                        artifact_id=write.get("artifact_id"),
                        linked_task=write.get("linked_task"),
                        relation_type=write.get("relation_type"),
                        parent_artifact_id=write.get("parent_artifact_id"),
                        node_type=write.get("node_type"),
                        hyperedge_type=write.get("hyperedge_type"),
                        hyperedge_id=write.get("hyperedge_id"),
                        hyperedge_status=hyperedge_statuses.get(write.get("hyperedge_id")),
                        confidence_tag=write.get("confidence_tag"),
                        task_phase=write.get("task_phase"),
                        procedure_type=write.get("procedure_type"),
                        reusability_class=write.get("reusability_class"),
                    )
                )
                seen.add(content)

        merged = list(items) + procedure_candidates
        merged.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe_and_truncate(merged, top_k)

    def _is_procedure_query(self, lowered_query: str, query_kind: str = "task") -> bool:
        return any(
            token in lowered_query
            for token in (
                "checklist",
                "playbook",
                "template",
                "format should we use",
                "before handoff",
                "before we close",
                "what checklist still applies",
                "what review format",
            )
        )


class ProcedureBlindRetrievalPolicy(RetrievalPolicy):
    """Ablation policy that suppresses procedure-typed nodes on explicit procedure queries."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        items = super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=max(top_k + 2, 6),
        )
        lowered_query = query.lower()
        if not self._is_procedure_query(lowered_query):
            return self._dedupe_and_truncate(items, top_k)
        filtered = [
            item for item in items
            if getattr(item, "node_type", None) not in {"playbook", "checklist", "template", "procedure"}
        ]
        return self._dedupe_and_truncate(filtered or items, top_k)


class TwoStageCompetitiveRetrievalPolicy(RetrievalPolicy):
    """Experiment-only policy: compete, expand, then re-compete."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        focal_items = super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=min(3, top_k),
        )
        if not turn_log:
            return focal_items

        expanded_candidates = list(focal_items)
        expanded_candidates.extend(self._expand_parent_artifacts(focal_items, turn_log, query.lower()))
        rescored = self._re_compete(query.lower(), focal_items, expanded_candidates)
        unique: List[RetrievedMemory] = []
        seen = set()
        must_keep = [
            item for item in rescored
            if item.kind == "preference" or (item.source == "relation_parent" and item.kind in {"plan", "hypothesis", "log"})
        ]
        for item in must_keep:
            if item.content in seen:
                continue
            seen.add(item.content)
            unique.append(item)
            if len(unique) >= min(4, top_k):
                return unique
        for item in rescored:
            if item.content in seen:
                continue
            seen.add(item.content)
            unique.append(item)
            if len(unique) >= min(4, top_k):
                break
        return unique

    def _re_compete(
        self,
        lowered_query: str,
        focal_items: List[RetrievedMemory],
        candidates: List[RetrievedMemory],
    ) -> List[RetrievedMemory]:
        focal_ids = {(item.artifact_id, item.content) for item in focal_items}
        rescored: List[RetrievedMemory] = []
        for item in candidates:
            score = item.score
            if (item.artifact_id, item.content) in focal_ids:
                score += 0.18
            if item.source == "relation_parent":
                score += 0.02
            if item.kind in {"task", "plan", "hypothesis", "log"}:
                score += 0.04
            if item.kind == "preference":
                score += 0.08
            if self._query_overlap_score(lowered_query, item.content) > 0:
                score += 0.05
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
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored


class HyperedgeExpansionRetrievalPolicy(RetrievalPolicy):
    """Experiment wrapper around the canonical hyperedge-expansion policy."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=top_k,
        )


class StateBlindRetrievalPolicy(RetrievalPolicy):
    """Canonical retrieval without hyperedge-state bonuses or penalties."""

    def _derive_hyperedge_status_map(self, turn_log: List[dict]) -> dict[str, str]:
        return {}


class StateAwareFocusedRetrievalPolicy(RetrievalPolicy):
    """State-aware retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )


class StateBlindFocusedRetrievalPolicy(StateBlindRetrievalPolicy):
    """State-blind retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )


class PhaseAwareFocusedRetrievalPolicy(RetrievalPolicy):
    """Phase-aware retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )

    def _re_rank_structured_candidates(
        self,
        lowered_query: str,
        query_kind: str,
        focal_items: Sequence[RetrievedMemory],
        candidates: Sequence[RetrievedMemory],
        *,
        conflict_roles: dict[str, str] | None = None,
    ) -> List[RetrievedMemory]:
        rescored = super()._re_rank_structured_candidates(
            lowered_query,
            query_kind,
            focal_items,
            candidates,
            conflict_roles=conflict_roles,
        )
        boosted: List[RetrievedMemory] = []
        implementation_query = any(token in lowered_query for token in ("continue the plan", "continue the scheduler plan", "next implementation step"))
        verification_query = any(token in lowered_query for token in ("verify next", "what should we verify", "validation next", "before closing this"))
        for item in rescored:
            score = item.score
            if implementation_query:
                if getattr(item, "task_phase", None) == "implementation":
                    score += 0.14
                elif getattr(item, "task_phase", None) == "analysis":
                    score -= 0.08
            if verification_query:
                if getattr(item, "task_phase", None) == "verification":
                    score += 0.16
                elif getattr(item, "task_phase", None) == "implementation":
                    score -= 0.05
            boosted.append(
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
                )
            )
        boosted.sort(key=lambda item: item.score, reverse=True)
        return boosted


class PhaseBlindFocusedRetrievalPolicy(RetrievalPolicy):
    """Phase-blind retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )

    def _re_rank_structured_candidates(
        self,
        lowered_query: str,
        query_kind: str,
        focal_items: Sequence[RetrievedMemory],
        candidates: Sequence[RetrievedMemory],
        *,
        conflict_roles: dict[str, str] | None = None,
    ) -> List[RetrievedMemory]:
        stripped = [
            RetrievedMemory(
                source=item.source,
                content=item.content,
                score=item.score,
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
                task_phase=None,
            )
            for item in candidates
        ]
        return super()._re_rank_structured_candidates(
            lowered_query,
            query_kind,
            focal_items,
            stripped,
            conflict_roles=conflict_roles,
        )


class ConfidenceBlindRetrievalPolicy(RetrievalPolicy):
    """Canonical retrieval without confidence bonuses or penalties."""

    def _collect_turn_log_items(self, lowered_query: str, turn_log: List[dict], query_kind: str) -> List[RetrievedMemory]:
        items = super()._collect_turn_log_items(lowered_query, turn_log, query_kind)
        return [
            RetrievedMemory(
                source=item.source,
                content=item.content,
                score=item.score,
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
                confidence_tag=None,
            )
            for item in items
        ]

    def _re_rank_structured_candidates(
        self,
        lowered_query: str,
        query_kind: str,
        focal_items: Sequence[RetrievedMemory],
        candidates: Sequence[RetrievedMemory],
        *,
        conflict_roles: dict[str, str] | None = None,
    ) -> List[RetrievedMemory]:
        stripped_candidates = [
            RetrievedMemory(
                source=item.source,
                content=item.content,
                score=item.score,
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
                confidence_tag=None,
            )
            for item in candidates
        ]
        return super()._re_rank_structured_candidates(
            lowered_query,
            query_kind,
            focal_items,
            stripped_candidates,
            conflict_roles=conflict_roles,
        )


class ConfidenceAwareFocusedRetrievalPolicy(RetrievalPolicy):
    """Confidence-aware retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )

    def _re_rank_structured_candidates(
        self,
        lowered_query: str,
        query_kind: str,
        focal_items: Sequence[RetrievedMemory],
        candidates: Sequence[RetrievedMemory],
        *,
        conflict_roles: dict[str, str] | None = None,
    ) -> List[RetrievedMemory]:
        rescored = super()._re_rank_structured_candidates(
            lowered_query,
            query_kind,
            focal_items,
            candidates,
            conflict_roles=conflict_roles,
        )
        boosted: List[RetrievedMemory] = []
        for item in rescored:
            score = item.score
            if item.confidence_tag == "verified":
                score += 0.12
            elif item.confidence_tag == "tentative":
                score += 0.02
            elif item.confidence_tag == "speculative":
                score -= 0.12
            elif item.confidence_tag == "contradicted":
                score -= 0.25
            if item.source == "relation_parent" and item.confidence_tag == "verified":
                score += 0.06
            boosted.append(
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
                )
            )
        boosted.sort(key=lambda item: item.score, reverse=True)
        return boosted


class ConfidenceBlindFocusedRetrievalPolicy(ConfidenceBlindRetrievalPolicy):
    """Confidence-blind retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )


class ContradictionBlindRetrievalPolicy(RetrievalPolicy):
    """Canonical retrieval without contradiction-link penalties."""

    def _derive_contradicted_artifact_ids(self, turn_log: List[dict]) -> set[str]:
        return set()


class ContradictionAwareFocusedRetrievalPolicy(RetrievalPolicy):
    """Contradiction-aware retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )

    def _derive_conflict_artifact_roles(self, turn_log: List[dict]) -> dict[str, str]:
        return {}


class ContradictionBlindFocusedRetrievalPolicy(ContradictionBlindRetrievalPolicy):
    """Contradiction-blind retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )


class ConflictHyperedgeAwareFocusedRetrievalPolicy(RetrievalPolicy):
    """Conflict-hyperedge-aware retrieval under a tight retrieval budget."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )


class ConflictHyperedgeBlindFocusedRetrievalPolicy(RetrievalPolicy):
    """Conflict-link-aware but conflict-hyperedge-blind retrieval."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )

    def _derive_conflict_artifact_roles(self, turn_log: List[dict]) -> dict[str, str]:
        return {}


class UncertaintyConflictAwareFocusedRetrievalPolicy(RetrievalPolicy):
    """Conflict-aware retrieval that also treats confidence tags as first-class ranking signals."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )

    def _re_rank_structured_candidates(
        self,
        lowered_query: str,
        query_kind: str,
        focal_items: Sequence[RetrievedMemory],
        candidates: Sequence[RetrievedMemory],
        *,
        conflict_roles: dict[str, str] | None = None,
    ) -> List[RetrievedMemory]:
        rescored = super()._re_rank_structured_candidates(
            lowered_query,
            query_kind,
            focal_items,
            candidates,
            conflict_roles=conflict_roles,
        )
        combined: List[RetrievedMemory] = []
        for item in rescored:
            score = item.score
            role = (conflict_roles or {}).get(item.artifact_id or "")
            if item.node_type == "hypothesis":
                if role == "dominant" and item.confidence_tag == "verified":
                    score += 0.14
                elif role == "dominant" and item.confidence_tag == "tentative":
                    score += 0.04
                elif role == "dominant" and item.confidence_tag in {"speculative", "contradicted"}:
                    score -= 0.10
                elif role == "active" and item.confidence_tag == "verified":
                    score += 0.06
            if item.source == "relation_parent" and item.confidence_tag == "verified":
                score += 0.05
            if item.confidence_tag == "speculative":
                score -= 0.08
            elif item.confidence_tag == "contradicted":
                score -= 0.18
            combined.append(
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
                )
            )
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined


class UncertaintyConflictBlindRetrievalPolicy(RetrievalPolicy):
    """Conflict-aware but confidence-blind counterpart for the uncertainty/conflict experiment."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        return super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=2,
        )

    def _collect_turn_log_items(self, lowered_query: str, turn_log: List[dict], query_kind: str) -> List[RetrievedMemory]:
        items = super()._collect_turn_log_items(lowered_query, turn_log, query_kind)
        return [
            RetrievedMemory(
                source=item.source,
                content=item.content,
                score=item.score,
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
                confidence_tag=None,
            )
            for item in items
        ]


class TwoStageConflictAwareRetrievalPolicy(RetrievalPolicy):
    """Experiment-only policy: focal conflict selection, local expansion, then conflict-aware re-competition."""

    def collect(
        self,
        query: str,
        *,
        memory,
        turn_log=None,
        embedding_mapper=None,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        lowered_query = query.lower()
        query_kind = self._classify_query_intent(lowered_query)
        focal_items = super().collect(
            query,
            memory=memory,
            turn_log=turn_log,
            embedding_mapper=embedding_mapper,
            top_k=min(2, top_k),
        )
        if not turn_log:
            return focal_items

        expanded_candidates = list(focal_items)
        expanded_candidates.extend(self._expand_parent_artifacts(focal_items, turn_log, lowered_query))
        expanded_candidates.extend(self._expand_hyperedge_members(focal_items, turn_log, lowered_query, query_kind))
        expanded_candidates.extend(self._expand_linked_task_members(focal_items, turn_log, lowered_query, query_kind))
        rescored = self._re_compete_conflict_focus(
            lowered_query,
            query_kind,
            focal_items,
            expanded_candidates,
            turn_log,
        )
        return self._dedupe_and_truncate(rescored, min(2, top_k))

    def _re_compete_conflict_focus(
        self,
        lowered_query: str,
        query_kind: str,
        focal_items: Sequence[RetrievedMemory],
        candidates: Sequence[RetrievedMemory],
        turn_log: List[dict],
    ) -> List[RetrievedMemory]:
        focal_contents = {item.content for item in focal_items}
        focal_hyperedge_ids = {item.hyperedge_id for item in focal_items if item.hyperedge_id}
        conflict_roles = self._derive_conflict_artifact_roles(turn_log)
        debugging_query = self._is_debugging_query(lowered_query, query_kind)
        rescored = super()._re_rank_structured_candidates(
            lowered_query,
            query_kind,
            focal_items,
            candidates,
            conflict_roles=conflict_roles,
        )
        tightened: List[RetrievedMemory] = []
        for item in rescored:
            score = item.score
            if item.content in focal_contents:
                score += 0.12
            if item.hyperedge_id and item.hyperedge_id in focal_hyperedge_ids:
                score += 0.08
            if item.source == "relation_parent":
                score += 0.03
            if debugging_query and item.kind == "preference":
                score -= 0.18
            if item.node_type in {"log", "issue"} and item.hyperedge_id in focal_hyperedge_ids:
                score += 0.05
            if item.node_type == "hypothesis":
                role = conflict_roles.get(item.artifact_id or "")
                if role == "dominant":
                    score += 0.12
                elif role == "active":
                    score += 0.05
                elif role == "contradicted":
                    score -= 0.12
            if item.linked_task and all(
                focal.linked_task and focal.linked_task != item.linked_task for focal in focal_items
            ):
                score -= 0.08
            if query_kind != "context" and item.kind == "context" and self._query_overlap_score(lowered_query, item.content) <= 0:
                score -= 0.15
            tightened.append(
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
                )
            )
        tightened.sort(key=lambda candidate: candidate.score, reverse=True)
        return tightened


def _run_competition_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    competition_results = run_eval_suite(
        scenarios=scenarios,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )
    direct_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=DirectRetrievalPolicy(),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )
    payload = {
        "competition_retrieval": competition_results["llm_runtime"] if include_llm else competition_results["runtime"],
        "direct_retrieval": direct_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "competition_vs_direct": _compare_metric_pair(
            payload["competition_retrieval"]["metrics"],
            payload["direct_retrieval"]["metrics"],
            payload["competition_retrieval"]["continuity"],
            payload["direct_retrieval"]["continuity"],
        ),
        "competition_vs_baseline": _compare_metric_pair(
            payload["competition_retrieval"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["competition_retrieval"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _run_associative_expansion_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    parent_expansion_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=RetrievalPolicy(strategy="parent_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="parent_expansion_llm" if include_llm else "parent_expansion",
    )
    single_hit_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=SingleHitRetrievalPolicy(),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="single_hit_llm" if include_llm else "single_hit",
    )
    hyperedge_expansion_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=HyperedgeExpansionRetrievalPolicy(),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="hyperedge_expansion_llm" if include_llm else "hyperedge_expansion",
    )
    payload = {
        "parent_expansion": parent_expansion_results,
        "single_hit": single_hit_results,
        "hyperedge_expansion": hyperedge_expansion_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "parent_vs_single_hit": _compare_metric_pair(
            payload["parent_expansion"]["metrics"],
            payload["single_hit"]["metrics"],
            payload["parent_expansion"]["continuity"],
            payload["single_hit"]["continuity"],
        ),
        "hyperedge_vs_parent": _compare_metric_pair(
            payload["hyperedge_expansion"]["metrics"],
            payload["parent_expansion"]["metrics"],
            payload["hyperedge_expansion"]["continuity"],
            payload["parent_expansion"]["continuity"],
        ),
        "hyperedge_vs_single_hit": _compare_metric_pair(
            payload["hyperedge_expansion"]["metrics"],
            payload["single_hit"]["metrics"],
            payload["hyperedge_expansion"]["continuity"],
            payload["single_hit"]["continuity"],
        ),
        "parent_vs_baseline": _compare_metric_pair(
            payload["parent_expansion"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["parent_expansion"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _state_experiment_scenarios() -> Sequence[EvalScenario]:
    return [
        EvalScenario(
            name="conflicted_hypothesis_debugging",
            description="State-aware retrieval should keep conflicting hypotheses active during debugging.",
            tier="stress",
            turns=[
                EvalTurn("Log: profile-sync job times out after redis reconnect."),
                EvalTurn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker."),
                EvalTurn("Hypothesis: the reconnect path is timing out because cursor reset happens too early."),
                EvalTurn("Hypothesis: the reconnect path is reusing a stale lease token after reconnect."),
                EvalTurn(
                    "Return to the profile-sync incident and compare the root causes.",
                    expected_retrievals=["stale cursor state", "cursor reset happens too early"],
                    expected_response_signals=["stale cursor state", "cursor reset"],
                ),
            ],
        ),
        EvalScenario(
            name="resolved_issue_deprioritization",
            description="State-aware retrieval should deprioritize resolved work in favor of active issues.",
            tier="stress",
            turns=[
                EvalTurn("Log: worker-7 checkpoint resume timeout after retry loop overflow."),
                EvalTurn("Resolved: worker-7 checkpoint resume timeout fixed after retry loop dedupe patch."),
                EvalTurn("Resolved: worker-7 checkpoint resume state mismatch fixed in the old scheduler path."),
                EvalTurn("Log: worker-7 duplicate backoff state still appears after checkpoint resume."),
                EvalTurn(
                    "Which checkpoint resume issue still needs investigation?",
                    expected_retrievals=["duplicate backoff state"],
                    expected_response_signals=["backoff state"],
                ),
            ],
        ),
    ]


def _run_hyperedge_state_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    state_aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=StateAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="state_aware_llm" if include_llm else "state_aware",
    )
    state_blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=StateBlindFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="state_blind_llm" if include_llm else "state_blind",
    )
    payload = {
        "state_aware": state_aware_results,
        "state_blind": state_blind_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "state_aware_vs_state_blind": _compare_metric_pair(
            payload["state_aware"]["metrics"],
            payload["state_blind"]["metrics"],
            payload["state_aware"]["continuity"],
            payload["state_blind"]["continuity"],
        ),
        "state_aware_vs_baseline": _compare_metric_pair(
            payload["state_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["state_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _phase_progress_experiment_scenarios() -> Sequence[EvalScenario]:
    return [
        EvalScenario(
            name="phase_resume_implementation_over_analysis",
            description="Phase-aware retrieval should favor implementation steps over earlier analysis when the user asks to continue the plan.",
            tier="stress",
            turns=[
                EvalTurn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume."),
                EvalTurn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice."),
                EvalTurn("Plan: inspect retry state restore, compare checkpoint IDs, then patch dedupe before requeue."),
                EvalTurn(
                    "Continue the scheduler plan from where we left off.",
                    expected_retrievals=["inspect retry state restore", "compare checkpoint IDs", "patch dedupe"],
                    expected_response_signals=["retry state restore", "checkpoint ids", "dedupe"],
                ),
            ],
        ),
        EvalScenario(
            name="phase_resume_verification_over_implementation",
            description="Phase-aware retrieval should favor verification evidence over older implementation steps when the user asks what to verify next.",
            tier="stress",
            turns=[
                EvalTurn("Decision: keep the retry policy stable and patch dedupe first."),
                EvalTurn("Plan: patch dedupe before touching the retry policy."),
                EvalTurn("Verified: scheduler dedupe patch reproduces cleanly and validation passed in staging."),
                EvalTurn(
                    "Before closing this, what should we verify next?",
                    expected_retrievals=["validation passed in staging", "patch dedupe before touching the retry policy"],
                    expected_response_signals=["validation", "staging", "patch dedupe"],
                ),
            ],
        ),
        EvalScenario(
            name="phase_resume_closure_over_verification",
            description="Phase-aware retrieval should favor closure state over older verification detail when the user asks whether the task is ready to close.",
            tier="stress",
            turns=[
                EvalTurn("Plan: patch dedupe before touching the retry policy."),
                EvalTurn("Verified: scheduler dedupe patch reproduces cleanly and validation passed in staging."),
                EvalTurn("Resolved: the scheduler incident is fixed, validation passed, and the rollout can be closed."),
                EvalTurn(
                    "Is this ready to close, or are we still in verification?",
                    expected_retrievals=["validation passed in staging", "rollout can be closed"],
                    expected_response_signals=["ready to close", "validation"],
                ),
            ],
        ),
    ]


def _run_phase_progress_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=PhaseAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="phase_progress_aware_llm" if include_llm else "phase_progress_aware",
    )
    blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=PhaseBlindFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="phase_progress_blind_llm" if include_llm else "phase_progress_blind",
    )
    payload = {
        "phase_progress_aware": aware_results,
        "phase_progress_blind": blind_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "aware_vs_blind": _compare_metric_pair(
            payload["phase_progress_aware"]["metrics"],
            payload["phase_progress_blind"]["metrics"],
            payload["phase_progress_aware"]["continuity"],
            payload["phase_progress_blind"]["continuity"],
        ),
        "aware_vs_baseline": _compare_metric_pair(
            payload["phase_progress_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["phase_progress_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _decision_residue_experiment_scenarios() -> Sequence[EvalScenario]:
    return [
        EvalScenario(
            name="decision_residue_resume",
            description="Decision residue should preserve the chosen approach, rationale, and constraint under unrelated preference noise.",
            tier="stress",
            turns=[
                EvalTurn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy."),
                EvalTurn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change."),
                EvalTurn("Constraint: avoid widening the change until scheduler dedupe is verified."),
                EvalTurn("Alternative: for the worker scheduler, rather than changing retry policy first, keep the retry settings stable."),
                EvalTurn("Remember that I prefer concise answers."),
                EvalTurn(
                    "Return to the scheduler work. What decision did we already make and why?",
                    expected_retrievals=[
                        "patch dedupe before touching the retry policy",
                        "duplicate backoff state is the current blocker",
                        "avoid widening the change",
                    ],
                    expected_response_signals=[
                        "patch dedupe",
                        "duplicate backoff state",
                        "avoid widening the change",
                    ],
                ),
            ],
        ),
        EvalScenario(
            name="decision_residue_rejected_alternative",
            description="Decision residue should preserve the rejected alternative and the reason it was rejected, not only the chosen plan.",
            tier="stress",
            turns=[
                EvalTurn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy."),
                EvalTurn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change."),
                EvalTurn("Alternative: for the worker scheduler, changing retry policy first would widen the change too early."),
                EvalTurn("Remember that I prefer concise answers."),
                EvalTurn(
                    "Back to the scheduler fix. Which option did we reject and why?",
                    expected_retrievals=[
                        "changing retry policy first would widen the change too early",
                        "avoid widening the change",
                    ],
                    expected_response_signals=[
                        "retry policy",
                        "widening the change",
                    ],
                ),
            ],
        ),
    ]


def _procedure_memory_experiment_scenarios() -> Sequence[EvalScenario]:
    return [
        EvalScenario(
            name="procedure_release_handoff",
            description="Procedure-aware retrieval should preserve a release handoff checklist under unrelated noise.",
            tier="stress",
            turns=[
                EvalTurn("Task: prepare the release handoff for the scheduler hotfix."),
                EvalTurn("Release handoff checklist: confirm staging verification, rollback notes, and migration notes before handoff."),
                EvalTurn("Remember that I prefer concise answers."),
                EvalTurn(
                    "Before handoff, what checklist still applies?",
                    expected_retrievals=[
                        "staging verification",
                        "rollback notes",
                        "migration notes",
                    ],
                    expected_response_signals=[
                        "staging verification",
                        "rollback notes",
                        "migration notes",
                    ],
                ),
            ],
        ),
        EvalScenario(
            name="procedure_review_template",
            description="Procedure-aware retrieval should preserve a reusable review template under task noise.",
            tier="stress",
            turns=[
                EvalTurn("Task: summarize the scheduler hotfix review follow-up."),
                EvalTurn("Review summary template: start with a diff-style summary, then list risks and follow-up."),
                EvalTurn("Decision: keep retry policy changes out of scope for this hotfix."),
                EvalTurn(
                    "What review format should we use for this follow-up?",
                    expected_retrievals=[
                        "diff-style summary",
                        "risks",
                        "follow-up",
                    ],
                    expected_response_signals=[
                        "diff-style",
                        "risks",
                        "follow-up",
                    ],
                ),
            ],
        ),
        EvalScenario(
            name="procedure_incident_closeout",
            description="Procedure-aware retrieval should preserve an incident closeout checklist for closure queries.",
            tier="stress",
            turns=[
                EvalTurn("Task: close out the profile-sync reconnect incident."),
                EvalTurn("Incident closeout checklist: verify reconnect ordering, confirm staging stability, then mark ready to close."),
                EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker."),
                EvalTurn(
                    "Before we close this incident, what checklist still applies?",
                    expected_retrievals=[
                        "verify reconnect ordering",
                        "confirm staging stability",
                        "ready to close",
                    ],
                    expected_response_signals=[
                        "reconnect ordering",
                        "staging",
                        "ready to close",
                    ],
                ),
            ],
        ),
    ]


def _run_procedure_memory_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ProcedureAwareRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="procedure_memory_aware_llm" if include_llm else "procedure_memory_aware",
    )
    blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ProcedureBlindRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="procedure_memory_blind_llm" if include_llm else "procedure_memory_blind",
    )
    payload = {
        "procedure_memory_aware": aware_results,
        "procedure_memory_blind": blind_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "aware_vs_blind": _compare_metric_pair(
            payload["procedure_memory_aware"]["metrics"],
            payload["procedure_memory_blind"]["metrics"],
            payload["procedure_memory_aware"]["continuity"],
            payload["procedure_memory_blind"]["continuity"],
        ),
        "aware_vs_baseline": _compare_metric_pair(
            payload["procedure_memory_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["procedure_memory_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _run_decision_residue_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=RetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="decision_residue_aware_llm" if include_llm else "decision_residue_aware",
    )
    blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=DecisionResidueBlindRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="decision_residue_blind_llm" if include_llm else "decision_residue_blind",
    )
    payload = {
        "decision_residue_aware": aware_results,
        "decision_residue_blind": blind_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "aware_vs_blind": _compare_metric_pair(
            payload["decision_residue_aware"]["metrics"],
            payload["decision_residue_blind"]["metrics"],
            payload["decision_residue_aware"]["continuity"],
            payload["decision_residue_blind"]["continuity"],
        ),
        "aware_vs_baseline": _compare_metric_pair(
            payload["decision_residue_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["decision_residue_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _confidence_experiment_scenarios() -> Sequence[EvalScenario]:
    return [
        EvalScenario(
            name="verified_vs_speculative_debugging",
            description="Confidence-aware retrieval should prefer verified evidence over speculative alternatives with similar wording.",
            tier="stress",
            turns=[
                EvalTurn("Log: profile-sync job times out after redis reconnect."),
                EvalTurn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker."),
                EvalTurn("Hypothesis: maybe dns jitter in the reconnect path is causing the timeout."),
                EvalTurn(
                    "Return to the profile-sync incident. What evidence should I trust first?",
                    expected_retrievals=["times out after redis reconnect", "stale cursor state"],
                    expected_response_signals=["redis reconnect", "stale cursor state"],
                ),
            ],
        ),
        EvalScenario(
            name="contradicted_hypothesis_filtering",
            description="Confidence-aware retrieval should demote contradicted explanations when a better active explanation exists.",
            tier="stress",
            turns=[
                EvalTurn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume."),
                EvalTurn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler."),
                EvalTurn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue."),
                EvalTurn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice."),
                EvalTurn(
                    "Back to the scheduler incident. Which explanation still looks strongest?",
                    expected_retrievals=["duplicate backoff state", "duplicate scheduler state"],
                    expected_response_signals=["duplicate scheduler state", "backoff state"],
                ),
            ],
        ),
    ]


def _run_confidence_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    confidence_aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ConfidenceAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="confidence_aware_llm" if include_llm else "confidence_aware",
    )
    confidence_blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ConfidenceBlindFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="confidence_blind_llm" if include_llm else "confidence_blind",
    )
    payload = {
        "confidence_aware": confidence_aware_results,
        "confidence_blind": confidence_blind_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "confidence_aware_vs_confidence_blind": _compare_metric_pair(
            payload["confidence_aware"]["metrics"],
            payload["confidence_blind"]["metrics"],
            payload["confidence_aware"]["continuity"],
            payload["confidence_blind"]["continuity"],
        ),
        "confidence_aware_vs_baseline": _compare_metric_pair(
            payload["confidence_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["confidence_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _uncertainty_conflict_experiment_scenarios() -> Sequence[EvalScenario]:
    return [
        EvalScenario(
            name="conflict_verified_vs_speculative",
            description="When conflict roles alone are ambiguous, verified support should beat speculative dominant phrasing.",
            tier="stress",
            turns=[
                EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
                EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
                EvalTurn("Hypothesis: maybe dns jitter after reconnect is the real root cause for the timeout."),
                EvalTurn("Verified: redis reconnect consistently leaves stale cursor state before the final checkpoint flush."),
                EvalTurn(
                    "Return to the profile-sync incident and tell me which root cause should I trust first.",
                    expected_retrievals=["stale cursor state remains", "redis reconnect consistently leaves stale cursor state"],
                    expected_response_signals=["stale cursor state", "redis reconnect", "checkpoint flush"],
                ),
            ],
        ),
        EvalScenario(
            name="conflict_contradicted_vs_verified_resume",
            description="A contradicted hypothesis in a conflict unit should stay down when a verified active explanation exists.",
            tier="stress",
            turns=[
                EvalTurn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume."),
                EvalTurn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler."),
                EvalTurn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue."),
                EvalTurn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice."),
                EvalTurn("Verified: duplicate scheduler state reproduces the duplicate backoff state immediately after resume."),
                EvalTurn(
                    "Back to the scheduler incident. Which explanation should I trust first now?",
                    expected_retrievals=["duplicate scheduler state", "duplicate backoff state"],
                    expected_response_signals=["duplicate scheduler state", "backoff state"],
                ),
            ],
        ),
    ]


def _run_uncertainty_conflict_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    uncertainty_conflict_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=UncertaintyConflictAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="uncertainty_conflict_aware_llm" if include_llm else "uncertainty_conflict_aware",
    )
    conflict_only_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ConflictHyperedgeAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="conflict_only_llm" if include_llm else "conflict_only",
    )
    confidence_blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=UncertaintyConflictBlindRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="uncertainty_conflict_blind_llm" if include_llm else "uncertainty_conflict_blind",
    )
    payload = {
        "uncertainty_conflict_aware": uncertainty_conflict_results,
        "conflict_only": conflict_only_results,
        "uncertainty_conflict_blind": confidence_blind_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "aware_vs_conflict_only": _compare_metric_pair(
            payload["uncertainty_conflict_aware"]["metrics"],
            payload["conflict_only"]["metrics"],
            payload["uncertainty_conflict_aware"]["continuity"],
            payload["conflict_only"]["continuity"],
        ),
        "aware_vs_blind": _compare_metric_pair(
            payload["uncertainty_conflict_aware"]["metrics"],
            payload["uncertainty_conflict_blind"]["metrics"],
            payload["uncertainty_conflict_aware"]["continuity"],
            payload["uncertainty_conflict_blind"]["continuity"],
        ),
        "aware_vs_baseline": _compare_metric_pair(
            payload["uncertainty_conflict_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["uncertainty_conflict_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _conflict_link_experiment_scenarios() -> Sequence[EvalScenario]:
    return [
        EvalScenario(
            name="contradiction_link_filtering",
            description="Contradiction links should suppress an invalidated hypothesis in favor of the still-active one.",
            tier="stress",
            turns=[
                EvalTurn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume."),
                EvalTurn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler."),
                EvalTurn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue."),
                EvalTurn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice."),
                EvalTurn(
                    "Back to the scheduler incident. Which explanation still looks strongest?",
                    expected_retrievals=["duplicate backoff state", "duplicate scheduler state"],
                    expected_response_signals=["duplicate scheduler state", "backoff state"],
                ),
            ],
        ),
        EvalScenario(
            name="conflict_unit_dominance",
            description="Conflict hyperedges should prefer the dominant remaining hypothesis when multiple competing explanations share the same task.",
            tier="stress",
            turns=[
                EvalTurn("Log: profile-sync job times out after redis reconnect and the worker stalls before final checkpoint flush."),
                EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
                EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
                EvalTurn("Hypothesis: the reconnect path is reusing a stale lease token after reconnect."),
                EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
                EvalTurn(
                    "Return to the profile-sync incident and name the strongest remaining root cause.",
                    expected_retrievals=["stale cursor state remains", "times out after redis reconnect"],
                    expected_response_signals=["stale cursor state", "redis reconnect", "checkpoint flush"],
                ),
            ],
        ),
        EvalScenario(
            name="contradiction_link_resume",
            description="Contradiction links should keep the invalidated root cause from resurfacing when resuming a task.",
            tier="stress",
            turns=[
                EvalTurn("Log: profile-sync job times out after redis reconnect."),
                EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
                EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
                EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker."),
                EvalTurn(
                    "Return to the profile-sync incident and tell me the strongest remaining root cause.",
                    expected_retrievals=["times out after redis reconnect", "stale cursor state remains"],
                    expected_response_signals=["stale cursor state", "redis reconnect"],
                ),
            ],
        ),
    ]


def _two_stage_conflict_experiment_scenarios() -> Sequence[EvalScenario]:
    base = list(_conflict_link_experiment_scenarios())
    base.append(
        EvalScenario(
            name="conflict_pair_preservation_under_preference_noise",
            description="Two-stage conflict retrieval should preserve the dominant hypothesis plus backing evidence even when unrelated preference memory competes for the small context budget.",
            tier="stress",
            turns=[
                EvalTurn("Remember that I prefer concise answers."),
                EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
                EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
                EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
                EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
                EvalTurn(
                    "Return to the profile-sync incident and name the strongest remaining root cause.",
                    expected_retrievals=["stale cursor state remains", "times out after redis reconnect"],
                    expected_response_signals=["stale cursor state", "redis reconnect", "checkpoint flush"],
                ),
            ],
        )
    )
    return base


def _run_conflict_link_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    conflict_hyperedge_aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ConflictHyperedgeAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="conflict_hyperedge_aware_llm" if include_llm else "conflict_hyperedge_aware",
    )
    contradiction_aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ContradictionAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="conflict_link_aware_llm" if include_llm else "conflict_link_aware",
    )
    conflict_hyperedge_blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ConflictHyperedgeBlindFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="conflict_hyperedge_blind_llm" if include_llm else "conflict_hyperedge_blind",
    )
    contradiction_blind_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ContradictionBlindFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="conflict_link_blind_llm" if include_llm else "conflict_link_blind",
    )
    payload = {
        "conflict_hyperedge_aware": conflict_hyperedge_aware_results,
        "conflict_hyperedge_blind": conflict_hyperedge_blind_results,
        "conflict_link_aware": contradiction_aware_results,
        "conflict_link_blind": contradiction_blind_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "hyperedge_aware_vs_link_aware": _compare_metric_pair(
            payload["conflict_hyperedge_aware"]["metrics"],
            payload["conflict_link_aware"]["metrics"],
            payload["conflict_hyperedge_aware"]["continuity"],
            payload["conflict_link_aware"]["continuity"],
        ),
        "hyperedge_aware_vs_hyperedge_blind": _compare_metric_pair(
            payload["conflict_hyperedge_aware"]["metrics"],
            payload["conflict_hyperedge_blind"]["metrics"],
            payload["conflict_hyperedge_aware"]["continuity"],
            payload["conflict_hyperedge_blind"]["continuity"],
        ),
        "aware_vs_blind": _compare_metric_pair(
            payload["conflict_link_aware"]["metrics"],
            payload["conflict_link_blind"]["metrics"],
            payload["conflict_link_aware"]["continuity"],
            payload["conflict_link_blind"]["continuity"],
        ),
        "aware_vs_baseline": _compare_metric_pair(
            payload["conflict_link_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["conflict_link_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _run_two_stage_conflict_experiment(
    scenarios: Sequence[EvalScenario],
    *,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
) -> Dict[str, object]:
    two_stage_conflict_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=TwoStageConflictAwareRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="two_stage_conflict_llm" if include_llm else "two_stage_conflict",
    )
    conflict_hyperedge_aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ConflictHyperedgeAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="conflict_hyperedge_aware_llm" if include_llm else "conflict_hyperedge_aware",
    )
    contradiction_aware_results = _run_runtime_suite_with_policy(
        scenarios,
        retrieval_policy=ContradictionAwareFocusedRetrievalPolicy(strategy="hyperedge_expansion"),
        include_llm=include_llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_force_powershell_transport=llm_force_powershell_transport,
        suite_name_override="conflict_link_aware_llm" if include_llm else "conflict_link_aware",
    )
    payload = {
        "two_stage_conflict_aware": two_stage_conflict_results,
        "conflict_hyperedge_aware": conflict_hyperedge_aware_results,
        "conflict_link_aware": contradiction_aware_results,
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    payload["comparison"] = {
        "two_stage_vs_hyperedge_aware": _compare_metric_pair(
            payload["two_stage_conflict_aware"]["metrics"],
            payload["conflict_hyperedge_aware"]["metrics"],
            payload["two_stage_conflict_aware"]["continuity"],
            payload["conflict_hyperedge_aware"]["continuity"],
        ),
        "two_stage_vs_link_aware": _compare_metric_pair(
            payload["two_stage_conflict_aware"]["metrics"],
            payload["conflict_link_aware"]["metrics"],
            payload["two_stage_conflict_aware"]["continuity"],
            payload["conflict_link_aware"]["continuity"],
        ),
        "two_stage_vs_baseline": _compare_metric_pair(
            payload["two_stage_conflict_aware"]["metrics"],
            payload["recent_history_baseline"]["metrics"],
            payload["two_stage_conflict_aware"]["continuity"],
            payload["recent_history_baseline"]["continuity"],
        ),
    }
    return payload


def _run_runtime_suite_with_policy(
    scenarios: Sequence[EvalScenario],
    *,
    retrieval_policy: RetrievalPolicy,
    include_llm: bool,
    llm_model: str,
    llm_base_url: str | None,
    llm_force_powershell_transport: bool,
    suite_name_override: str | None = None,
) -> Dict[str, object]:
    scenario_results = []
    totals = {
        "expected": 0,
        "matched": 0,
        "retrieved": 0,
        "retrieved_hits": 0,
        "response_expected": 0,
        "response_matched": 0,
        "provider_errors": 0,
    }
    suite_name = suite_name_override or ("direct_retrieval_llm" if include_llm else "direct_retrieval")

    for scenario in scenarios:
        agent = HypergraphAgent(
            k=4,
            L=2,
            use_embeddings=False,
            name=f"{suite_name}_{scenario.name}",
            llm_api_key="" if not include_llm else None,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
        agent.turn_processor.retrieval_policy = retrieval_policy
        turn_records: List[EvalTurnRecord] = []
        session_path = Path(f"_{suite_name}_{scenario.name}.json").resolve()

        for turn_index, turn in enumerate(scenario.turns):
            if scenario.name == "session_recovery" and turn_index == len(scenario.turns) - 1:
                agent.save(str(session_path))
                restored = HypergraphAgent(
                    k=4,
                    L=2,
                    use_embeddings=False,
                    name=f"{suite_name}_{scenario.name}",
                    llm_api_key="" if not include_llm else None,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    llm_force_powershell_transport=llm_force_powershell_transport,
                )
                restored.turn_processor.retrieval_policy = retrieval_policy
                restored.load(str(session_path))
                agent = restored

            if not turn.expected_retrievals and not turn.expected_response_signals:
                _seed_eval_turn(agent, turn.user_input)
                continue

            result = agent.process_turn(turn.user_input)
            matched_expected, matched_items = _match_expected(turn.expected_retrievals, result.retrieved_items)
            matched_response_signals, _ = _match_expected(
                turn.expected_response_signals,
                [result.assistant_response],
            )
            turn_records.append(
                EvalTurnRecord(
                    user_input=turn.user_input,
                    expected_retrievals=list(turn.expected_retrievals),
                    retrieved_items=list(result.retrieved_items),
                    matched_expected=matched_expected,
                    matched_retrieved_items=matched_items,
                    assistant_response=result.assistant_response,
                    expected_response_signals=list(turn.expected_response_signals),
                    matched_response_signals=matched_response_signals,
                    llm_debug=agent.get_last_llm_debug_snapshot(),
                )
            )
            totals["expected"] += len(turn.expected_retrievals)
            totals["matched"] += len(matched_expected)
            totals["retrieved"] += len(result.retrieved_items)
            totals["retrieved_hits"] += len(matched_items)
            totals["response_expected"] += len(turn.expected_response_signals)
            totals["response_matched"] += len(matched_response_signals)

        scenario_results.append(
            _scenario_payload(
                scenario,
                turn_records,
                artifact_graph_snapshot=_artifact_graph_snapshot(agent.get_artifact_graph()),
                hypergraph_view_snapshot=_hypergraph_view_snapshot(agent.get_hypergraph_view()),
            )
        )
        _cleanup_session_files(session_path)

    return _suite_payload(suite_name, scenario_results, totals)


def _compare_metric_pair(
    left: Dict[str, float],
    right: Dict[str, float],
    left_continuity: Dict[str, float] | None = None,
    right_continuity: Dict[str, float] | None = None,
) -> Dict[str, float]:
    comparison = {
        "recall_delta": left["recall"] - right["recall"],
        "precision_proxy_delta": left["precision_proxy"] - right["precision_proxy"],
        "irrelevant_context_rate_delta": left["irrelevant_context_rate"] - right["irrelevant_context_rate"],
        "response_recall_delta": left["response_recall"] - right["response_recall"],
        "avg_retrieved_delta": _safe_ratio(left["retrieved_count"], left["expected_count"] or 1)
        - _safe_ratio(right["retrieved_count"], right["expected_count"] or 1),
    }
    if left_continuity is not None and right_continuity is not None:
        comparison.update(
            {
                "task_continuation_delta": float(left_continuity.get("task_continuation", 0.0))
                - float(right_continuity.get("task_continuation", 0.0)),
                "blocker_preservation_delta": float(left_continuity.get("blocker_preservation", 0.0))
                - float(right_continuity.get("blocker_preservation", 0.0)),
                "decision_continuity_delta": float(left_continuity.get("decision_continuity", 0.0))
                - float(right_continuity.get("decision_continuity", 0.0)),
                "repeated_work_avoidance_delta": float(left_continuity.get("repeated_work_avoidance_proxy", 0.0))
                - float(right_continuity.get("repeated_work_avoidance_proxy", 0.0)),
            }
        )
    return comparison
