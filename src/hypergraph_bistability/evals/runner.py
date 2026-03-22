"""Evaluation runners for practical agent progress."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from hypergraph_bistability.agent import HypergraphAgent
from hypergraph_bistability.evals.config import (
    DEFAULT_LLM_EVAL_BASE_URL,
    DEFAULT_LLM_EVAL_MODEL,
    DEFAULT_LLM_EVAL_TEMPERATURE,
)
from hypergraph_bistability.evals.scenarios import DEFAULT_EVAL_SCENARIOS, EvalScenario

CONTINUITY_SCENARIO_NAMES = (
    "debugging_resume_with_preference",
    "blocker_resume_no_reexplaining",
    "task_continuity",
    "decision_resume_after_interruption",
    "artifact_chain_resume",
    "plan_resume_without_restart",
    "contradiction_link_resume",
    "conflict_unit_dominance",
)

PRODUCT_SCENARIO_NAMES = (
    "release_hotfix_handoff",
    "debug_fix_verify_close_loop",
    "coding_review_commitment_chain",
    "incident_root_cause_handoff",
    "release_scope_guardrail_handoff",
    "review_scope_followup_chain",
    "sidecar_incident_close_packet_paraphrase",
    "sidecar_release_packet_followthrough_paraphrase",
    "procedure_release_handoff_chain",
    "procedure_review_handoff_chain",
    "procedure_release_gate_review_chain",
    "procedure_review_validation_handoff_chain",
)

CONFLICT_PRACTICAL_SCENARIO_NAMES = (
    "incident_root_cause_handoff",
    "sidecar_incident_handoff_bundle_paraphrase",
    "incident_conflict_packet_resolution",
)

LONG_TASK_SCENARIO_NAMES = (
    "hotfix_full_lifecycle_replay",
    "procedure_incident_closeout_replay",
    "procedure_incident_handoff_closeout_replay",
    "incident_debug_handoff_replay",
    "review_to_release_followthrough_replay",
)

PRACTICAL_SIDECAR_SCENARIO_NAMES = (
    "sidecar_incident_handoff_bundle_paraphrase",
)

CONFLICT_SIDECAR_SCENARIO_NAMES = (
    "sidecar_incident_handoff_bundle_paraphrase",
)

PRACTICAL_ROBUSTNESS_SCENARIO_NAMES = (
    "robustness_incident_story_bundle",
    "robustness_release_scope_bundle",
)


def _resolve_llm_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    return explicit_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class EvalTurnRecord:
    """Recorded retrieval output for one evaluated turn."""

    user_input: str
    expected_retrievals: List[str]
    retrieved_items: List[str]
    matched_expected: List[str]
    matched_retrieved_items: List[str]
    assistant_response: str = ""
    expected_response_signals: List[str] = None
    matched_response_signals: List[str] = None
    llm_debug: Dict[str, object] | None = None


class RecentHistoryBaselineRunner:
    """Simple lexical baseline over recent conversation history."""

    def __init__(self, window_size: int = 4) -> None:
        self.window_size = window_size
        self.history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self.history = []

    def process_turn(self, user_input: str) -> List[str]:
        retrieved = self._retrieve(user_input)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": f"[baseline] {user_input}"})
        return retrieved

    def _retrieve(self, query: str) -> List[str]:
        query_tokens = _tokenize(query)
        scored: List[tuple[float, str]] = []
        for item in self.history[-self.window_size * 2:]:
            content = item["content"]
            overlap = len(query_tokens & _tokenize(content))
            if overlap <= 0:
                continue
            scored.append((float(overlap), content))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [content for _, content in scored[:4]]


def run_eval_suite(
    *,
    scenarios: Sequence[EvalScenario] | None = None,
    output_path: str | None = None,
    tier: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    """Run the default eval suite and optionally persist JSON results."""
    scenarios = list(scenarios or DEFAULT_EVAL_SCENARIOS)
    if tier:
        scenarios = [scenario for scenario in scenarios if scenario.tier == tier]
    results = {
        "runtime": _run_runtime_suite(scenarios),
        "recent_history_baseline": _run_baseline_suite(scenarios),
    }
    if include_llm:
        llm_results = _run_runtime_suite(
            scenarios,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_temperature=llm_temperature,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
        llm_results["used_real_llm"] = bool(_resolve_llm_api_key(llm_api_key))
        results["llm_runtime"] = llm_results
    results["comparison"] = _build_comparison(results)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def run_continuity_regression(
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    scenarios = [
        scenario for scenario in DEFAULT_EVAL_SCENARIOS if scenario.name in CONTINUITY_SCENARIO_NAMES
    ]
    return run_eval_suite(
        scenarios=scenarios,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )


def run_product_regression(
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    scenarios = [
        scenario for scenario in DEFAULT_EVAL_SCENARIOS if scenario.name in PRODUCT_SCENARIO_NAMES
    ]
    return run_eval_suite(
        scenarios=scenarios,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )


def run_long_task_regression(
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    scenarios = [
        scenario for scenario in DEFAULT_EVAL_SCENARIOS if scenario.name in LONG_TASK_SCENARIO_NAMES
    ]
    return run_eval_suite(
        scenarios=scenarios,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )


def run_conflict_regression(
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    scenarios = [
        scenario for scenario in DEFAULT_EVAL_SCENARIOS if scenario.name in CONFLICT_PRACTICAL_SCENARIO_NAMES
    ]
    return run_eval_suite(
        scenarios=scenarios,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )


def run_practical_sidecar_regression(
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    scenarios = [
        scenario for scenario in DEFAULT_EVAL_SCENARIOS if scenario.name in PRACTICAL_SIDECAR_SCENARIO_NAMES
    ]
    return run_eval_suite(
        scenarios=scenarios,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )


def run_conflict_sidecar_regression(
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    scenarios = [
        scenario for scenario in DEFAULT_EVAL_SCENARIOS if scenario.name in CONFLICT_SIDECAR_SCENARIO_NAMES
    ]
    return run_eval_suite(
        scenarios=scenarios,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )


def run_practical_robustness_regression(
    *,
    output_path: str | None = None,
    include_llm: bool = False,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
) -> Dict[str, object]:
    scenarios = [
        scenario for scenario in DEFAULT_EVAL_SCENARIOS if scenario.name in PRACTICAL_ROBUSTNESS_SCENARIO_NAMES
    ]
    return run_eval_suite(
        scenarios=scenarios,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_force_powershell_transport=llm_force_powershell_transport,
    )


def _run_runtime_suite(
    scenarios: Sequence[EvalScenario],
    *,
    llm_model: str = DEFAULT_LLM_EVAL_MODEL,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = DEFAULT_LLM_EVAL_BASE_URL,
    llm_temperature: float = DEFAULT_LLM_EVAL_TEMPERATURE,
    llm_force_powershell_transport: bool = False,
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
    resolved_llm_api_key = _resolve_llm_api_key(llm_api_key)
    use_llm = bool(resolved_llm_api_key)

    for scenario in scenarios:
        agent = HypergraphAgent(
            k=4,
            L=2,
            use_embeddings=False,
            name=f"eval_{scenario.name}",
            llm_api_key=resolved_llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_temperature=llm_temperature,
            llm_force_powershell_transport=llm_force_powershell_transport,
        )
        turn_records: List[EvalTurnRecord] = []
        session_path = Path(f"_eval_{scenario.name}.json").resolve()

        for turn_index, turn in enumerate(scenario.turns):
            if scenario.name == "session_recovery" and turn_index == len(scenario.turns) - 1:
                agent.save(str(session_path))
                restored = HypergraphAgent(
                    k=4,
                    L=2,
                    use_embeddings=False,
                    name=f"eval_{scenario.name}",
                    llm_api_key=resolved_llm_api_key,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    llm_temperature=llm_temperature,
                    llm_force_powershell_transport=llm_force_powershell_transport,
                )
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
            totals["provider_errors"] += int(_is_llm_error_response(result.assistant_response))

        scenario_results.append(
            _scenario_payload(
        scenario,
        turn_records,
        artifact_graph_snapshot=_artifact_graph_snapshot(agent.get_artifact_graph()),
        hypergraph_view_snapshot=_hypergraph_view_snapshot(agent.get_hypergraph_view()),
    )
        )
        _cleanup_session_files(session_path)

    suite_name = "llm_runtime" if use_llm else "runtime"
    return _suite_payload(suite_name, scenario_results, totals)


def _run_baseline_suite(scenarios: Sequence[EvalScenario]) -> Dict[str, object]:
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

    for scenario in scenarios:
        baseline = RecentHistoryBaselineRunner()
        turn_records: List[EvalTurnRecord] = []

        for turn in scenario.turns:
            retrieved_items = baseline.process_turn(turn.user_input)
            if not turn.expected_retrievals and not turn.expected_response_signals:
                continue

            matched_expected, matched_items = _match_expected(turn.expected_retrievals, retrieved_items)
            turn_records.append(
                EvalTurnRecord(
                    user_input=turn.user_input,
                    expected_retrievals=list(turn.expected_retrievals),
                    retrieved_items=list(retrieved_items),
                    matched_expected=matched_expected,
                    matched_retrieved_items=matched_items,
                    expected_response_signals=list(turn.expected_response_signals),
                    matched_response_signals=[],
                )
            )
            totals["expected"] += len(turn.expected_retrievals)
            totals["matched"] += len(matched_expected)
            totals["retrieved"] += len(retrieved_items)
            totals["retrieved_hits"] += len(matched_items)
            totals["response_expected"] += len(turn.expected_response_signals)

        scenario_results.append(_scenario_payload(scenario, turn_records))

    return _suite_payload("recent_history_baseline", scenario_results, totals)


def _seed_eval_turn(agent: HypergraphAgent, user_input: str) -> None:
    """Seed deterministic memory state for eval setup turns without LLM generation."""
    agent.record_external_turn(
        user_input=user_input,
        assistant_response="",
    )


def _match_expected(expected_retrievals: Sequence[str], retrieved_items: Sequence[str]) -> tuple[List[str], List[str]]:
    matched_expected = []
    matched_items = []
    lowered_items = [item.lower() for item in retrieved_items]
    for expected in expected_retrievals:
        lowered_expected = expected.lower()
        for index, item in enumerate(lowered_items):
            if lowered_expected in item:
                matched_expected.append(expected)
                matched_items.append(retrieved_items[index])
                break
    unique_items = list(dict.fromkeys(matched_items))
    return matched_expected, unique_items


def _scenario_payload(
    scenario: EvalScenario,
    turn_records: Sequence[EvalTurnRecord],
    *,
    artifact_graph_snapshot: Optional[Dict[str, object]] = None,
    hypergraph_view_snapshot: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    expected = sum(len(record.expected_retrievals) for record in turn_records)
    matched = sum(len(record.matched_expected) for record in turn_records)
    matched_items = sum(len(record.matched_retrieved_items) for record in turn_records)
    retrieved_items = sum(len(record.retrieved_items) for record in turn_records)
    irrelevant_retrieved = max(0, retrieved_items - matched_items)
    provider_error_count = sum(int(_is_llm_error_response(record.assistant_response)) for record in turn_records)
    continuity = _continuity_metrics(
        turn_records,
        scenario=scenario,
        hypergraph_view_snapshot=hypergraph_view_snapshot,
    )
    payload = {
        "name": scenario.name,
        "description": scenario.description,
        "turns": [
            {
                "user_input": record.user_input,
                "expected_retrievals": record.expected_retrievals,
                "retrieved_items": record.retrieved_items,
                "matched_expected": record.matched_expected,
                "matched_retrieved_items": record.matched_retrieved_items,
                "assistant_response": record.assistant_response,
                "expected_response_signals": record.expected_response_signals,
                "matched_response_signals": record.matched_response_signals,
                "llm_debug": record.llm_debug,
            }
            for record in turn_records
        ],
        "metrics": {
            "expected_count": expected,
            "matched_count": matched,
            "retrieved_count": retrieved_items,
            "retrieved_hit_count": matched_items,
            "irrelevant_retrieved_count": irrelevant_retrieved,
            "recall": _safe_ratio(matched, expected),
            "precision_proxy": _safe_ratio(matched_items, retrieved_items),
            "irrelevant_context_rate": _safe_ratio(irrelevant_retrieved, retrieved_items),
            "response_signal_count": sum(len(record.expected_response_signals or []) for record in turn_records),
            "response_signal_hits": sum(len(record.matched_response_signals or []) for record in turn_records),
            "provider_error_count": provider_error_count,
            "response_recall": _safe_ratio(
                sum(len(record.matched_response_signals or []) for record in turn_records),
                sum(len(record.expected_response_signals or []) for record in turn_records),
            ),
        },
        "continuity": continuity,
    }
    if artifact_graph_snapshot is not None:
        payload["artifact_graph"] = artifact_graph_snapshot
    if hypergraph_view_snapshot is not None:
        payload["hypergraph_view"] = hypergraph_view_snapshot
    payload["analysis"] = _scenario_analysis(
        turn_records,
        artifact_graph_snapshot=artifact_graph_snapshot,
        hypergraph_view_snapshot=hypergraph_view_snapshot,
    )
    return payload


def _suite_payload(name: str, scenario_results: Sequence[Dict[str, object]], totals: Dict[str, int]) -> Dict[str, object]:
    irrelevant_retrieved = max(0, totals["retrieved"] - totals["retrieved_hits"])
    continuity_scores = [scenario.get("continuity", {}) for scenario in scenario_results]
    return {
        "name": name,
        "scenarios": list(scenario_results),
        "metrics": {
            "expected_count": totals["expected"],
            "matched_count": totals["matched"],
            "retrieved_count": totals["retrieved"],
            "retrieved_hit_count": totals["retrieved_hits"],
            "irrelevant_retrieved_count": irrelevant_retrieved,
            "recall": _safe_ratio(totals["matched"], totals["expected"]),
            "precision_proxy": _safe_ratio(totals["retrieved_hits"], totals["retrieved"]),
            "irrelevant_context_rate": _safe_ratio(irrelevant_retrieved, totals["retrieved"]),
            "response_signal_count": totals["response_expected"],
            "response_signal_hits": totals["response_matched"],
            "provider_error_count": totals["provider_errors"],
            "response_recall": _safe_ratio(totals["response_matched"], totals["response_expected"]),
        },
        "continuity": {
            "scenario_count": len(continuity_scores),
            "blocker_eligible_scenarios": sum(1 for score in continuity_scores if int(score.get("blocker_eligible_count", 0)) > 0),
            "decision_eligible_scenarios": sum(1 for score in continuity_scores if int(score.get("decision_eligible_count", 0)) > 0),
            "procedure_eligible_scenarios": sum(1 for score in continuity_scores if int(score.get("procedure_eligible_count", 0)) > 0),
            "conflict_eligible_scenarios": sum(1 for score in continuity_scores if int(score.get("conflict_eligible_count", 0)) > 0),
            "task_continuation": _average_continuity_metric(continuity_scores, "task_continuation"),
            "blocker_preservation": _average_continuity_metric(continuity_scores, "blocker_preservation"),
            "decision_continuity": _average_continuity_metric(continuity_scores, "decision_continuity"),
            "procedure_continuity": _average_continuity_metric(continuity_scores, "procedure_continuity"),
            "conflict_continuity": _average_continuity_metric(continuity_scores, "conflict_continuity"),
            "repeated_work_avoidance_proxy": _average_continuity_metric(
                continuity_scores,
                "repeated_work_avoidance_proxy",
            ),
        },
    }


def _continuity_metrics(
    turn_records: Sequence[EvalTurnRecord],
    *,
    scenario: EvalScenario | None = None,
    hypergraph_view_snapshot: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    evaluated_turns = [record for record in turn_records if record.expected_retrievals or record.expected_response_signals]
    conflict_expected = int(_has_scenario_tag(scenario, "conflict_heavy"))
    if not evaluated_turns:
        return {
            "task_continuation": 0.0,
            "blocker_preservation": 0.0,
            "decision_continuity": 0.0,
            "procedure_continuity": 0.0,
            "conflict_continuity": 0.0,
            "repeated_work_avoidance_proxy": 0.0,
            "blocker_eligible_count": 0,
            "decision_eligible_count": 0,
            "procedure_eligible_count": 0,
            "conflict_eligible_count": conflict_expected,
        }

    task_hits = 0
    blocker_expected = 0
    blocker_hits = 0
    decision_expected = 0
    decision_hits = 0
    procedure_expected = 0
    procedure_hits = 0
    repeated_work_avoided = 0

    for record in evaluated_turns:
        retrieval_hit = bool(record.matched_expected)
        response_hit = bool(record.matched_response_signals)
        if retrieval_hit and (response_hit or not record.expected_response_signals):
            task_hits += 1

        expected_texts = [*record.expected_retrievals, *(record.expected_response_signals or [])]
        matched_texts = [*record.matched_expected, *(record.matched_response_signals or [])]

        if any(_is_blocker_signal(text) for text in expected_texts):
            blocker_expected += 1
            if any(_is_blocker_signal(text) for text in matched_texts):
                blocker_hits += 1

        if any(_is_decision_signal(text) for text in expected_texts):
            decision_expected += 1
            if any(_is_decision_signal(text) for text in matched_texts):
                decision_hits += 1

        if any(_is_procedure_signal(text) for text in expected_texts):
            procedure_expected += 1
            if any(_is_procedure_signal(text) for text in matched_texts):
                procedure_hits += 1

        if retrieval_hit and not _looks_like_restart_response(record.assistant_response):
            repeated_work_avoided += 1

    task_continuation = _safe_ratio(task_hits, len(evaluated_turns))
    conflict_hits = int(
        conflict_expected > 0
        and task_continuation >= 1.0
        and _conflict_structure_preserved(hypergraph_view_snapshot)
    )

    return {
        "task_continuation": task_continuation,
        "blocker_preservation": _safe_ratio(blocker_hits, blocker_expected),
        "decision_continuity": _safe_ratio(decision_hits, decision_expected),
        "procedure_continuity": _safe_ratio(procedure_hits, procedure_expected),
        "conflict_continuity": _safe_ratio(conflict_hits, conflict_expected),
        "repeated_work_avoidance_proxy": _safe_ratio(repeated_work_avoided, len(evaluated_turns)),
        "blocker_eligible_count": blocker_expected,
        "decision_eligible_count": decision_expected,
        "procedure_eligible_count": procedure_expected,
        "conflict_eligible_count": conflict_expected,
    }


def _average_continuity_metric(continuity_scores: Sequence[Dict[str, float]], key: str) -> float:
    if not continuity_scores:
        return 0.0
    eligibility_key = None
    if key == "blocker_preservation":
        eligibility_key = "blocker_eligible_count"
    elif key == "decision_continuity":
        eligibility_key = "decision_eligible_count"
    elif key == "procedure_continuity":
        eligibility_key = "procedure_eligible_count"
    elif key == "conflict_continuity":
        eligibility_key = "conflict_eligible_count"

    if eligibility_key is None:
        return sum(float(score.get(key, 0.0)) for score in continuity_scores) / len(continuity_scores)

    eligible_scores = [score for score in continuity_scores if int(score.get(eligibility_key, 0)) > 0]
    if not eligible_scores:
        return 0.0
    return sum(float(score.get(key, 0.0)) for score in eligible_scores) / len(eligible_scores)


def _is_blocker_signal(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "bug",
            "failure",
            "rollback",
            "stale",
            "backoff",
            "timeout",
            "migration",
            "cache invalidation",
            "checkpoint",
            "error",
            "issue",
            "root cause",
        )
    )


def _is_decision_signal(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "plan",
            "next step",
            "next steps",
            "rollback",
            "migration notes",
            "checklist",
            "decision",
            "inspect",
            "patch",
            "summary",
        )
    )


def _is_procedure_signal(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "checklist",
            "template",
            "playbook",
            "procedure",
            "runbook",
            "workflow",
            "diff-style summary",
        )
    )


def _has_scenario_tag(scenario: EvalScenario | None, tag: str) -> bool:
    if scenario is None:
        return False
    return tag in getattr(scenario, "tags", [])


def _conflict_structure_preserved(hypergraph_view_snapshot: Optional[Dict[str, object]]) -> bool:
    if not hypergraph_view_snapshot:
        return False
    conflict_hyperedges = hypergraph_view_snapshot.get("conflict_hyperedges", [])
    if not conflict_hyperedges:
        return False
    return any(
        conflict.get("dominant_node_id") is not None
        or bool(conflict.get("active_hypothesis_node_ids"))
        or str(conflict.get("status", "")) in {"active_conflict", "resolved_conflict", "competing_hypotheses"}
        for conflict in conflict_hyperedges
    )


def _looks_like_restart_response(response: str) -> bool:
    lowered = response.lower()
    if not lowered.strip():
        return True
    return any(
        token in lowered
        for token in (
            "what aspect interests you most",
            "how can i help",
            "tell me more",
            "what would you like to explore",
            "what should we focus on",
        )
    )


def _artifact_graph_snapshot(graph: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [
            {
                "artifact_id": node.get("artifact_id"),
                "artifact_type": node.get("artifact_type"),
                "linked_task": node.get("linked_task"),
                "turn_index": node.get("turn_index"),
            }
            for node in nodes[:12]
        ],
        "edges": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation_type": edge.get("relation_type"),
                "linked_task": edge.get("linked_task"),
            }
            for edge in edges[:12]
        ],
    }


def _hypergraph_view_snapshot(graph: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    hyperedges = graph.get("hyperedges", [])
    conflict_hyperedges = graph.get("conflict_hyperedges", [])
    procedure_residues = graph.get("procedure_residues", [])
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "hyperedge_count": len(hyperedges),
        "conflict_hyperedge_count": len(conflict_hyperedges),
        "procedure_residue_count": len(procedure_residues),
        "nodes": [
            {
                "node_id": node.get("node_id"),
                "node_type": node.get("node_type"),
                "hyperedge_id": node.get("hyperedge_id"),
                "hyperedge_type": node.get("hyperedge_type"),
                "linked_task": node.get("linked_task"),
                "turn_index": node.get("turn_index"),
            }
            for node in nodes[:12]
        ],
        "edges": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation_type": edge.get("relation_type"),
                "linked_task": edge.get("linked_task"),
            }
            for edge in edges[:12]
        ],
        "hyperedges": [
            {
                "hyperedge_id": hyperedge.get("hyperedge_id"),
                "hyperedge_type": hyperedge.get("hyperedge_type"),
                "linked_task": hyperedge.get("linked_task"),
                "member_node_ids": list(hyperedge.get("member_node_ids", []))[:8],
            }
            for hyperedge in hyperedges[:12]
        ],
        "conflict_hyperedges": [
            {
                "conflict_hyperedge_id": conflict.get("conflict_hyperedge_id"),
                "linked_task": conflict.get("linked_task"),
                "backing_hyperedge_id": conflict.get("backing_hyperedge_id"),
                "status": conflict.get("status"),
                "dominant_node_id": conflict.get("dominant_node_id"),
                "hypothesis_node_ids": list(conflict.get("hypothesis_node_ids", []))[:8],
                "contradicted_node_ids": list(conflict.get("contradicted_node_ids", []))[:8],
                "active_hypothesis_node_ids": list(conflict.get("active_hypothesis_node_ids", []))[:8],
            }
            for conflict in conflict_hyperedges[:12]
        ],
        "procedure_residues": [
            {
                "procedure_residue_id": residue.get("procedure_residue_id"),
                "linked_task": residue.get("linked_task"),
                "backing_hyperedge_id": residue.get("backing_hyperedge_id"),
                "status": residue.get("status"),
                "dominant_procedure_node_id": residue.get("dominant_procedure_node_id"),
                "procedure_types": list(residue.get("procedure_types", []))[:8],
                "task_anchor_node_ids": list(residue.get("task_anchor_node_ids", []))[:8],
                "decision_anchor_node_ids": list(residue.get("decision_anchor_node_ids", []))[:8],
                "plan_anchor_node_ids": list(residue.get("plan_anchor_node_ids", []))[:8],
            }
            for residue in procedure_residues[:12]
        ],
    }


def _scenario_analysis(
    turn_records: Sequence[EvalTurnRecord],
    *,
    artifact_graph_snapshot: Optional[Dict[str, object]] = None,
    hypergraph_view_snapshot: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    expected_retrievals = [item for record in turn_records for item in record.expected_retrievals]
    matched_retrievals = [item for record in turn_records for item in record.matched_expected]
    expected_signals = [item for record in turn_records for item in (record.expected_response_signals or [])]
    matched_signals = [item for record in turn_records for item in (record.matched_response_signals or [])]
    provider_error_count = sum(int(_is_llm_error_response(record.assistant_response)) for record in turn_records)

    missing_expected = [item for item in expected_retrievals if item not in matched_retrievals]
    missing_response_signals = [item for item in expected_signals if item not in matched_signals]

    stage_failure = _infer_stage_failure(
        missing_expected=missing_expected,
        missing_response_signals=missing_response_signals,
        artifact_graph_snapshot=artifact_graph_snapshot,
        provider_error_count=provider_error_count,
    )
    graph_observation = _graph_observation(
        artifact_graph_snapshot,
        hypergraph_view_snapshot=hypergraph_view_snapshot,
        stage_failure=stage_failure,
    )

    return {
        "stage_failure": stage_failure,
        "missing_expected": missing_expected,
        "missing_response_signals": missing_response_signals,
        "provider_error_count": provider_error_count,
        "graph_observation": graph_observation,
        "graph_profile": _graph_profile(hypergraph_view_snapshot),
        "suspected_cause": _suspected_cause(
            stage_failure=stage_failure,
            missing_expected=missing_expected,
            missing_response_signals=missing_response_signals,
            graph_observation=graph_observation,
            provider_error_count=provider_error_count,
        ),
    }


def _infer_stage_failure(
    *,
    missing_expected: Sequence[str],
    missing_response_signals: Sequence[str],
    artifact_graph_snapshot: Optional[Dict[str, object]],
    provider_error_count: int,
) -> str:
    if provider_error_count > 0:
        return "provider"
    if missing_expected:
        if artifact_graph_snapshot and artifact_graph_snapshot.get("node_count", 0) > 1 and artifact_graph_snapshot.get("edge_count", 0) <= 0:
            return "relation_chain"
        return "retrieval"
    if missing_response_signals:
        return "response"
    return "none"


def _graph_observation(
    artifact_graph_snapshot: Optional[Dict[str, object]],
    *,
    hypergraph_view_snapshot: Optional[Dict[str, object]] = None,
    stage_failure: str,
) -> str:
    if not artifact_graph_snapshot:
        return "no artifact graph available"

    node_count = int(artifact_graph_snapshot.get("node_count", 0))
    edge_count = int(artifact_graph_snapshot.get("edge_count", 0))
    nodes = artifact_graph_snapshot.get("nodes", [])
    node_types = [str(node.get("artifact_type")) for node in nodes if node.get("artifact_type")]
    conflict_hyperedge_count = int((hypergraph_view_snapshot or {}).get("conflict_hyperedge_count", 0))
    procedure_residue_count = int((hypergraph_view_snapshot or {}).get("procedure_residue_count", 0))

    if node_count == 0:
        return "artifact graph empty"
    if stage_failure == "none":
        if conflict_hyperedge_count > 0:
            if procedure_residue_count > 0:
                return f"relation chain recovered with {edge_count} edge(s), {conflict_hyperedge_count} conflict unit(s), and {procedure_residue_count} procedure residue(s)"
            return f"relation chain recovered with {edge_count} edge(s) and {conflict_hyperedge_count} conflict unit(s)"
        if procedure_residue_count > 0 and edge_count > 0:
            return f"relation chain recovered with {edge_count} edge(s) and {procedure_residue_count} procedure residue(s)"
        if procedure_residue_count > 0:
            return f"artifact context recovered with {node_count} node(s) and {procedure_residue_count} procedure residue(s)"
        if edge_count > 0:
            return f"relation chain recovered with {edge_count} edge(s)"
        return f"artifact context recovered with {node_count} node(s)"
    if conflict_hyperedge_count > 0 and edge_count > 0:
        return f"relation chain recovered with {edge_count} edge(s) and {conflict_hyperedge_count} conflict unit(s)"
    if edge_count == 0 and node_count > 1:
        return "multiple artifacts stored but no relation edges recovered"
    if "plan" in node_types and "hypothesis" not in node_types:
        return "plan artifact present but linked hypothesis missing"
    if "hypothesis" in node_types and "log" not in node_types:
        return "hypothesis artifact present but source log missing"
    if edge_count > 0:
        return f"relation chain recovered with {edge_count} edge(s)"
    return f"single artifact recovered ({node_types[0] if node_types else 'unknown'})"


def _graph_profile(hypergraph_view_snapshot: Optional[Dict[str, object]]) -> str:
    if not hypergraph_view_snapshot:
        return "none"
    conflict_hyperedge_count = int(hypergraph_view_snapshot.get("conflict_hyperedge_count", 0))
    procedure_residue_count = int(hypergraph_view_snapshot.get("procedure_residue_count", 0))
    if conflict_hyperedge_count > 0 and procedure_residue_count > 0:
        return "mixed_conflict_procedure"
    if conflict_hyperedge_count > 0:
        return "conflict_heavy"
    if procedure_residue_count > 0:
        return "procedure_heavy"
    return "artifact_context"


def _suspected_cause(
    *,
    stage_failure: str,
    missing_expected: Sequence[str],
    missing_response_signals: Sequence[str],
    graph_observation: str,
    provider_error_count: int,
) -> str:
    if stage_failure == "none":
        return "no issue detected"
    if stage_failure == "provider":
        return f"provider returned {provider_error_count} error response(s)"
    if stage_failure == "relation_chain":
        return "artifact relations were not recovered strongly enough for multi-step context"
    if stage_failure == "retrieval":
        if missing_expected:
            return f"query overlap or kind scoring was too weak for: {', '.join(missing_expected[:3])}"
        return "retrieval missed expected context"
    if stage_failure == "response":
        if "relation chain recovered" in graph_observation:
            return "retrieved artifacts were available but the response did not use them explicitly"
        if missing_response_signals:
            return f"response omitted expected signals: {', '.join(missing_response_signals[:3])}"
        return "response did not ground itself in retrieved memory"
    return "undetermined"


def _build_comparison(results: Dict[str, object]) -> Dict[str, object]:
    runtime_metrics = results["runtime"]["metrics"]
    baseline_metrics = results["recent_history_baseline"]["metrics"]
    return {
        "recall_delta": runtime_metrics["recall"] - baseline_metrics["recall"],
        "precision_proxy_delta": runtime_metrics["precision_proxy"] - baseline_metrics["precision_proxy"],
        "response_recall_delta_vs_baseline": runtime_metrics.get("response_recall", 0.0),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _tokenize(text: str) -> set[str]:
    return {token.strip(".,!?;:()[]{}'\"").lower() for token in text.split() if len(token.strip(".,!?;:()[]{}'\"")) > 2}


def _is_llm_error_response(text: str) -> bool:
    return text.startswith("[LLM error:")


def _cleanup_session_files(session_path: Path) -> None:
    for path in (session_path, Path(str(session_path).replace(".json", "_history.json"))):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
