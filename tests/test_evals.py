"""
Tests for the practical evaluation harness.
"""

import json
import os
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.evals import (
    CONFLICT_PRACTICAL_SCENARIO_NAMES,
    CONFLICT_SIDECAR_SCENARIO_NAMES,
    CONTINUITY_SCENARIO_NAMES,
    DEFAULT_EVAL_SCENARIOS,
    DEFAULT_LLM_EVAL_BASE_URL,
    DEFAULT_LLM_EVAL_MODEL,
    DEFAULT_LLM_EVAL_TEMPERATURE,
    LONG_TASK_SCENARIO_NAMES,
    PRACTICAL_ROBUSTNESS_SCENARIO_NAMES,
    PRACTICAL_SIDECAR_SCENARIO_NAMES,
    run_conflict_regression,
    run_conflict_sidecar_regression,
    run_continuity_regression,
    run_long_task_regression,
    run_practical_robustness_regression,
    run_practical_sidecar_regression,
    run_product_regression,
    run_eval_suite,
)
from hypergraph_bistability.experiments import run_mechanism_experiment


def test_eval_suite_returns_runtime_and_baseline_metrics():
    results = run_eval_suite()

    assert "runtime" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert results["runtime"]["metrics"]["expected_count"] > 0
    assert results["recent_history_baseline"]["metrics"]["expected_count"] > 0
    assert 0.0 <= results["runtime"]["metrics"]["precision_proxy"] <= 1.0
    assert 0.0 <= results["runtime"]["metrics"]["irrelevant_context_rate"] <= 1.0
    assert 0.0 <= results["recent_history_baseline"]["metrics"]["precision_proxy"] <= 1.0
    assert 0.0 <= results["runtime"]["metrics"]["response_recall"] <= 1.0
    assert "analysis" in results["runtime"]["scenarios"][0]
    assert "continuity" in results["runtime"]
    assert "task_continuation" in results["runtime"]["continuity"]
    assert "procedure_continuity" in results["runtime"]["continuity"]
    assert "conflict_continuity" in results["runtime"]["continuity"]


def test_eval_setup_turns_seed_memory_without_assistant_turn_generation():
    from hypergraph_bistability.agent import HypergraphAgent
    from hypergraph_bistability.evals.runner import _seed_eval_turn

    agent = HypergraphAgent(use_embeddings=False)
    _seed_eval_turn(agent, "Remember that I prefer concise answers.")

    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["role"] == "user"
    assert any("prefer concise answers" in value.lower() for value in agent.memory.content_map.values())


def test_eval_suite_can_persist_json_output():
    output_path = os.path.join(os.getcwd(), "_eval_results_test.json")
    try:
        results = run_eval_suite(output_path=output_path)
        assert os.path.exists(output_path)

        with open(output_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        assert payload["comparison"] == results["comparison"]
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_eval_suite_marks_llm_usage_when_requested():
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    old_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        results = run_eval_suite(include_llm=True)
        assert "llm_runtime" in results
        assert results["llm_runtime"]["used_real_llm"] is False
        llm_scenarios = {scenario["name"]: scenario for scenario in results["llm_runtime"]["scenarios"]}
        relation_llm = llm_scenarios["artifact_relation_chain"]
        assert "artifact_graph" in relation_llm
        assert relation_llm["artifact_graph"]["node_count"] >= 3
        assert relation_llm["artifact_graph"]["edge_count"] >= 2
        assert results["llm_runtime"]["metrics"]["provider_error_count"] == 0
    finally:
        if old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key
        if old_anthropic_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_anthropic_key


def test_runtime_agent_accepts_configurable_llm_temperature():
    from hypergraph_bistability.agent import HypergraphAgent

    agent = HypergraphAgent(use_embeddings=False, llm_temperature=0.01)
    assert agent.llm_temperature == 0.01


def test_default_llm_eval_config_is_minimax_deterministic():
    assert DEFAULT_LLM_EVAL_MODEL == "MiniMax-M2.7"
    assert DEFAULT_LLM_EVAL_BASE_URL == "https://api.minimaxi.com/anthropic"
    assert DEFAULT_LLM_EVAL_TEMPERATURE == 0.01


def test_eval_suite_recognizes_anthropic_api_key_for_real_llm_usage():
    old_openai_key = os.environ.pop("OPENAI_API_KEY", None)
    old_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    try:
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        results = run_eval_suite(include_llm=True)
        assert results["llm_runtime"]["used_real_llm"] is True
    finally:
        if old_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = old_openai_key
        if old_anthropic_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_anthropic_key


def test_run_llm_evals_passes_force_powershell_flag():
    from hypergraph_bistability import cli

    captured = {}
    old_run_eval_suite = cli.run_eval_suite
    old_stdout = sys.stdout
    stream = StringIO()
    try:
        sys.stdout = stream

        def _fake_run_eval_suite(**kwargs):
            captured.update(kwargs)
            return {
                "llm_runtime": {
                    "used_real_llm": True,
                    "metrics": {
                        "recall": 1.0,
                        "precision_proxy": 1.0,
                        "response_recall": 1.0,
                        "provider_error_count": 0,
                    },
                    "scenarios": [],
                }
            }

        cli.run_eval_suite = _fake_run_eval_suite
        cli._run_llm_evals(force_powershell=True)
    finally:
        cli.run_eval_suite = old_run_eval_suite
        sys.stdout = old_stdout

    assert captured["llm_force_powershell_transport"] is True


def test_eval_turn_payload_includes_llm_debug_snapshot():
    from hypergraph_bistability.evals.runner import EvalTurnRecord, _scenario_payload
    from hypergraph_bistability.evals.scenarios import EvalScenario, EvalTurn

    scenario = EvalScenario(
        name="debug",
        description="debug",
        tier="core",
        turns=[EvalTurn(user_input="q", expected_retrievals=["foo"])],
    )
    payload = _scenario_payload(
        scenario,
        [
            EvalTurnRecord(
                user_input="q",
                expected_retrievals=["foo"],
                retrieved_items=["foo"],
                matched_expected=["foo"],
                matched_retrieved_items=["foo"],
                assistant_response="OK",
                expected_response_signals=[],
                matched_response_signals=[],
                llm_debug={"transport": "anthropic-powershell", "content_types": ["thinking", "text"]},
            )
        ],
    )

    assert payload["turns"][0]["llm_debug"]["transport"] == "anthropic-powershell"


def test_competition_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("competition")

    results = payload["results"]
    assert "competition_retrieval" in results
    assert "direct_retrieval" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert results["competition_retrieval"]["metrics"]["recall"] >= 0.0
    assert "irrelevant_context_rate_delta" in results["comparison"]["competition_vs_direct"]


def test_associative_expansion_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("associative_expansion")

    results = payload["results"]
    assert "parent_expansion" in results
    assert "single_hit" in results
    assert "hyperedge_expansion" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert "parent_vs_single_hit" in results["comparison"]
    assert "hyperedge_vs_parent" in results["comparison"]
    relation_scenario = {
        scenario["name"]: scenario for scenario in results["hyperedge_expansion"]["scenarios"]
    }["artifact_relation_chain"]
    assert "hypergraph_view" in relation_scenario
    assert relation_scenario["hypergraph_view"]["hyperedge_count"] >= 1


def test_hyperedge_state_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("hyperedge_state")

    results = payload["results"]
    assert "state_aware" in results
    assert "state_blind" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert "state_aware_vs_state_blind" in results["comparison"]
    conflicted = {scenario["name"]: scenario for scenario in results["state_aware"]["scenarios"]}[
        "conflicted_hypothesis_debugging"
    ]
    assert conflicted["hypergraph_view"]["hyperedge_count"] >= 1


def test_confidence_tags_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("confidence_tags")

    results = payload["results"]
    assert "confidence_aware" in results
    assert "confidence_blind" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert "confidence_aware_vs_confidence_blind" in results["comparison"]
    scenario = {scenario["name"]: scenario for scenario in results["confidence_aware"]["scenarios"]}[
        "contradicted_hypothesis_filtering"
    ]
    assert scenario["hypergraph_view"]["hyperedge_count"] >= 1


def test_procedure_memory_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("procedure_memory")

    results = payload["results"]
    assert "procedure_memory_aware" in results
    assert "procedure_memory_blind" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert "aware_vs_blind" in results["comparison"]
    scenario = {scenario["name"]: scenario for scenario in results["procedure_memory_aware"]["scenarios"]}[
        "procedure_release_handoff"
    ]
    assert scenario["hypergraph_view"]["hyperedge_count"] >= 1
    assert scenario["hypergraph_view"]["procedure_residues"]


def test_conflict_links_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("conflict_links")

    results = payload["results"]
    assert "conflict_hyperedge_aware" in results
    assert "conflict_hyperedge_blind" in results
    assert "conflict_link_aware" in results
    assert "conflict_link_blind" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert "aware_vs_blind" in results["comparison"]
    assert "hyperedge_aware_vs_link_aware" in results["comparison"]
    assert "hyperedge_aware_vs_hyperedge_blind" in results["comparison"]
    scenario = {scenario["name"]: scenario for scenario in results["conflict_hyperedge_aware"]["scenarios"]}[
        "contradiction_link_filtering"
    ]
    assert scenario["hypergraph_view"]["hyperedge_count"] >= 1
    assert scenario["hypergraph_view"]["conflict_hyperedge_count"] >= 1
    assert scenario["hypergraph_view"]["conflict_hyperedges"]


def test_uncertainty_conflict_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("uncertainty_conflict")

    results = payload["results"]
    assert "uncertainty_conflict_aware" in results
    assert "conflict_only" in results
    assert "uncertainty_conflict_blind" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert "aware_vs_conflict_only" in results["comparison"]
    assert "aware_vs_blind" in results["comparison"]
    scenario = {scenario["name"]: scenario for scenario in results["uncertainty_conflict_aware"]["scenarios"]}[
        "conflict_verified_vs_speculative"
    ]
    assert scenario["hypergraph_view"]["conflict_hyperedge_count"] >= 1
    assert scenario["hypergraph_view"]["conflict_hyperedges"]
    assert "task_continuation_delta" in results["comparison"]["aware_vs_conflict_only"]


def test_two_stage_conflict_mechanism_experiment_produces_ablation_outputs():
    payload = run_mechanism_experiment("two_stage_conflict")

    results = payload["results"]
    assert "two_stage_conflict_aware" in results
    assert "conflict_hyperedge_aware" in results
    assert "conflict_link_aware" in results
    assert "recent_history_baseline" in results
    assert "comparison" in results
    assert "two_stage_vs_hyperedge_aware" in results["comparison"]
    assert "two_stage_vs_link_aware" in results["comparison"]
    assert "two_stage_vs_baseline" in results["comparison"]
    scenario = {scenario["name"]: scenario for scenario in results["two_stage_conflict_aware"]["scenarios"]}[
        "conflict_unit_dominance"
    ]
    assert scenario["hypergraph_view"]["conflict_hyperedge_count"] >= 1
    assert scenario["hypergraph_view"]["conflict_hyperedges"]
    assert "conflict_pair_preservation_under_preference_noise" in {
        scenario["name"] for scenario in results["two_stage_conflict_aware"]["scenarios"]
    }


def test_default_eval_scenarios_include_harder_cases():
    names = {scenario.name for scenario in DEFAULT_EVAL_SCENARIOS}
    assert "layered_preferences" in names
    assert "debugging_resume_with_preference" in names
    assert "coding_agent_resume" in names
    assert "artifact_chain_resume" in names
    assert "artifact_relation_chain" in names
    assert "blocker_resume_no_reexplaining" in names
    assert "decision_resume_after_interruption" in names
    assert "plan_resume_without_restart" in names
    assert "contradiction_link_resume" in names
    assert "conflict_unit_dominance" in names


def test_runtime_core_continuity_metrics_outperform_baseline():
    results = run_eval_suite(tier="core")

    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    assert runtime["task_continuation"] >= baseline["task_continuation"]
    assert runtime["decision_continuity"] >= baseline["decision_continuity"]
    assert runtime["procedure_continuity"] >= baseline["procedure_continuity"]


def test_runtime_stress_continuity_metrics_outperform_baseline_on_resume_behavior():
    results = run_eval_suite(tier="stress")

    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    assert runtime["task_continuation"] >= baseline["task_continuation"]
    assert runtime["repeated_work_avoidance_proxy"] >= baseline["repeated_work_avoidance_proxy"]
    assert runtime["procedure_continuity"] >= baseline["procedure_continuity"]


def test_continuity_regression_uses_dedicated_scenario_subset():
    results = run_continuity_regression()

    names = {scenario["name"] for scenario in results["runtime"]["scenarios"]}
    assert names == set(CONTINUITY_SCENARIO_NAMES)
    assert len(names) == 8


def test_eval_suite_supports_tier_filtering():
    core_results = run_eval_suite(tier="core")
    stress_results = run_eval_suite(tier="stress")

    core_names = {scenario["name"] for scenario in core_results["runtime"]["scenarios"]}
    stress_names = {scenario["name"] for scenario in stress_results["runtime"]["scenarios"]}

    assert "preference_recall" in core_names
    assert "coding_agent_resume" not in core_names
    assert "coding_agent_resume" in stress_names
    assert "artifact_chain_resume" in stress_names
    assert "artifact_relation_chain" in stress_names
    assert "preference_recall" not in stress_names


def test_runtime_eval_includes_artifact_graph_snapshot_for_relation_scenario():
    results = run_eval_suite(tier="stress")

    runtime_scenarios = {scenario["name"]: scenario for scenario in results["runtime"]["scenarios"]}
    baseline_scenarios = {scenario["name"]: scenario for scenario in results["recent_history_baseline"]["scenarios"]}

    relation_runtime = runtime_scenarios["artifact_relation_chain"]
    relation_baseline = baseline_scenarios["artifact_relation_chain"]

    assert "artifact_graph" in relation_runtime
    assert "hypergraph_view" in relation_runtime
    assert relation_runtime["artifact_graph"]["node_count"] >= 3
    assert relation_runtime["artifact_graph"]["edge_count"] >= 2
    assert relation_runtime["hypergraph_view"]["node_count"] >= 3
    assert relation_runtime["hypergraph_view"]["hyperedge_count"] >= 1
    assert all("artifact_type" in node for node in relation_runtime["artifact_graph"]["nodes"])
    assert all("node_type" in node for node in relation_runtime["hypergraph_view"]["nodes"])
    assert relation_baseline.get("artifact_graph") is None
    assert relation_baseline.get("hypergraph_view") is None
    assert relation_runtime["analysis"]["stage_failure"] == "none"
    assert "relation chain recovered" in relation_runtime["analysis"]["graph_observation"]


def test_conflict_link_experiment_snapshot_includes_conflict_units():
    payload = run_mechanism_experiment("conflict_links")
    scenario = {scenario["name"]: scenario for scenario in payload["results"]["conflict_hyperedge_aware"]["scenarios"]}[
        "contradiction_link_resume"
    ]

    assert scenario["hypergraph_view"]["conflict_hyperedge_count"] >= 1
    assert any(unit["status"] in {"active_conflict", "resolved_conflict", "competing_hypotheses"} for unit in scenario["hypergraph_view"]["conflict_hyperedges"])
    assert "conflict unit" in scenario["analysis"]["graph_observation"]


def test_conflict_hyperedge_experiment_includes_dominance_scenario():
    payload = run_mechanism_experiment("conflict_links")
    scenario = {scenario["name"]: scenario for scenario in payload["results"]["conflict_hyperedge_aware"]["scenarios"]}[
        "conflict_unit_dominance"
    ]

    assert scenario["hypergraph_view"]["conflict_hyperedge_count"] >= 1
    assert any(unit["dominant_node_id"] is not None for unit in scenario["hypergraph_view"]["conflict_hyperedges"])


def test_baseline_eval_includes_failure_analysis_for_missed_context():
    results = run_eval_suite(tier="stress")

    baseline_scenarios = {scenario["name"]: scenario for scenario in results["recent_history_baseline"]["scenarios"]}
    relation_baseline = baseline_scenarios["artifact_relation_chain"]

    assert relation_baseline["analysis"]["stage_failure"] == "retrieval"
    assert relation_baseline["analysis"]["missing_expected"]
    assert "query overlap or kind scoring was too weak" in relation_baseline["analysis"]["suspected_cause"]


def test_cli_run_evals_prints_failure_analysis():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._run_evals(tier="stress")
    finally:
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "Baseline Failure Analysis" in output
    assert "artifact_relation_chain: retrieval" in output


def test_provider_error_is_classified_separately():
    from hypergraph_bistability.evals.runner import EvalTurnRecord, _scenario_analysis

    analysis = _scenario_analysis(
        [
            EvalTurnRecord(
                user_input="q",
                expected_retrievals=["foo"],
                retrieved_items=["foo"],
                matched_expected=["foo"],
                matched_retrieved_items=["foo"],
                assistant_response="[LLM error: Connection error.] I understand your message. How can I help?",
                expected_response_signals=["bar"],
                matched_response_signals=[],
            )
        ]
    )

    assert analysis["stage_failure"] == "provider"
    assert analysis["provider_error_count"] == 1
    assert "provider returned 1 error response" in analysis["suspected_cause"]


def test_cli_failure_analysis_skips_when_no_failures():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._print_failure_analysis("Runtime", [{"name": "ok", "analysis": {"stage_failure": "none"}}])
    finally:
        sys.stdout = old_stdout

    assert stream.getvalue() == ""


def test_cli_run_llm_evals_prints_provider_error_count():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    old_run_eval_suite = cli.run_eval_suite
    try:
        sys.stdout = stream
        cli.run_eval_suite = lambda **kwargs: {
            "llm_runtime": {
                "used_real_llm": True,
                "metrics": {
                    "recall": 1.0,
                    "precision_proxy": 1.0,
                    "response_recall": 0.0,
                    "provider_error_count": 2,
                },
                "scenarios": [],
            }
        }
        cli._run_llm_evals()
    finally:
        cli.run_eval_suite = old_run_eval_suite
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "provider errors: 2" in output


def test_cli_minimax_smoke_reports_python_runtime_socket_issue():
    from hypergraph_bistability import cli

    class _Result:
        returncode = 0
        stdout = "OK"
        stderr = ""

    stream = StringIO()
    old_stdout = sys.stdout
    old_run = cli.subprocess.run
    old_env = os.environ.get("OPENAI_API_KEY")
    old_agent = cli.HypergraphAgent
    try:
        sys.stdout = stream
        os.environ["OPENAI_API_KEY"] = "test-key"
        cli.subprocess.run = lambda *args, **kwargs: _Result()

        class _FailingAgent:
            def __init__(self, *args, **kwargs):
                self.llm_base_url = kwargs.get("llm_base_url")

            def _call_llm_via_client(self, messages):
                if self.llm_base_url.endswith("/anthropic"):
                    return "OK from anthropic"
                raise RuntimeError("WinError 10013")

        cli.HypergraphAgent = _FailingAgent
        cli._run_minimax_smoke()
    finally:
        cli.subprocess.run = old_run
        cli.HypergraphAgent = old_agent
        if old_env is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_env
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "openai_powershell_transport: ok" in output
    assert "openai_python_transport: failed" in output
    assert "anthropic_python_transport: ok" in output
    assert "Anthropic-compatible transport is healthier than OpenAI-compatible" in output


def test_cli_run_continuity_regression_prints_continuity_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._run_continuity_regression()
    finally:
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "Continuity Regression" in output
    assert "blocker_resume_no_reexplaining" in output
    assert "runtime task continuation" in output


def test_product_regression_returns_runtime_and_baseline():
    results = run_product_regression()

    assert "runtime" in results
    assert "recent_history_baseline" in results
    names = {scenario["name"] for scenario in results["runtime"]["scenarios"]}
    assert "release_hotfix_handoff" in names
    assert "debug_fix_verify_close_loop" in names
    assert "coding_review_commitment_chain" in names
    assert "incident_root_cause_handoff" in names
    assert "release_scope_guardrail_handoff" in names
    assert "review_scope_followup_chain" in names
    assert "sidecar_incident_close_packet_paraphrase" in names
    assert "sidecar_release_packet_followthrough_paraphrase" in names
    assert "procedure_release_handoff_chain" in names
    assert "procedure_review_handoff_chain" in names
    assert "procedure_release_gate_review_chain" in names
    assert "procedure_review_validation_handoff_chain" in names
    assert results["runtime"]["continuity"]["procedure_continuity"] >= results["recent_history_baseline"]["continuity"]["procedure_continuity"]


def test_cli_run_product_regression_prints_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._run_product_regression()
    finally:
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "Product Regression" in output
    assert "release_hotfix_handoff" in output
    assert "incident_root_cause_handoff" in output
    assert "runtime task continuation" in output


def test_cli_run_llm_product_regression_prints_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    old_run_product_regression = cli.run_product_regression
    try:
        sys.stdout = stream
        cli.run_product_regression = lambda **kwargs: {
            "llm_runtime": {
                "used_real_llm": True,
                "metrics": {
                    "recall": 1.0,
                    "precision_proxy": 1.0,
                    "response_recall": 1.0,
                    "provider_error_count": 0,
                },
                "continuity": {
                    "task_continuation": 1.0,
                    "blocker_preservation": 1.0,
                    "decision_continuity": 1.0,
                    "procedure_continuity": 1.0,
                    "repeated_work_avoidance_proxy": 1.0,
                },
                "scenarios": [{"name": "release_hotfix_handoff", "analysis": {"stage_failure": "none"}}],
            }
        }
        cli._run_llm_product_regression()
    finally:
        cli.run_product_regression = old_run_product_regression
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "LLM Product Regression" in output
    assert "release_hotfix_handoff" in output
    assert "runtime decision continuity" in output
    assert "runtime procedure continuity" in output


def test_long_task_regression_uses_dedicated_scenario_subset():
    results = run_long_task_regression()

    names = {scenario["name"] for scenario in results["runtime"]["scenarios"]}
    assert names == set(LONG_TASK_SCENARIO_NAMES)
    assert len(names) == 5
    assert "procedure_incident_closeout_replay" in names
    assert "procedure_incident_handoff_closeout_replay" in names
    assert results["runtime"]["continuity"]["procedure_continuity"] >= results["recent_history_baseline"]["continuity"]["procedure_continuity"]


def test_conflict_regression_uses_dedicated_practical_scenario_subset():
    results = run_conflict_regression()

    names = {scenario["name"] for scenario in results["runtime"]["scenarios"]}
    assert names == set(CONFLICT_PRACTICAL_SCENARIO_NAMES)
    assert len(names) == 3
    assert results["runtime"]["continuity"]["conflict_continuity"] >= 1.0
    assert results["recent_history_baseline"]["continuity"]["conflict_continuity"] == 0.0


def test_practical_sidecar_regression_uses_dedicated_scenario_subset():
    results = run_practical_sidecar_regression()

    names = {scenario["name"] for scenario in results["runtime"]["scenarios"]}
    assert names == set(PRACTICAL_SIDECAR_SCENARIO_NAMES)
    assert len(names) == 1
    assert results["runtime"]["continuity"]["procedure_continuity"] >= results["recent_history_baseline"]["continuity"]["procedure_continuity"]


def test_conflict_sidecar_regression_uses_dedicated_scenario_subset():
    results = run_conflict_sidecar_regression()

    names = {scenario["name"] for scenario in results["runtime"]["scenarios"]}
    assert names == set(CONFLICT_SIDECAR_SCENARIO_NAMES)
    assert len(names) == 1
    scenario = results["runtime"]["scenarios"][0]
    assert scenario["analysis"]["graph_profile"] == "conflict_heavy"


def test_practical_robustness_regression_uses_dedicated_scenario_subset():
    results = run_practical_robustness_regression()

    names = {scenario["name"] for scenario in results["runtime"]["scenarios"]}
    assert names == set(PRACTICAL_ROBUSTNESS_SCENARIO_NAMES)
    assert len(names) == 2
    assert results["runtime"]["continuity"]["task_continuation"] >= results["recent_history_baseline"]["continuity"]["task_continuation"]


def test_cli_run_practical_sidecar_regression_prints_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._run_practical_sidecar_regression()
    finally:
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "Practical Sidecar Regression" in output
    assert "sidecar_incident_handoff_bundle_paraphrase" in output
    assert "runtime procedure continuity" in output


def test_cli_run_conflict_regression_prints_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._run_conflict_regression()
    finally:
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "Conflict Practical Regression" in output
    assert "incident_root_cause_handoff" in output
    assert "runtime conflict continuity" in output


def test_cli_run_conflict_sidecar_regression_prints_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._run_conflict_sidecar_regression()
    finally:
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "Conflict Sidecar Regression" in output
    assert "sidecar_incident_handoff_bundle_paraphrase" in output
    assert "runtime conflict continuity" in output


def test_cli_run_practical_robustness_regression_prints_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stream
        cli._run_practical_robustness_regression()
    finally:
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "Practical Robustness Regression" in output
    assert "robustness_incident_story_bundle" in output
    assert "runtime task continuation" in output


def test_cli_run_llm_long_task_regression_prints_summary():
    from hypergraph_bistability import cli

    stream = StringIO()
    old_stdout = sys.stdout
    old_run_long_task_regression = cli.run_long_task_regression
    try:
        sys.stdout = stream
        cli.run_long_task_regression = lambda **kwargs: {
            "llm_runtime": {
                "used_real_llm": True,
                "metrics": {
                    "recall": 1.0,
                    "precision_proxy": 1.0,
                    "response_recall": 1.0,
                    "provider_error_count": 0,
                },
                "continuity": {
                    "task_continuation": 1.0,
                    "blocker_preservation": 1.0,
                    "decision_continuity": 1.0,
                    "procedure_continuity": 1.0,
                    "repeated_work_avoidance_proxy": 1.0,
                },
                "scenarios": [{"name": "hotfix_full_lifecycle_replay", "analysis": {"stage_failure": "none"}}],
            }
        }
        cli._run_llm_long_task_regression()
    finally:
        cli.run_long_task_regression = old_run_long_task_regression
        sys.stdout = old_stdout

    output = stream.getvalue()
    assert "LLM Long-Task Regression" in output
    assert "hotfix_full_lifecycle_replay" in output
    assert "runtime decision continuity" in output
    assert "runtime procedure continuity" in output


def test_cli_write_from_docs_saves_output():
    from hypergraph_bistability import cli

    output_path = Path(os.getcwd()) / "_write_from_docs_test_output.md"
    doc_path = Path(os.getcwd()) / "_write_from_docs_test_doc.md"
    doc_path.write_text("# Doc\nstable_v1 stays formal.\n", encoding="utf-8")

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def write_from_documents(self, **kwargs):
            output = kwargs.get("output_path")
            if output:
                Path(output).write_text("written", encoding="utf-8")
            return {"response": "written"}

    stream = StringIO()
    old_stdout = sys.stdout
    old_agent = cli.HypergraphAgent
    try:
        sys.stdout = stream
        cli.HypergraphAgent = _FakeAgent
        cli._write_from_docs(
            document_paths=[str(doc_path)],
            instruction="Write a summary.",
            output_path=str(output_path),
        )
    finally:
        cli.HypergraphAgent = old_agent
        sys.stdout = old_stdout
        if output_path.exists():
            output_path.unlink()
        if doc_path.exists():
            doc_path.unlink()

    assert "Saved document-driven output" in stream.getvalue()
    assert not doc_path.exists() or True


def test_cli_write_from_docs_reads_utf8_instruction_file():
    from hypergraph_bistability import cli

    output_path = Path(os.getcwd()) / "_write_from_docs_instruction_output.md"
    doc_path = Path(os.getcwd()) / "_write_from_docs_instruction_doc.md"
    instruction_path = Path(os.getcwd()) / "_write_from_docs_instruction.txt"
    doc_path.write_text("# Doc\nstable_v1 stays formal.\n", encoding="utf-8")
    instruction_path.write_text("请基于文档写一段中文交接说明。", encoding="utf-8")
    captured = {}

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def write_from_documents(self, **kwargs):
            captured.update(kwargs)
            output = kwargs.get("output_path")
            if output:
                Path(output).write_text("written", encoding="utf-8")
            return {"response": "written"}

    stream = StringIO()
    old_stdout = sys.stdout
    old_agent = cli.HypergraphAgent
    try:
        sys.stdout = stream
        cli.HypergraphAgent = _FakeAgent
        cli._write_from_docs(
            document_paths=[str(doc_path)],
            instruction="placeholder",
            instruction_path=str(instruction_path),
            output_path=str(output_path),
        )
    finally:
        cli.HypergraphAgent = old_agent
        sys.stdout = old_stdout
        if output_path.exists():
            output_path.unlink()
        if doc_path.exists():
            doc_path.unlink()
        if instruction_path.exists():
            instruction_path.unlink()

    assert captured["instruction"] == "请基于文档写一段中文交接说明。"
    assert "Saved document-driven output" in stream.getvalue()


def test_product_regression_procedure_scenarios_expose_procedure_residues():
    results = run_product_regression()

    scenarios = {scenario["name"]: scenario for scenario in results["runtime"]["scenarios"]}
    release = scenarios["procedure_release_handoff_chain"]
    review = scenarios["procedure_review_handoff_chain"]
    release_gate = scenarios["procedure_release_gate_review_chain"]
    review_validation = scenarios["procedure_review_validation_handoff_chain"]
    close_packet = scenarios["sidecar_incident_close_packet_paraphrase"]
    release_packet = scenarios["sidecar_release_packet_followthrough_paraphrase"]

    assert release["hypergraph_view"]["procedure_residue_count"] >= 1
    assert review["hypergraph_view"]["procedure_residue_count"] >= 1
    assert release_gate["hypergraph_view"]["procedure_residue_count"] >= 1
    assert review_validation["hypergraph_view"]["procedure_residue_count"] >= 1
    assert close_packet["hypergraph_view"]["procedure_residue_count"] >= 1
    assert release_packet["hypergraph_view"]["procedure_residue_count"] >= 1
    assert release["analysis"]["stage_failure"] == "none"
    assert "procedure residue" in release["analysis"]["graph_observation"]


def test_practical_sidecar_regression_exposes_procedure_residues():
    results = run_practical_sidecar_regression()

    scenarios = {scenario["name"]: scenario for scenario in results["runtime"]["scenarios"]}
    handoff_bundle = scenarios["sidecar_incident_handoff_bundle_paraphrase"]

    assert handoff_bundle["hypergraph_view"]["conflict_hyperedge_count"] >= 1
    assert handoff_bundle["analysis"]["graph_profile"] == "conflict_heavy"


def test_conflict_regression_tracks_conflict_heavy_profiles():
    results = run_conflict_regression()

    scenarios = {scenario["name"]: scenario for scenario in results["runtime"]["scenarios"]}
    assert scenarios["incident_root_cause_handoff"]["analysis"]["graph_profile"] == "conflict_heavy"
    assert scenarios["sidecar_incident_handoff_bundle_paraphrase"]["analysis"]["graph_profile"] == "conflict_heavy"
    assert scenarios["incident_conflict_packet_resolution"]["analysis"]["graph_profile"] == "conflict_heavy"
    assert results["runtime"]["continuity"]["conflict_eligible_scenarios"] == 3


def test_long_task_regression_procedure_closeout_scenario_exposes_procedure_residue():
    results = run_long_task_regression()

    scenario = {scenario["name"]: scenario for scenario in results["runtime"]["scenarios"]}[
        "procedure_incident_closeout_replay"
    ]
    handoff_closeout = {scenario["name"]: scenario for scenario in results["runtime"]["scenarios"]}[
        "procedure_incident_handoff_closeout_replay"
    ]

    assert scenario["hypergraph_view"]["procedure_residue_count"] >= 1
    assert scenario["analysis"]["stage_failure"] == "none"
    assert "procedure residue" in scenario["analysis"]["graph_observation"]
    assert handoff_closeout["hypergraph_view"]["procedure_residue_count"] >= 1
    assert handoff_closeout["analysis"]["stage_failure"] == "none"


def test_decision_residue_mechanism_experiment_produces_comparison():
    payload = run_mechanism_experiment("decision_residue")

    assert payload["experiment"] == "decision_residue"
    assert "decision_residue_aware" in payload["results"]
    assert "decision_residue_blind" in payload["results"]
    assert "comparison" in payload["results"]
    assert "aware_vs_blind" in payload["results"]["comparison"]


def test_phase_progress_mechanism_experiment_produces_comparison():
    payload = run_mechanism_experiment("phase_progress")

    assert payload["experiment"] == "phase_progress"
    assert "phase_progress_aware" in payload["results"]
    assert "phase_progress_blind" in payload["results"]
    assert "comparison" in payload["results"]
    assert "aware_vs_blind" in payload["results"]["comparison"]
