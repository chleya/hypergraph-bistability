"""Simple CLI entry points for demos and visualization."""

from __future__ import annotations

import argparse
import json
import locale
import os
import subprocess
from pathlib import Path

from hypergraph_bistability.agent import HypergraphAgent
from hypergraph_bistability.evals import (
    CONFLICT_PRACTICAL_SCENARIO_NAMES,
    CONFLICT_SIDECAR_SCENARIO_NAMES,
    CONTINUITY_SCENARIO_NAMES,
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
from hypergraph_bistability.memory import AgentMemory


def _derive_minimax_base_urls(base_url: str | None) -> tuple[str, str]:
    normalized = (base_url or DEFAULT_LLM_EVAL_BASE_URL).rstrip("/")
    if normalized.endswith("/anthropic"):
        return normalized[:-len("/anthropic")] + "/v1", normalized
    if normalized.endswith("/v1"):
        return normalized, normalized[:-len("/v1")] + "/anthropic"
    return normalized + "/v1", normalized + "/anthropic"


def _run_single_minimax_smoke_transport(*, label: str, base_url: str, model: str, api_key: str) -> dict:
    powershell_ok = False
    powershell_output = ""

    if label == "anthropic":
        body = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply with exactly OK"}]}],
            "temperature": DEFAULT_LLM_EVAL_TEMPERATURE,
            "max_tokens": 20,
        }
        script = (
            "$headers = @{ 'x-api-key' = '"
            + api_key
            + "'; 'anthropic-version' = '2023-06-01'; 'Content-Type' = 'application/json' }; "
            + "$body = @'\n"
            + json.dumps(body, ensure_ascii=False)
            + "\n'@; "
            + "$resp = Invoke-WebRequest -Uri '"
            + base_url.rstrip("/")
            + "/v1/messages' -Headers $headers -Method POST -Body $body -UseBasicParsing; "
            + "$json = $resp.Content | ConvertFrom-Json; "
            + "($json.content | Where-Object { $_.type -eq 'text' } | Select-Object -ExpandProperty text) -join \"`n\""
        )
    else:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly OK"}],
            "temperature": DEFAULT_LLM_EVAL_TEMPERATURE,
            "max_tokens": 20,
        }
        script = (
            "$headers = @{ Authorization = 'Bearer "
            + api_key
            + "'; 'Content-Type' = 'application/json' }; "
            + "$body = @'\n"
            + json.dumps(body, ensure_ascii=False)
            + "\n'@; "
            + "$resp = Invoke-WebRequest -Uri '"
            + base_url.rstrip("/")
            + "/chat/completions' -Headers $headers -Method POST -Body $body -UseBasicParsing; "
            + "$json = $resp.Content | ConvertFrom-Json; "
            + "$json.choices[0].message.content"
        )

    result = subprocess.run(
        ["powershell", "-Command", script],
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        timeout=90,
        check=False,
    )
    if result.returncode == 0:
        powershell_output = result.stdout.strip()
        powershell_ok = bool(powershell_output)
        if not powershell_ok:
            powershell_output = "empty response body"
    else:
        powershell_output = (result.stderr or result.stdout or "unknown PowerShell error").strip()

    python_ok = False
    python_output = ""
    try:
        agent = HypergraphAgent(
            use_embeddings=False,
            llm_api_key=api_key,
            llm_model=model,
            llm_base_url=base_url,
            llm_max_retries=0,
            llm_retry_backoff_seconds=0.0,
        )
        response = agent._call_llm_via_client([{"role": "user", "content": "Reply with exactly OK"}])
        python_ok = True
        python_output = response
    except Exception as exc:
        python_output = str(exc)

    return {
        "label": label,
        "base_url": base_url,
        "powershell_ok": powershell_ok,
        "powershell_output": powershell_output,
        "python_ok": python_ok,
        "python_output": python_output,
    }


def _print_minimax_smoke_diagnosis(results: list[dict]) -> None:
    for result in results:
        print(f"- {result['label']}_base_url: {result['base_url']}")
        print(f"- {result['label']}_powershell_transport: {'ok' if result['powershell_ok'] else 'failed'}")
        print(f"  output: {result['powershell_output']}")
        print(f"- {result['label']}_python_transport: {'ok' if result['python_ok'] else 'failed'}")
        print(f"  output: {result['python_output']}")

    openai_result = next((result for result in results if result["label"] == "openai"), None)
    anthropic_result = next((result for result in results if result["label"] == "anthropic"), None)
    if openai_result and anthropic_result:
        if anthropic_result["python_ok"] and not openai_result["python_ok"]:
            print("Diagnosis: Anthropic-compatible transport is healthier than OpenAI-compatible for this MiniMax setup.")
            return
        if openai_result["powershell_ok"] and not openai_result["python_ok"] and not anthropic_result["python_ok"]:
            print("Diagnosis: MiniMax is reachable from PowerShell, but this Python runtime still cannot open outbound HTTPS sockets.")
            return
        if not openai_result["powershell_ok"] and not anthropic_result["powershell_ok"] and not openai_result["python_ok"] and not anthropic_result["python_ok"]:
            print("Diagnosis: both MiniMax compatibility paths failed; check endpoint, key, or host/network policy.")
            return
        if openai_result["python_ok"] or anthropic_result["python_ok"]:
            print("Diagnosis: at least one MiniMax compatibility path is healthy.")
            return

    print("Diagnosis: MiniMax compatibility diagnostics were inconclusive.")


def _demo_memory() -> None:
    memory = AgentMemory(k=3, L=2, use_physics_control=True)
    memory.group_labels = ["professional", "personal", "technical"]
    memory.layer_labels = ["preferences", "context"]
    memory.write("User prefers concise answers", group=0, layer=0)
    print(memory.status())
    print(memory.get_context_for_llm())


def _demo_agent() -> None:
    agent = HypergraphAgent(k=4, L=2, use_embeddings=False)
    print(agent.get_memory_state())
    print(agent.chat("Help me debug a Python function"))


def _print_turn_summary(result) -> None:
    print("\nAssistant")
    print(result.assistant_response)
    print("\nRuntime")
    print(f"- mode: {result.controller_state['mode']}")
    print(f"- conflict: {result.conflict_level:.2f}")

    if result.retrieved_items:
        print("- retrieved:")
        for item in result.retrieved_items:
            print(f"  - {item}")
    else:
        print("- retrieved: none")

    if result.writes:
        print("- writes:")
        for write in result.writes:
            print(
                f"  - {write['reason']} -> group={write['group']} layer={write['layer']} "
                f"importance={write['importance']:.2f}"
            )
    else:
        print("- writes: none")


def _chat_demo(session_path: str | None = None) -> None:
    agent = HypergraphAgent(k=4, L=2, use_embeddings=False, name="terminal_chat")

    resolved_path = Path(session_path).resolve() if session_path else None
    if resolved_path and resolved_path.exists():
        agent.load(str(resolved_path))
        print(f"Loaded session from {resolved_path}")

    print("Hypergraph Agent Terminal Demo")
    print("Commands: /exit, /state, /graph, /save, /reset")

    while True:
        try:
            user_input = input("\nYou\n> ").strip()
        except EOFError:
            print()
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/state":
            print(agent.visualize_state())
            continue
        if user_input == "/graph":
            print(agent.summarize_artifact_graph())
            continue
        if user_input == "/reset":
            agent.reset_memory()
            print("Session reset.")
            continue
        if user_input == "/save":
            target = resolved_path or Path(f"{agent.name}_state.json").resolve()
            agent.save(str(target))
            print(f"Saved session to {target}")
            continue

        result = agent.process_turn(user_input)
        _print_turn_summary(result)

    if resolved_path:
        agent.save(str(resolved_path))
        print(f"Saved session to {resolved_path}")


def _print_failure_analysis(label: str, scenarios: list[dict]) -> None:
    failures = []
    for scenario in scenarios:
        analysis = scenario.get("analysis") or {}
        stage_failure = analysis.get("stage_failure", "none")
        if stage_failure == "none":
            continue
        failures.append(
            (
                scenario.get("name", "unknown"),
                stage_failure,
                analysis.get("suspected_cause", "undetermined"),
            )
        )

    if not failures:
        return

    print(f"{label} Failure Analysis")
    for name, stage_failure, suspected_cause in failures:
        print(f"- {name}: {stage_failure} -> {suspected_cause}")


def _run_evals(output_path: str | None = None, tier: str | None = None) -> None:
    results = run_eval_suite(output_path=output_path, tier=tier)
    runtime = results["runtime"]["metrics"]
    baseline = results["recent_history_baseline"]["metrics"]
    comparison = results["comparison"]

    print("Evaluation Summary")
    print(f"- runtime recall: {runtime['recall']:.3f}")
    print(f"- baseline recall: {baseline['recall']:.3f}")
    print(f"- runtime precision proxy: {runtime['precision_proxy']:.3f}")
    print(f"- baseline precision proxy: {baseline['precision_proxy']:.3f}")
    print(f"- recall delta: {comparison['recall_delta']:.3f}")
    print(f"- precision proxy delta: {comparison['precision_proxy_delta']:.3f}")
    _print_failure_analysis("Baseline", results["recent_history_baseline"]["scenarios"])

    if output_path:
        print(f"Saved eval results to {Path(output_path).resolve()}")
    else:
        print(json.dumps(results, indent=2))


def _run_continuity_regression(output_path: str | None = None) -> None:
    results = run_continuity_regression(output_path=output_path)
    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    print("Continuity Regression")
    print(f"- scenarios: {', '.join(CONTINUITY_SCENARIO_NAMES)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- baseline task continuation: {baseline['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- baseline blocker preservation: {baseline['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- baseline decision continuity: {baseline['decision_continuity']:.3f}")
    print(f"- runtime procedure continuity: {runtime['procedure_continuity']:.3f}")
    print(f"- baseline procedure continuity: {baseline['procedure_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    print(f"- baseline repeated-work avoidance: {baseline['repeated_work_avoidance_proxy']:.3f}")

    if output_path:
        print(f"Saved continuity regression results to {Path(output_path).resolve()}")


def _run_product_regression(output_path: str | None = None) -> None:
    results = run_product_regression(output_path=output_path)
    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]
    scenario_names = [scenario["name"] for scenario in results["runtime"]["scenarios"]]

    print("Product Regression")
    print(f"- scenarios: {', '.join(scenario_names)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- baseline task continuation: {baseline['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- baseline blocker preservation: {baseline['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- baseline decision continuity: {baseline['decision_continuity']:.3f}")
    print(f"- runtime procedure continuity: {runtime['procedure_continuity']:.3f}")
    print(f"- baseline procedure continuity: {baseline['procedure_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    print(f"- baseline repeated-work avoidance: {baseline['repeated_work_avoidance_proxy']:.3f}")

    if output_path:
        print(f"Saved product regression results to {Path(output_path).resolve()}")


def _run_long_task_regression(output_path: str | None = None) -> None:
    results = run_long_task_regression(output_path=output_path)
    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    print("Long-Task Regression")
    print(f"- scenarios: {', '.join(LONG_TASK_SCENARIO_NAMES)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- baseline task continuation: {baseline['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- baseline blocker preservation: {baseline['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- baseline decision continuity: {baseline['decision_continuity']:.3f}")
    print(f"- runtime procedure continuity: {runtime['procedure_continuity']:.3f}")
    print(f"- baseline procedure continuity: {baseline['procedure_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    print(f"- baseline repeated-work avoidance: {baseline['repeated_work_avoidance_proxy']:.3f}")

    if output_path:
        print(f"Saved long-task regression results to {Path(output_path).resolve()}")


def _run_conflict_regression(output_path: str | None = None) -> None:
    results = run_conflict_regression(output_path=output_path)
    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    print("Conflict Practical Regression")
    print(f"- scenarios: {', '.join(CONFLICT_PRACTICAL_SCENARIO_NAMES)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- baseline task continuation: {baseline['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- baseline blocker preservation: {baseline['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- baseline decision continuity: {baseline['decision_continuity']:.3f}")
    print(f"- runtime conflict continuity: {runtime['conflict_continuity']:.3f}")
    print(f"- baseline conflict continuity: {baseline['conflict_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    print(f"- baseline repeated-work avoidance: {baseline['repeated_work_avoidance_proxy']:.3f}")

    if output_path:
        print(f"Saved conflict practical regression results to {Path(output_path).resolve()}")


def _run_practical_sidecar_regression(output_path: str | None = None) -> None:
    results = run_practical_sidecar_regression(output_path=output_path)
    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    print("Practical Sidecar Regression")
    print(f"- scenarios: {', '.join(PRACTICAL_SIDECAR_SCENARIO_NAMES)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- baseline task continuation: {baseline['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- baseline blocker preservation: {baseline['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- baseline decision continuity: {baseline['decision_continuity']:.3f}")
    print(f"- runtime procedure continuity: {runtime['procedure_continuity']:.3f}")
    print(f"- baseline procedure continuity: {baseline['procedure_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    print(f"- baseline repeated-work avoidance: {baseline['repeated_work_avoidance_proxy']:.3f}")

    if output_path:
        print(f"Saved practical sidecar regression results to {Path(output_path).resolve()}")


def _run_conflict_sidecar_regression(output_path: str | None = None) -> None:
    results = run_conflict_sidecar_regression(output_path=output_path)
    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    print("Conflict Sidecar Regression")
    print(f"- scenarios: {', '.join(CONFLICT_SIDECAR_SCENARIO_NAMES)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- baseline task continuation: {baseline['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- baseline blocker preservation: {baseline['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- baseline decision continuity: {baseline['decision_continuity']:.3f}")
    print(f"- runtime conflict continuity: {runtime['conflict_continuity']:.3f}")
    print(f"- baseline conflict continuity: {baseline['conflict_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    print(f"- baseline repeated-work avoidance: {baseline['repeated_work_avoidance_proxy']:.3f}")

    if output_path:
        print(f"Saved conflict sidecar regression results to {Path(output_path).resolve()}")


def _run_practical_robustness_regression(output_path: str | None = None) -> None:
    results = run_practical_robustness_regression(output_path=output_path)
    runtime = results["runtime"]["continuity"]
    baseline = results["recent_history_baseline"]["continuity"]

    print("Practical Robustness Regression")
    print(f"- scenarios: {', '.join(PRACTICAL_ROBUSTNESS_SCENARIO_NAMES)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- baseline task continuation: {baseline['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- baseline blocker preservation: {baseline['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- baseline decision continuity: {baseline['decision_continuity']:.3f}")
    print(f"- runtime procedure continuity: {runtime['procedure_continuity']:.3f}")
    print(f"- baseline procedure continuity: {baseline['procedure_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    print(f"- baseline repeated-work avoidance: {baseline['repeated_work_avoidance_proxy']:.3f}")

    if output_path:
        print(f"Saved practical robustness regression results to {Path(output_path).resolve()}")


def _run_llm_product_regression(
    output_path: str | None = None,
    model: str = DEFAULT_LLM_EVAL_MODEL,
    base_url: str | None = DEFAULT_LLM_EVAL_BASE_URL,
    force_powershell: bool = False,
) -> None:
    results = run_product_regression(
        output_path=output_path,
        include_llm=True,
        llm_model=model,
        llm_base_url=base_url,
        llm_temperature=DEFAULT_LLM_EVAL_TEMPERATURE,
        llm_force_powershell_transport=force_powershell,
    )
    llm_runtime = results.get("llm_runtime")
    if not llm_runtime:
        print("No LLM product regression results were produced.")
        return
    if not llm_runtime.get("used_real_llm"):
        print("LLM product regression skipped: no OPENAI_API_KEY or ANTHROPIC_API_KEY configured.")
        if output_path:
            print(f"Saved placeholder product regression results to {Path(output_path).resolve()}")
        return

    runtime = llm_runtime["continuity"]
    scenario_names = [scenario["name"] for scenario in llm_runtime["scenarios"]]
    metrics = llm_runtime["metrics"]

    print("LLM Product Regression")
    print(f"- scenarios: {', '.join(scenario_names)}")
    print(f"- retrieval recall: {metrics['recall']:.3f}")
    print(f"- retrieval precision proxy: {metrics['precision_proxy']:.3f}")
    print(f"- response recall: {metrics['response_recall']:.3f}")
    print(f"- provider errors: {metrics.get('provider_error_count', 0)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- runtime procedure continuity: {runtime['procedure_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    _print_failure_analysis("LLM Product", llm_runtime["scenarios"])

    if output_path:
        print(f"Saved LLM product regression results to {Path(output_path).resolve()}")


def _run_llm_long_task_regression(
    output_path: str | None = None,
    model: str = DEFAULT_LLM_EVAL_MODEL,
    base_url: str | None = DEFAULT_LLM_EVAL_BASE_URL,
    force_powershell: bool = False,
) -> None:
    results = run_long_task_regression(
        output_path=output_path,
        include_llm=True,
        llm_model=model,
        llm_base_url=base_url,
        llm_temperature=DEFAULT_LLM_EVAL_TEMPERATURE,
        llm_force_powershell_transport=force_powershell,
    )
    llm_runtime = results.get("llm_runtime")
    if not llm_runtime:
        print("No LLM long-task regression results were produced.")
        return
    if not llm_runtime.get("used_real_llm"):
        print("LLM long-task regression skipped: no OPENAI_API_KEY or ANTHROPIC_API_KEY configured.")
        if output_path:
            print(f"Saved placeholder long-task regression results to {Path(output_path).resolve()}")
        return

    runtime = llm_runtime["continuity"]
    scenario_names = [scenario["name"] for scenario in llm_runtime["scenarios"]]
    metrics = llm_runtime["metrics"]

    print("LLM Long-Task Regression")
    print(f"- scenarios: {', '.join(scenario_names)}")
    print(f"- retrieval recall: {metrics['recall']:.3f}")
    print(f"- retrieval precision proxy: {metrics['precision_proxy']:.3f}")
    print(f"- response recall: {metrics['response_recall']:.3f}")
    print(f"- provider errors: {metrics.get('provider_error_count', 0)}")
    print(f"- runtime task continuation: {runtime['task_continuation']:.3f}")
    print(f"- runtime blocker preservation: {runtime['blocker_preservation']:.3f}")
    print(f"- runtime decision continuity: {runtime['decision_continuity']:.3f}")
    print(f"- runtime procedure continuity: {runtime['procedure_continuity']:.3f}")
    print(f"- runtime repeated-work avoidance: {runtime['repeated_work_avoidance_proxy']:.3f}")
    _print_failure_analysis("LLM Long-Task", llm_runtime["scenarios"])

    if output_path:
        print(f"Saved LLM long-task regression results to {Path(output_path).resolve()}")


def _run_llm_evals(
    output_path: str | None = None,
    model: str = DEFAULT_LLM_EVAL_MODEL,
    base_url: str | None = DEFAULT_LLM_EVAL_BASE_URL,
    tier: str | None = None,
    force_powershell: bool = False,
) -> None:
    results = run_eval_suite(
        output_path=output_path,
        tier=tier,
        include_llm=True,
        llm_model=model,
        llm_base_url=base_url,
        llm_temperature=DEFAULT_LLM_EVAL_TEMPERATURE,
        llm_force_powershell_transport=force_powershell,
    )
    llm_runtime = results.get("llm_runtime")
    if not llm_runtime:
        print("No LLM runtime results were produced.")
        return
    if not llm_runtime.get("used_real_llm"):
        print("LLM eval skipped: no OPENAI_API_KEY or ANTHROPIC_API_KEY configured.")
        if output_path:
            print(f"Saved placeholder eval results to {Path(output_path).resolve()}")
        return

    metrics = llm_runtime["metrics"]
    print("LLM Evaluation Summary")
    print(f"- retrieval recall: {metrics['recall']:.3f}")
    print(f"- retrieval precision proxy: {metrics['precision_proxy']:.3f}")
    print(f"- response recall: {metrics['response_recall']:.3f}")
    print(f"- provider errors: {metrics.get('provider_error_count', 0)}")
    _print_failure_analysis("LLM", llm_runtime["scenarios"])

    if output_path:
        print(f"Saved LLM eval results to {Path(output_path).resolve()}")
    else:
        print(json.dumps(results, indent=2))


def _run_mechanism_experiment(
    experiment: str,
    output_path: str | None = None,
    include_llm: bool = False,
    model: str = DEFAULT_LLM_EVAL_MODEL,
    base_url: str | None = DEFAULT_LLM_EVAL_BASE_URL,
    force_powershell: bool = False,
) -> None:
    results = run_mechanism_experiment(
        experiment,
        output_path=output_path,
        include_llm=include_llm,
        llm_model=model,
        llm_base_url=base_url,
        llm_force_powershell_transport=force_powershell,
    )
    print(f"Mechanism experiment: {experiment}")
    print(f"- scenarios: {', '.join(results['scenario_names'])}")
    if output_path:
        print(f"Saved experiment results to {Path(output_path).resolve()}")
    else:
        print(json.dumps(results, indent=2))


def _run_minimax_smoke(
    model: str = DEFAULT_LLM_EVAL_MODEL,
    base_url: str | None = DEFAULT_LLM_EVAL_BASE_URL,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("MiniMax smoke skipped: no OPENAI_API_KEY or ANTHROPIC_API_KEY configured.")
        return

    print("MiniMax Smoke Check")
    print(f"- model: {model}")
    openai_base_url, anthropic_base_url = _derive_minimax_base_urls(base_url)
    results = [
        _run_single_minimax_smoke_transport(
            label="openai",
            base_url=openai_base_url,
            model=model,
            api_key=api_key,
        ),
        _run_single_minimax_smoke_transport(
            label="anthropic",
            base_url=anthropic_base_url,
            model=model,
            api_key=api_key,
        ),
    ]
    _print_minimax_smoke_diagnosis(results)


def _write_from_docs(
    *,
    document_paths: list[str],
    instruction: str,
    instruction_path: str | None = None,
    output_path: str | None = None,
    model: str = DEFAULT_LLM_EVAL_MODEL,
    base_url: str | None = DEFAULT_LLM_EVAL_BASE_URL,
    force_powershell: bool = False,
    per_doc_char_limit: int = 2200,
) -> None:
    if not document_paths:
        raise ValueError("write-from-docs requires at least one --doc path.")
    if instruction_path:
        instruction = Path(instruction_path).read_text(encoding="utf-8").strip()
    if not instruction.strip():
        raise ValueError("write-from-docs requires a non-empty instruction.")

    agent = HypergraphAgent(
        k=4,
        L=2,
        use_embeddings=False,
        llm_model=model,
        llm_base_url=base_url,
        llm_force_powershell_transport=force_powershell,
        llm_max_output_tokens=900,
        name="doc_writer",
    )
    result = agent.write_from_documents(
        document_paths=document_paths,
        instruction=instruction,
        output_path=output_path,
        per_doc_char_limit=per_doc_char_limit,
    )
    if output_path:
        print(f"Saved document-driven output to {Path(output_path).resolve()}")
    else:
        print(result["response"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Hypergraph bistability CLI")
    parser.add_argument(
        "command",
        choices=[
            "demo-memory",
            "demo-agent",
            "chat-demo",
            "run-evals",
            "run-continuity-regression",
            "run-product-regression",
            "run-long-task-regression",
            "run-conflict-regression",
            "run-practical-sidecar-regression",
            "run-conflict-sidecar-regression",
            "run-practical-robustness-regression",
            "run-llm-product-regression",
            "run-llm-long-task-regression",
            "run-llm-evals",
            "run-mechanism-experiment",
            "minimax-smoke",
            "write-from-docs",
        ],
        help="Which built-in demo to run.",
    )
    parser.add_argument(
        "--session",
        help="Optional path for loading and saving a terminal chat session.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for saving eval results as JSON.",
    )
    parser.add_argument(
        "--tier",
        choices=["core", "stress"],
        help="Optional eval scenario tier filter.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_EVAL_MODEL,
        help="LLM model to use for LLM-backed evals.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_LLM_EVAL_BASE_URL,
        help="LLM base URL for OpenAI-compatible or Anthropic-compatible endpoints.",
    )
    parser.add_argument(
        "--force-powershell",
        action="store_true",
        help="Force MiniMax requests through the PowerShell transport on Windows.",
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "competition",
            "associative_expansion",
            "mode_specific",
            "hyperedge_state",
            "phase_progress",
            "confidence_tags",
            "decision_residue",
            "procedure_memory",
            "uncertainty_conflict",
            "conflict_links",
            "two_stage_conflict",
        ],
        help="Mechanism experiment to run.",
    )
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        help="Path to a local document to include in write-from-docs. Repeat for multiple files.",
    )
    parser.add_argument(
        "--instruction",
        default="Write a concise grounded summary from the provided documents.",
        help="Writing instruction for write-from-docs.",
    )
    parser.add_argument(
        "--instruction-file",
        help="UTF-8 text file for write-from-docs instructions. Prefer this for non-ASCII prompts on Windows.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=2200,
        help="Per-document character budget for write-from-docs.",
    )
    args = parser.parse_args()

    if args.command == "demo-memory":
        _demo_memory()
    elif args.command == "demo-agent":
        _demo_agent()
    elif args.command == "chat-demo":
        _chat_demo(args.session)
    elif args.command == "run-evals":
        _run_evals(args.output, tier=args.tier)
    elif args.command == "run-continuity-regression":
        _run_continuity_regression(args.output)
    elif args.command == "run-product-regression":
        _run_product_regression(args.output)
    elif args.command == "run-long-task-regression":
        _run_long_task_regression(args.output)
    elif args.command == "run-conflict-regression":
        _run_conflict_regression(args.output)
    elif args.command == "run-practical-sidecar-regression":
        _run_practical_sidecar_regression(args.output)
    elif args.command == "run-conflict-sidecar-regression":
        _run_conflict_sidecar_regression(args.output)
    elif args.command == "run-practical-robustness-regression":
        _run_practical_robustness_regression(args.output)
    elif args.command == "run-llm-product-regression":
        _run_llm_product_regression(
            args.output,
            model=args.model,
            base_url=args.base_url,
            force_powershell=args.force_powershell,
        )
    elif args.command == "run-llm-long-task-regression":
        _run_llm_long_task_regression(
            args.output,
            model=args.model,
            base_url=args.base_url,
            force_powershell=args.force_powershell,
        )
    elif args.command == "run-llm-evals":
        _run_llm_evals(
            args.output,
            model=args.model,
            base_url=args.base_url,
            tier=args.tier,
            force_powershell=args.force_powershell,
        )
    elif args.command == "run-mechanism-experiment":
        _run_mechanism_experiment(
            args.experiment or "competition",
            output_path=args.output,
            include_llm=True,
            model=args.model,
            base_url=args.base_url,
            force_powershell=args.force_powershell,
        )
    elif args.command == "minimax-smoke":
        _run_minimax_smoke(model=args.model, base_url=args.base_url)
    elif args.command == "write-from-docs":
        _write_from_docs(
            document_paths=args.docs or [],
            instruction=args.instruction,
            instruction_path=args.instruction_file,
            output_path=args.output,
            model=args.model,
            base_url=args.base_url,
            force_powershell=args.force_powershell,
            per_doc_char_limit=args.context_chars,
        )


if __name__ == "__main__":
    main()
