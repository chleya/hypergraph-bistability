"""
Tests for the practical agent runtime pipeline.
"""

import os
import sys
import json
import shutil
from pathlib import Path
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypergraph_bistability.agent import HypergraphAgent, HypergraphMemoryAgent
from hypergraph_bistability.agent.runtime import ContextAssembler
from hypergraph_bistability.memory.policies import RetrievalPolicy, RetrievedMemory, WritePolicy


def test_process_turn_returns_structured_result():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    result = agent.process_turn("Remember that I prefer concise code reviews.")

    assert result.user_input.startswith("Remember")
    assert isinstance(result.assistant_response, str)
    assert isinstance(result.memory_context, str)
    assert isinstance(result.retrieved_items, list)
    assert isinstance(result.writes, list)
    assert len(agent.conversation_history) == 2
    assert len(agent.turn_log) == 1


def test_chat_preserves_simple_interface():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent.chat("Help me plan tomorrow's tasks.")

    assert isinstance(response, str)
    state = agent.get_memory_state()
    assert state["conversation_turns"] == 1
    assert state["runtime"]["write_policy"] == "WritePolicy"
    assert state["runtime"]["turn_log_size"] == 1
    assert state["runtime"]["retrieval_strategy"] == "hyperedge_expansion"
    assert state["runtime"]["runtime_profile"] == "stable_v1"


def test_agent_exposes_stable_runtime_profile():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)

    profile = agent.get_runtime_profile()

    assert profile["name"] == "stable_v1"
    assert profile["retrieval_strategy"] == "hyperedge_expansion"
    assert "conflict_hyperedges" in profile["enabled_structures"]
    assert "decision_residues" in profile["enabled_structures"]


def test_agent_save_and_load_round_trip():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False, name="runtime_test")
    agent.chat("Remember that I am working on a release checklist.")

    temp_dir = os.path.join(os.getcwd(), "_agent_runtime_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        path = os.path.join(temp_dir, "agent_state.json")
        agent.save(path)

        restored = HypergraphAgent(k=3, L=2, use_embeddings=False, name="runtime_test")
        restored.load(path)

        assert restored.conversation_history == agent.conversation_history
        assert restored.system_prompt == agent.system_prompt
        assert restored.memory.content_map == agent.memory.content_map
        assert restored.turn_log == agent.turn_log
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_retrieval_policy_prefers_typed_turn_log_matches():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Remember that I prefer concise answers.")
    agent.process_turn("Help me plan a release checklist.")

    policy = RetrievalPolicy()
    items = policy.collect(
        "What are my preferences and release plan?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=6,
    )

    contents = [item.content.lower() for item in items]
    assert any("prefer concise answers" in content for content in contents)
    assert any("release checklist" in content for content in contents)


def test_context_assembler_groups_retrieved_memory_by_kind():
    assembler = ContextAssembler()
    messages = assembler.build_messages(
        system_prompt="System",
        memory_context="[Memory State: neutral]",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="User prefers concise answers", score=0.9, kind="preference"),
            RetrievedMemory(source="turn_log", content="Need a release checklist", score=0.8, kind="task"),
        ],
        conversation_history=[],
        user_input="Help me",
    )

    system_content = messages[0]["content"]
    assert "Preference:" in system_content or "Preferences:" in system_content
    assert "Task:" in system_content or "Tasks:" in system_content
    assert "Do not output <think>" in system_content
    assert "use it explicitly" in system_content
    assert "explicitly name the recalled task" in system_content


def test_context_assembler_adds_literal_root_cause_contract_for_debug_preference():
    assembler = ContextAssembler()
    messages = assembler.build_messages(
        system_prompt="System",
        memory_context="[Memory State: neutral]",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Remember that when debugging, I want the root cause hypothesis first.",
                score=0.9,
                kind="preference",
            ),
        ],
        conversation_history=[],
        user_input="What should I investigate first?",
    )

    system_content = messages[0]["content"]
    assert 'begin with the literal phrase "Root cause hypothesis:"' in system_content


def test_context_assembler_adds_literal_diff_style_contract_for_coding_preference():
    assembler = ContextAssembler()
    messages = assembler.build_messages(
        system_prompt="System",
        memory_context="[Memory State: neutral]",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Remember that when giving code changes, I want a short diff-style summary first.",
                score=0.9,
                kind="preference",
            ),
        ],
        conversation_history=[],
        user_input="What should I inspect first?",
    )

    system_content = messages[0]["content"]
    assert 'begin with the literal phrase "Diff-style summary:"' in system_content


def test_retrieval_policy_penalizes_irrelevant_context_after_topic_shift():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Remember that I prefer concise code reviews.")
    agent.process_turn("I need help planning an onboarding checklist for a new backend engineer.")
    agent.process_turn("Also, I want dinner ideas for tonight.")

    policy = RetrievalPolicy()
    items = policy.collect(
        "Back to the onboarding plan. Keep my preference in mind.",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=3,
    )

    contents = [item.content.lower() for item in items]
    assert any("onboarding checklist" in content for content in contents)
    assert any("prefer concise code reviews" in content for content in contents)
    assert not any("dinner ideas" in content for content in contents)


def test_write_policy_skips_low_value_assistant_questions():
    policy = WritePolicy()
    decision = policy.decide(
        "How can I help?",
        role="assistant",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert decision is None


def test_write_policy_skips_non_explicit_assistant_memory_writeback():
    policy = WritePolicy()
    decision = policy.decide(
        "Investigate cache invalidation ordering after deploy.",
        role="assistant",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert decision is None


def test_write_policy_allows_explicit_assistant_plan_persistence():
    policy = WritePolicy()
    decision = policy.decide(
        "Plan: inspect retry state restore, compare checkpoint IDs, then patch dedupe before requeue.",
        role="assistant",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert decision is not None
    assert decision.kind == "plan"


def test_mock_response_uses_retrieved_task_details():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Remember that when giving code changes, I want a short diff-style summary first.")
    agent.process_turn("I am fixing a flaky retry loop in the worker scheduler.")
    agent.process_turn("The suspected problem is duplicate backoff state after task resume.")

    result = agent.process_turn("Back to the scheduler fix. What should I inspect first?")

    lowered = result.assistant_response.lower()
    assert "retry loop" in lowered
    assert "backoff state" in lowered
    assert "diff-style" in lowered


def test_response_contracts_force_root_cause_prefix_when_required():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Check cache invalidation ordering after deploy.",
        user_input="Return to the bug and tell me what to investigate first.",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Remember that when debugging, I want the root cause hypothesis first.",
                score=0.9,
                kind="preference",
            ),
        ],
    )

    assert response.startswith("Root cause hypothesis:")


def test_response_contracts_force_diff_style_prefix_when_required():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Inspect the task resume path and duplicate backoff state first.",
        user_input="Back to the scheduler fix. What should I inspect first?",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Remember that when giving code changes, I want a short diff-style summary first.",
                score=0.9,
                kind="preference",
            ),
        ],
    )

    assert response.startswith("Diff-style summary:")


def test_write_from_documents_builds_grounded_prompt_and_saves_utf8_output():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False, llm_max_output_tokens=700)
    doc_path = Path(os.getcwd()) / "_write_from_documents_plan.md"
    output_path = Path(os.getcwd()) / "_write_from_documents_out.md"
    doc_path.write_text("# Plan\nKeep stable_v1 as the formal runtime.\n", encoding="utf-8")

    captured = {}

    def _fake_generate_response(*, user_input, memory_context, messages, retrieved_items=None):
        captured["user_input"] = user_input
        captured["memory_context"] = memory_context
        captured["messages"] = messages
        return "Grounded write-up."

    agent.generate_response = _fake_generate_response  # type: ignore[method-assign]
    try:
        payload = agent.write_from_documents(
            document_paths=[str(doc_path)],
            instruction="Write a concise roadmap summary.",
            output_path=str(output_path),
            per_doc_char_limit=120,
        )

        assert payload["response"] == "Grounded write-up."
        assert output_path.read_text(encoding="utf-8") == "Grounded write-up."
        assert captured["user_input"] == "Write a concise roadmap summary."
        assert captured["memory_context"] == "[Memory State: document_writer]"
        assert "Match the language of the user's instruction." in captured["messages"][0]["content"]
        assert "## _write_from_documents_plan.md" in captured["messages"][1]["content"]
        assert "stable_v1" in captured["messages"][1]["content"]
        assert "If the instruction is in Chinese" in captured["messages"][1]["content"]
    finally:
        if output_path.exists():
            output_path.unlink()
        if doc_path.exists():
            doc_path.unlink()


def test_write_from_documents_can_use_dedicated_minimax_powershell_writer(monkeypatch, tmp_path):
    import subprocess

    doc_path = tmp_path / "plan.md"
    output_path = tmp_path / "out.md"
    doc_path.write_text("# Plan\nKeep stable_v1 as the formal runtime.\n", encoding="utf-8")
    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*args, **kwargs):
        captured["args"] = args[0]
        prompt_path = args[0][10]
        response_path = args[0][12]
        captured["prompt"] = Path(prompt_path).read_text(encoding="utf-8")
        Path(response_path).write_text(
            json.dumps(
                {
                    "model": "MiniMax-M2.7",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "中文交接说明"}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    agent = HypergraphAgent(
        k=3,
        L=2,
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
        llm_force_powershell_transport=True,
    )

    payload = agent.write_from_documents(
        document_paths=[str(doc_path)],
        instruction="请写中文交接说明。",
        output_path=str(output_path),
        per_doc_char_limit=120,
    )

    assert payload["response"] == "中文交接说明"
    assert payload["llm_debug"]["transport"] == "anthropic-powershell-doc-writer"
    assert output_path.read_text(encoding="utf-8") == "中文交接说明"
    assert captured["args"][:4] == ["powershell", "-ExecutionPolicy", "Bypass", "-File"]
    assert captured["args"][4].endswith("minimax_prompt_request.ps1")
    assert "请写中文交接说明。" in captured["prompt"]


def test_response_contracts_force_concise_prefix_when_preference_requires_it():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "I will present the rollout plan as bullet points with phases and risks.",
        user_input="Now summarize how you should present the rollout plan back to me.",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Remember that I prefer concise answers.",
                score=0.9,
                kind="preference",
            ),
            RetrievedMemory(
                source="turn_log",
                content="Also remember that action items should be in bullet lists.",
                score=0.9,
                kind="preference",
            ),
            RetrievedMemory(
                source="turn_log",
                content="I am planning a production rollout for a billing service.",
                score=0.9,
                kind="task",
            ),
        ],
    )

    assert response.startswith("Concise summary:")


def test_response_contracts_append_debugging_signals_when_missing():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Root cause hypothesis: investigate the deploy path first.",
        user_input="Return to the bug and tell me what to investigate first.",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="I am investigating a flaky cache invalidation bug in the user profile service.", score=0.9, kind="fact"),
            RetrievedMemory(source="turn_log", content="The bug appears after deploys when stale profile data persists.", score=0.9, kind="fact"),
        ],
    )

    lowered = response.lower()
    assert "cache invalidation" in lowered
    assert "stale profile data" in lowered


def test_response_contracts_append_coding_resume_signals_when_missing():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Diff-style summary:\nInspect the resume path first.",
        user_input="Back to the scheduler fix. What should I inspect first?",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="I am fixing a flaky retry loop in the worker scheduler.", score=0.9, kind="task"),
            RetrievedMemory(source="turn_log", content="The suspected problem is duplicate backoff state after task resume.", score=0.9, kind="fact"),
        ],
    )

    lowered = response.lower()
    assert "retry loop" in lowered
    assert "backoff state" in lowered


def test_response_contracts_append_artifact_plan_signals_when_missing():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Continue the remediation plan in order.",
        user_input="Return to the profile-sync incident and continue the plan.",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="Plan: inspect cursor reset, verify reconnect ordering, then patch stale cursor cleanup.", score=0.9, kind="plan"),
        ],
    )

    lowered = response.lower()
    assert "cursor reset" in lowered
    assert "reconnect ordering" in lowered
    assert "stale cursor cleanup" in lowered


def test_response_contracts_append_conflict_backing_evidence_for_scheduler_root_cause():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Root cause hypothesis: duplicate scheduler state is being rehydrated twice.",
        user_input="Back to the scheduler incident. Which explanation still looks strongest?",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="Log: worker scheduler duplicate backoff state still appears after checkpoint resume.", score=0.9, kind="log"),
            RetrievedMemory(source="turn_log", content="Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice.", score=0.9, kind="hypothesis"),
        ],
    )

    assert "backoff state" in response.lower()


def test_response_contracts_append_conflict_backing_evidence_for_profile_sync_root_cause():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Root cause hypothesis: stale cursor state remains after reconnect and blocks the sync worker.",
        user_input="Return to the profile-sync incident and name the strongest remaining root cause.",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="Log: profile-sync job times out after redis reconnect and the worker stalls before final checkpoint flush.", score=0.9, kind="log"),
            RetrievedMemory(source="turn_log", content="Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.", score=0.9, kind="hypothesis"),
        ],
    )

    lowered = response.lower()
    assert "redis reconnect" in lowered
    assert "checkpoint flush" in lowered


def test_response_contracts_append_ready_to_hand_off_signal_when_verification_is_complete():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Release handoff:\n- Shipping scope: scheduler dedupe patch only.\n- Next: verify scheduler dedupe in staging before handoff.",
        user_input="Summarize the hotfix handoff: what did we ship, what did we reject, and are we ready to hand off?",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="Verified: scheduler dedupe looks stable in staging and rollback notes are now complete.", score=0.9, kind="fact"),
        ],
    )

    assert "ready to hand off" in response.lower()


def test_response_contracts_append_ready_to_close_signal_when_closure_query_uses_closing_language():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Before closing, verify reconnect ordering and carry forward the stale cursor cleanup fix.",
        user_input="Before we close this incident, what should we verify and what fix are we carrying forward?",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging before closing the incident.",
                score=0.9,
                kind="fact",
            ),
        ],
    )

    assert "ready to close" in response.lower()


def test_response_contracts_append_handoff_bundle_signals_for_survived_explanation_queries():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Incident handoff:\n- Carry forward the stale cursor cleanup fix.",
        user_input="For the handoff bundle, which explanation survived, which patch survived, and what evidence stays citeable?",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.",
                score=0.9,
                kind="hypothesis",
            ),
            RetrievedMemory(
                source="turn_log",
                content="Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging.",
                score=0.9,
                kind="fact",
            ),
            RetrievedMemory(
                source="turn_log",
                content="Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush.",
                score=0.9,
                kind="log",
            ),
        ],
    )

    lowered = response.lower()
    assert "stale cursor state" in lowered
    assert "stale cursor cleanup" in lowered
    assert "checkpoint flush" in lowered


def test_response_contracts_append_ruled_out_theory_for_conflict_packet_queries():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Incident handoff:\n- Carry forward the stale cursor cleanup fix.",
        user_input="For the incident packet, which theory still survives, which theory stays ruled out, and which proof points still travel with it?",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Hypothesis: cursor reset timing is causing the timeout in the sync worker.",
                score=0.8,
                kind="hypothesis",
            ),
            RetrievedMemory(
                source="turn_log",
                content="Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.",
                score=0.9,
                kind="hypothesis",
            ),
            RetrievedMemory(
                source="turn_log",
                content="Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging.",
                score=0.9,
                kind="fact",
            ),
        ],
    )

    lowered = response.lower()
    assert "stale cursor state" in lowered
    assert "cursor reset timing" in lowered
    assert "stale cursor cleanup" in lowered


def test_response_contracts_force_diff_style_for_review_handoff_queries():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "We committed to patch dedupe first and rejected changing the retry policy first.",
        user_input="For the release review handoff, what did we commit to, what stays out of scope, and what validation already passed?",
        retrieved_items=[
            RetrievedMemory(
                source="turn_log",
                content="Remember that when giving code changes, I want a short diff-style summary first.",
                score=0.9,
                kind="preference",
            ),
        ],
    )

    assert response.lower().startswith("diff-style summary:")


def test_response_contracts_build_incident_handoff_summary():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    response = agent._apply_response_contracts(
        "Carry forward the root cause and fix.",
        user_input="Before I hand this incident off, what root cause are we carrying forward, what fix are we carrying forward, and what evidence should I cite?",
        retrieved_items=[
            RetrievedMemory(source="turn_log", content="Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.", score=0.9, kind="hypothesis"),
            RetrievedMemory(source="turn_log", content="Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff.", score=0.9, kind="plan"),
            RetrievedMemory(source="turn_log", content="Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging.", score=0.9, kind="fact"),
            RetrievedMemory(source="turn_log", content="Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush.", score=0.9, kind="log"),
        ],
    )

    lowered = response.lower()
    assert "incident handoff" in lowered
    assert "ready to hand off" in lowered


def test_generate_response_retries_transient_llm_failures():
    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary connection error")
            return _Response("Recovered answer")

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

    agent = HypergraphAgent(
        k=3,
        L=2,
        use_embeddings=False,
        llm_max_retries=1,
        llm_retry_backoff_seconds=0.0,
    )
    agent.llm_client = _FakeClient()

    response = agent.generate_response(
        user_input="Help me debug this.",
        memory_context="[Memory State: neutral]",
        messages=[{"role": "user", "content": "Help me debug this."}],
        retrieved_items=[],
    )

    assert response == "Recovered answer"
    assert agent.llm_client.chat.completions.calls == 2


def test_runtime_agent_exposes_llm_retry_configuration():
    agent = HypergraphAgent(
        use_embeddings=False,
        llm_max_retries=3,
        llm_retry_backoff_seconds=0.25,
    )

    assert agent.llm_max_retries == 3
    assert agent.llm_retry_backoff_seconds == 0.25


def test_generate_response_falls_back_to_powershell_transport_for_minimax_connection_error():
    class _FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("Connection error.")

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

    agent = HypergraphAgent(
        k=3,
        L=2,
        use_embeddings=False,
        llm_model="MiniMax-M2.7",
        llm_api_key="test-key",
        llm_base_url="https://api.minimaxi.com/v1",
        llm_max_retries=0,
    )
    agent.llm_client = _FakeClient()
    agent._call_llm_via_powershell_transport = lambda messages: "Recovered via PowerShell"

    response = agent.generate_response(
        user_input="Help me debug this.",
        memory_context="[Memory State: neutral]",
        messages=[{"role": "user", "content": "Help me debug this."}],
        retrieved_items=[],
    )

    assert response == "Recovered via PowerShell"


def test_call_llm_via_client_supports_anthropic_compatible_transport(monkeypatch):
    import json
    import urllib.request

    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [
                        {"type": "text", "text": "OK from anthropic"},
                    ]
                }
            ).encode("utf-8")

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    agent = HypergraphAgent(
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
        llm_temperature=0.01,
    )

    response = agent._call_llm_via_client(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Reply with exactly OK"},
        ]
    )

    assert response == "OK from anthropic"
    assert captured["url"] == "https://api.minimaxi.com/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "MiniMax-M2.7"
    assert captured["body"]["system"] == "Be concise."
    assert captured["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Reply with exactly OK"}]}
    ]


def test_powershell_anthropic_transport_rejects_empty_content(monkeypatch):
    import subprocess

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*args, **kwargs):
        output_path = args[0][12]
        Path(output_path).write_text("", encoding="utf-8")
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    agent = HypergraphAgent(
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
    )

    with pytest.raises(RuntimeError, match="empty content"):
        agent._call_llm_via_powershell_transport([{"role": "user", "content": "Reply with exactly OK"}])


def test_generate_response_can_force_powershell_transport():
    agent = HypergraphAgent(
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
        llm_force_powershell_transport=True,
    )
    agent._call_llm_via_powershell_transport = lambda messages: "Forced PowerShell answer"

    response = agent.generate_response(
        user_input="Reply with exactly OK",
        memory_context="",
        messages=[{"role": "user", "content": "Reply with exactly OK"}],
        retrieved_items=[],
    )

    assert response == "Forced PowerShell answer"


def test_powershell_anthropic_transport_captures_raw_response(monkeypatch):
    import subprocess

    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["script_path"] = args[0][4]
        payload_path = args[0][10]
        output_path = args[0][12]
        captured["payload"] = Path(payload_path).read_text(encoding="utf-8")
        Path(output_path).write_text(
            '{"model":"MiniMax-M2.7","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"draft"},{"type":"text","text":"OK"}]}',
            encoding="utf-8",
        )
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    agent = HypergraphAgent(
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
    )

    output = agent._call_llm_via_powershell_transport([{"role": "user", "content": "Reply with exactly OK"}])
    debug = agent.get_last_llm_debug_snapshot()

    assert output == "OK"
    assert debug["transport"] == "anthropic-powershell"
    assert debug["content_types"] == ["thinking", "text"]
    assert debug["content_blocks"][0]["thinking"] == "draft"
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["args"][0][:4] == ["powershell", "-ExecutionPolicy", "Bypass", "-File"]
    assert captured["script_path"].endswith("minimax_payload_request.ps1")


def test_powershell_anthropic_transport_accepts_utf8_chinese_json(monkeypatch):
    import subprocess

    chinese_json = json.dumps(
        {
            "model": "MiniMax-M2.7",
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": "当前阶段：stable_v1 仍是正式 runtime。"},
            ],
        },
        ensure_ascii=False,
    )

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*args, **kwargs):
        output_path = args[0][12]
        Path(output_path).write_text(chinese_json, encoding="utf-8")
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    agent = HypergraphAgent(
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
    )

    output = agent._call_llm_via_powershell_transport([{"role": "user", "content": "请简述当前阶段"}])

    assert "stable_v1" in output


def test_mock_response_uses_session_recovery_details():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Remember that I am debugging a flaky deployment pipeline.")
    agent.process_turn("The failure happens during the database migration step.")

    result = agent.process_turn("After reloading this session, what issue was I working on?")

    lowered = result.assistant_response.lower()
    assert "flaky deployment pipeline" in lowered
    assert "database migration step" in lowered


def test_write_policy_classifies_structured_artifacts():
    policy = WritePolicy()

    assert policy._classify_kind("log: worker retry overflow", "user") == "log"
    assert policy._classify_kind("hypothesis: duplicate scheduler state", "user") == "hypothesis"
    assert policy._classify_kind("plan: patch dedupe before requeue", "user") == "plan"

    log_decision = policy.decide(
        "Log: worker-7 retry loop overflow after checkpoint resume.",
        role="user",
        turn_index=0,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    assert log_decision is not None
    assert log_decision.artifact_type == "log"
    assert log_decision.artifact_id is not None
    assert log_decision.linked_task in {"worker scheduler", "retry loop"}
    assert log_decision.node_type == "log"
    assert log_decision.hyperedge_type == "evidence_hyperedge"
    assert log_decision.hyperedge_id is not None


def test_mock_response_uses_artifact_chain_details():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker-7 retry loop overflow after checkpoint resume.")
    agent.process_turn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice.")
    agent.process_turn("Plan: inspect retry state restore, compare checkpoint IDs, then patch dedupe before requeue.")

    result = agent.process_turn("Return to the scheduler incident. What should I inspect first?")

    lowered = result.assistant_response.lower()
    assert "retry state restore" in lowered
    assert "checkpoint ids" in lowered
    assert "dedupe" in lowered


def test_turn_log_persists_artifact_metadata():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker-7 retry loop overflow after checkpoint resume.")
    entry = agent.turn_log[-1]
    writes = entry["writes"]

    assert writes
    assert writes[0]["artifact_type"] == "log"
    assert writes[0]["artifact_id"] is not None
    assert writes[0]["linked_task"] in {"worker scheduler", "retry loop"}
    assert writes[0]["node_type"] == "log"
    assert writes[0]["hyperedge_type"] == "evidence_hyperedge"
    assert writes[0]["hyperedge_id"] is not None


def test_artifact_relations_link_log_hypothesis_and_plan():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker.")
    agent.process_turn("Plan: inspect cursor reset, verify reconnect ordering, then patch stale cursor cleanup.")

    log_write = agent.turn_log[0]["writes"][0]
    hypothesis_write = agent.turn_log[1]["writes"][0]
    plan_write = agent.turn_log[2]["writes"][0]

    assert log_write["artifact_type"] == "log"
    assert hypothesis_write["artifact_type"] == "hypothesis"
    assert hypothesis_write["relation_type"] == "derived_from"
    assert hypothesis_write["parent_artifact_id"] == log_write["artifact_id"]
    assert plan_write["artifact_type"] == "plan"
    assert plan_write["relation_type"] == "tests"
    assert plan_write["parent_artifact_id"] == hypothesis_write["artifact_id"]
    assert log_write["hyperedge_type"] == "evidence_hyperedge"
    assert hypothesis_write["hyperedge_type"] == "evidence_hyperedge"
    assert log_write["hyperedge_id"] == hypothesis_write["hyperedge_id"]
    assert plan_write["hyperedge_type"] == "change_hyperedge"
    assert plan_write["hyperedge_id"] != hypothesis_write["hyperedge_id"]


def test_agent_exposes_artifact_graph_view():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker.")
    agent.process_turn("Plan: inspect cursor reset, verify reconnect ordering, then patch stale cursor cleanup.")

    graph = agent.get_artifact_graph()
    node_types = {node["artifact_type"] for node in graph["nodes"]}
    edge_types = {edge["relation_type"] for edge in graph["edges"]}

    assert {"log", "hypothesis", "plan"} <= node_types
    assert "derived_from" in edge_types
    assert "tests" in edge_types


def test_agent_exposes_hypergraph_view():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker.")
    agent.process_turn("Plan: inspect cursor reset, verify reconnect ordering, then patch stale cursor cleanup.")

    graph = agent.get_hypergraph_view()

    assert graph["nodes"]
    assert graph["hyperedges"]
    assert any(node["node_type"] == "log" for node in graph["nodes"])
    assert any(edge["relation_type"] == "derived_from" for edge in graph["edges"])
    assert any(h["hyperedge_type"] in {"evidence_hyperedge", "change_hyperedge"} for h in graph["hyperedges"])
    evidence_hyperedges = [h for h in graph["hyperedges"] if h["hyperedge_type"] == "evidence_hyperedge"]
    assert any(len(h["member_node_ids"]) >= 2 for h in evidence_hyperedges)
    assert any(h["status"] == "active" for h in graph["hyperedges"])
    assert any(node["confidence_tag"] in {"verified", "tentative", "speculative"} for node in graph["nodes"])
    assert any(h["confidence_summary"] in {"verified", "tentative", "speculative", "contradicted"} for h in graph["hyperedges"])


def test_hypergraph_view_marks_conflicted_evidence_hyperedges():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker.")
    agent.process_turn("Hypothesis: the reconnect path is timing out because cursor reset happens too early.")
    graph = agent.get_hypergraph_view()

    evidence_hyperedges = [h for h in graph["hyperedges"] if h["hyperedge_type"] == "evidence_hyperedge"]
    assert any(h["status"] == "conflicted" for h in evidence_hyperedges)


def test_retrieval_policy_exposes_conflicted_hyperedge_status():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker.")
    agent.process_turn("Hypothesis: the reconnect path is timing out because cursor reset happens too early.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Return to the profile-sync incident and inspect the root cause.",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=5,
    )

    conflicted_items = [item for item in items if item.hyperedge_status == "conflicted"]
    assert conflicted_items
    assert any(item.hyperedge_type == "evidence_hyperedge" for item in conflicted_items)


def test_retrieval_policy_exposes_confidence_tags():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path might leave stale cursor state in the sync worker.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Return to the profile-sync incident and inspect the root cause.",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=5,
    )

    assert any(item.confidence_tag == "verified" for item in items)
    assert any(item.confidence_tag in {"tentative", "speculative"} for item in items)


def test_write_policy_groups_same_task_evidence_into_shared_hyperedge():
    policy = WritePolicy()

    log_decision = policy.decide(
        "Log: profile-sync job times out after redis reconnect.",
        role="user",
        turn_index=0,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    hypothesis_decision = policy.decide(
        "Hypothesis: the reconnect path leaves stale cursor state in the sync worker.",
        role="user",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert log_decision is not None
    assert hypothesis_decision is not None
    assert log_decision.hyperedge_type == "evidence_hyperedge"
    assert hypothesis_decision.hyperedge_type == "evidence_hyperedge"
    assert log_decision.hyperedge_id == hypothesis_decision.hyperedge_id


def test_write_policy_infers_confidence_tags():
    policy = WritePolicy()

    log_decision = policy.decide(
        "Log: profile-sync job times out after redis reconnect.",
        role="user",
        turn_index=0,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    hypothesis_decision = policy.decide(
        "Hypothesis: the reconnect path might leave stale cursor state in the sync worker.",
        role="user",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    contradicted_decision = policy.decide(
        "The earlier reconnect theory was wrong and ruled out after reproducing the issue.",
        role="user",
        turn_index=2,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert log_decision is not None and log_decision.confidence_tag == "verified"
    assert hypothesis_decision is not None and hypothesis_decision.confidence_tag == "speculative"
    assert contradicted_decision is not None and contradicted_decision.confidence_tag == "contradicted"


def test_write_policy_builds_decision_residue_nodes_in_shared_hyperedge():
    policy = WritePolicy()

    decision = policy.decide(
        "Decision: for the worker scheduler fix, patch dedupe before touching the retry policy.",
        role="user",
        turn_index=0,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    rationale = policy.decide(
        "Reason: duplicate backoff state is the current blocker and we should avoid widening the change.",
        role="user",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    constraint = policy.decide(
        "Constraint: avoid widening the change until scheduler dedupe is verified.",
        role="user",
        turn_index=2,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    alternative = policy.decide(
        "Alternative: for the worker scheduler, rather than changing retry policy first, keep the retry settings stable.",
        role="user",
        turn_index=3,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert decision is not None and decision.node_type == "decision"
    assert rationale is not None and rationale.node_type == "rationale"
    assert constraint is not None and constraint.node_type == "constraint"
    assert alternative is not None and alternative.node_type == "alternative"
    assert decision.hyperedge_type == "decision_hyperedge"
    assert rationale.hyperedge_type == "decision_hyperedge"
    assert constraint.hyperedge_type == "decision_hyperedge"
    assert alternative.hyperedge_type == "decision_hyperedge"
    assert decision.hyperedge_id == constraint.hyperedge_id == alternative.hyperedge_id


def test_memory_state_reports_artifact_graph_counts():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker.")
    agent.process_turn("The reconnect path theory was wrong and ruled out after reproducing the issue.")
    agent.process_turn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker.")
    state = agent.get_memory_state()

    assert state["runtime"]["artifact_nodes"] >= 1
    assert state["runtime"]["artifact_edges"] >= 0
    assert state["runtime"]["conflict_hyperedges"] >= 1


def test_turn_log_links_decision_residue_relations():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy.")
    agent.process_turn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change.")
    agent.process_turn("Constraint: avoid widening the change until scheduler dedupe is verified.")
    agent.process_turn("Alternative: for the worker scheduler, rather than changing retry policy first, keep the retry settings stable.")

    decision_write = agent.turn_log[0]["writes"][0]
    rationale_write = agent.turn_log[1]["writes"][0]
    constraint_write = agent.turn_log[2]["writes"][0]
    alternative_write = agent.turn_log[3]["writes"][0]

    assert decision_write["node_type"] == "decision"
    assert rationale_write["relation_type"] == "rationale_for"
    assert rationale_write["parent_artifact_id"] == decision_write["artifact_id"]
    assert constraint_write["relation_type"] == "constrains"
    assert constraint_write["parent_artifact_id"] == decision_write["artifact_id"]
    assert alternative_write["relation_type"] == "alternative_to"
    assert alternative_write["parent_artifact_id"] == decision_write["artifact_id"]


def test_turn_log_links_verified_result_back_to_plan_or_decision():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: ship only the scheduler dedupe patch in tonight's hotfix.")
    agent.process_turn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the hotfix.")
    agent.process_turn("Verified: scheduler dedupe looks stable in staging and rollback notes are now complete.")

    verified_write = agent.turn_log[2]["writes"][0]

    assert verified_write["task_phase"] == "verification"
    assert verified_write["relation_type"] == "verifies"
    assert verified_write["parent_artifact_id"] is not None


def test_agent_exposes_decision_residue_units():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy.")
    agent.process_turn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change.")
    agent.process_turn("Constraint: avoid widening the change until scheduler dedupe is verified.")
    agent.process_turn("Alternative: for the worker scheduler, rather than changing retry policy first, keep the retry settings stable.")

    graph = agent.get_hypergraph_view()
    state = agent.get_memory_state()

    assert graph["decision_residues"]
    residue = graph["decision_residues"][0]
    assert residue["dominant_decision_node_id"] is not None
    assert residue["decision_node_ids"]
    assert residue["rationale_node_ids"]
    assert residue["constraint_node_ids"]
    assert residue["alternative_node_ids"]
    assert state["runtime"]["decision_residues"] >= 1


def test_write_policy_builds_procedural_nodes_in_shared_hyperedge():
    policy = WritePolicy()

    checklist = policy.decide(
        "Release handoff checklist: confirm staging verification, rollback notes, and migration notes before handoff.",
        role="user",
        turn_index=0,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    template = policy.decide(
        "Review summary template: start with a diff-style summary, then list risks and follow-up.",
        role="user",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert checklist is not None and checklist.node_type == "checklist"
    assert checklist.procedure_type == "release_handoff_checklist"
    assert checklist.hyperedge_type == "procedure_hyperedge"
    assert checklist.reusability_class in {"project_reusable", "cross_task_reusable"}
    assert template is not None and template.node_type == "template"
    assert template.procedure_type == "review_summary_template"
    assert template.hyperedge_type == "procedure_hyperedge"
    assert template.reusability_class == "cross_task_reusable"


def test_write_policy_prefers_explicit_procedure_nodes_over_constraint_keywords():
    policy = WritePolicy()

    procedure = policy.decide(
        "Release packet procedure: start with a diff-style summary, then list committed scope, deferred retry policy follow-up, validated checks, avoid widening the change, and the ship call.",
        role="user",
        turn_index=0,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert procedure is not None
    assert procedure.node_type == "template"
    assert procedure.procedure_type == "review_summary_template"
    assert procedure.hyperedge_type == "procedure_hyperedge"


def test_turn_log_links_procedural_relations():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the hotfix.")
    agent.process_turn("Release handoff checklist: confirm staging verification, rollback notes, and migration notes before handoff.")

    plan_write = agent.turn_log[0]["writes"][0]
    checklist_write = agent.turn_log[1]["writes"][0]

    assert plan_write["node_type"] == "plan"
    assert checklist_write["node_type"] == "checklist"
    assert checklist_write["relation_type"] == "procedure_for"
    assert checklist_write["parent_artifact_id"] == plan_write["artifact_id"]
    assert checklist_write["procedure_type"] == "release_handoff_checklist"


def test_agent_exposes_procedure_residue_units():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Task: prepare the release handoff for the scheduler hotfix.")
    agent.process_turn("Release handoff checklist: confirm staging verification, rollback notes, and migration notes before handoff.")
    agent.process_turn("Review summary template: start with a diff-style summary, then list risks and follow-up.")

    graph = agent.get_hypergraph_view()
    state = agent.get_memory_state()

    assert graph["procedure_residues"]
    residue = graph["procedure_residues"][0]
    assert residue["dominant_procedure_node_id"] is not None
    assert residue["procedure_node_ids"]
    assert residue["procedure_types"]
    assert state["runtime"]["procedure_residues"] >= 1


def test_runtime_retrieval_prioritizes_procedure_residue_for_procedure_queries():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Task: prepare the release handoff for the scheduler hotfix.")
    agent.process_turn("Release handoff checklist: confirm staging verification, rollback notes, and migration notes before handoff.")
    agent.process_turn("Remember that I prefer concise answers.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Before handoff, what checklist still applies?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=3,
    )

    node_types = [item.node_type for item in items]
    contents = [item.content.lower() for item in items]
    assert "checklist" in node_types
    assert any("staging verification" in content for content in contents)
    assert any("rollback notes" in content for content in contents)


def test_runtime_retrieval_prioritizes_review_template_for_review_format_queries():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Plan: rerun scheduler resume tests, then prepare the release review handoff.")
    agent.process_turn("Review summary template: start with a diff-style summary, then list risks and follow-up.")
    agent.process_turn("Remember that I prefer concise answers.")

    items = agent.turn_processor.retrieval_policy.collect(
        "For the release review handoff, what review format should we use?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=3,
    )

    node_types = [item.node_type for item in items]
    contents = [item.content.lower() for item in items]
    assert "template" in node_types
    assert any("diff-style summary" in content for content in contents)
    assert any("follow-up" in content for content in contents)


def test_write_policy_infers_task_phase_for_progress_states():
    policy = WritePolicy()

    log_decision = policy.decide(
        "Log: worker scheduler duplicate backoff state still appears after checkpoint resume.",
        role="user",
        turn_index=0,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    decision = policy.decide(
        "Decision: for the worker scheduler fix, patch dedupe before touching the retry policy.",
        role="user",
        turn_index=1,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    plan = policy.decide(
        "Plan: inspect retry state restore, compare checkpoint IDs, then patch dedupe before requeue.",
        role="user",
        turn_index=2,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    verified = policy.decide(
        "Verified: scheduler dedupe patch reproduces cleanly and validation passed.",
        role="user",
        turn_index=3,
        k=3,
        L=2,
        embedding_mapper=None,
    )
    closed = policy.decide(
        "The scheduler incident is resolved and the rollout is completed.",
        role="user",
        turn_index=4,
        k=3,
        L=2,
        embedding_mapper=None,
    )

    assert log_decision is not None and log_decision.task_phase == "analysis"
    assert decision is not None and decision.task_phase == "decision"
    assert plan is not None and plan.task_phase == "implementation"
    assert verified is not None and verified.task_phase == "verification"
    assert closed is not None and closed.task_phase == "closure"


def test_hypergraph_view_exposes_phase_summary():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy.")
    agent.process_turn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change.")
    agent.process_turn("Plan: inspect retry state restore, compare checkpoint IDs, then patch dedupe before requeue.")

    graph = agent.get_hypergraph_view()

    assert any(node.get("task_phase") in {"decision", "implementation"} for node in graph["nodes"])
    assert any(hyperedge.get("phase_summary") in {"implementation", "decision"} for hyperedge in graph["hyperedges"])


def test_get_working_set_surfaces_active_task_decisions_and_procedures():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: keep the scheduler fix scoped to patch dedupe only.")
    agent.process_turn("Plan: patch dedupe, rerun scheduler resume tests, then prepare the release review handoff.")
    agent.process_turn("Review summary template: start with a diff-style summary, then list risks and follow-up.")
    agent.process_turn("Verified: scheduler resume tests are green after the dedupe patch.")

    working_set = agent.get_working_set()

    assert working_set["linked_task"]
    assert working_set["active_nodes"]
    assert working_set["next_step_candidates"]
    assert working_set["active_decisions"]
    assert working_set["active_procedures"]


def test_query_current_task_state_reports_phase_and_status():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("The blocker is that rollback notes for the old retry policy are still incomplete.")
    agent.process_turn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the hotfix.")

    state = agent.query_current_task_state()

    assert state["linked_task"]
    assert state["blocker_count"] >= 1
    assert state["next_step_count"] >= 1
    assert state["status"] == "blocked"
    assert state["phase_summary"] in {"implementation", "analysis"}


def test_query_dominant_conflict_returns_dominant_and_contradicted_hypotheses():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush.")
    agent.process_turn("Hypothesis: cursor reset timing is causing the timeout in the sync worker.")
    agent.process_turn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout.")
    agent.process_turn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.")

    conflict = agent.query_dominant_conflict()

    assert conflict is not None
    assert "stale cursor state" in (conflict["dominant_content"] or "").lower()
    assert conflict["contradicted_hypothesis_count"] >= 1
    assert any("cursor reset timing" in str(content).lower() for content in conflict["contradicted_contents"])


def test_query_decision_residue_returns_dominant_decision_summary():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy.")
    agent.process_turn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change.")
    agent.process_turn("Constraint: avoid widening the change until scheduler dedupe is verified.")
    agent.process_turn("Alternative: rather than changing retry policy first, keep the retry settings stable.")

    residues = agent.query_decision_residue()

    assert residues
    assert any("patch dedupe" in str(item["dominant_decision_content"]).lower() for item in residues)
    assert any(item["constraint_count"] >= 1 for item in residues)


def test_query_applicable_procedures_returns_active_procedure_summary():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Plan: rerun scheduler resume tests, then prepare the release review handoff.")
    agent.process_turn("Review summary template: start with a diff-style summary, then list risks and follow-up.")

    procedures = agent.query_applicable_procedures()

    assert procedures
    assert any("review summary template" in str(item["dominant_procedure_content"]).lower() for item in procedures)
    assert any("review_summary_template" in item["procedure_types"] for item in procedures)


def test_query_handoff_bundle_combines_conflict_procedure_and_evidence():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush.")
    agent.process_turn("Hypothesis: cursor reset timing is causing the timeout in the sync worker.")
    agent.process_turn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout.")
    agent.process_turn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.")
    agent.process_turn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff.")
    agent.process_turn("Incident closeout checklist: verify reconnect ordering, confirm staging stability, then mark ready to close.")
    agent.process_turn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging.")

    bundle = agent.query_handoff_bundle()

    assert bundle["linked_task"]
    assert bundle["dominant_conflict"] is not None
    assert bundle["applicable_procedures"]
    assert any("reconnect ordering" in str(item).lower() for item in bundle["evidence"])
    assert bundle["ready_signals"]


def test_runtime_retrieval_prioritizes_decision_residue_for_decision_queries():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy.")
    agent.process_turn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change.")
    agent.process_turn("Constraint: avoid widening the change until scheduler dedupe is verified.")
    agent.process_turn("Alternative: for the worker scheduler, rather than changing retry policy first, keep the retry settings stable.")
    agent.process_turn("Remember that I prefer concise answers.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Return to the scheduler work. What decision did we already make and why?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=3,
    )

    node_types = [item.node_type for item in items]
    contents = [item.content.lower() for item in items]
    assert "decision" in node_types
    assert any(node_type in {"rationale", "constraint"} for node_type in node_types)
    assert any("patch dedupe" in content for content in contents)
    assert any("duplicate backoff state" in content or "avoid widening the change" in content for content in contents)


def test_runtime_retrieval_promotes_verified_handoff_evidence():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Decision: ship only the scheduler dedupe patch in tonight's hotfix.")
    agent.process_turn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the hotfix.")
    agent.process_turn("Verified: scheduler dedupe looks stable in staging and rollback notes are now complete.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Summarize the hotfix handoff: what did we ship, what did we reject, and are we ready to hand off?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=6,
    )

    assert any("stable in staging" in item.content.lower() for item in items)


def test_runtime_retrieval_keeps_verified_incident_handoff_evidence_when_query_overlap_is_weak():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush.")
    agent.process_turn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.")
    agent.process_turn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff.")
    agent.process_turn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Before handoff, what root cause are we carrying forward, what fix are we carrying forward, and what evidence should I cite?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=6,
    )

    contents = [item.content.lower() for item in items]
    assert any("reconnect ordering looks stable in staging" in content for content in contents)


def test_runtime_retrieval_keeps_verified_review_validation_evidence_for_handoff_queries():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Remember that when giving code changes, I want a short diff-style summary first.")
    agent.process_turn("Decision: keep the scheduler fix scoped to patch dedupe only.")
    agent.process_turn("Alternative: retry policy rewrite goes into a follow-up patch after the hotfix.")
    agent.process_turn("Constraint: avoid widening the change before tonight's rollout.")
    agent.process_turn("Plan: patch dedupe, rerun scheduler resume tests, then prepare the release review handoff.")
    agent.process_turn("Verified: scheduler resume tests are green after the dedupe patch.")

    items = agent.turn_processor.retrieval_policy.collect(
        "For the release review handoff, what did we commit to, what stays out of scope, and what validation already passed?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=6,
    )

    contents = [item.content.lower() for item in items]
    assert any("scheduler resume tests are green" in content for content in contents)


def test_langchain_wrapper_updates_agent_memory_runtime():
    memory = HypergraphMemoryAgent(k=3, L=2, use_embeddings=False)
    memory.save_context(
        {"input": "Remember that I prefer concise answers."},
        {"response": "Understood."},
    )

    variables = memory.load_memory_variables({})
    lowered_history = variables["history"].lower()
    content_values = " ".join(memory.agent.memory.content_map.values()).lower()

    assert "prefer concise answers" in lowered_history
    assert "prefer concise answers" in content_values
    assert memory.agent.turn_log


def test_retrieval_policy_expands_parent_artifact_relations():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect.")
    agent.process_turn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker.")
    agent.process_turn("Plan: inspect cursor reset, verify reconnect ordering, then patch stale cursor cleanup.")

    policy = RetrievalPolicy()
    items = policy.collect(
        "Return to the profile-sync incident and continue the plan.",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=6,
    )

    contents = [item.content.lower() for item in items]
    assert any("cursor reset" in content for content in contents)
    assert any("stale cursor state" in content for content in contents)


def test_retrieval_policy_defaults_to_hyperedge_expansion():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Remember that when debugging, I want the root cause hypothesis first.")
    agent.process_turn("I am investigating a flaky cache invalidation bug in the user profile service.")
    agent.process_turn("The bug appears after deploys when stale profile data persists.")
    agent.process_turn("Also remind me to book train tickets tomorrow.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Return to the bug and tell me what to investigate first.",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=3,
    )

    contents = [item.content.lower() for item in items]
    assert any("root cause hypothesis first" in content for content in contents)
    assert any("flaky cache invalidation bug" in content for content in contents)
    assert any("stale profile data" in content for content in contents)
    assert not any("book train tickets" in content for content in contents)


def test_runtime_retrieval_demotes_explicitly_contradicted_items():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume.")
    agent.process_turn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler.")
    agent.process_turn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue.")
    agent.process_turn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Back to the scheduler incident. Which explanation still looks strongest?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=2,
    )

    contents = [item.content.lower() for item in items]
    assert any("duplicate backoff state" in content for content in contents)
    assert not any("was wrong and ruled out" in content for content in contents)


def test_contradiction_note_links_to_prior_hypothesis():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume.")
    agent.process_turn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler.")
    agent.process_turn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue.")

    contradiction_write = agent.turn_log[-1]["writes"][0]
    prior_hypothesis = agent.turn_log[1]["writes"][0]

    assert contradiction_write["relation_type"] == "contradicts"
    assert contradiction_write["parent_artifact_id"] == prior_hypothesis["artifact_id"]


def test_hypergraph_view_exposes_contradiction_edges():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume.")
    agent.process_turn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler.")
    agent.process_turn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue.")

    graph = agent.get_hypergraph_view()

    assert any(edge["relation_type"] == "contradicts" for edge in graph["edges"])


def test_hypergraph_view_derives_conflict_hyperedges():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume.")
    agent.process_turn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler.")
    agent.process_turn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue.")
    agent.process_turn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice.")

    graph = agent.get_hypergraph_view()
    conflict_units = graph["conflict_hyperedges"]

    assert conflict_units
    assert any(unit["status"] == "active_conflict" for unit in conflict_units)
    assert any(unit["dominant_node_id"] is not None for unit in conflict_units)
    assert any(unit["contradicted_node_ids"] for unit in conflict_units)
    assert any(unit["active_hypothesis_node_ids"] for unit in conflict_units)


def test_runtime_retrieval_demotes_contradicted_hypothesis_target():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume.")
    agent.process_turn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler.")
    agent.process_turn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue.")
    agent.process_turn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Back to the scheduler incident. Which explanation still looks strongest?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=2,
    )

    contents = [item.content.lower() for item in items]
    assert any("duplicate scheduler state" in content for content in contents)
    assert not any("stale checkpoint ids are causing" in content for content in contents)


def test_runtime_retrieval_prefers_active_conflict_hypothesis():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume.")
    agent.process_turn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler.")
    agent.process_turn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue.")
    agent.process_turn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Back to the scheduler incident. Which root cause should I focus on now?",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=1,
    )

    assert items
    assert "duplicate scheduler state" in items[0].content.lower()
    assert items[0].node_type == "hypothesis"


def test_runtime_retrieval_marks_dominant_conflict_candidate_above_contradicted_one():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: worker scheduler duplicate backoff state still appears after checkpoint resume.")
    agent.process_turn("Hypothesis: stale checkpoint IDs are causing the duplicate backoff state in the worker scheduler.")
    agent.process_turn("The stale checkpoint IDs theory for the worker scheduler was wrong and ruled out after reproducing the issue.")
    agent.process_turn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Return to the worker scheduler root cause and name the strongest hypothesis.",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=3,
    )

    active_index = next(
        index for index, item in enumerate(items)
        if "duplicate scheduler state" in item.content.lower()
    )
    contradicted_index = next(
        index for index, item in enumerate(items)
        if "stale checkpoint ids are causing" in item.content.lower()
    )

    assert active_index < contradicted_index


def test_debugging_query_keeps_conflict_dominant_hypothesis_and_backing_log():
    agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
    agent.process_turn("Log: profile-sync job times out after redis reconnect and the worker stalls before final checkpoint flush.")
    agent.process_turn("Hypothesis: cursor reset timing is causing the timeout in the sync worker.")
    agent.process_turn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout.")
    agent.process_turn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush.")

    items = agent.turn_processor.retrieval_policy.collect(
        "Return to the profile-sync incident and name the strongest remaining root cause.",
        memory=agent.memory,
        turn_log=agent.turn_log,
        embedding_mapper=None,
        top_k=2,
    )

    contents = [item.content.lower() for item in items]
    assert any("stale cursor state remains" in content for content in contents)
    assert any("times out after redis reconnect" in content for content in contents)


def test_powershell_script_runner_preserves_utf8_script_content(monkeypatch):
    import subprocess

    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["script_path"] = args[0][4]
        payload_path = args[0][10]
        output_path = args[0][12]
        captured["payload"] = Path(payload_path).read_text(encoding="utf-8")
        Path(output_path).write_text(
            '{"content":[{"type":"text","text":"当前阶段：stable_v1 仍是正式 runtime。"}]}',
            encoding="utf-8",
        )
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    agent = HypergraphAgent(
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
    )

    output = agent._call_llm_via_powershell_anthropic_transport(
        [{"role": "user", "content": "请简述当前阶段。"}]
    )

    assert output == "当前阶段：stable_v1 仍是正式 runtime。"
    assert captured["args"][0][:4] == ["powershell", "-ExecutionPolicy", "Bypass", "-File"]
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["script_path"].endswith("minimax_payload_request.ps1")


def test_powershell_anthropic_transport_accepts_utf8_chinese_json_override(monkeypatch):
    import subprocess

    chinese_json = json.dumps(
        {
            "model": "MiniMax-M2.7",
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": "当前阶段：stable_v1 仍是正式 runtime。"},
            ],
        },
        ensure_ascii=False,
    )

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*args, **kwargs):
        output_path = args[0][12]
        Path(output_path).write_text(chinese_json, encoding="utf-8")
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    agent = HypergraphAgent(
        use_embeddings=False,
        llm_api_key="test-key",
        llm_model="MiniMax-M2.7",
        llm_base_url="https://api.minimaxi.com/anthropic",
    )

    output = agent._call_llm_via_powershell_transport([{"role": "user", "content": "Reply with exactly OK"}])

    assert output == "当前阶段：stable_v1 仍是正式 runtime。"
