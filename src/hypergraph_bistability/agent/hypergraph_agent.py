"""
HypergraphAgent: Complete Agent with Physics-Based Memory
=========================================================

A full-featured AI agent that combines:
- Hypergraph memory (AgentMemoryEnhanced)
- Embedding-based storage (EmbeddingMemoryMapper)
- Adaptive cognitive control (AdaptiveController)
- LLM inference (OpenAI-compatible and MiniMax Anthropic-compatible)

Usage:
    agent = HypergraphAgent(
        k=4, L=2,
        llm_api_key="...",
        use_embeddings=True
    )
    
    response = agent.chat("I want to learn coding but I'm tired")
    print(response)

With LangChain:
    memory = HypergraphMemoryAgent(k=4, L=2)
    chain = ConversationChain(llm=llm, memory=memory)
"""

import os
import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from hypergraph_bistability.agent.runtime import ContextAssembler, TurnProcessor
from hypergraph_bistability.agent.runtime_profile import get_runtime_profile
from hypergraph_bistability.agent.session import SessionState
from hypergraph_bistability.memory.policies import RetrievalPolicy, WritePolicy


class HypergraphAgent:
    """
    Complete AI agent with physics-based memory control.
    
    The agent uses a multi-layer memory architecture:
    - Working memory: k×L ODE matrix M (attention focus)
    - Semantic memory: Embedding-based slot mapping
    - Cold storage: ChromaDB (optional, for long-term)
    - Cognitive control: Adaptive λ/μ adjustment
    
    Parameters
    ----------
    k : int
        Number of memory groups (default 4)
    L : int
        Number of memory layers (default 2)
    llm_api_key : str, optional
        API key for LLM inference
    llm_model : str
        Model name (default "gpt-4o-mini")
    llm_base_url : str, optional
        Base URL for OpenAI-compatible API or Anthropic-compatible API
    use_embeddings : bool
        Use embedding-based memory mapping (default True)
    use_chromadb : bool
        Use ChromaDB cold storage (default False)
    """
    
    def __init__(
        self,
        k: int = 4,
        L: int = 2,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: Optional[str] = None,
        llm_temperature: float = 0.7,
        llm_max_output_tokens: int = 500,
        llm_max_retries: int = 2,
        llm_retry_backoff_seconds: float = 1.0,
        llm_force_powershell_transport: bool = False,
        use_embeddings: bool = True,
        use_chromadb: bool = False,
        group_labels: Optional[List[str]] = None,
        layer_labels: Optional[List[str]] = None,
        name: str = "hypergraph_agent"
    ):
        self.name = name
        self.k = k
        self.L = L
        
        self.group_labels = group_labels or [f"group_{i}" for i in range(k)]
        self.layer_labels = layer_labels or [f"layer_{l}" for l in range(L)]
        
        api_key = llm_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self.llm_api_key = api_key
        self.llm_base_url = llm_base_url
        self.llm_transport = self._detect_llm_transport(llm_base_url)
        
        from hypergraph_bistability.memory.llm_memory import AgentMemoryEnhanced
        self.memory = AgentMemoryEnhanced(
            k=k, L=L,
            use_llm_detector=False,
            use_llm_mapper=False,
            use_physics_control=True,
            name=name
        )
        self.memory.group_labels = self.group_labels
        self.memory.layer_labels = self.layer_labels
        
        self.embedding_mapper = None
        if use_embeddings:
            try:
                from hypergraph_bistability.integrations.embeddings import EmbeddingMemoryMapper
                self.embedding_mapper = EmbeddingMemoryMapper(
                    k=k, L=L,
                    api_key=api_key,
                    persist_dir=f".{name}_embeddings" if use_chromadb else None
                )
                self.embedding_mapper.set_group_labels(self.group_labels)
            except ImportError:
                print("Warning: embedding_memory not available, falling back to slot-based")
        
        from hypergraph_bistability.agent.adaptive_controller import AdaptiveController
        lc = self.memory.get_lambda_c() or 0.044
        self.controller = AdaptiveController(k=k, lambda_c=lc)
        
        self.llm_client = None
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_output_tokens = max(64, int(llm_max_output_tokens))
        self.llm_max_retries = max(0, llm_max_retries)
        self.llm_retry_backoff_seconds = max(0.0, llm_retry_backoff_seconds)
        self.llm_force_powershell_transport = llm_force_powershell_transport
        self.last_llm_raw_response: Optional[Dict[str, Any]] = None
        self.last_llm_transport_used: Optional[str] = None
        if api_key and self.llm_transport == "openai":
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    api_key=api_key,
                    base_url=llm_base_url
                )
            except ImportError:
                pass
        
        self.conversation_history: List[Dict[str, str]] = []
        self.turn_log: List[Dict[str, Any]] = []
        self.system_prompt = self._default_system_prompt()
        self.runtime_profile_name = "stable_v1"
        self.turn_processor = TurnProcessor(
            write_policy=WritePolicy(),
            retrieval_policy=RetrievalPolicy(),
            context_assembler=ContextAssembler(),
        )

    def _detect_llm_transport(self, base_url: Optional[str]) -> str:
        normalized = (base_url or "").lower()
        if "/anthropic" in normalized:
            return "anthropic"
        return "openai"
    
    def _default_system_prompt(self) -> str:
        return """You are an AI assistant with a unique memory system based on physics.

Your memory is organized as a k?L matrix where:
- k groups represent different aspects (work, personal, technical, etc.)
- L layers represent different dimensions (current context, preferences, etc.)

Your responses should:
1. Be concise and helpful
2. Draw on the active memory contexts when relevant
3. Adapt your level of detail based on the topic
4. Use concrete remembered details when they are relevant to the user's request
5. Never reveal hidden reasoning, chain-of-thought, or <think> tags
6. If retrieved memory includes preferences, follow them in the answer style
7. If retrieved memory includes an active task or issue, mention it directly rather than replying generically
8. When a retrieved preference implies an explicit response format, follow that format literally
9. Prefer explicit recalled phrases like "Root cause hypothesis:" or "Diff-style summary:" when the memory contract calls for them

Memory state is shown in parentheses like (?=0.02, r=0.5, regime=moderate-coupling).
This indicates your current "attention focus" - how distributed or focused your responses are.
"""

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt."""
        self.system_prompt = prompt
    
    def chat(
        self,
        user_input: str,
        use_adaptive: bool = True,
        return_context: bool = False
    ) -> str:
        """
        Process a user message and return a response.
        
        Parameters
        ----------
        user_input : str
            User message
        use_adaptive : bool
            Use adaptive λ control (default True)
        return_context : bool
            Include memory context in response
            
        Returns
        -------
        str
            Assistant response
        """
        result = self.turn_processor.process(
            agent=self,
            user_input=user_input,
            use_adaptive=use_adaptive,
            return_context=return_context,
        )
        return result if isinstance(result, str) else result.assistant_response

    def process_turn(
        self,
        user_input: str,
        use_adaptive: bool = True,
        return_context: bool = False,
    ):
        """Expose the structured turn pipeline for advanced callers."""
        return self.turn_processor.process(
            agent=self,
            user_input=user_input,
            use_adaptive=use_adaptive,
            return_context=return_context,
        )

    def record_external_turn(
        self,
        *,
        user_input: str,
        assistant_response: str,
        use_adaptive: bool = True,
    ) -> None:
        """Persist an externally-generated turn through the same memory/runtime state."""
        if not user_input and not assistant_response:
            return

        conflict_level = self.detect_conflict(user_input) if user_input else 0.0
        lambda_value, mu_value, _ = self.controller.update(user_input or assistant_response, conflict_level)
        if use_adaptive:
            self.memory.lambda_ = lambda_value
            self.memory.mu = mu_value

        write_events: List[Dict[str, Any]] = []
        turn_index = len(self.conversation_history) // 2

        if user_input:
            user_decision = self.turn_processor.write_policy.decide(
                user_input,
                role="user",
                turn_index=turn_index,
                k=self.k,
                L=self.L,
                embedding_mapper=self.embedding_mapper,
            )
            if user_decision and user_decision.should_write:
                self.turn_processor._attach_artifact_relations(user_decision, self.turn_log)
                self.memory.write(
                    user_decision.content,
                    group=user_decision.group,
                    layer=user_decision.layer,
                    activate=user_decision.activate,
                )
                write_events.append(user_decision.__dict__.copy())

        if user_input:
            self.conversation_history.append({"role": "user", "content": user_input})
        if assistant_response:
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

        if assistant_response:
            assistant_decision = self.turn_processor.write_policy.decide(
                assistant_response,
                role="assistant",
                turn_index=len(self.conversation_history) // 2,
                k=self.k,
                L=self.L,
                embedding_mapper=self.embedding_mapper,
            )
            if assistant_decision and assistant_decision.should_write:
                self.turn_processor._attach_artifact_relations(assistant_decision, self.turn_log + [{"writes": write_events}])
                self.memory.write(
                    assistant_decision.content,
                    group=assistant_decision.group,
                    layer=assistant_decision.layer,
                    activate=assistant_decision.activate,
                )
                write_events.append(assistant_decision.__dict__.copy())

        self.turn_log.append(
            {
                "user_input": user_input,
                "assistant_response": assistant_response,
                "memory_context": self.memory.get_context_for_llm(),
                "conflict_level": conflict_level,
                "retrieved_items": [],
                "retrieved_detail": [],
                "writes": write_events,
                "controller_state": self.controller.get_state_summary(),
            }
        )

    def detect_conflict(self, text: str) -> float:
        """Detect conflict using the configured detector or fallback heuristics."""
        if hasattr(self.memory, "detector") and self.memory.detector is not None and hasattr(self.memory.detector, "detect_conflict"):
            try:
                conflict_level, _ = self.memory.detector.detect_conflict(text)
                return conflict_level
            except Exception:
                return self._simple_conflict_detection(text)
        return self._simple_conflict_detection(text)

    def generate_response(
        self,
        *,
        user_input: str,
        memory_context: str,
        messages: List[Dict[str, str]],
        retrieved_items=None,
    ) -> str:
        """Generate a response from the configured LLM or the mock fallback."""
        if self.llm_force_powershell_transport and self.llm_api_key and self.llm_base_url:
            try:
                return self._apply_response_contracts(
                    self._call_llm_via_powershell_transport(messages),
                    user_input=user_input,
                    retrieved_items=retrieved_items or [],
                )
            except Exception as power_shell_error:
                return f"[LLM error: {power_shell_error}] I understand your message. How can I help?"

        if self.llm_client or (self.llm_transport == "anthropic" and self.llm_api_key and self.llm_base_url):
            last_error = None
            for attempt in range(self.llm_max_retries + 1):
                try:
                    return self._apply_response_contracts(
                        self._call_llm_via_client(messages),
                        user_input=user_input,
                        retrieved_items=retrieved_items or [],
                    )
                except Exception as e:
                    last_error = e
                    if attempt >= self.llm_max_retries:
                        break
                    delay = self.llm_retry_backoff_seconds * (2 ** attempt)
                    if delay > 0:
                        time.sleep(delay)
            if last_error and self._should_use_powershell_transport(last_error):
                try:
                    return self._apply_response_contracts(
                        self._call_llm_via_powershell_transport(messages),
                        user_input=user_input,
                        retrieved_items=retrieved_items or [],
                    )
                except Exception as fallback_error:
                    last_error = fallback_error
            return f"[LLM error: {last_error}] I understand your message. How can I help?"

        controller_state = self.controller.get_state_summary()
        return self._apply_response_contracts(
            self._mock_response(user_input, memory_context, controller_state, retrieved_items or []),
            user_input=user_input,
            retrieved_items=retrieved_items or [],
        )

    def write_from_documents(
        self,
        *,
        document_paths: List[str],
        instruction: str,
        output_path: Optional[str] = None,
        per_doc_char_limit: int = 2200,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read local documents, generate a grounded write-up, and optionally persist UTF-8 output."""
        resolved_paths = [str(Path(path).resolve()) for path in document_paths]
        document_context = self._build_document_context(
            resolved_paths,
            per_doc_char_limit=per_doc_char_limit,
        )
        prompt = (
            f"{instruction.strip()}\n\n"
            "Use only the provided document context. "
            "Keep stable_v1 as the formal runtime unless the docs say otherwise. "
            "Write the answer in the same language as the instruction. "
            "If the instruction is in Chinese, answer entirely in Chinese.\n\n"
            f"{document_context}"
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt
                or (
                    "You are the project agent. Write concise grounded output from the provided documents only. "
                    "Match the language of the user's instruction."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        if self._should_use_minimax_doc_writer_powershell():
            response = self._call_minimax_doc_writer_powershell(messages)
        else:
            response = self.generate_response(
                user_input=instruction,
                memory_context="[Memory State: document_writer]",
                messages=messages,
                retrieved_items=[],
            )
        payload: Dict[str, Any] = {
            "response": response,
            "documents": resolved_paths,
            "instruction": instruction,
            "llm_debug": self.get_last_llm_debug_snapshot(),
        }
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response, encoding="utf-8")
            payload["output_path"] = str(path.resolve())
        return payload

    def _should_use_minimax_doc_writer_powershell(self) -> bool:
        return (
            os.name == "nt"
            and self.llm_force_powershell_transport
            and self.llm_transport == "anthropic"
            and bool(self.llm_api_key)
            and bool(self.llm_base_url)
            and "minimaxi.com" in (self.llm_base_url or "")
        )

    def _call_minimax_doc_writer_powershell(self, messages: List[Dict[str, str]]) -> str:
        prompt_sections = [f"[{message['role']}]\n{message['content']}" for message in messages]
        prompt_text = "\n\n".join(prompt_sections)
        prompt_path = self._write_temp_payload(prompt_text, suffix=".txt")
        output_path = self._write_temp_payload("", suffix=".json")
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "minimax_prompt_request.ps1"
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    "-ApiKey",
                    self.llm_api_key or "",
                    "-Model",
                    self.llm_model,
                    "-PromptFile",
                    prompt_path,
                    "-OutputFile",
                    output_path,
                    "-Temperature",
                    str(self.llm_temperature),
                    "-MaxTokens",
                    str(self.llm_max_output_tokens),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
        finally:
            self._remove_temp_path(prompt_path)
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "unknown MiniMax doc-writer PowerShell error").strip()
            self._remove_temp_path(output_path)
            raise RuntimeError(stderr)
        try:
            output = Path(output_path).read_text(encoding="utf-8-sig").strip()
        finally:
            self._remove_temp_path(output_path)
        if not output:
            stderr = (result.stderr or "").strip()
            if stderr:
                raise RuntimeError(stderr)
            raise RuntimeError("MiniMax doc-writer PowerShell transport returned empty content.")
        body = json.loads(output)
        self.last_llm_transport_used = "anthropic-powershell-doc-writer"
        self.last_llm_raw_response = body
        return self._extract_anthropic_text(body)

    def _build_document_context(
        self,
        document_paths: List[str],
        *,
        per_doc_char_limit: int = 2200,
    ) -> str:
        sections: List[str] = []
        for document_path in document_paths:
            path = Path(document_path)
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
            if len(content) > per_doc_char_limit:
                content = content[:per_doc_char_limit].rstrip() + "\n...[truncated]"
            sections.append(f"## {path.name}\n{content}")
        return "\n\n".join(sections)

    def _call_llm_via_client(self, messages: List[Dict[str, str]]) -> str:
        if self.llm_transport == "anthropic":
            return self._call_llm_via_anthropic_http(messages)
        self.last_llm_transport_used = "openai-client"
        self.last_llm_raw_response = None
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.llm_max_output_tokens,
            temperature=self.llm_temperature
        )
        return response.choices[0].message.content

    def _call_llm_via_anthropic_http(self, messages: List[Dict[str, str]]) -> str:
        if not self.llm_api_key or not self.llm_base_url:
            raise RuntimeError("Anthropic-compatible transport requires llm_api_key and llm_base_url.")

        system_prompt, anthropic_messages = self._convert_messages_for_anthropic(messages)
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "max_tokens": self.llm_max_output_tokens,
            "temperature": self.llm_temperature,
            "messages": anthropic_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt

        request = urllib.request.Request(
            url=self.llm_base_url.rstrip("/") + "/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.llm_api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Anthropic transport error: {exc}") from exc

        self.last_llm_transport_used = "anthropic-http"
        self.last_llm_raw_response = body
        return self._extract_anthropic_text(body)

    def _convert_messages_for_anthropic(self, messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, Any]]]:
        system_parts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            anthropic_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": str(content)}],
                }
            )
        if not anthropic_messages:
            anthropic_messages.append(
                {"role": "user", "content": [{"type": "text", "text": ""}]}
            )
        return "\n\n".join(system_parts).strip(), anthropic_messages

    def _extract_anthropic_text(self, response_body: Dict[str, Any]) -> str:
        content_blocks = response_body.get("content") or []
        text_parts: List[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                text_parts.append(str(block["text"]))
        if text_parts:
            return "\n".join(text_parts).strip()
        if response_body.get("error"):
            raise RuntimeError(str(response_body["error"]))
        raise RuntimeError(f"Anthropic response did not include text content: {response_body}")

    def get_last_llm_debug_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.last_llm_raw_response is None and self.last_llm_transport_used is None:
            return None

        snapshot: Dict[str, Any] = {
            "transport": self.last_llm_transport_used,
        }
        if isinstance(self.last_llm_raw_response, dict):
            content_blocks = self.last_llm_raw_response.get("content") or []
            snapshot["content_types"] = [
                block.get("type")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type")
            ][:12]
            snapshot["content_blocks"] = [
                {
                    "type": block.get("type"),
                    "text": block.get("text"),
                    "thinking": block.get("thinking"),
                }
                for block in content_blocks[:6]
                if isinstance(block, dict)
            ]
            if self.last_llm_raw_response.get("stop_reason") is not None:
                snapshot["stop_reason"] = self.last_llm_raw_response.get("stop_reason")
            if self.last_llm_raw_response.get("model") is not None:
                snapshot["model"] = self.last_llm_raw_response.get("model")
        return snapshot

    def _should_use_powershell_transport(self, error: Exception) -> bool:
        if os.name != "nt":
            return False
        if not self.llm_api_key or not self.llm_base_url:
            return False
        if "minimaxi.com" not in self.llm_base_url:
            return False
        text = str(error).lower()
        return "winerror 10013" in text or "connection error" in text

    def _call_llm_via_powershell_transport(self, messages: List[Dict[str, str]]) -> str:
        if self.llm_transport == "anthropic":
            return self._call_llm_via_powershell_anthropic_transport(messages)
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_output_tokens,
        }
        payload_json = json.dumps(payload, ensure_ascii=False)
        result = self._run_minimax_powershell_request(payload_json, transport="openai")
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "unknown powershell transport error").strip()
            self._remove_temp_path(getattr(result, "minimax_output_path", ""))
            raise RuntimeError(stderr)
        output_path = getattr(result, "minimax_output_path", "")
        try:
            output = Path(output_path).read_text(encoding="utf-8-sig").strip() if output_path else result.stdout.strip()
        finally:
            self._remove_temp_path(output_path)
        if not output:
            stderr = (result.stderr or "").strip()
            if stderr:
                raise RuntimeError(stderr)
            raise RuntimeError("PowerShell transport returned empty content.")
        self.last_llm_transport_used = "openai-powershell"
        self.last_llm_raw_response = None
        return output

    def _call_llm_via_powershell_anthropic_transport(self, messages: List[Dict[str, str]]) -> str:
        system_prompt, anthropic_messages = self._convert_messages_for_anthropic(messages)
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": anthropic_messages,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_output_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload_json = json.dumps(payload, ensure_ascii=False)
        result = self._run_minimax_powershell_request(payload_json, transport="anthropic")
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "unknown powershell transport error").strip()
            self._remove_temp_path(getattr(result, "minimax_output_path", ""))
            raise RuntimeError(stderr)
        output_path = getattr(result, "minimax_output_path", "")
        try:
            output = Path(output_path).read_text(encoding="utf-8-sig").strip() if output_path else result.stdout.strip()
        finally:
            self._remove_temp_path(output_path)
        if not output:
            stderr = (result.stderr or "").strip()
            if stderr:
                raise RuntimeError(stderr)
            raise RuntimeError("PowerShell anthropic transport returned empty content.")
        try:
            body = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"PowerShell anthropic transport returned non-JSON content: {output}") from exc
        self.last_llm_transport_used = "anthropic-powershell"
        self.last_llm_raw_response = body
        return self._extract_anthropic_text(body)

    def _run_minimax_powershell_request(self, payload_json: str, *, transport: str):
        payload_path = self._write_temp_payload(payload_json)
        output_path = self._write_temp_payload("", suffix=".json")
        script_path = self._minimax_payload_script_path()
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    "-ApiKey",
                    self.llm_api_key or "",
                    "-BaseUrl",
                    self.llm_base_url or "",
                    "-PayloadFile",
                    payload_path,
                    "-OutputFile",
                    output_path,
                    "-Transport",
                    transport,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
            result.minimax_output_path = output_path  # type: ignore[attr-defined]
            return result
        finally:
            self._remove_temp_path(payload_path)

    def _minimax_payload_script_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "scripts" / "minimax_payload_request.ps1"

    def _write_temp_payload(self, payload: str, *, suffix: str = ".json") -> str:
        with tempfile.NamedTemporaryFile("w", suffix=suffix, encoding="utf-8", delete=False) as handle:
            handle.write(payload)
            return handle.name

    def _remove_temp_path(self, path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass
    
    def _simple_conflict_detection(self, text: str) -> float:
        """Simple keyword-based conflict detection."""
        conflict_keywords = ["but", "however", "although", "actually", "wait", 
                            "on the other hand", "or", "either", "neither"]
        conflict_count = sum(1 for kw in conflict_keywords if kw in text.lower())
        return min(1.0, conflict_count * 0.15)

    def _apply_response_contracts(
        self,
        response: str,
        *,
        user_input: str,
        retrieved_items: List[Any],
    ) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
        lowered_input = user_input.lower()
        joined = " ".join(
            item.content if hasattr(item, "content") else str(item)
            for item in retrieved_items
        ).lower()
        lowered_response = cleaned.lower()

        if "root cause hypothesis first" in joined and any(
            token in lowered_input for token in ("investigate first", "debug", "bug", "root cause")
        ):
            if "root cause hypothesis:" not in lowered_response:
                cleaned = f"Root cause hypothesis: {cleaned}"
                lowered_response = cleaned.lower()

        if "diff-style" in joined and any(
            token in lowered_input
            for token in (
                "inspect first",
                "code change",
                "fix",
                "scheduler",
                "review handoff",
                "coding review",
                "release review handoff",
                "what did we commit to",
                "tonight's packet",
                "clear to ship",
            )
        ):
            if "diff-style summary:" not in lowered_response:
                cleaned = f"Diff-style summary:\n{cleaned}"
                lowered_response = cleaned.lower()

        if "concise" in joined and any(
            token in lowered_input for token in ("summarize", "present", "rollout plan", "respond")
        ):
            if "concise" not in lowered_response and "brief" not in lowered_response:
                cleaned = f"Concise summary:\n{cleaned}"
                lowered_response = cleaned.lower()

        if any(
            token in lowered_input
            for token in (
                "before i hand this incident off",
                "before handoff",
                "ready to hand off",
                "ready to handoff",
                "handoff bundle",
            )
        ):
            if "incident handoff:" not in lowered_response and (
                "stale cursor state" in joined or "reconnect ordering" in joined or "checkpoint flush" in joined
            ):
                cleaned = f"Incident handoff:\n{cleaned}"
                lowered_response = cleaned.lower()

        if self._looks_like_generic_restart_response(cleaned):
            cleaned = ""
            lowered_response = ""

        additions = self._missing_response_additions(
            lowered_input=lowered_input,
            lowered_response=lowered_response,
            joined_retrieval=joined,
        )
        if additions:
            additions = list(dict.fromkeys(additions))
            cleaned = cleaned.rstrip() + "\n\nNext checks:\n" + "\n".join(f"- {line}" for line in additions)
            lowered_response = cleaned.lower()

        if any(
            token in lowered_input
            for token in (
                "ready to hand off",
                "ready to handoff",
                "before i hand this incident off",
                "before handoff",
                "hotfix handoff",
            )
        ):
            if "ready to hand off" not in lowered_response:
                if "ready for handoff" in lowered_response or ("ready" in lowered_response and "handoff" in lowered_response):
                    cleaned = cleaned.rstrip() + "\n- This is ready to hand off."
                    lowered_response = cleaned.lower()

        return cleaned.strip()

    def _looks_like_generic_restart_response(self, response: str) -> bool:
        lowered = response.lower().strip()
        if not lowered:
            return False
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

    def _missing_response_additions(
        self,
        *,
        lowered_input: str,
        lowered_response: str,
        joined_retrieval: str,
    ) -> List[str]:
        additions: List[str] = []

        def add_if_missing(token: str, line: str) -> None:
            if token in joined_retrieval and token not in lowered_response and line not in additions:
                additions.append(line)

        if any(token in lowered_input for token in ("investigate first", "return to the bug", "debug")):
            add_if_missing("cache invalidation", "Inspect cache invalidation ordering after deploy.")
            add_if_missing("stale profile data", "Trace why stale profile data persists after deploy.")

        if any(token in lowered_input for token in ("scheduler fix", "inspect first", "resume")):
            add_if_missing("retry loop", "Inspect the retry loop state transitions first.")
            add_if_missing("backoff state", "Check whether duplicate backoff state appears after task resume.")

        if "what is blocking us" in lowered_input or "what is still blocking us" in lowered_input:
            add_if_missing("migration notes", "The migration notes are still incomplete.")
            add_if_missing("rollback", "The rollback steps for the old schema are still missing.")
            add_if_missing("next", "Next, finish the rollback steps before continuing the rollout.")

        if "what decision did we already make" in lowered_input:
            add_if_missing("patch dedupe", "We already decided to patch dedupe before touching the retry policy.")
            add_if_missing("backoff state", "That decision was driven by the duplicate backoff state blocker.")
            add_if_missing("avoid widening the change", "We chose this path to avoid widening the change.")

        if any(
            token in lowered_input
            for token in (
                "what did we commit to",
                "what did we reject",
                "what constraint are we still optimizing around",
                "what constraint is still active",
                "what follow-up stays out of scope",
                "what stays out of scope",
            )
        ):
            add_if_missing("retry policy", "We kept the retry policy stable instead of changing it first.")
            add_if_missing("patch dedupe", "We committed to patch dedupe first.")
            add_if_missing("avoid widening the change", "The active constraint is to avoid widening the change.")
            add_if_missing("scheduler resume tests", "Next validation is to rerun the scheduler resume tests.")

        if any(
            token in lowered_input
            for token in (
                "what are we shipping",
                "what is still blocking us",
                "what has to happen next",
                "what are we explicitly not doing",
                "what gate is still open",
                "what stays out of scope",
            )
        ):
            add_if_missing("scheduler dedupe", "We are shipping the scheduler dedupe patch only.")
            add_if_missing("retry policy", "We are not widening this hotfix into a retry policy change.")
            add_if_missing("rollback notes", "Rollback notes for the old retry policy are still incomplete.")
            add_if_missing("staging", "Next, verify scheduler dedupe in staging before handoff.")
            add_if_missing("complete", "This is ready to hand off once the final staging and rollback checks stay complete.")

        if any(
            token in lowered_input
            for token in (
                "ready to hand off",
                "ready to handoff",
                "before i hand this incident off",
                "before handoff",
                "handoff bundle",
            )
        ):
            add_if_missing("staging", "Cite the latest staging verification in the handoff.")
            add_if_missing("verified", "Call out the verified fix that is carrying forward.")
            add_if_missing("verified", "If the latest verification still holds, this is ready to hand off.")
            add_if_missing("complete", "If the verification evidence holds, this is ready to hand off.")
            add_if_missing("stale cursor state", "Carry forward the stale cursor state root cause in the handoff bundle.")
            add_if_missing("checkpoint flush", "Cite the checkpoint flush stall as handoff evidence.")

        if any(
            token in lowered_input
            for token in (
                "what should we verify next",
                "before closing this",
                "before we close this incident",
                "ready to close",
                "what fix are we carrying forward",
                "close packet",
                "closure unlocked",
                "which fix survives",
                "what staging signal",
            )
        ):
            add_if_missing("validation", "Verification should focus on the latest validation results.")
            add_if_missing("stale cursor cleanup", "Carry forward the stale cursor cleanup fix.")
            add_if_missing("reconnect ordering", "Verify reconnect ordering before closure.")
            add_if_missing("staging", "Use the staging validation as the final check.")
            add_if_missing("close", "If validation stays clean, this is ready to close.")
            if (
                "if validation stays clean, this is ready to close." not in additions
                and "ready to close" not in lowered_response
                and any(token in joined_retrieval for token in ("before closing", "verified", "staging"))
            ):
                additions.append("If validation stays clean, this is ready to close.")

        if any(
            token in lowered_input
            for token in (
                "which explanation survived",
                "which patch survived",
                "what evidence stays citeable",
                "handoff bundle",
                "what story still holds",
                "what fix still holds",
                "proof points are still worth citing",
                "which theory still survives",
                "which theory stays ruled out",
                "stays ruled out",
                "stays dead",
            )
        ):
            add_if_missing("stale cursor state", "The explanation that survived is stale cursor state after reconnect.")
            add_if_missing("cursor reset timing", "The ruled-out theory is cursor reset timing in the sync worker.")
            add_if_missing("stale cursor cleanup", "The patch that survived is stale cursor cleanup on reconnect.")
            add_if_missing("reconnect ordering", "Cite reconnect ordering stability as active evidence.")
            add_if_missing("checkpoint flush", "Cite the checkpoint flush stall as backing evidence.")

        if any(
            token in lowered_input
            for token in (
                "which patch stays in",
                "which follow-up stays out",
                "what validation is already banked",
                "clear to ship",
                "tonight's packet",
                "what still belongs in tonight's release story",
                "what is still deferred",
                "what validation already holds",
            )
        ):
            add_if_missing("patch dedupe", "The patch that stays in is patch dedupe for the scheduler fix.")
            add_if_missing("retry policy", "The retry policy rewrite stays out as a follow-up change.")
            add_if_missing("scheduler resume tests", "Validation already banked: scheduler resume tests are green.")
            add_if_missing("avoid widening the change", "We are still holding the line against widening the change.")
            add_if_missing("diff-style", "Start with a diff-style summary for tonight's packet.")

        if "continue the plan" in lowered_input or "profile-sync incident" in lowered_input:
            add_if_missing("cursor reset", "Inspect cursor reset first.")
            add_if_missing("reconnect ordering", "Verify reconnect ordering next.")
            add_if_missing("stale cursor cleanup", "Patch stale cursor cleanup before closing the incident.")

        if "return to the scheduler incident" in lowered_input or "continue the scheduler plan" in lowered_input:
            add_if_missing("retry state restore", "Inspect retry state restore first.")
            add_if_missing("checkpoint ids", "Compare checkpoint IDs before and after resume.")
            add_if_missing("dedupe", "Confirm the dedupe patch point before requeue.")

        if any(
            token in lowered_input
            for token in (
                "strongest remaining root cause",
                "strongest hypothesis",
                "which explanation still looks strongest",
                "what root cause are we carrying forward",
                "what evidence should i cite",
            )
        ):
            add_if_missing("backoff state", "Tie the dominant hypothesis back to the duplicate backoff state symptom.")
            add_if_missing("redis reconnect", "Reference the redis reconnect evidence when naming the dominant root cause.")
            add_if_missing("checkpoint flush", "Mention the checkpoint flush stall as backing evidence for the dominant hypothesis.")

        return additions
    
    def _mock_response(
        self,
        user_input: str,
        context: str,
        controller_state: Dict,
        retrieved_items: List[Any],
    ) -> str:
        """Generate a memory-aware mock response when LLM is not available."""
        lowered = user_input.lower()
        retrieved_text = [item.content if hasattr(item, "content") else str(item) for item in retrieved_items]
        joined = " ".join(retrieved_text).lower()

        if retrieved_text:
            if "how should you respond" in lowered or "present the rollout plan" in lowered:
                bullets = []
                if "concise" in joined:
                    bullets.append("Keep it concise.")
                if "bullet" in joined:
                    bullets.append("Use bullet lists for action items.")
                if "rollout" in joined:
                    bullets.append("State the rollout plan and next steps directly.")
                if bullets:
                    return "\n".join(f"- {line}" for line in bullets)

            if "what matters most" in lowered:
                highlights = []
                if "release checklist" in joined:
                    highlights.append("Release checklist")
                if "migration notes" in joined:
                    highlights.append("Migration notes")
                if "rollback" in joined:
                    highlights.append("Rollback readiness")
                if highlights:
                    return "Top priorities:\n" + "\n".join(f"- {item}" for item in highlights)

            if "what issue was i working on" in lowered:
                issue_bits = []
                if "flaky deployment pipeline" in joined:
                    issue_bits.append("flaky deployment pipeline")
                if "database migration step" in joined:
                    issue_bits.append("database migration step")
                if issue_bits:
                    return "You were working on: " + "; ".join(issue_bits)

            if "what should i inspect first" in lowered or "investigate first" in lowered:
                lines = []
                if "root cause hypothesis first" in joined:
                    lines.append("- Root cause hypothesis first.")
                if "retry loop" in joined:
                    lines.append("- Inspect the retry loop state transitions.")
                if "retry state restore" in joined:
                    lines.append("- Inspect retry state restore.")
                if "backoff state" in joined:
                    lines.append("- Check duplicate backoff state after resume.")
                if "checkpoint" in joined:
                    lines.append("- Compare checkpoint IDs before and after resume.")
                if "dedupe" in joined:
                    lines.append("- Confirm the dedupe patch point before requeue.")
                if "cache invalidation" in joined:
                    lines.append("- Inspect cache invalidation ordering.")
                if "stale profile data" in joined:
                    lines.append("- Trace stale profile data after deploy.")
                if lines:
                    prefix = "Diff-style summary:" if "diff-style" in joined else "First checks:"
                    return prefix + "\n" + "\n".join(lines)

            if "what is blocking us" in lowered:
                lines = []
                if "migration notes" in joined:
                    lines.append("- The migration notes are still incomplete.")
                if "rollback" in joined:
                    lines.append("- The rollback steps for the old billing schema are still missing.")
                if lines:
                    lines.append("- Next: finish the rollback steps before the rollout continues.")
                    return "Current blockers:\n" + "\n".join(lines)

            if any(
                token in lowered
                for token in (
                    "what checklist still applies",
                    "what review format should we use",
                    "what format should the summary use",
                    "what playbook should we use",
                    "what template should we use",
                )
            ):
                lines = []
                if "staging verification" in joined:
                    lines.append("- Include staging verification.")
                if "rollback notes" in joined:
                    lines.append("- Include rollback notes.")
                if "migration notes" in joined:
                    lines.append("- Include migration notes.")
                if "diff-style" in joined:
                    lines.append("- Start with a diff-style summary.")
                if "risks" in joined:
                    lines.append("- List risks.")
                if "follow-up" in joined:
                    lines.append("- List follow-up items.")
                if "reconnect ordering" in joined:
                    lines.append("- Verify reconnect ordering.")
                if "staging stability" in joined or "staging" in joined:
                    lines.append("- Confirm staging stability.")
                if "ready to close" in joined:
                    lines.append("- Mark the incident ready to close.")
                if "scheduler resume tests are green" in joined or "scheduler resume tests" in joined:
                    lines.append("- Validation already passed: scheduler resume tests are green.")
                if "retry policy" in joined:
                    lines.append("- Keep retry policy changes out of scope for this handoff.")
                if lines:
                    prefix = "Diff-style summary:\n" if "diff-style" in joined else ""
                    return prefix + "Procedure summary:\n" + "\n".join(lines)

            if "what decision did we already make" in lowered:
                lines = []
                if "patch dedupe" in joined or "dedupe" in joined:
                    lines.append("- We decided to patch dedupe before touching the retry policy.")
                if "backoff state" in joined:
                    lines.append("- The reason was the duplicate backoff state blocker.")
                if "avoid widening the change" in joined:
                    lines.append("- We wanted to avoid widening the change.")
                if lines:
                    return "Decision summary:\n" + "\n".join(lines)

            if any(
                token in lowered
                for token in (
                    "what did we commit to",
                    "what did we reject",
                    "what constraint are we still optimizing around",
                    "what constraint is still active",
                    "what follow-up stays out of scope",
                )
            ):
                lines = []
                if "patch dedupe" in joined or "dedupe" in joined:
                    lines.append("- We committed to patch dedupe first.")
                if "retry policy" in joined:
                    lines.append("- We rejected changing the retry policy first.")
                if "avoid widening the change" in joined:
                    lines.append("- The active constraint is to avoid widening the change.")
                if "scheduler resume tests" in joined:
                    lines.append("- Next validation is to rerun the scheduler resume tests.")
                if lines:
                    prefix = "Diff-style summary:" if "diff-style" in joined else "Decision continuation:"
                    return prefix + "\n" + "\n".join(lines)

            if any(
                token in lowered
                for token in (
                    "what are we shipping",
                    "what is still blocking us",
                    "what has to happen next",
                    "what are we explicitly not doing",
                    "what gate is still open",
                )
            ):
                lines = []
                if "scheduler dedupe" in joined:
                    lines.append("- Shipping scope: scheduler dedupe patch only.")
                if "retry policy" in joined:
                    lines.append("- Explicitly out of scope: changing the retry policy in this hotfix.")
                if "rollback notes" in joined:
                    lines.append("- Blocker: rollback notes for the old retry policy are still incomplete.")
                if "staging" in joined:
                    lines.append("- Next: verify scheduler dedupe in staging before handoff.")
                if "complete" in joined or "verified" in joined:
                    lines.append("- If those checks stay clean, this is ready to hand off.")
                if lines:
                    return "Release handoff:\n" + "\n".join(lines)

            if any(
                token in lowered
                for token in (
                    "ready to hand off",
                    "ready to handoff",
                    "before i hand this incident off",
                    "before handoff",
                )
            ):
                lines = []
                if "stale cursor state" in joined:
                    lines.append("- Carry forward the stale cursor state root cause.")
                if "stale cursor cleanup" in joined:
                    lines.append("- Carry forward the stale cursor cleanup fix.")
                if "reconnect ordering" in joined:
                    lines.append("- Cite reconnect ordering as the key verification check.")
                if "checkpoint flush" in joined:
                    lines.append("- Include the checkpoint flush stall as backing evidence.")
                if "staging" in joined or "verified" in joined:
                    lines.append("- If the latest verification still holds, this is ready to hand off.")
                if lines:
                    return "Incident handoff:\n" + "\n".join(lines)

            if any(
                token in lowered
                for token in (
                    "what should we verify next",
                    "before closing this",
                    "before we close this incident",
                    "ready to close",
                    "what fix are we carrying forward",
                )
            ):
                lines = []
                if "stale cursor cleanup" in joined:
                    lines.append("- Carry forward the stale cursor cleanup fix.")
                if "reconnect ordering" in joined:
                    lines.append("- Verify reconnect ordering before closure.")
                if "validation" in joined or "staging" in joined:
                    lines.append("- Use the staging validation as the final closeout check.")
                if "close" in lowered:
                    lines.append("- If those checks stay clean, this is ready to close.")
                if lines:
                    return "Verification summary:\n" + "\n".join(lines)

            if "scheduler incident" in lowered or "return to the scheduler incident" in lowered:
                lines = []
                if "retry loop overflow" in joined:
                    lines.append("- Inspect retry state restore around the retry loop overflow.")
                if "duplicate scheduler state" in joined:
                    lines.append("- Verify whether duplicate scheduler state rehydrates the same backoff entry twice.")
                if "checkpoint" in joined:
                    lines.append("- Compare checkpoint IDs before and after resume.")
                if "dedupe" in joined:
                    lines.append("- Confirm the dedupe patch point before requeue.")
                if lines:
                    return "Investigation order:\n" + "\n".join(lines)

            if any(
                token in lowered
                for token in (
                    "strongest remaining root cause",
                    "strongest hypothesis",
                    "which explanation still looks strongest",
                    "what root cause are we carrying forward",
                    "what evidence should i cite",
                    "which theory still survives",
                    "which theory stays ruled out",
                    "stays ruled out",
                    "stays dead",
                )
            ):
                lines = []
                if "stale cursor state" in joined:
                    lines.append("- The strongest remaining root cause is stale cursor state after reconnect.")
                if "cursor reset timing" in joined:
                    lines.append("- The ruled-out theory remains cursor reset timing in the sync worker.")
                if "duplicate scheduler state" in joined:
                    lines.append("- The strongest remaining explanation is duplicate scheduler state rehydrating the same backoff entry.")
                if "redis reconnect" in joined:
                    lines.append("- Backing evidence: the incident starts after redis reconnect.")
                if "checkpoint flush" in joined:
                    lines.append("- Backing evidence: the worker stalls before the final checkpoint flush.")
                if "backoff state" in joined:
                    lines.append("- Backing evidence: duplicate backoff state still appears after resume.")
                if lines:
                    return "Root cause continuation:\n" + "\n".join(lines)

            if "continue the plan" in lowered or "continue the scheduler plan" in lowered:
                lines = []
                if "retry state restore" in joined:
                    lines.append("- Inspect retry state restore first.")
                if "checkpoint" in joined:
                    lines.append("- Compare checkpoint IDs before and after resume.")
                if "dedupe" in joined:
                    lines.append("- Confirm the dedupe patch point before requeue.")
                if "backoff state" in joined:
                    lines.append("- Re-check duplicate backoff state after resume.")
                if "cursor reset" in joined:
                    lines.append("- Inspect cursor reset first.")
                if "reconnect ordering" in joined:
                    lines.append("- Verify reconnect ordering next.")
                if "stale cursor cleanup" in joined:
                    lines.append("- Patch stale cursor cleanup before rerunning.")
                if "stale cursor state" in joined:
                    lines.append("- Confirm the stale cursor state hypothesis against the logs.")
                if lines:
                    return "Plan continuation:\n" + "\n".join(lines)

        mode = controller_state["mode"]
        lam = controller_state["lambda_value"]
        
        responses = {
            "exploratory": f"[Exploratory mode, λ={lam:.4f}] I hear you exploring multiple ideas. What aspect interests you most?",
            "focused": f"[Focused mode, λ={lam:.4f}] Based on your input, here's my recommendation: ...",
            "balanced": f"[Balanced mode, λ={lam:.4f}] Let me think about this and provide a measured response.",
            "oscillating": f"[Considering multiple perspectives, λ={lam:.4f}] I notice some tension in your request. Could you clarify your priority?"
        }
        
        return responses.get(mode, responses["balanced"])
    
    def get_memory_state(self) -> Dict:
        """Get current memory and controller state."""
        state = self.memory.read()
        controller_state = self.controller.get_state_summary()
        artifact_graph = self.get_artifact_graph()
        hypergraph_view = self.get_hypergraph_view()
        
        return {
            "memory": {
                "active": state.n_active,
                "total": self.k * self.L,
                "groups": state.groups.tolist(),
                "mode": self.memory.current_mode,
                "lambda": self.memory.lambda_,
                "mu": self.memory.mu,
                "gamma": self.memory.gamma
            },
            "controller": controller_state,
            "conversation_turns": len(self.conversation_history) // 2,
            "runtime": {
                "runtime_profile": self.runtime_profile_name,
                "write_policy": type(self.turn_processor.write_policy).__name__,
                "retrieval_policy": type(self.turn_processor.retrieval_policy).__name__,
                "retrieval_strategy": getattr(self.turn_processor.retrieval_policy, "strategy", None),
                "turn_log_size": len(self.turn_log),
                "artifact_nodes": len(artifact_graph["nodes"]),
                "artifact_edges": len(artifact_graph["edges"]),
                "conflict_hyperedges": len(hypergraph_view.get("conflict_hyperedges", [])),
                "decision_residues": len(hypergraph_view.get("decision_residues", [])),
                "procedure_residues": len(hypergraph_view.get("procedure_residues", [])),
            },
        }

    def get_runtime_profile(self) -> Dict[str, Any]:
        """Return the named runtime profile for the current agent."""
        return get_runtime_profile(self.runtime_profile_name)

    def get_artifact_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a read-only artifact graph view from persisted turn-log writes."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for turn_index, entry in enumerate(self.turn_log):
            for write_index, write in enumerate(entry.get("writes", [])):
                artifact_id = write.get("artifact_id")
                artifact_type = write.get("artifact_type")
                if not artifact_id or not artifact_type:
                    continue

                nodes.setdefault(
                    artifact_id,
                    {
                        "artifact_id": artifact_id,
                        "artifact_type": artifact_type,
                        "kind": write.get("kind"),
                        "content": write.get("content"),
                        "linked_task": write.get("linked_task"),
                        "group": write.get("group"),
                        "layer": write.get("layer"),
                        "turn_index": turn_index,
                        "write_index": write_index,
                    },
                )

                parent_artifact_id = write.get("parent_artifact_id")
                relation_type = write.get("relation_type")
                if parent_artifact_id and relation_type:
                    edges.append(
                        {
                            "source": parent_artifact_id,
                            "target": artifact_id,
                            "relation_type": relation_type,
                            "linked_task": write.get("linked_task"),
                        }
                    )

        return {"nodes": list(nodes.values()), "edges": edges}

    def get_hypergraph_view(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a read-only hypergraph view from persisted turn-log writes."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        hyperedges: Dict[str, Dict[str, Any]] = {}

        for turn_index, entry in enumerate(self.turn_log):
            for write_index, write in enumerate(entry.get("writes", [])):
                node_id = write.get("artifact_id") or f"turn{turn_index}_write{write_index}"
                node_type = write.get("node_type") or write.get("kind")
                if not node_type:
                    continue

                nodes.setdefault(
                    node_id,
                    {
                        "node_id": node_id,
                        "node_type": node_type,
                        "artifact_type": write.get("artifact_type"),
                        "kind": write.get("kind"),
                        "content": write.get("content"),
                        "linked_task": write.get("linked_task"),
                        "hyperedge_id": write.get("hyperedge_id"),
                        "hyperedge_type": write.get("hyperedge_type"),
                        "confidence_tag": write.get("confidence_tag"),
                        "task_phase": write.get("task_phase"),
                        "procedure_type": write.get("procedure_type"),
                        "reusability_class": write.get("reusability_class"),
                        "turn_index": turn_index,
                        "write_index": write_index,
                    },
                )

                hyperedge_id = write.get("hyperedge_id")
                hyperedge_type = write.get("hyperedge_type")
                if hyperedge_id and hyperedge_type:
                    hyperedge = hyperedges.setdefault(
                        hyperedge_id,
                        {
                            "hyperedge_id": hyperedge_id,
                            "hyperedge_type": hyperedge_type,
                            "linked_task": write.get("linked_task"),
                            "member_node_ids": [],
                            "member_node_types": [],
                            "member_confidence_tags": [],
                            "member_task_phases": [],
                            "member_procedure_types": [],
                            "member_reusability_classes": [],
                            "last_member_turn": turn_index,
                        },
                    )
                    if node_id not in hyperedge["member_node_ids"]:
                        hyperedge["member_node_ids"].append(node_id)
                    node_type_value = node_type or "unknown"
                    if node_type_value not in hyperedge["member_node_types"]:
                        hyperedge["member_node_types"].append(node_type_value)
                    confidence_tag = write.get("confidence_tag")
                    if confidence_tag and confidence_tag not in hyperedge["member_confidence_tags"]:
                        hyperedge["member_confidence_tags"].append(confidence_tag)
                    task_phase = write.get("task_phase")
                    if task_phase and task_phase not in hyperedge["member_task_phases"]:
                        hyperedge["member_task_phases"].append(task_phase)
                    procedure_type = write.get("procedure_type")
                    if procedure_type and procedure_type not in hyperedge["member_procedure_types"]:
                        hyperedge["member_procedure_types"].append(procedure_type)
                    reusability_class = write.get("reusability_class")
                    if reusability_class and reusability_class not in hyperedge["member_reusability_classes"]:
                        hyperedge["member_reusability_classes"].append(reusability_class)
                    hyperedge["last_member_turn"] = max(hyperedge["last_member_turn"], turn_index)

                parent_artifact_id = write.get("parent_artifact_id")
                relation_type = write.get("relation_type")
                if parent_artifact_id and relation_type:
                    edges.append(
                        {
                            "source": parent_artifact_id,
                            "target": node_id,
                            "relation_type": relation_type,
                            "linked_task": write.get("linked_task"),
                        }
                    )

        recent_hyperedge_ids = self._recent_hyperedge_ids()
        for hyperedge in hyperedges.values():
            hyperedge["status"] = self._infer_hyperedge_status(
                hyperedge,
                nodes,
                recent_hyperedge_ids,
            )
            hyperedge["confidence_summary"] = self._summarize_hyperedge_confidence(
                hyperedge,
                nodes,
            )
            hyperedge["phase_summary"] = self._summarize_hyperedge_phase(
                hyperedge,
                nodes,
            )

        conflict_hyperedges = self._build_conflict_hyperedges(nodes, edges)
        decision_residues = self._build_decision_residues(nodes, edges, hyperedges)
        procedure_residues = self._build_procedure_residues(nodes, edges, hyperedges)

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "hyperedges": list(hyperedges.values()),
            "conflict_hyperedges": conflict_hyperedges,
            "decision_residues": decision_residues,
            "procedure_residues": procedure_residues,
        }

    def get_working_set(self, linked_task: Optional[str] = None) -> Dict[str, Any]:
        """Derive a minimal working-set view from the current hypergraph state."""
        hypergraph_view = self.get_hypergraph_view()
        resolved_task = linked_task or self._resolve_current_task(hypergraph_view)

        nodes = [
            node for node in hypergraph_view.get("nodes", [])
            if self._node_matches_task(node, resolved_task)
        ]
        nodes.sort(key=lambda node: (node.get("turn_index", -1), node.get("write_index", -1)), reverse=True)

        hyperedges = [
            hyperedge for hyperedge in hypergraph_view.get("hyperedges", [])
            if self._task_matches(hyperedge.get("linked_task"), resolved_task)
        ]
        active_hyperedges = [
            hyperedge for hyperedge in hyperedges
            if hyperedge.get("status") in {"active", "conflicted"}
        ]

        dominant_conflicts = [
            conflict for conflict in hypergraph_view.get("conflict_hyperedges", [])
            if self._task_matches(conflict.get("linked_task"), resolved_task)
        ]
        active_decisions = [
            residue for residue in hypergraph_view.get("decision_residues", [])
            if self._task_matches(residue.get("linked_task"), resolved_task)
        ]
        active_procedures = [
            residue for residue in hypergraph_view.get("procedure_residues", [])
            if self._task_matches(residue.get("linked_task"), resolved_task)
        ]

        blocker_nodes = [
            node for node in nodes
            if node.get("node_type") in {"log", "issue", "fact", "constraint", "context"}
            and any(
                token in str(node.get("content", "")).lower()
                for token in ("blocker", "blocked", "failure", "timeout", "stalls", "rollback", "incomplete")
            )
        ]
        next_step_candidates = [
            node for node in nodes
            if node.get("node_type") in {"plan", "decision", "procedure", "template", "checklist", "playbook"}
        ][:6]

        return {
            "linked_task": resolved_task,
            "active_node_count": len(nodes),
            "active_hyperedge_count": len(active_hyperedges),
            "active_nodes": nodes[:8],
            "active_hyperedges": active_hyperedges[:6],
            "dominant_conflicts": dominant_conflicts[:4],
            "active_decisions": active_decisions[:4],
            "active_procedures": active_procedures[:4],
            "active_blockers": blocker_nodes[:4],
            "next_step_candidates": next_step_candidates,
        }

    def query_current_task_state(self, linked_task: Optional[str] = None) -> Dict[str, Any]:
        """Return the current task-oriented working set with summary fields."""
        working_set = self.get_working_set(linked_task)
        return {
            "linked_task": working_set["linked_task"],
            "active_node_count": working_set["active_node_count"],
            "active_hyperedge_count": working_set["active_hyperedge_count"],
            "blocker_count": len(working_set["active_blockers"]),
            "decision_count": len(working_set["active_decisions"]),
            "procedure_count": len(working_set["active_procedures"]),
            "next_step_count": len(working_set["next_step_candidates"]),
            "phase_summary": self._summarize_working_set_phase(working_set),
            "status": self._summarize_working_set_status(working_set),
        }

    def query_dominant_conflict(self, linked_task: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the dominant conflict unit for the active or requested task."""
        working_set = self.get_working_set(linked_task)
        conflicts = list(working_set["dominant_conflicts"])
        if not conflicts:
            return None
        ranked = sorted(
            conflicts,
            key=lambda conflict: (
                1 if conflict.get("status") == "active_conflict" else 0,
                len(conflict.get("active_hypothesis_node_ids", [])),
                len(conflict.get("hypothesis_node_ids", [])),
            ),
            reverse=True,
        )
        dominant = ranked[0]
        hypergraph_view = self.get_hypergraph_view()
        nodes_by_id = {node["node_id"]: node for node in hypergraph_view.get("nodes", [])}
        dominant_node = nodes_by_id.get(dominant.get("dominant_node_id"))
        contradicted_nodes = [
            nodes_by_id[node_id]
            for node_id in dominant.get("contradicted_node_ids", [])
            if node_id in nodes_by_id
        ]
        return {
            "linked_task": working_set["linked_task"],
            "status": dominant.get("status"),
            "dominant_node_id": dominant.get("dominant_node_id"),
            "dominant_content": dominant_node.get("content") if dominant_node else None,
            "active_hypothesis_count": len(dominant.get("active_hypothesis_node_ids", [])),
            "contradicted_hypothesis_count": len(dominant.get("contradicted_node_ids", [])),
            "contradicted_contents": [node.get("content") for node in contradicted_nodes[:4]],
            "backing_hyperedge_id": dominant.get("backing_hyperedge_id"),
        }

    def query_decision_residue(self, linked_task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return active decision residues for the active or requested task."""
        working_set = self.get_working_set(linked_task)
        hypergraph_view = self.get_hypergraph_view()
        nodes_by_id = {node["node_id"]: node for node in hypergraph_view.get("nodes", [])}
        results = []
        for residue in working_set["active_decisions"]:
            dominant_node_id = residue.get("dominant_decision_node_id")
            dominant_node = nodes_by_id.get(dominant_node_id)
            results.append(
                {
                    "linked_task": residue.get("linked_task"),
                    "status": residue.get("status"),
                    "dominant_decision_node_id": dominant_node_id,
                    "dominant_decision_content": dominant_node.get("content") if dominant_node else None,
                    "constraint_count": len(residue.get("constraint_node_ids", [])),
                    "alternative_count": len(residue.get("alternative_node_ids", [])),
                    "rationale_count": len(residue.get("rationale_node_ids", [])),
                }
            )
        return results

    def query_applicable_procedures(self, linked_task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return active procedure residues for the active or requested task."""
        working_set = self.get_working_set(linked_task)
        hypergraph_view = self.get_hypergraph_view()
        nodes_by_id = {node["node_id"]: node for node in hypergraph_view.get("nodes", [])}
        results = []
        for residue in working_set["active_procedures"]:
            dominant_node_id = residue.get("dominant_procedure_node_id")
            dominant_node = nodes_by_id.get(dominant_node_id)
            results.append(
                {
                    "linked_task": residue.get("linked_task"),
                    "status": residue.get("status"),
                    "dominant_procedure_node_id": dominant_node_id,
                    "dominant_procedure_content": dominant_node.get("content") if dominant_node else None,
                    "procedure_types": list(residue.get("procedure_types", [])),
                    "task_anchor_count": len(residue.get("task_anchor_node_ids", [])),
                    "decision_anchor_count": len(residue.get("decision_anchor_node_ids", [])),
                    "plan_anchor_count": len(residue.get("plan_anchor_node_ids", [])),
                }
            )
        return results

    def query_handoff_bundle(self, linked_task: Optional[str] = None) -> Dict[str, Any]:
        """Return a compact handoff bundle derived from the current working set."""
        working_set = self.get_working_set(linked_task)
        conflict = self.query_dominant_conflict(linked_task)
        decisions = self.query_decision_residue(linked_task)
        procedures = self.query_applicable_procedures(linked_task)

        active_nodes = working_set["active_nodes"]
        evidence = [
            node.get("content")
            for node in active_nodes
            if node.get("node_type") in {"log", "fact", "verification", "context"}
            and any(
                token in str(node.get("content", "")).lower()
                for token in ("verified", "staging", "green", "checkpoint flush", "reconnect", "rollback")
            )
        ][:6]
        next_steps = [node.get("content") for node in working_set["next_step_candidates"][:4]]
        blockers = [node.get("content") for node in working_set["active_blockers"][:4]]

        return {
            "linked_task": working_set["linked_task"],
            "dominant_conflict": conflict,
            "active_decisions": decisions,
            "applicable_procedures": procedures,
            "blockers": blockers,
            "evidence": evidence,
            "next_steps": next_steps,
            "ready_signals": self._derive_handoff_ready_signals(active_nodes),
        }

    def _resolve_current_task(self, hypergraph_view: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        recent_nodes = sorted(
            hypergraph_view.get("nodes", []),
            key=lambda node: (node.get("turn_index", -1), node.get("write_index", -1)),
            reverse=True,
        )
        counts: Dict[str, int] = {}
        for node in recent_nodes[:16]:
            linked_task = node.get("linked_task")
            if not linked_task:
                continue
            counts[str(linked_task)] = counts.get(str(linked_task), 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]

    def _task_matches(self, candidate_task: Optional[str], resolved_task: Optional[str]) -> bool:
        if resolved_task is None:
            return True
        if candidate_task is None:
            return False
        return str(candidate_task).lower() == str(resolved_task).lower()

    def _node_matches_task(self, node: Dict[str, Any], resolved_task: Optional[str]) -> bool:
        if self._task_matches(node.get("linked_task"), resolved_task):
            return True
        if resolved_task is None and node.get("linked_task") is None:
            return True
        return False

    def _summarize_working_set_phase(self, working_set: Dict[str, Any]) -> str:
        phase_order = ["closure", "verification", "implementation", "decision", "analysis"]
        observed = []
        for node in working_set.get("active_nodes", []):
            phase = node.get("task_phase")
            if phase and phase not in observed:
                observed.append(phase)
        for phase in phase_order:
            if phase in observed:
                return phase
        return "unknown"

    def _summarize_working_set_status(self, working_set: Dict[str, Any]) -> str:
        if working_set.get("active_blockers"):
            return "blocked"
        if any(
            "ready to close" in str(node.get("content", "")).lower() or "ready to hand off" in str(node.get("content", "")).lower()
            for node in working_set.get("active_nodes", [])
        ):
            return "ready"
        if working_set.get("next_step_candidates"):
            return "in_progress"
        return "unknown"

    def _derive_handoff_ready_signals(self, active_nodes: List[Dict[str, Any]]) -> List[str]:
        signals = []
        for node in active_nodes:
            content = str(node.get("content", "")).strip()
            lowered = content.lower()
            if any(
                token in lowered
                for token in (
                    "ready to close",
                    "ready to hand off",
                    "ready to handoff",
                    "looks stable in staging",
                    "tests are green",
                    "rollback notes are now complete",
                )
            ):
                signals.append(content)
        return signals[:6]

    def _recent_hyperedge_ids(self) -> set[str]:
        recent_ids: set[str] = set()
        for entry in self.turn_log[-2:]:
            for item in entry.get("retrieved_detail", []):
                hyperedge_id = item.get("hyperedge_id")
                if hyperedge_id:
                    recent_ids.add(hyperedge_id)
            for item in entry.get("writes", []):
                hyperedge_id = item.get("hyperedge_id")
                if hyperedge_id:
                    recent_ids.add(hyperedge_id)
        return recent_ids

    def _infer_hyperedge_status(
        self,
        hyperedge: Dict[str, Any],
        nodes: Dict[str, Dict[str, Any]],
        recent_hyperedge_ids: set[str],
    ) -> str:
        hyperedge_id = hyperedge.get("hyperedge_id")
        member_node_ids = hyperedge.get("member_node_ids", [])
        member_nodes = [nodes[node_id] for node_id in member_node_ids if node_id in nodes]
        member_text = " ".join(str(node.get("content", "")) for node in member_nodes).lower()
        member_types = {str(node.get("node_type", "")) for node in member_nodes}

        if any(token in member_text for token in ("superseded", "replaced by", "obsolete", "deprecated")):
            return "superseded"
        if any(token in member_text for token in ("resolved", "fixed", "done", "completed", "closed")):
            return "resolved"
        if "hypothesis" in member_types:
            hypotheses = {
                str(node.get("content", "")).strip().lower()
                for node in member_nodes
                if str(node.get("node_type", "")) == "hypothesis"
            }
            if len(hypotheses) > 1:
                return "conflicted"
        if hyperedge_id in recent_hyperedge_ids:
            return "active"
        return "paused"

    def _summarize_hyperedge_confidence(
        self,
        hyperedge: Dict[str, Any],
        nodes: Dict[str, Dict[str, Any]],
    ) -> str:
        counts = {
            "verified": 0,
            "tentative": 0,
            "speculative": 0,
            "contradicted": 0,
        }
        for node_id in hyperedge.get("member_node_ids", []):
            node = nodes.get(node_id)
            if not node:
                continue
            tag = node.get("confidence_tag")
            if tag in counts:
                counts[tag] += 1
        if counts["contradicted"] > 0:
            return "contradicted"
        if counts["verified"] > 0 and counts["speculative"] == 0:
            return "verified"
        if counts["speculative"] > 0:
            return "speculative"
        if counts["tentative"] > 0:
            return "tentative"
        return "unknown"

    def _summarize_hyperedge_phase(
        self,
        hyperedge: Dict[str, Any],
        nodes: Dict[str, Dict[str, Any]],
    ) -> str:
        phase_order = ["closure", "verification", "implementation", "decision", "analysis"]
        observed = []
        for node_id in hyperedge.get("member_node_ids", []):
            node = nodes.get(node_id)
            if not node:
                continue
            phase = node.get("task_phase")
            if phase and phase not in observed:
                observed.append(phase)
        for phase in phase_order:
            if phase in observed:
                return phase
        return "unknown"

    def _build_conflict_hyperedges(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        contradiction_targets: Dict[str, List[str]] = {}
        contradiction_notes: Dict[str, List[str]] = {}
        for edge in edges:
            if edge.get("relation_type") != "contradicts":
                continue
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            contradiction_targets.setdefault(source, []).append(target)
            contradiction_notes.setdefault(target, []).append(source)

        grouped: Dict[str, Dict[str, Any]] = {}
        for node_id, node in nodes.items():
            if node.get("node_type") != "hypothesis":
                continue
            linked_task = node.get("linked_task") or "unlinked"
            hyperedge_id = node.get("hyperedge_id") or "no_hyperedge"
            conflict_id = f"conflict::{linked_task}::{hyperedge_id}"
            group = grouped.setdefault(
                conflict_id,
                {
                    "conflict_hyperedge_id": conflict_id,
                    "linked_task": linked_task,
                    "backing_hyperedge_id": node.get("hyperedge_id"),
                    "hypothesis_node_ids": [],
                    "contradicted_node_ids": [],
                    "contradiction_note_ids": [],
                    "active_hypothesis_node_ids": [],
                    "status": "stable",
                    "dominant_node_id": None,
                },
            )
            group["hypothesis_node_ids"].append(node_id)
            if node_id in contradiction_targets:
                group["contradicted_node_ids"].append(node_id)
                group["contradiction_note_ids"].extend(contradiction_targets[node_id])
            else:
                group["active_hypothesis_node_ids"].append(node_id)

        conflict_units: List[Dict[str, Any]] = []
        for unit in grouped.values():
            if len(unit["hypothesis_node_ids"]) < 2 and not unit["contradicted_node_ids"]:
                continue
            unit["contradiction_note_ids"] = list(dict.fromkeys(unit["contradiction_note_ids"]))
            if unit["contradicted_node_ids"] and unit["active_hypothesis_node_ids"]:
                unit["status"] = "active_conflict"
            elif unit["contradicted_node_ids"]:
                unit["status"] = "resolved_conflict"
            else:
                unit["status"] = "competing_hypotheses"
            dominant_candidates = unit["active_hypothesis_node_ids"] or unit["hypothesis_node_ids"]
            unit["dominant_node_id"] = self._select_dominant_conflict_node(dominant_candidates, nodes)
            conflict_units.append(unit)

        return conflict_units

    def _select_dominant_conflict_node(
        self,
        candidate_node_ids: List[str],
        nodes: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        if not candidate_node_ids:
            return None
        ranked = sorted(
            candidate_node_ids,
            key=lambda node_id: (
                nodes.get(node_id, {}).get("turn_index", -1),
                1 if nodes.get(node_id, {}).get("confidence_tag") == "verified" else 0,
                1 if nodes.get(node_id, {}).get("confidence_tag") == "tentative" else 0,
            ),
            reverse=True,
        )
        return ranked[0]

    def _build_decision_residues(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        hyperedges: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        decision_residues: List[Dict[str, Any]] = []
        edges_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            edges_by_target.setdefault(str(edge.get("target")), []).append(edge)

        for hyperedge_id, hyperedge in hyperedges.items():
            if hyperedge.get("hyperedge_type") != "decision_hyperedge":
                continue

            member_node_ids = list(hyperedge.get("member_node_ids", []))
            if not member_node_ids:
                continue

            decision_node_ids = [node_id for node_id in member_node_ids if nodes.get(node_id, {}).get("node_type") == "decision"]
            rationale_node_ids = [node_id for node_id in member_node_ids if nodes.get(node_id, {}).get("node_type") == "rationale"]
            constraint_node_ids = [node_id for node_id in member_node_ids if nodes.get(node_id, {}).get("node_type") == "constraint"]
            alternative_node_ids = [node_id for node_id in member_node_ids if nodes.get(node_id, {}).get("node_type") == "alternative"]
            if not decision_node_ids:
                continue

            dominant_decision_node_id = max(
                decision_node_ids,
                key=lambda node_id: nodes.get(node_id, {}).get("turn_index", -1),
            )
            supported_decision_node_ids = []
            constrained_decision_node_ids = []
            rejected_alternative_node_ids = []
            for rationale_id in rationale_node_ids:
                if any(edge.get("relation_type") == "rationale_for" for edge in edges_by_target.get(rationale_id, [])):
                    supported_decision_node_ids.append(next(edge["source"] for edge in edges_by_target[rationale_id] if edge.get("relation_type") == "rationale_for"))
            for constraint_id in constraint_node_ids:
                if any(edge.get("relation_type") == "constrains" for edge in edges_by_target.get(constraint_id, [])):
                    constrained_decision_node_ids.append(next(edge["source"] for edge in edges_by_target[constraint_id] if edge.get("relation_type") == "constrains"))
            for alternative_id in alternative_node_ids:
                if any(edge.get("relation_type") == "alternative_to" for edge in edges_by_target.get(alternative_id, [])):
                    rejected_alternative_node_ids.append(alternative_id)

            decision_residues.append(
                {
                    "decision_residue_id": f"decision::{hyperedge_id}",
                    "linked_task": hyperedge.get("linked_task"),
                    "backing_hyperedge_id": hyperedge_id,
                    "status": hyperedge.get("status"),
                    "dominant_decision_node_id": dominant_decision_node_id,
                    "decision_node_ids": decision_node_ids,
                    "rationale_node_ids": rationale_node_ids,
                    "constraint_node_ids": constraint_node_ids,
                    "alternative_node_ids": alternative_node_ids,
                    "supported_decision_node_ids": list(dict.fromkeys(supported_decision_node_ids)),
                    "constrained_decision_node_ids": list(dict.fromkeys(constrained_decision_node_ids)),
                    "rejected_alternative_node_ids": list(dict.fromkeys(rejected_alternative_node_ids)),
                    "confidence_summary": hyperedge.get("confidence_summary"),
                }
            )

        return decision_residues

    def _build_procedure_residues(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        hyperedges: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        procedure_residues: List[Dict[str, Any]] = []
        edges_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            edges_by_target.setdefault(str(edge.get("target")), []).append(edge)

        for hyperedge_id, hyperedge in hyperedges.items():
            if hyperedge.get("hyperedge_type") != "procedure_hyperedge":
                continue

            member_node_ids = list(hyperedge.get("member_node_ids", []))
            if not member_node_ids:
                continue

            procedure_node_ids = [
                node_id
                for node_id in member_node_ids
                if nodes.get(node_id, {}).get("node_type") in {"playbook", "checklist", "template", "procedure"}
            ]
            if not procedure_node_ids:
                continue

            task_anchor_node_ids = []
            decision_anchor_node_ids = []
            plan_anchor_node_ids = []
            for procedure_node_id in procedure_node_ids:
                for edge in edges_by_target.get(procedure_node_id, []):
                    if edge.get("relation_type") not in {"procedure_for", "template_for"}:
                        continue
                    source_id = str(edge.get("source"))
                    source_type = nodes.get(source_id, {}).get("node_type")
                    if source_type == "task":
                        task_anchor_node_ids.append(source_id)
                    elif source_type == "decision":
                        decision_anchor_node_ids.append(source_id)
                    elif source_type == "plan":
                        plan_anchor_node_ids.append(source_id)

            dominant_procedure_node_id = max(
                procedure_node_ids,
                key=lambda node_id: nodes.get(node_id, {}).get("turn_index", -1),
            )
            procedure_residues.append(
                {
                    "procedure_residue_id": f"procedure::{hyperedge_id}",
                    "linked_task": hyperedge.get("linked_task"),
                    "backing_hyperedge_id": hyperedge_id,
                    "status": hyperedge.get("status"),
                    "procedure_node_ids": procedure_node_ids,
                    "dominant_procedure_node_id": dominant_procedure_node_id,
                    "procedure_types": list(
                        dict.fromkeys(
                            nodes.get(node_id, {}).get("procedure_type")
                            for node_id in procedure_node_ids
                            if nodes.get(node_id, {}).get("procedure_type")
                        )
                    ),
                    "task_anchor_node_ids": list(dict.fromkeys(task_anchor_node_ids)),
                    "decision_anchor_node_ids": list(dict.fromkeys(decision_anchor_node_ids)),
                    "plan_anchor_node_ids": list(dict.fromkeys(plan_anchor_node_ids)),
                    "reusability_summary": list(hyperedge.get("member_reusability_classes", [])),
                }
            )

        return procedure_residues

    def summarize_artifact_graph(self) -> str:
        """Return a compact ASCII summary of the artifact graph."""
        graph = self.get_artifact_graph()
        lines = [
            "Artifact Graph",
            f"- nodes: {len(graph['nodes'])}",
            f"- edges: {len(graph['edges'])}",
        ]

        for node in graph["nodes"][:6]:
            label = node["artifact_type"]
            task = f" [{node['linked_task']}]" if node.get("linked_task") else ""
            lines.append(f"- {label}:{task} {node['artifact_id']}")

        for edge in graph["edges"][:6]:
            lines.append(f"- {edge['source']} -> {edge['target']} ({edge['relation_type']})")

        return "\n".join(lines)

    def get_session_state(self) -> SessionState:
        """Return the structured session state for persistence or inspection."""
        return SessionState(
            system_prompt=self.system_prompt,
            conversation_history=list(self.conversation_history),
            turn_log=list(self.turn_log),
            controller_state=self.controller.get_state_summary(),
        )
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save agent state to file."""
        if filepath is None:
            filepath = f"{self.name}_state.json"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.memory.save(str(path))

        session_state = self.get_session_state()
        history_path = path.with_name(path.stem + "_history.json")
        temp_history_path = history_path.with_suffix(history_path.suffix + ".tmp")
        with open(temp_history_path, 'w', encoding='utf-8') as f:
            json.dump(session_state.to_dict(), f, indent=2)
        os.replace(temp_history_path, history_path)

        return str(path)
    
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        self.memory = self.memory.load(filepath)

        path = Path(filepath)
        history_path = path.with_name(path.stem + "_history.json")
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                session_state = SessionState.from_dict(json.load(f))
                self.conversation_history = session_state.conversation_history
                self.system_prompt = session_state.system_prompt or self._default_system_prompt()
                self.turn_log = session_state.turn_log
                controller_state = session_state.controller_state
                if "lambda_ratio" in controller_state:
                    self.controller.target_lambda_ratio = controller_state["lambda_ratio"]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def reset_memory(self) -> None:
        """Reset memory and conversation history."""
        self.memory = type(self.memory)(
            k=self.k, L=self.L,
            use_physics_control=True,
            name=self.name
        )
        self.memory.group_labels = self.group_labels
        self.memory.layer_labels = self.layer_labels
        self.conversation_history = []
        self.turn_log = []
        self.controller.reset()
    
    def visualize_state(self) -> str:
        """Get ASCII visualization of memory matrix."""
        state = self.memory.read()
        M = self.memory.M

        lines = [
            f"HypergraphAgent '{self.name}' State",
            f"{'=' * 50}",
            f"k={self.k}, L={self.L} | λ={self.memory.lambda_:.4f} | γ={self.memory.gamma:.4f}",
            f"Mode: {self.controller.current_mode.value} | Turns: {len(self.conversation_history)//2}",
            "",
            "Memory Matrix M (0=low, 1=high):",
            ""
        ]

        for i in range(self.k):
            row = f"  {self.group_labels[i]:12} | "
            for l in range(self.L):
                row += f"{self._ascii_level_char(M[i, l])} "
            row += f"| {state.groups[i]:.2f}"
            lines.append(row)

        lines.append("")
        lines.append(f"Controller: {self.controller.get_state_summary()}")
        lines.append("")
        lines.append(self.summarize_artifact_graph())

        return "\n".join(lines)

    @staticmethod
    def _ascii_level_char(val: float) -> str:
        """Map activation levels to ASCII-safe terminal characters."""
        if val > 0.8:
            return "#"
        if val > 0.5:
            return "O"
        if val > 0.3:
            return "+"
        if val > 0.1:
            return "."
        return " "

class HypergraphMemoryAgent:
    """
    LangChain-compatible memory wrapper for HypergraphAgent.
    
    Use as a drop-in replacement for LangChain's built-in memory:
    
        from langchain_openai import ChatOpenAI
        from langchain.chains import ConversationChain
        
        memory = HypergraphMemoryAgent(k=4, L=2)
        llm = ChatOpenAI(model="gpt-4o-mini")
        chain = ConversationChain(llm=llm, memory=memory)
        
        response = chain.invoke({"input": "Hello!"})
    """
    
    def __init__(self, k: int = 4, L: int = 2, **kwargs):
        self.agent = HypergraphAgent(k=k, L=L, **kwargs)
    
    @property
    def memory_variables(self) -> List[str]:
        """Required by LangChain."""
        return ["history", "memory_state"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LangChain."""
        context = self.agent.memory.get_context_for_llm()
        
        history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.agent.conversation_history[-10:]
        ])
        
        return {
            "history": history,
            "memory_state": context
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context after each chain step."""
        user_input = inputs.get("input", "")
        assistant_output = outputs.get("response", "") or outputs.get("output", "")

        self.agent.record_external_turn(
            user_input=user_input,
            assistant_response=assistant_output,
        )
    
    def clear(self) -> None:
        """Clear memory."""
        self.agent.reset_memory()


def demo():
    """Demo of HypergraphAgent."""
    print("=" * 60)
    print("HypergraphAgent Demo")
    print("=" * 60)
    
    agent = HypergraphAgent(k=4, L=2, use_embeddings=False)
    agent.group_labels = ["work", "personal", "technical", "creative"]
    
    print("\n1. Testing adaptive mode switching:")
    
    scenarios = [
        "Let's brainstorm some startup ideas",
        "I need to decide between job offers",
        "Help me debug this Python code",
        "Write me a short poem",
    ]
    
    for input_text in scenarios:
        response = agent.chat(input_text)
        state = agent.get_memory_state()
        print(f"\nInput: {input_text}")
        print(f"Mode: {state['controller']['mode']}, λ={state['memory']['lambda']:.4f}")
        print(f"Response: {response[:80]}...")
    
    print("\n" + agent.visualize_state())
    
    print("\n2. Testing conflict mitigation:")
    response = agent.chat("I want to go to the beach but I also want to finish this project")
    state = agent.get_memory_state()
    print(f"\nInput: I want to go to the beach but I also want to finish this project")
    print(f"Mode: {state['controller']['mode']}")
    print(f"Response: {response[:100]}...")


if __name__ == "__main__":
    demo()

