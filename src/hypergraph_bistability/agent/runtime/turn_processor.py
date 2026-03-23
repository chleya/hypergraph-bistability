"""Turn lifecycle for practical agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hypergraph_bistability.memory.policies import RetrievalPolicy, WritePolicy


@dataclass
class TurnResult:
    """Materialized output of a processed agent turn."""

    user_input: str
    assistant_response: str
    memory_context: str
    conflict_level: float
    retrieved_items: List[str]
    writes: List[Dict[str, object]]
    controller_state: Dict[str, object]


class TurnProcessor:
    """Drive the per-turn agent pipeline."""

    def __init__(
        self,
        *,
        write_policy: Optional[WritePolicy] = None,
        retrieval_policy: Optional[RetrievalPolicy] = None,
        context_assembler=None,
    ) -> None:
        self.write_policy = write_policy or WritePolicy()
        self.retrieval_policy = retrieval_policy or RetrievalPolicy()
        self.context_assembler = context_assembler
        self._last_handoff_validation: Dict[str, Any] = {}

    def _validate_handoff_continuity(
        self,
        agent,
        assistant_response: str,
        working_set_context: str,
    ) -> None:
        """Validate that response demonstrates meaningful continuity with handoff bundle.
        
        Logs warnings for failures/weak passes and records to file for later analysis.
        """
        import json
        import logging
        import os
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        
        # Skip if no working set context (empty string)
        if not working_set_context:
            self._last_handoff_validation = {"status": "not_applicable", "reason": "no_working_set_context"}
            return
        
        # Skip if response is empty (shouldn't happen but just in case)
        if not assistant_response:
            self._last_handoff_validation = {"status": "not_applicable", "reason": "empty_response"}
            return
        
        try:
            # Parse the working set context
            ws_data = json.loads(working_set_context)
            
            # Skip if the parsed context is empty - but this is now "not_applicable", not pass!
            if not ws_data or not ws_data.get("task"):
                # Has no task but might have blockers/decisions/evidence - still validate if present
                has_continuity_material = (
                    ws_data.get("blockers") or 
                    ws_data.get("active_decisions") or 
                    ws_data.get("applicable_procedures")
                )
                if not has_continuity_material:
                    self._last_handoff_validation = {"status": "not_applicable", "reason": "empty_handoff_data"}
                    return
                # Fall through to validate if there are continuity materials but no task
        except json.JSONDecodeError:
            self._last_handoff_validation = {"status": "not_applicable", "reason": "invalid_json"}
            return
        
        # Get handoff bundle from agent
        handoff_bundle = agent.query_handoff_bundle()
        
        # Validate using context assembler
        if self.context_assembler:
            result = self.context_assembler.validate_handoff_continuity(
                response=assistant_response,
                handoff_bundle=handoff_bundle,
            )
        else:
            result = {"status": "not_applicable", "reason": "no_context_assembler"}
        
        self._last_handoff_validation = result
        
        # Get status for logging
        status = result.get("status", "fail")
        
        # Log warning for failures and weak passes
        if status in ["fail", "weak_pass"]:
            logger.warning(
                f"Handoff continuity {status}: {result.get('reason', 'unknown')}, "
                f"task={result.get('task_reference')}, "
                f"blocker={result.get('blocker_reference')}, "
                f"next_step={result.get('next_step_reference')}, "
                f"decision={result.get('decision_reference')}, "
                f"quality={result.get('content_quality')}"
            )
        
        # Log to file for later analysis
        try:
            # Find project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            log_dir = os.path.join(project_root, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "handoff_validation.jsonl")
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "reason": result.get("reason", ""),
                "task_reference": result.get("task_reference", False),
                "blocker_reference": result.get("blocker_reference", False),
                "next_step_reference": result.get("next_step_reference", False),
                "decision_reference": result.get("decision_reference", False),
                "content_quality": result.get("content_quality", "unknown"),
                "response_length": len(assistant_response),
                "response_preview": assistant_response[:200] if assistant_response else "",
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Don't fail validation if file logging fails

    def _generate_working_set_context(self, agent) -> str:
        """Generate working-set context string for prompt injection (JSON format)."""
        import json
        
        try:
            from hypergraph_bistability.agent.query import get_query_layer
            query = get_query_layer(agent)
            
            ws_data = {}
            
            # Current task state
            task_state = query.query_current_task_state()
            if task_state.linked_task or task_state.blocker_count > 0 or task_state.active_decisions_count > 0 or task_state.active_procedures_count > 0:
                ws_data["task"] = {
                    "name": task_state.linked_task or None,
                    "status": task_state.status or "idle",
                    "phase": task_state.phase_summary or "none",
                    "blockers": task_state.blocker_count,
                    "decisions": task_state.active_decisions_count,
                    "procedures": task_state.active_procedures_count,
                }
            
            # Dominant conflict
            conflict = query.query_dominant_conflict()
            if conflict and (getattr(conflict, 'has_conflict', None) or getattr(conflict, 'status', None)):
                ws_data["conflict"] = {
                    "dominant": getattr(conflict, 'dominant_content', None) or getattr(conflict, 'dominant_hypothesis', None),
                    "contradicted_count": len(getattr(conflict, 'contradicted_contents', []) or getattr(conflict, 'contradicted_hypotheses', [])),
                }
            
            # Active decisions
            decisions = query.query_decision_residue()
            if decisions:
                ws_data["decisions_detail"] = [
                    {"content": d.decision_content[:100], "phase": getattr(d, 'phase', None)}
                    for d in decisions[:3]
                ]
            
            # Active procedures
            procedures = query.query_applicable_procedures()
            if procedures:
                ws_data["procedures_detail"] = [
                    {"content": p.procedure_content[:100], "phase": p.phase}
                    for p in procedures[:3]
                ]
            
            # Handoff readiness
            if ws_data.get("task"):
                blockers = ws_data["task"].get("blockers", 0)
                ws_data["handoff_ready"] = blockers == 0 and ws_data["task"].get("status") == "completed"
            
            return json.dumps(ws_data, ensure_ascii=False, indent=2) if ws_data else ""
            
        except Exception:
            return ""

    def process(
        self,
        *,
        agent,
        user_input: str,
        use_adaptive: bool = True,
        return_context: bool = False,
    ) -> TurnResult | str:
        conflict_level = agent.detect_conflict(user_input)
        lambda_value, mu_value, suggestion = agent.controller.update(user_input, conflict_level)

        if use_adaptive:
            agent.memory.lambda_ = lambda_value
            agent.memory.mu = mu_value

        retrieved = self.retrieval_policy.collect(
            user_input,
            memory=agent.memory,
            turn_log=agent.turn_log,
            embedding_mapper=agent.embedding_mapper,
        )

        write_events: List[Dict[str, object]] = []
        user_decision = self.write_policy.decide(
            user_input,
            role="user",
            turn_index=len(agent.conversation_history) // 2,
            k=agent.k,
            L=agent.L,
            embedding_mapper=agent.embedding_mapper,
        )
        if user_decision and user_decision.should_write:
            self._attach_artifact_relations(user_decision, agent.turn_log)
            agent.memory.write(
                user_decision.content,
                group=user_decision.group,
                layer=user_decision.layer,
                activate=user_decision.activate,
            )
            if agent.embedding_mapper and getattr(agent.embedding_mapper, "store", None):
                agent.embedding_mapper.store_memory(
                    user_decision.content,
                    user_decision.group,
                    user_decision.layer,
                    {
                        "role": "user",
                        "kind": user_decision.kind,
                        "artifact_type": user_decision.artifact_type,
                        "artifact_id": user_decision.artifact_id,
                        "linked_task": user_decision.linked_task,
                        "relation_type": user_decision.relation_type,
                        "parent_artifact_id": user_decision.parent_artifact_id,
                        "node_type": user_decision.node_type,
                        "hyperedge_type": user_decision.hyperedge_type,
                        "hyperedge_id": user_decision.hyperedge_id,
                        "confidence_tag": user_decision.confidence_tag,
                        "task_phase": user_decision.task_phase,
                        "procedure_type": user_decision.procedure_type,
                        "reusability_class": user_decision.reusability_class,
                    },
                )
            write_events.append(user_decision.__dict__.copy())

        memory_context = agent.memory.get_context_for_llm()
        if return_context:
            return memory_context

        effective_input = f"{user_input}\n\n{suggestion}" if suggestion else user_input
        
        # Generate working-set context for the prompt
        working_set_context = self._generate_working_set_context(agent)
        
        messages = self.context_assembler.build_messages(
            system_prompt=agent.system_prompt,
            memory_context=memory_context,
            retrieved_items=retrieved,
            conversation_history=agent.conversation_history,
            user_input=effective_input,
            working_set_context=working_set_context,
        )
        assistant_response = agent.generate_response(
            user_input=effective_input,
            memory_context=memory_context,
            messages=messages,
            retrieved_items=retrieved,
        )

        # Optional: Validate handoff continuity (log warning if failed, don't block)
        self._validate_handoff_continuity(agent, assistant_response, working_set_context)

        agent.conversation_history.append({"role": "user", "content": user_input})
        agent.conversation_history.append({"role": "assistant", "content": assistant_response})

        assistant_decision = self.write_policy.decide(
            assistant_response,
            role="assistant",
            turn_index=len(agent.conversation_history) // 2,
            k=agent.k,
            L=agent.L,
            embedding_mapper=agent.embedding_mapper,
        )
        if assistant_decision and assistant_decision.should_write:
            self._attach_artifact_relations(assistant_decision, agent.turn_log + [{"writes": write_events}])
            agent.memory.write(
                assistant_decision.content,
                group=assistant_decision.group,
                layer=assistant_decision.layer,
                activate=assistant_decision.activate,
            )
            if agent.embedding_mapper and getattr(agent.embedding_mapper, "store", None):
                agent.embedding_mapper.store_memory(
                    assistant_decision.content,
                    assistant_decision.group,
                    assistant_decision.layer,
                    {
                        "role": "assistant",
                        "kind": assistant_decision.kind,
                        "artifact_type": assistant_decision.artifact_type,
                        "artifact_id": assistant_decision.artifact_id,
                        "linked_task": assistant_decision.linked_task,
                        "relation_type": assistant_decision.relation_type,
                        "parent_artifact_id": assistant_decision.parent_artifact_id,
                        "node_type": assistant_decision.node_type,
                        "hyperedge_type": assistant_decision.hyperedge_type,
                        "hyperedge_id": assistant_decision.hyperedge_id,
                        "confidence_tag": assistant_decision.confidence_tag,
                        "task_phase": assistant_decision.task_phase,
                        "procedure_type": assistant_decision.procedure_type,
                        "reusability_class": assistant_decision.reusability_class,
                    },
                )
            write_events.append(assistant_decision.__dict__.copy())

        turn_result = TurnResult(
            user_input=user_input,
            assistant_response=assistant_response,
            memory_context=memory_context,
            conflict_level=conflict_level,
            retrieved_items=[item.content for item in retrieved],
            writes=write_events,
            controller_state=agent.controller.get_state_summary(),
        )
        agent.turn_log.append({
            "user_input": turn_result.user_input,
            "assistant_response": turn_result.assistant_response,
            "memory_context": turn_result.memory_context,
            "conflict_level": turn_result.conflict_level,
            "retrieved_items": turn_result.retrieved_items,
            "retrieved_detail": [
                {
                    "source": item.source,
                    "content": item.content,
                    "score": item.score,
                    "kind": item.kind,
                    "group": item.group,
                    "layer": item.layer,
                    "artifact_type": getattr(item, "artifact_type", None),
                    "artifact_id": getattr(item, "artifact_id", None),
                    "linked_task": getattr(item, "linked_task", None),
                    "relation_type": getattr(item, "relation_type", None),
                    "parent_artifact_id": getattr(item, "parent_artifact_id", None),
                    "node_type": getattr(item, "node_type", None),
                    "hyperedge_type": getattr(item, "hyperedge_type", None),
                    "hyperedge_id": getattr(item, "hyperedge_id", None),
                    "hyperedge_status": getattr(item, "hyperedge_status", None),
                    "confidence_tag": getattr(item, "confidence_tag", None),
                    "task_phase": getattr(item, "task_phase", None),
                    "procedure_type": getattr(item, "procedure_type", None),
                    "reusability_class": getattr(item, "reusability_class", None),
                }
                for item in retrieved
            ],
            "writes": turn_result.writes,
            "controller_state": turn_result.controller_state,
        })
        return turn_result

    def _attach_artifact_relations(self, decision, turn_log: List[dict]) -> None:
        if getattr(decision, "confidence_tag", None) == "contradicted":
            for entry in reversed(turn_log):
                for write in reversed(entry.get("writes", [])):
                    if write.get("node_type") != "hypothesis":
                        continue
                    if decision.linked_task and write.get("linked_task") != decision.linked_task:
                        continue
                    parent_id = write.get("artifact_id")
                    if not parent_id:
                        continue
                    if not decision.linked_task and write.get("linked_task"):
                        decision.linked_task = write.get("linked_task")
                    decision.parent_artifact_id = parent_id
                    decision.relation_type = "contradicts"
                    return

        node_type = getattr(decision, "node_type", None)
        if node_type in {"rationale", "constraint", "alternative"}:
            relation_map = {
                "rationale": "rationale_for",
                "constraint": "constrains",
                "alternative": "alternative_to",
            }
            for entry in reversed(turn_log):
                for write in reversed(entry.get("writes", [])):
                    if write.get("node_type") != "decision":
                        continue
                    if decision.linked_task and write.get("linked_task") != decision.linked_task:
                        continue
                    parent_id = write.get("artifact_id")
                    if not parent_id:
                        continue
                    if not decision.linked_task and write.get("linked_task"):
                        decision.linked_task = write.get("linked_task")
                    if write.get("hyperedge_id"):
                        decision.hyperedge_id = write.get("hyperedge_id")
                    if write.get("hyperedge_type"):
                        decision.hyperedge_type = write.get("hyperedge_type")
                    decision.parent_artifact_id = parent_id
                    decision.relation_type = relation_map[node_type]
                    return

        if node_type in {"playbook", "checklist", "template", "procedure"}:
            relation_map = {
                "playbook": "procedure_for",
                "checklist": "procedure_for",
                "template": "template_for",
                "procedure": "procedure_for",
            }
            for entry in reversed(turn_log):
                for write in reversed(entry.get("writes", [])):
                    if decision.linked_task and write.get("linked_task") != decision.linked_task:
                        continue
                    if write.get("node_type") not in {"task", "plan", "decision"}:
                        continue
                    parent_id = write.get("artifact_id")
                    if not parent_id:
                        continue
                    if not decision.linked_task and write.get("linked_task"):
                        decision.linked_task = write.get("linked_task")
                    parent_hyperedge_id = write.get("hyperedge_id")
                    parent_hyperedge_type = write.get("hyperedge_type")
                    if parent_hyperedge_type == "procedure_hyperedge" and parent_hyperedge_id:
                        decision.hyperedge_id = parent_hyperedge_id
                    elif parent_hyperedge_id:
                        decision.hyperedge_id = f"procedure::{parent_hyperedge_id}"
                    elif parent_id:
                        decision.hyperedge_id = f"procedure::{parent_id}"
                    decision.hyperedge_type = "procedure_hyperedge"
                    decision.parent_artifact_id = parent_id
                    decision.relation_type = relation_map[node_type]
                    return

        if getattr(decision, "task_phase", None) == "verification" and getattr(decision, "confidence_tag", None) == "verified":
            for entry in reversed(turn_log):
                for write in reversed(entry.get("writes", [])):
                    if write.get("artifact_type") not in {"plan", "decision"}:
                        continue
                    if decision.linked_task and write.get("linked_task") != decision.linked_task:
                        continue
                    parent_id = write.get("artifact_id")
                    if not parent_id:
                        continue
                    if not decision.linked_task and write.get("linked_task"):
                        decision.linked_task = write.get("linked_task")
                    decision.parent_artifact_id = parent_id
                    decision.relation_type = "verifies"
                    return

        if not getattr(decision, "artifact_type", None):
            return

        relation_targets = {
            "hypothesis": ("derived_from", {"log"}),
            "plan": ("tests", {"hypothesis", "log"}),
        }
        target = relation_targets.get(decision.artifact_type)
        if not target:
            return

        relation_type, accepted_parent_types = target
        for entry in reversed(turn_log):
            for write in reversed(entry.get("writes", [])):
                if write.get("artifact_type") not in accepted_parent_types:
                    continue
                if decision.linked_task and write.get("linked_task") != decision.linked_task:
                    continue
                parent_id = write.get("artifact_id")
                if not parent_id:
                    continue
                if not decision.linked_task and write.get("linked_task"):
                    decision.linked_task = write.get("linked_task")
                decision.parent_artifact_id = parent_id
                decision.relation_type = relation_type
                return
