"""Turn lifecycle for practical agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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
        messages = self.context_assembler.build_messages(
            system_prompt=agent.system_prompt,
            memory_context=memory_context,
            retrieved_items=retrieved,
            conversation_history=agent.conversation_history,
            user_input=effective_input,
        )
        assistant_response = agent.generate_response(
            user_input=effective_input,
            memory_context=memory_context,
            messages=messages,
            retrieved_items=retrieved,
        )

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
