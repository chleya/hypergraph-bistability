"""
A/B Evaluation: Working-set Context Impact (Mock LLM)

Tests whether working-set context is properly injected into prompts.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hypergraph_bistability.agent import HypergraphAgent
from hypergraph_bistability.agent.runtime import ContextAssembler


@dataclass
class ABResult:
    """Result of A/B comparison."""
    test_name: str
    with_ws_in_prompt: bool
    without_ws_in_prompt: bool
    working_set_detected: bool


class ABWorkingSetPromptEvaluator:
    """Evaluate whether working-set context is properly injected into prompts."""
    
    def __init__(self):
        pass
    
    def _get_prompt_with_ws(self, agent: HypergraphAgent, user_input: str) -> str:
        """Get the full prompt with working-set context."""
        # Simulate what happens in process_turn
        from hypergraph_bistability.memory.policies import RetrievalPolicy
        retrieval_policy = RetrievalPolicy()
        
        retrieved = retrieval_policy.collect(
            user_input,
            memory=agent.memory,
            turn_log=agent.turn_log,
            embedding_mapper=None,
            top_k=3,
        )
        
        memory_context = agent.memory.get_context_for_llm()
        
        # Get working-set context
        working_set_context = agent.turn_processor._generate_working_set_context(agent)
        
        # Build messages
        messages = agent.turn_processor.context_assembler.build_messages(
            system_prompt=agent.system_prompt,
            memory_context=memory_context,
            retrieved_items=retrieved,
            conversation_history=agent.conversation_history,
            user_input=user_input,
            working_set_context=working_set_context,
        )
        
        # Return system prompt
        return messages[0]["content"] if messages else ""
    
    def _get_prompt_without_ws(self, agent: HypergraphAgent, user_input: str) -> str:
        """Get the prompt without working-set context."""
        from hypergraph_bistability.memory.policies import RetrievalPolicy
        retrieval_policy = RetrievalPolicy()
        
        retrieved = retrieval_policy.collect(
            user_input,
            memory=agent.memory,
            turn_log=agent.turn_log,
            embedding_mapper=None,
            top_k=3,
        )
        
        memory_context = agent.memory.get_context_for_llm()
        
        # Empty working-set context
        working_set_context = ""
        
        messages = agent.turn_processor.context_assembler.build_messages(
            system_prompt=agent.system_prompt,
            memory_context=memory_context,
            retrieved_items=retrieved,
            conversation_history=agent.conversation_history,
            user_input=user_input,
            working_set_context=working_set_context,
        )
        
        return messages[0]["content"] if messages else ""
    
    def run_prompt_injection_test(self) -> List[ABResult]:
        """Test prompt injection with working-set context."""
        results = []
        
        # Setup agent with some memory
        agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
        
        # Add some memory
        agent.memory.write("决定使用 Python 开发项目", group=0, layer=1)
        agent.memory.write("Blocker: 需要生产日志", group=0, layer=1)
        agent.memory.write("Plan: 1.检查 2.修复 3.测试", group=0, layer=1)
        
        test_input = "我之前决定用什么语言？"
        
        # Get prompts
        prompt_with = self._get_prompt_with_ws(agent, test_input)
        prompt_without = self._get_prompt_without_ws(agent, test_input)
        
        # Check for working-set indicators
        ws_indicators = [
            "Current task",
            "Blockers:",
            "Decisions:",
            "Active working set",
            "task_1",
        ]
        
        with_ws_found = []
        for indicator in ws_indicators:
            if indicator in prompt_with:
                with_ws_found.append(indicator)
        
        without_ws_found = []
        for indicator in ws_indicators:
            if indicator in prompt_without:
                without_ws_found.append(indicator)
        
        result = ABResult(
            test_name="prompt_injection",
            with_ws_in_prompt=len(with_ws_found) > 0,
            without_ws_in_prompt=len(without_ws_found) > 0,
            working_set_detected=len(with_ws_found) > 0,
        )
        results.append(result)
        
        # Print details
        print("\n" + "="*60)
        print("A/B Prompt Injection Test")
        print("="*60)
        
        print(f"\nWith Working-set Context:")
        print(f"  - Found indicators: {with_ws_found}")
        
        print(f"\nWithout Working-set Context:")
        print(f"  - Found indicators: {without_ws_found}")
        
        # Show prompt snippets
        print(f"\nPrompt WITH ws (first 500 chars):")
        print(prompt_with[:500])
        
        print(f"\nPrompt WITHOUT ws (first 500 chars):")
        print(prompt_without[:500])
        
        print("\n" + "="*60)
        print(f"RESULT: Working-set context {'INJECTED' if result.working_set_detected else 'NOT INJECTED'}")
        print("="*60)
        
        return results


class ABWorkingSetBehaviorEvaluator:
    """Evaluate actual behavior difference with mock LLM."""
    
    def __init__(self):
        pass
    
    def _create_mock_agent(self) -> HypergraphAgent:
        """Create agent with mock response."""
        agent = HypergraphAgent(k=3, L=2, use_embeddings=False)
        
        # Override generate_response to return mock
        def mock_response(user_input, memory_context, messages, retrieved_items):
            # Check if working-set context is in the messages
            full_prompt = str(messages)
            has_ws = "Active working set" in full_prompt or "Current task" in full_prompt
            
            if has_ws:
                return "Mock: Using working-set context"
            else:
                return "Mock: No working-set context"
        
        agent.generate_response = mock_response
        return agent
    
    def run_mock_test(self) -> Dict[str, Any]:
        """Run test with mock LLM."""
        
        # Setup agent with memory
        agent = self._create_mock_agent()
        agent.memory.write("决定使用 Python", group=0, layer=1)
        agent.memory.write("Blocker: 需要日志", group=0, layer=1)
        
        # With working-set context
        result_with = agent.process_turn("现在做什么？")
        
        # Without working-set context  
        original_method = agent.turn_processor._generate_working_set_context
        agent.turn_processor._generate_working_set_context = lambda a: ""
        result_without = agent.process_turn("现在做什么？")
        agent.turn_processor._generate_working_set_context = original_method
        
        with_text = result_with.assistant_response if hasattr(result_with, 'assistant_response') else str(result_with)
        without_text = result_without.assistant_response if hasattr(result_without, 'assistant_response') else str(result_without)
        
        print("\n" + "="*60)
        print("A/B Behavior Test (Mock LLM)")
        print("="*60)
        print(f"With WS:    {with_text}")
        print(f"Without WS: {without_text}")
        print("="*60)
        
        return {
            "with_ws": with_text,
            "without_ws": without_text,
            "different": with_text != without_text,
        }


if __name__ == "__main__":
    # Test 1: Prompt injection
    prompt_evaluator = ABWorkingSetPromptEvaluator()
    prompt_results = prompt_evaluator.run_prompt_injection_test()
    
    # Test 2: Mock behavior
    behavior_evaluator = ABWorkingSetBehaviorEvaluator()
    behavior_result = behavior_evaluator.run_mock_test()
