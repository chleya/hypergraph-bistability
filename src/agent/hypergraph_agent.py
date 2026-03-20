"""
HypergraphAgent: Complete Agent with Physics-Based Memory
=========================================================

A full-featured AI agent that combines:
- Hypergraph memory (AgentMemoryEnhanced)
- Embedding-based storage (EmbeddingMemoryMapper)
- Adaptive cognitive control (AdaptiveController)
- LLM inference (OpenAI-compatible)

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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np


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
        Base URL for OpenAI-compatible API
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
        
        api_key = llm_api_key or os.environ.get("OPENAI_API_KEY")
        
        from agent.agent_memory_enhanced import AgentMemoryEnhanced
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
                from agent.embedding_memory import EmbeddingMemoryMapper
                self.embedding_mapper = EmbeddingMemoryMapper(
                    k=k, L=L,
                    api_key=api_key,
                    persist_dir=f".{name}_embeddings" if use_chromadb else None
                )
                self.embedding_mapper.set_group_labels(self.group_labels)
            except ImportError:
                print("Warning: embedding_memory not available, falling back to slot-based")
        
        from agent.adaptive_controller import AdaptiveController
        lc = self.memory.get_lambda_c() or 0.044
        self.controller = AdaptiveController(k=k, lambda_c=lc)
        
        self.llm_client = None
        self.llm_model = llm_model
        if api_key:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    api_key=api_key,
                    base_url=llm_base_url
                )
            except ImportError:
                pass
        
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """You are an AI assistant with a unique memory system based on physics.

Your memory is organized as a k×L matrix where:
- k groups represent different aspects (work, personal, technical, etc.)
- L layers represent different dimensions (current context, preferences, etc.)

Your responses should:
1. Be concise and helpful
2. Draw on the active memory contexts when relevant
3. Adapt your level of detail based on the topic

Memory state is shown in parentheses like (λ=0.02, r=0.5, regime=moderate-coupling).
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
        conflict_level = 0.0
        
        if hasattr(self.memory.detector, 'detect_conflict'):
            try:
                conflict_level, _ = self.memory.detector.detect_conflict(user_input)
            except:
                conflict_level = self._simple_conflict_detection(user_input)
        else:
            conflict_level = self._simple_conflict_detection(user_input)
        
        lambda_value, mu_value, suggestion = self.controller.update(
            user_input, conflict_level
        )
        
        if use_adaptive:
            self.memory.lambda_ = lambda_value
            self.memory.mu = mu_value
        
        if self.embedding_mapper:
            group, layer = self.embedding_mapper.find_slot(user_input, action="write")
        else:
            group = len(self.conversation_history) % self.k
            layer = (len(self.conversation_history) // self.k) % self.L
        
        self.memory.write(user_input, group=group, layer=layer)
        
        if self.embedding_mapper and self.embedding_mapper.store:
            self.embedding_mapper.store_memory(
                user_input, group, layer,
                metadata={"turn": len(self.conversation_history)}
            )
        
        context = self.memory.get_context_for_llm()
        
        if return_context:
            return context
        
        if suggestion:
            user_input = f"{user_input}\n\n{suggestion}"
        
        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n\nCurrent memory state:\n{context}"},
            *self.conversation_history[-5:],
            {"role": "user", "content": user_input}
        ]
        
        if self.llm_client:
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                assistant_response = response.choices[0].message.content
            except Exception as e:
                assistant_response = f"[LLM error: {e}] I understand your message. How can I help?"
        else:
            controller_state = self.controller.get_state_summary()
            assistant_response = self._mock_response(
                user_input, context, controller_state
            )
        
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        self.memory.write(assistant_response, group=group, layer=layer)
        
        return assistant_response
    
    def _simple_conflict_detection(self, text: str) -> float:
        """Simple keyword-based conflict detection."""
        conflict_keywords = ["but", "however", "although", "actually", "wait", 
                            "on the other hand", "or", "either", "neither"]
        conflict_count = sum(1 for kw in conflict_keywords if kw in text.lower())
        return min(1.0, conflict_count * 0.15)
    
    def _mock_response(
        self,
        user_input: str,
        context: str,
        controller_state: Dict
    ) -> str:
        """Generate a mock response when LLM is not available."""
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
            "conversation_turns": len(self.conversation_history) // 2
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save agent state to file."""
        if filepath is None:
            filepath = f"{self.name}_state.json"
        
        self.memory.save(filepath)
        
        with open(filepath.replace(".json", "_history.json"), 'w') as f:
            json.dump({
                "conversation_history": self.conversation_history,
                "system_prompt": self.system_prompt
            }, f)
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        self.memory = self.memory.load(filepath)
        
        history_path = filepath.replace(".json", "_history.json")
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
                self.conversation_history = data.get("conversation_history", [])
                self.system_prompt = data.get("system_prompt", self._default_system_prompt())
        except FileNotFoundError:
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
                val = M[i, l]
                if val > 0.8:
                    ch = "█"
                elif val > 0.5:
                    ch = "▓"
                elif val > 0.3:
                    ch = "▒"
                elif val > 0.1:
                    ch = "░"
                else:
                    ch = " "
                row += f"{ch} "
            row += f"| {state.groups[i]:.2f}"
            lines.append(row)
        
        lines.append("")
        lines.append(f"Controller: {self.controller.get_state_summary()}")
        
        return "\n".join(lines)


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
        self.chat_history: List[Dict[str, str]] = []
    
    @property
    def memory_variables(self) -> List[str]:
        """Required by LangChain."""
        return ["history", "memory_state"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LangChain."""
        state = self.agent.get_memory_state()
        context = self.agent.memory.get_context_for_llm()
        
        history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.chat_history[-10:]
        ])
        
        return {
            "history": history,
            "memory_state": context
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context after each chain step."""
        user_input = inputs.get("input", "")
        assistant_output = outputs.get("response", "")
        
        if user_input:
            self.chat_history.append({"role": "user", "content": user_input})
        if assistant_output:
            self.chat_history.append({"role": "assistant", "content": assistant_output})
    
    def clear(self) -> None:
        """Clear memory."""
        self.agent.reset_memory()
        self.chat_history = []


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
