"""
LLM Integration Example - Agent Memory + OpenAI
================================================

Shows how to integrate AgentMemory with an LLM API
for context-aware response generation.

Requirements:
    pip install openai langchain langchain-openai
"""

from typing import Optional, List, Dict
import os
import numpy as np


class LLMAgentWithMemory:
    """
    Simple LLM agent with hypergraph memory integration.
    
    This is a reference implementation showing the integration pattern.
    For production use, implement proper error handling and rate limiting.
    """
    
    def __init__(
        self,
        memory_module,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.memory = memory_module
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        
        self._has_openai = False
        self.client = None
        
        if api_key or os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                actual_key = api_key or os.environ.get("OPENAI_API_KEY")
                actual_base = (base_url or os.environ.get("OPENAI_API_BASE")
                             or "https://api.openai.com/v1")
                self.client = OpenAI(api_key=actual_key, base_url=actual_base)
                self._has_openai = True
            except ImportError:
                print("Warning: OpenAI not installed properly.")
    
    def chat(
        self,
        user_message: str,
        include_memory_context: bool = True,
        max_history: int = 10
    ) -> str:
        """
        Send a message and get a response.
        
        Parameters
        ----------
        user_message : str
            User's message
        include_memory_context : bool
            Include memory state in context
        max_history : int
            Max conversation turns to include
        
        Returns
        -------
        str
            Assistant response
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
        
        if include_memory_context:
            adjust_result = self.memory.process_prompt(user_message)
            memory_context = self.memory.get_context_for_llm()
            context_prompt = (
                f"Current memory state:\n{memory_context}\n\n"
                f"User message: {user_message}"
            )
        else:
            context_prompt = user_message
        
        if self._has_openai:
            response = self._call_openai(context_prompt)
        else:
            response = self._mock_response(context_prompt)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def _mock_response(self, prompt: str) -> str:
        """Mock response for testing without API."""
        memory_context = self.memory.get_context_for_llm()
        return (
            f"[Mock response based on memory context]\n"
            f"Memory state: {self.memory.current_mode}\n"
            f"λ={self.memory.lambda_:.3f}\n"
            f"Prompt was: {prompt[:100]}..."
        )
    
    def get_memory_status(self) -> str:
        """Get current memory status."""
        return self.memory.status()
    
    def reset_memory(self) -> None:
        """Reset memory to neutral state."""
        self.memory.M = np.full((self.memory.k, self.memory.L), 0.5)
        self.memory.lambda_ = 0.0
        self.memory.mu = 0.0
        self.memory.current_mode = "neutral"


class LangChainAgentWithMemory:
    """
    LangChain agent with hypergraph memory integration.
    
    Shows how to use AgentMemory as a LangChain tool.
    Includes physics-based control via set_n_high.
    """
    
    def __init__(self, memory_module, llm_provider: str = "openai",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.memory = memory_module
        self.llm_provider = llm_provider
        
        try:
            from langchain.agents import AgentExecutor, create_openai_functions_agent
            from langchain_openai import ChatOpenAI
            from langchain.tools import Tool
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            self._has_langchain = True
            self._api_key = api_key
            self._base_url = base_url
            self._setup_tools()
        except ImportError as e:
            self._has_langchain = False
            print(f"Warning: LangChain not properly installed: {e}")
    
    def _setup_tools(self):
        """Setup LangChain tools from memory functions."""
        tools = [
            Tool(
                name="read_memory",
                func=lambda x: self.memory.read().to_dict(),
                description="Read current memory state. Returns active patterns and group/layer info."
            ),
            Tool(
                name="get_memory_context",
                func=lambda x: self.memory.get_context_for_llm(),
                description="Get memory context string for LLM. Use before responding to know current state."
            ),
            Tool(
                name="switch_memory_mode",
                func=lambda mode: self.memory.switch_mode(mode) or f"Switched to {mode}",
                description="Switch memory mode. Modes: neutral (multi-attractor), exploratory (light coupling), focused (near λ_c), creative (moderate coupling + anti-sync)."
            ),
            Tool(
                name="set_high_groups",
                func=lambda n_str: str(self.memory.set_n_high(int(n_str))),
                description="Set the number of HIGH active groups (n_high). Input: integer 0 to k. Higher = more contexts collapse. Use to force focus or expand memory. Example: set_high_groups(1) for maximum focus."
            ),
            Tool(
                name="get_collapse_status",
                func=lambda _: (
                    f"λ={self.memory.lambda_:.4f}, "
                    f"λ_c={self.memory.get_lambda_c():.4f}, "
                    f"r={self.memory.lambda_/self.memory.get_lambda_c():.2f}"
                ),
                description="Get current collapse ratio r=λ/λ_c. Shows how close the system is to WTA collapse."
            ),
        ]
        self.tools = tools
    
    def create_agent(self, system_prompt: str):
        """Create a LangChain agent with memory tools."""
        if not self._has_langchain:
            raise ImportError("LangChain is required for this function")
        
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        llm_kwargs = {"model": "gpt-4", "temperature": 0.7}
        if self._api_key:
            llm_kwargs["api_key"] = self._api_key
        if self._base_url:
            llm_kwargs["base_url"] = self._base_url
        llm = ChatOpenAI(**llm_kwargs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)


def demo():
    """Demo showing the integration pattern with physics-based memory control."""
    print("=" * 60)
    print("LLM Integration Demo - Physics-Based Memory")
    print("=" * 60)
    
    from hypergraph_bistability.memory.agent_memory import AgentMemory, CollapsController, _regime_name
    
    print("\n1. Setup Agent Memory (physics-enabled):")
    memory = AgentMemory(k=3, L=2, name="assistant", use_physics_control=True)
    memory.group_labels = ["professional", "personal", "technical"]
    memory.layer_labels = ["preferences", "context"]
    
    memory.write("User prefers concise answers", group=0, layer=0)
    memory.write("User likes code examples", group=2, layer=0)
    
    print(memory.status())
    print(f"   λ_c = {memory.get_lambda_c():.4f}")
    
    print("\n2. Physics-based mode switching:")
    for mode in ["neutral", "exploratory", "focused", "creative"]:
        memory.switch_mode(mode)
        lc = memory.get_lambda_c()
        r = memory.lambda_ / lc if lc else 0
        print(f"   {mode:15s}: λ={memory.lambda_:.4f}, r={r:.2f}, regime={_regime_name(r)}")
    
    print("\n3. n_high control:")
    memory.set_n_high(1)
    print(f"   set_n_high(1): n_high={memory.get_n_high_groups()}, λ={memory.lambda_:.4f}")
    memory.set_n_high(2)
    print(f"   set_n_high(2): n_high={memory.get_n_high_groups()}, λ={memory.lambda_:.4f}")
    
    print("\n4. Memory context with physics info:")
    print(memory.get_context_for_llm())
    
    print("\n5. Chat with memory context (mock mode):")
    agent = LLMAgentWithMemory(memory, model="gpt-4")
    
    prompts = [
        "Hello!",
        "How do I sort a list in Python?",
        "Actually wait, I meant JavaScript. But also keep it brief.",
    ]
    
    for prompt in prompts:
        print(f"\n   User: {prompt}")
        response = agent.chat(prompt)
        print(f"   Assistant: {response[:80]}...")
        print(f"   Memory: λ={memory.lambda_:.4f}, r={memory.lambda_/memory.get_lambda_c():.2f}")
    
    print("\n" + "=" * 60)
    print("For real API integration:")
    print("1. pip install openai langchain langchain-openai")
    print("2. Use LLMAgentWithMemory(api_key=..., base_url=...)")
    print("3. Supports MiniMax: base_url='https://api.minimaxi.com/v1'")


if __name__ == "__main__":
    demo()
