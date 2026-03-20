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
        api_key: Optional[str] = None
    ):
        self.memory = memory_module
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        
        self._has_openai = False
        self.client = None
        
        if api_key or os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
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
    """
    
    def __init__(self, memory_module, llm_provider: str = "openai"):
        self.memory = memory_module
        self.llm_provider = llm_provider
        
        try:
            from langchain.agents import AgentExecutor, create_openai_functions_agent
            from langchain_openai import ChatOpenAI
            from langchain.tools import Tool
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            self._has_langchain = True
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
                description="Read current memory state. Returns active patterns."
            ),
            Tool(
                name="get_memory_context",
                func=lambda x: self.memory.get_context_for_llm(),
                description="Get memory context for LLM. Use before responding."
            ),
            Tool(
                name="switch_memory_mode",
                func=lambda mode: self.memory.switch_mode(mode) or f"Switched to {mode}",
                description="Switch memory mode. Modes: neutral, exploratory, focused, professional, casual."
            )
        ]
        self.tools = tools
    
    def create_agent(self, system_prompt: str):
        """Create a LangChain agent with memory tools."""
        if not self._has_langchain:
            raise ImportError("LangChain is required for this function")
        
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)


def demo():
    """Demo showing the integration pattern."""
    print("=" * 60)
    print("LLM Integration Demo")
    print("=" * 60)
    
    from agent.agent_memory import AgentMemory
    
    print("\n1. Setup Agent Memory:")
    memory = AgentMemory(k=3, L=2, name="assistant", auto_adjust=True)
    memory.group_labels = ["professional", "personal", "technical"]
    memory.layer_labels = ["preferences", "context"]
    
    memory.write("User prefers concise answers", group=0, layer=0)
    memory.write("User likes code examples", group=2, layer=0)
    
    print(memory.status())
    
    print("\n2. Setup LLM Agent (mock mode):")
    agent = LLMAgentWithMemory(memory, model="gpt-4")
    
    print("\n3. Chat with memory context:")
    prompts = [
        "Hello!",
        "How do I sort a list in Python?",
        "Actually wait, I meant JavaScript. But also keep it brief.",
    ]
    
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        response = agent.chat(prompt)
        print(f"Assistant: {response}")
        print(f"Memory: {agent.get_memory_status()}")
    
    print("\n4. Memory context string for LLM:")
    print(memory.get_context_for_llm())
    
    print("\n" + "=" * 60)
    print("For real OpenAI integration:")
    print("1. pip install openai langchain langchain-openai")
    print("2. export OPENAI_API_KEY=your_key")
    print("3. Use LLMAgentWithMemory with _has_openai=True")


if __name__ == "__main__":
    demo()