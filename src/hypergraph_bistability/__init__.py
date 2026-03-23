"""Public package entry points for the hypergraph bistability project.

This package provides:
- HypergraphAgent: Main agent with memory and control
- AgentMemory: Core memory system
- AgentMemoryEnhanced: Extended memory with LLM integration
- Theoretical control functions: compute_lambda_c, get_all_lambda_c

For detailed APIs, access submodules:
- hypergraph_bistability.agent: QueryLayer, WorkingSet, etc.
- hypergraph_bistability.memory: DurableMemory, UnifiedNode, etc.
- hypergraph_bistability.evals: Evaluation runners
"""

# High-level APIs only
from .agent.hypergraph_agent import HypergraphAgent
from .memory.agent_memory import AgentMemory
from .memory.llm_memory import AgentMemoryEnhanced
from .control import compute_lambda_c, get_all_lambda_c, power_law_approximation

__all__ = [
    # Core objects (high-level entry points)
    "HypergraphAgent",
    "AgentMemory",
    "AgentMemoryEnhanced",
    # Theoretical control
    "compute_lambda_c",
    "get_all_lambda_c",
    "power_law_approximation",
]

__version__ = "0.1.0"
