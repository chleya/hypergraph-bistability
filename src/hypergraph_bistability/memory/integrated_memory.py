"""
Integration: Unified Node System + AgentMemoryEnhanced
=====================================================

This module integrates the unified node system (skills + memories)
with the existing AgentMemoryEnhanced for persistent,
learnable memory management.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import json
import os
import logging

from .llm_memory import AgentMemoryEnhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from .unified_node import (
    UnifiedNodeManager,
    NodeType,
    SkillDefinition,
    UnifiedNode,
)


@dataclass
class IntegrationConfig:
    """Configuration for the integrated system."""
    # Unified node settings
    enable_unified_nodes: bool = True
    unified_db_path: str = "unified_memory.db"
    
    # When to use unified nodes
    use_for_persistence: bool = True      # Store to unified nodes for durability
    use_for_skill_storage: bool = True    # Store skills in unified nodes
    use_for_unified_retrieval: bool = True # Use unified retrieval for queries
    
    # Learning
    learn_from_execution: bool = True       # Track skill execution success
    learn_from_retrieval: bool = True      # Track retrieval usefulness


class IntegratedAgentMemory:
    """
    Integrated Agent Memory that combines:
    
    1. AgentMemoryEnhanced (hypergraph dynamics, conflict detection)
    2. UnifiedNodeManager (persistent skills + memories, learning)
    
    Benefits:
    - Skills persist across sessions like memories
    - Effectiveness tracking for both skills and memories
    - Unified retrieval interface
    - Automatic learning from usage
    """
    
    def __init__(
        self,
        # AgentMemoryEnhanced params
        k: int = 3,
        L: int = 2,
        alpha: float = 1.0,
        a_bistable: float = 0.5,
        lambda_: float = 0.0,
        mu: float = 0.0,
        name: str = "integrated",
        use_llm_detector: bool = False,
        use_llm_mapper: bool = False,
        api_key: Optional[str] = None,
        use_physics_control: bool = True,
        gamma: float = 0.0,
        # Integration params
        config: Optional[IntegrationConfig] = None,
    ):
        self.config = config or IntegrationConfig()
        
        # Core hypergraph memory
        self.hypergraph = AgentMemoryEnhanced(
            k=k, L=L, alpha=alpha, a_bistable=a_bistable,
            lambda_=lambda_, mu=mu, name=name,
            use_llm_detector=use_llm_detector,
            use_llm_mapper=use_llm_mapper,
            api_key=api_key,
            use_physics_control=use_physics_control,
            gamma=gamma,
        )
        
        # Unified node system (skills + persistent memories)
        self.unified: Optional[UnifiedNodeManager] = None
        if self.config.enable_unified_nodes:
            try:
                self.unified = UnifiedNodeManager(self.config.unified_db_path)
                logger.info(f"Unified nodes enabled: {self.config.unified_db_path}")
            except Exception as e:
                logger.warning(f"Failed to init unified nodes: {e}")
                self.unified = None
        
        # Skill registry (for quick lookup)
        self._skill_registry: Dict[str, int] = {}
        
        # Load existing skills
        if self.unified:
            self._load_skill_registry()
    
    def _load_skill_registry(self):
        """Load skill IDs into registry for quick lookup."""
        if not self.unified:
            return
        
        stats = self.unified.get_stats()
        if "skill" in stats.get("by_type", {}):
            skills = self.unified.store.retrieve_skills("")
            for skill in skills:
                self._skill_registry[skill.content] = skill.id
    
    # ==================== Memory Operations ====================
    
    def remember(
        self,
        content: str,
        importance: float = 0.5,
        kind: str = "memory",
        group: Optional[int] = None,
        layer: Optional[int] = None,
        **kwargs
    ) -> int:
        """
        Remember something (add to memory).
        
        This stores to BOTH hypergraph and unified nodes.
        """
        # Auto-assign group/layer if not specified
        if group is None or layer is None:
            # Use internal mapper or default
            group = group if group is not None else 0
            layer = layer if layer is not None else 0
        
        # Store in hypergraph
        self.hypergraph.write(content, group=group, layer=layer)
        
        # Store in unified nodes for persistence
        unified_id = None
        if self.unified and self.config.use_for_persistence:
            node_type_map = {
                "preference": NodeType.PREFERENCE,
                "fact": NodeType.FACT,
                "decision": NodeType.DECISION,
                "task": NodeType.TASK,
                "memory": NodeType.MEMORY,
            }
            node_type = node_type_map.get(kind, NodeType.MEMORY)
            
            unified_id = self.unified.remember(
                content=content,
                node_type=node_type,
                importance=importance,
                **kwargs
            )
        
        return unified_id if unified_id else group * 100 + layer
    
    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories.
        
        Uses unified retrieval if enabled.
        """
        results = []
        
        # Use unified retrieval
        if self.unified and self.config.use_for_unified_retrieval:
            unified_results = self.unified.recall(query, limit=limit)
            
            for node in unified_results:
                results.append({
                    "content": node.content,
                    "type": node.node_type.value,
                    "effectiveness": node.effectiveness,
                    "source": "unified",
                    "node_id": node.id,
                })
            
            return results
        
        return results
    
    # ==================== Skill Operations ====================
    
    def register_skill(
        self,
        name: str,
        code: str,
        description: str = "",
        category: str = "general",
        parameters: Optional[Dict] = None,
    ) -> int:
        """
        Register a skill in the system.
        
        Skills are stored in unified nodes for:
        - Persistence across sessions
        - Effectiveness tracking
        - Unified retrieval
        """
        if not self.unified or not self.config.use_for_skill_storage:
            raise RuntimeError("Unified nodes not enabled")
        
        skill_id = self.unified.learn_skill(
            name=name,
            code=code,
            description=description,
            category=category,
            parameters=parameters,
        )
        
        self._skill_registry[name] = skill_id
        return skill_id
    
    def use_skill(self, skill_name: str, **kwargs) -> Tuple[Any, bool]:
        """
        Use a skill by name.
        
        Tracks execution and learns from success/failure.
        """
        if not self.unified:
            return ("Unified nodes not enabled", False)
        
        result, success = self.unified.use_skill(skill_name, **kwargs)
        
        # Learn from execution
        if self.config.learn_from_execution and success:
            self.unified.provide_feedback(skill_name, 0.2)  # Positive
        elif self.config.learn_from_execution:
            self.unified.provide_feedback(skill_name, -0.2)  # Negative
        
        return result, success
    
    def get_skill(self, name: str) -> Optional[UnifiedNode]:
        """Get a skill by name."""
        if not self.unified:
            return None
        
        return self.unified.store.get_node(self._skill_registry.get(name, -1))
    
    def list_skills(self) -> List[Dict]:
        """List all registered skills."""
        if not self.unified:
            return []
        
        stats = self.unified.get_stats()
        return stats.get("top_skills", [])
    
    # ==================== Hybrid Operations ====================
    
    def recall_everything(self, query: str, limit: int = 5) -> Dict[str, List]:
        """
        Recall both memories and skills.
        
        Returns dict with 'memories' and 'skills' keys.
        """
        results = {
            "memories": [],
            "skills": [],
        }
        
        if not self.unified:
            # Fallback to hypergraph
            results["memories"] = self.retrieve(query, limit)
            return results
        
        # Unified retrieval
        all_nodes = self.unified.recall(query, limit=limit * 2)
        
        for node in all_nodes:
            if node.is_skill():
                results["skills"].append({
                    "name": node.content,
                    "description": node.skill_def.description if node.skill_def else "",
                    "category": node.metadata.get("category", "general"),
                    "effectiveness": node.effectiveness,
                    "usage_count": node.activation_count,
                })
            else:
                results["memories"].append({
                    "content": node.content,
                    "type": node.node_type.value,
                    "importance": node.effectiveness,
                })
        
        return results
    
    def suggest_skill(self, task: str) -> Optional[Dict]:
        """
        Suggest the best skill for a task.
        
        Uses effectiveness ranking to find the most reliable skill.
        """
        if not self.unified:
            return None
        
        skills = self.unified.recall_skills(task, limit=3)
        
        if not skills:
            return None
        
        best = skills[0]
        return {
            "name": best.content,
            "description": best.skill_def.description if best.skill_def else "",
            "effectiveness": best.effectiveness,
            "usage_count": best.activation_count,
        }
    
    # ==================== Persistence ====================
    
    def save(self, path: Optional[str] = None) -> None:
        """Save state to file."""
        # Save hypergraph state
        self.hypergraph.save(path)
        
        # Unified nodes are already persistent (SQLite)
        logger.info("State saved (hypergraph + unified nodes)")
    
    def load(self, path: Optional[str] = None) -> None:
        """Load state from file."""
        # Load hypergraph state - AgentMemoryEnhanced.load returns a NEW instance
        from hypergraph_bistability.memory.llm_memory import AgentMemoryEnhanced
        self.hypergraph = AgentMemoryEnhanced.load(path)
        
        # Reload skill registry
        self._load_skill_registry()
        
        logger.info("State loaded")
    
    def close(self) -> None:
        """Close all resources."""
        if self.unified:
            self.unified.close()
    
    # ==================== Statistics ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        # Get hypergraph info
        h_state = self.hypergraph.read()
        hypergraph_info = {
            "name": self.hypergraph.name,
            "k": self.hypergraph.k,
            "L": self.hypergraph.L,
            "mode": self.hypergraph.current_mode,
            "n_high_groups": int(np.sum(h_state.groups > 0.5)),
        }
        
        status = {
            "hypergraph": hypergraph_info,
        }
        
        if self.unified:
            status["unified"] = self.unified.get_stats()
        
        return status
    
    def __repr__(self) -> str:
        return f"IntegratedAgentMemory(name={self.hypergraph.name}, unified={self.unified is not None})"


def create_integrated_memory(
    name: str = "integrated",
    db_path: str = "unified_memory.db",
    **hypergraph_kwargs
) -> IntegratedAgentMemory:
    """Factory function to create integrated memory."""
    config = IntegrationConfig(
        enable_unified_nodes=True,
        unified_db_path=db_path,
    )
    
    return IntegratedAgentMemory(
        name=name,
        config=config,
        **hypergraph_kwargs
    )
