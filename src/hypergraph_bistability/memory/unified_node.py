"""
Unified Node System - Skills and Memories as the Same
=====================================================

This module implements a unified memory system where:
- Skills and memories are both "nodes" in the memory graph
- Skills can be retrieved and executed like memories are recalled
- The system learns which nodes (skills + memories) are most effective

Core Concept:
- MemoryNode: represents both "knowledge" and "capabilities"
- Unified retrieval: search without type distinction
- Execution: skills execute, memories recall
- Effectiveness tracking: learn from usage
"""

from __future__ import annotations

import sqlite3
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect


class NodeType(Enum):
    """Types of nodes in the unified system."""
    MEMORY = "memory"           # Pure knowledge/preference
    SKILL = "skill"             # Executable capability
    PROCEDURE = "procedure"      # Multi-step process
    DECISION = "decision"       # Decision record
    FACT = "fact"               # Verified fact
    PREFERENCE = "preference"    # User preference
    TASK = "task"               # Task state
    CHAT = "chat"               # Conversational content


class NodeStatus(Enum):
    """Node lifecycle status."""
    ACTIVE = "active"
    DORMANT = "dormant"
    DEPRECATED = "deprecated"
    LEARNING = "learning"  # Being evaluated


@dataclass
class SkillDefinition:
    """Definition of a skill that can be executed."""
    name: str
    code: str                   # Python code or reference
    parameters: Dict[str, Any]  # Parameter schema
    handler: Optional[Callable] = None  # Actual execution function
    description: str = ""
    category: str = "general"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "code": self.code,
            "parameters": self.parameters,
            "description": self.description,
            "category": self.category,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> SkillDefinition:
        return cls(
            name=data["name"],
            code=data["code"],
            parameters=data.get("parameters", {}),
            description=data.get("description", ""),
            category=data.get("category", "general"),
        )


@dataclass
class UnifiedNode:
    """
    Unified node representing both memory and skill.
    
    This is the core abstraction that treats:
    - Memories as "passive" nodes that provide context
    - Skills as "active" nodes that can be executed
    """
    
    id: int
    content: str                # Text content (for memories) or skill name (for skills)
    node_type: NodeType        # Type of node
    status: NodeStatus          # Lifecycle status
    
    # Content
    skill_def: Optional[SkillDefinition] = None  # For skill nodes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Effectiveness tracking
    effectiveness: float = 0.5  # 0-1, learned over time
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Temporal
    created_at: float = 0
    last_used: float = 0
    last_evaluated: float = 0
    
    # Context
    session_id: Optional[str] = None
    parent_id: Optional[int] = None  # For derived nodes
    
    # Layer (0=working, 1=episodic, 2=durable)
    layer: int = 0
    
    def is_skill(self) -> bool:
        return self.node_type in [NodeType.SKILL, NodeType.PROCEDURE]
    
    def is_memory(self) -> bool:
        return not self.is_skill()
    
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    def should_promote(self) -> bool:
        """Decide if this node should be promoted to higher layer."""
        if self.is_skill():
            # Skills promoted based on effectiveness
            return self.effectiveness > 0.7 and self.activation_count > 5
        else:
            # Memories promoted based on importance and access
            return self.effectiveness > 0.6 and self.activation_count > 3
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "skill_def": self.skill_def.to_dict() if self.skill_def else None,
            "metadata": self.metadata,
            "effectiveness": self.effectiveness,
            "activation_count": self.activation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "layer": self.layer,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> UnifiedNode:
        skill_def = None
        if data.get("skill_def"):
            skill_def = SkillDefinition.from_dict(data["skill_def"])
        
        return cls(
            id=data["id"],
            content=data["content"],
            node_type=NodeType(data["node_type"]),
            status=NodeStatus(data.get("status", "active")),
            skill_def=skill_def,
            metadata=data.get("metadata", {}),
            effectiveness=data.get("effectiveness", 0.5),
            activation_count=data.get("activation_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            created_at=data.get("created_at", 0),
            last_used=data.get("last_used", 0),
            layer=data.get("layer", 0),
            session_id=data.get("session_id"),
            parent_id=data.get("parent_id"),
        )


class UnifiedNodeStore:
    """
    Persistent store for unified nodes (memories + skills).
    
    Features:
    - SQLite-based storage
    - Unified search across all node types
    - Effectiveness-based retrieval ranking
    - Automatic learning from usage
    """
    
    def __init__(
        self,
        db_path: str = "unified_nodes.db",
        enable_embeddings: bool = False,
    ):
        self.db_path = db_path
        self.enable_embeddings = enable_embeddings
        self.conn: Optional[sqlite3.Connection] = None
        self._skill_handlers: Dict[str, Callable] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Unified nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                skill_def TEXT,
                metadata TEXT DEFAULT '{}',
                effectiveness REAL DEFAULT 0.5,
                activation_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL,
                last_evaluated REAL NOT NULL,
                layer INTEGER DEFAULT 0,
                session_id TEXT,
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES nodes(id)
            )
        """)
        
        # Indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_layer ON nodes(layer)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_effectiveness ON nodes(effectiveness DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id)
        """)
        
        # Usage history for learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL,
                used_at REAL NOT NULL,
                success INTEGER,
                feedback REAL,
                context TEXT,
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            )
        """)
        
        self.conn.commit()
    
    def register_skill_handler(self, name: str, handler: Callable) -> None:
        """Register a handler function for executing skills."""
        self._skill_handlers[name] = handler
    
    def add_node(
        self,
        content: str,
        node_type: NodeType,
        skill_def: Optional[SkillDefinition] = None,
        metadata: Optional[Dict] = None,
        layer: int = 0,
        session_id: Optional[str] = None,
        parent_id: Optional[int] = None,
        effectiveness: float = 0.5,
    ) -> int:
        """Add a new unified node."""
        now = time.time()
        
        cursor = self.conn.cursor()
        
        # Check for duplicate
        cursor.execute(
            "SELECT id FROM nodes WHERE content = ? AND node_type = ?",
            (content, node_type.value)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            cursor.execute("""
                UPDATE nodes 
                SET last_used = ?, activation_count = activation_count + 1,
                    effectiveness = MAX(effectiveness, ?)
                WHERE id = ?
            """, (now, effectiveness, existing["id"]))
            self.conn.commit()
            return existing["id"]
        
        # Serialize skill definition
        skill_def_json = None
        if skill_def:
            skill_def_json = json.dumps(skill_def.to_dict())
        
        # Use empty string instead of None to avoid NULL issues in SQLite
        session_id_str = session_id if session_id is not None else ""
        
        cursor.execute("""
            INSERT INTO nodes 
            (content, node_type, status, skill_def, metadata, effectiveness,
             activation_count, success_count, failure_count, created_at, 
             last_used, last_evaluated, layer, session_id, parent_id)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?, ?, ?, ?, ?)
        """, (
            content, node_type.value, "active", skill_def_json,
            json.dumps(metadata or {}), effectiveness,
            now, now, now, layer, session_id_str, parent_id
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_node(self, node_id: int) -> Optional[UnifiedNode]:
        """Get a node by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_node(row)
    
    def retrieve(
        self,
        query: str,
        node_types: Optional[List[NodeType]] = None,
        layer: Optional[int] = None,
        min_effectiveness: float = 0.0,
        limit: int = 10,
    ) -> List[UnifiedNode]:
        """
        Unified retrieval - search across all node types.
        
        Returns nodes ranked by:
        1. Relevance to query (keyword match)
        2. Effectiveness score
        3. Recency
        """
        cursor = self.conn.cursor()
        
        sql = """
            SELECT * FROM nodes 
            WHERE LOWER(content) LIKE ?
            AND effectiveness >= ?
            AND status != 'deprecated'
        """
        params = [f"%{query.lower()}%", min_effectiveness]
        
        if node_types:
            type_strs = [t.value for t in node_types]
            placeholders = ",".join(["?"] * len(type_strs))
            sql += f" AND node_type IN ({placeholders})"
            params.extend(type_strs)
        
        if layer is not None:
            sql += " AND layer = ?"
            params.append(layer)
        
        # Rank by effectiveness and recency
        sql += " ORDER BY effectiveness DESC, last_used DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        
        return [self._row_to_node(row) for row in cursor.fetchall()]
    
    def retrieve_skills(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[UnifiedNode]:
        """Retrieve only skill nodes."""
        skills = self.retrieve(query, node_types=[NodeType.SKILL, NodeType.PROCEDURE], limit=limit)
        
        if category:
            skills = [s for s in skills if s.metadata.get("category") == category]
        
        return skills
    
    def retrieve_memories(
        self,
        query: str,
        limit: int = 10,
    ) -> List[UnifiedNode]:
        """Retrieve only memory nodes."""
        return self.retrieve(
            query, 
            node_types=[NodeType.MEMORY, NodeType.FACT, NodeType.PREFERENCE, NodeType.DECISION],
            limit=limit
        )
    
    def execute_skill(self, node_id: int, **kwargs) -> Any:
        """
        Execute a skill node.
        
        Returns:
            - Success: (result, True)
            - Failure: (error_message, False)
        """
        node = self.get_node(node_id)
        
        if not node or not node.is_skill():
            return (f"Node {node_id} is not a skill", False)
        
        if not node.skill_def:
            return (f"Node {node_id} has no skill definition", False)
        
        skill_name = node.skill_def.name
        
        # Try registered handler first
        if skill_name in self._skill_handlers:
            try:
                result = self._skill_handlers[skill_name](**kwargs)
                self._record_activation(node_id, success=True)
                return (result, True)
            except Exception as e:
                self._record_activation(node_id, success=False, feedback=0.0)
                return (str(e), False)
        
        # SECURITY FIX: Removed exec-based code execution
        # Only registered handlers are allowed to execute skills
        # This prevents remote code execution (RCE) vulnerabilities
        
        return (f"No registered handler for skill {skill_name}", False)
        
        return ("No handler or code found for skill", False)
    
    def _record_activation(
        self, 
        node_id: int, 
        success: Optional[bool],
        feedback: Optional[float] = None,
    ) -> None:
        """Record node usage for learning."""
        now = time.time()
        
        cursor = self.conn.cursor()
        
        # Update node stats
        if success is not None:
            if success:
                cursor.execute("""
                    UPDATE nodes 
                    SET activation_count = activation_count + 1,
                        success_count = success_count + 1,
                        last_used = ?,
                        effectiveness = MIN(1.0, effectiveness + 0.05)
                    WHERE id = ?
                """, (now, node_id))
            else:
                cursor.execute("""
                    UPDATE nodes 
                    SET activation_count = activation_count + 1,
                        failure_count = failure_count + 1,
                        last_used = ?,
                        effectiveness = MAX(0.0, effectiveness - 0.05)
                    WHERE id = ?
                """, (now, node_id))
        else:
            cursor.execute("""
                UPDATE nodes 
                SET activation_count = activation_count + 1,
                    last_used = ?
                WHERE id = ?
            """, (now, node_id))
        
        # Record in history
        cursor.execute("""
            INSERT INTO usage_history (node_id, used_at, success, feedback)
            VALUES (?, ?, ?, ?)
        """, (node_id, now, 1 if success else 0 if success is False else None, feedback))
        
        self.conn.commit()
    
    def learn_from_feedback(self, node_id: int, feedback: float) -> None:
        """
        Learn from explicit feedback.
        
        Feedback: -1 (very bad) to 1 (very good)
        """
        cursor = self.conn.cursor()
        
        # Update effectiveness based on feedback
        delta = feedback * 0.1  # 10% adjustment per feedback
        
        # SQLite doesn't have CLAMP, use CASE
        cursor.execute("""
            UPDATE nodes 
            SET effectiveness = CASE 
                WHEN effectiveness + ? > 1.0 THEN 1.0
                WHEN effectiveness + ? < 0.0 THEN 0.0
                ELSE effectiveness + ?
            END,
            last_evaluated = ?
            WHERE id = ?
        """, (delta, delta, delta, time.time(), node_id))
        
        self.conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the node store."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total nodes
        cursor.execute("SELECT COUNT(*) as count FROM nodes")
        stats["total"] = cursor.fetchone()["count"]
        
        # By type
        cursor.execute("""
            SELECT node_type, COUNT(*) as count, AVG(effectiveness) as avg_effectiveness
            FROM nodes GROUP BY node_type
        """)
        stats["by_type"] = {
            row["node_type"]: {
                "count": row["count"],
                "avg_effectiveness": row["avg_effectiveness"]
            }
            for row in cursor.fetchall()
        }
        
        # Top skills
        cursor.execute("""
            SELECT content, activation_count, success_count, effectiveness
            FROM nodes WHERE node_type = 'skill'
            ORDER BY activation_count DESC LIMIT 5
        """)
        stats["top_skills"] = [
            dict(row) for row in cursor.fetchall()
        ]
        
        return stats
    
    def _row_to_node(self, row: sqlite3.Row) -> UnifiedNode:
        """Convert database row to UnifiedNode."""
        skill_def = None
        if row["skill_def"]:
            skill_def = SkillDefinition.from_dict(json.loads(row["skill_def"]))
        
        return UnifiedNode(
            id=row["id"],
            content=row["content"],
            node_type=NodeType(row["node_type"]),
            status=NodeStatus(row["status"]),
            skill_def=skill_def,
            metadata=json.loads(row["metadata"]),
            effectiveness=row["effectiveness"],
            activation_count=row["activation_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            created_at=row["created_at"],
            last_used=row["last_used"],
            layer=row["layer"],
            session_id=row["session_id"],
            parent_id=row["parent_id"],
        )
    
    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


class UnifiedNodeManager:
    """
    High-level manager for unified nodes.
    
    Provides:
    - Simplified API for common operations
    - Automatic memory/skill handling
    - Learning and adaptation
    """
    
    def __init__(self, db_path: str = "unified_nodes.db"):
        self.store = UnifiedNodeStore(db_path)
    
    def remember(self, content: str, node_type: NodeType = NodeType.MEMORY, **kwargs) -> int:
        """Add a memory node."""
        return self.store.add_node(
            content=content,
            node_type=node_type,
            metadata=kwargs,
            layer=0,
        )
    
    def learn_skill(
        self,
        name: str,
        code: str,
        description: str = "",
        category: str = "general",
        parameters: Optional[Dict] = None,
    ) -> int:
        """Add a skill node."""
        skill_def = SkillDefinition(
            name=name,
            code=code,
            description=description,
            category=category,
            parameters=parameters or {},
        )
        
        return self.store.add_node(
            content=name,
            node_type=NodeType.SKILL,
            skill_def=skill_def,
            metadata={"category": category},
            layer=1,  # Skills start in episodic layer
        )
    
    def recall(self, query: str, limit: int = 5) -> List[UnifiedNode]:
        """Recall relevant nodes (memories and skills)."""
        return self.store.retrieve(query, limit=limit)
    
    def recall_skills(self, query: str, limit: int = 3) -> List[UnifiedNode]:
        """Recall only skills."""
        return self.store.retrieve_skills(query, limit=limit)
    
    def use_skill(self, skill_name: str, **kwargs) -> tuple:
        """Use a skill by name."""
        # Find the skill node
        skills = self.store.retrieve_skills(skill_name)
        
        if not skills:
            return (f"Skill '{skill_name}' not found", False)
        
        # Use the most effective one
        skill = skills[0]
        return self.store.execute_skill(skill.id, **kwargs)
    
    def provide_feedback(self, content: str, feedback: float) -> None:
        """Provide feedback on a node (by content)."""
        cursor = self.store.conn.cursor()
        cursor.execute(
            "SELECT id FROM nodes WHERE content = ? LIMIT 1",
            (content,)
        )
        row = cursor.fetchone()
        
        if row:
            self.store.learn_from_feedback(row["id"], feedback)
    
    def get_stats(self) -> Dict[str, Any]:
        return self.store.get_statistics()
    
    def close(self):
        self.store.close()


# Factory function
def create_unified_memory(db_path: str = "unified_nodes.db") -> UnifiedNodeManager:
    """Create a unified memory manager."""
    return UnifiedNodeManager(db_path)
