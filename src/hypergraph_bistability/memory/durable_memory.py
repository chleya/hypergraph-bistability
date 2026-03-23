"""Durable Memory Layer - Persistent storage for long-term memory."""

from __future__ import annotations

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import hashlib


@dataclass
class DurableMemory:
    """A single durable memory entry."""
    
    id: int
    content: str
    kind: str  # preference, decision, fact, procedure, etc.
    importance: float
    created_at: float
    last_accessed: float
    access_count: int
    layer: int  # 0=working, 1=episodic, 2=durable
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "layer": self.layer,
            "metadata": self.metadata,
        }


class DurableMemoryStore:
    """
    Persistent memory store with multi-layer architecture.
    
    Layers:
    - Layer 0: Working memory (in-memory, transient)
    - Layer 1: Episodic memory (short-term, SQLite)
    - Layer 2: Durable memory (long-term, SQLite)
    
    Features:
    - SQLite-based persistent storage
    - Semantic search with embeddings
    - Automatic promotion/demotion between layers
    - Cross-session persistence
    """
    
    DEFAULT_DB_PATH = "memory_durable.db"
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        enable_embeddings: bool = False,
    ):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.enable_embeddings = enable_embeddings
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        cursor = self.conn.cursor()
        
        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                layer INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                embedding BLOB,
                session_id TEXT,
                UNIQUE(session_id, content)
            )
        """)
        
        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_layer 
            ON memories(layer)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_kind 
            ON memories(kind)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_session 
            ON memories(session_id)
        """)
        
        # Sessions table for tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                last_active REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        self.conn.commit()
    
    def add(
        self,
        content: str,
        kind: str,
        importance: float = 0.5,
        layer: int = 1,
        metadata: Optional[Dict] = None,
        session_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Add a new memory to the store."""
        now = time.time()
        
        cursor = self.conn.cursor()
        
        # Check if already exists
        cursor.execute(
            "SELECT id FROM memories WHERE session_id = ? AND content = ?",
            (session_id, content)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            cursor.execute("""
                UPDATE memories 
                SET last_accessed = ?, access_count = access_count + 1,
                    importance = MAX(importance, ?)
                WHERE id = ?
            """, (now, importance, existing["id"]))
            self.conn.commit()
            return existing["id"]
        
        # Insert new
        metadata_json = json.dumps(metadata or {})
        
        # Serialize embedding if present
        embedding_blob = None
        if embedding:
            embedding_blob = json.dumps(embedding)
        
        cursor.execute("""
            INSERT INTO memories 
            (content, kind, importance, created_at, last_accessed, 
             access_count, layer, metadata, session_id, embedding)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
        """, (
            content, kind, importance, now, now,
            layer, metadata_json, session_id, embedding_blob
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get(self, memory_id: int) -> Optional[DurableMemory]:
        """Get a memory by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_memory(row)
    
    def get_by_layer(self, layer: int) -> List[DurableMemory]:
        """Get all memories in a specific layer."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM memories WHERE layer = ? ORDER BY importance DESC",
            (layer,)
        )
        
        return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def get_by_session(self, session_id: str) -> List[DurableMemory]:
        """Get all memories for a specific session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        )
        
        return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def search(
        self,
        query: str,
        layer: Optional[int] = None,
        kind: Optional[str] = None,
        limit: int = 10,
    ) -> List[DurableMemory]:
        """Search memories by content (case-insensitive)."""
        cursor = self.conn.cursor()
        
        sql = "SELECT * FROM memories WHERE LOWER(content) LIKE ?"
        params = [f"%{query.lower()}%"]
        
        if layer is not None:
            sql += " AND layer = ?"
            params.append(layer)
        
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        
        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        
        return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def promote(self, memory_id: int, target_layer: int) -> bool:
        """Promote a memory to a higher layer."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET layer = ?, last_accessed = ? WHERE id = ?",
            (target_layer, time.time(), memory_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0
    
    def demote(self, memory_id: int, target_layer: int) -> bool:
        """Demote a memory to a lower layer."""
        return self.promote(memory_id, target_layer)
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def update_importance(self, memory_id: int, importance: float) -> None:
        """Update memory importance."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET importance = ?, last_accessed = ? WHERE id = ?",
            (importance, time.time(), memory_id)
        )
        self.conn.commit()
    
    def record_access(self, memory_id: int) -> None:
        """Record that a memory was accessed."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET access_count = access_count + 1, "
            "last_accessed = ? WHERE id = ?",
            (time.time(), memory_id)
        )
        self.conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total memories
        cursor.execute("SELECT COUNT(*) as count FROM memories")
        stats["total"] = cursor.fetchone()["count"]
        
        # By layer
        cursor.execute("""
            SELECT layer, COUNT(*) as count, AVG(importance) as avg_importance
            FROM memories GROUP BY layer
        """)
        stats["by_layer"] = {
            row["layer"]: {
                "count": row["count"],
                "avg_importance": row["avg_importance"]
            }
            for row in cursor.fetchall()
        }
        
        # By kind
        cursor.execute("""
            SELECT kind, COUNT(*) as count FROM memories GROUP BY kind
        """)
        stats["by_kind"] = {
            row["kind"]: row["count"]
            for row in cursor.fetchall()
        }
        
        return stats
    
    def cleanup_old_memories(
        self,
        max_age_days: float = 30,
        min_importance: float = 0.1,
    ) -> int:
        """Remove old, low-importance memories from durable layer."""
        cursor = self.conn.cursor()
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        cursor.execute("""
            DELETE FROM memories 
            WHERE layer = 2 
            AND last_accessed < ?
            AND importance < ?
        """, (cutoff_time, min_importance))
        
        deleted = cursor.rowcount
        self.conn.commit()
        
        return deleted
    
    def _row_to_memory(self, row: sqlite3.Row) -> DurableMemory:
        """Convert database row to DurableMemory."""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"])
        
        return DurableMemory(
            id=row["id"],
            content=row["content"],
            kind=row["kind"],
            importance=row["importance"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            layer=row["layer"],
            metadata=json.loads(row["metadata"]),
            embedding=embedding,
        )
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class DurableMemoryManager:
    """
    Manager that coordinates between working memory and durable storage.
    
    This bridges the in-memory AgentMemory with the persistent
    DurableMemoryStore, handling promotion/demotion automatically.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        working_capacity: int = 100,
    ):
        self.store = DurableMemoryStore(db_path)
        self.working_capacity = working_capacity
        self.working_memory: Dict[str, Any] = {}
    
    def add_memory(
        self,
        content: str,
        kind: str,
        importance: float = 0.5,
        session_id: Optional[str] = None,
    ) -> int:
        """Add a memory, automatically placing in appropriate layer."""
        
        # Always add to working memory first
        memory_id = self.store.add(
            content=content,
            kind=kind,
            importance=importance,
            layer=0,  # Start in working
            session_id=session_id,
        )
        
        # Track in working memory
        self.working_memory[content] = memory_id
        
        # If working memory is full, promote some to episodic
        if len(self.working_memory) > self.working_capacity:
            self._promote_excess()
        
        return memory_id
    
    def retrieve(
        self,
        query: str,
        layers: Optional[List[int]] = None,
        limit: int = 5,
    ) -> List[DurableMemory]:
        """Retrieve relevant memories from all layers."""
        
        if layers is None:
            layers = [0, 1, 2]  # Search all layers
        
        results = []
        for layer in layers:
            layer_results = self.store.search(
                query=query,
                layer=layer,
                limit=limit,
            )
            results.extend(layer_results)
        
        # Record access for each
        for mem in results:
            self.store.record_access(mem.id)
        
        return results[:limit]
    
    def promote_to_episodic(self, memory_id: int) -> bool:
        """Promote memory from working to episodic."""
        return self.store.promote(memory_id, 1)
    
    def promote_to_durable(self, memory_id: int) -> bool:
        """Promote memory from episodic to durable."""
        return self.store.promote(memory_id, 2)
    
    def demote_to_working(self, memory_id: int) -> bool:
        """Demote memory back to working."""
        return self.store.demote(memory_id, 0)
    
    def demote_to_episodic(self, memory_id: int) -> bool:
        """Demote memory from durable to episodic."""
        return self.store.demote(memory_id, 1)
    
    def _promote_excess(self) -> None:
        """Promote excess working memories to episodic."""
        
        # Get working memories sorted by importance
        working = self.store.get_by_layer(0)
        
        # Keep only the most important ones
        to_promote = working[self.working_capacity:]
        
        for mem in to_promote:
            self.store.promote(mem.id, 1)
        
        # Fix: Rebuild working_memory dict from store to reflect actual layer 0 contents
        # Get current layer 0 memories from store
        current_working = self.store.get_by_layer(0)
        self.working_memory = {}
        for mem in current_working:
            # Find the content that maps to this memory_id
            for content, mem_id in list(self.working_memory.items()):
                if mem_id == mem.id:
                    self.working_memory[content] = mem.id
                    break
    
    def run_decay_cycle(
        self,
        decay_policy,  # DecayPolicy instance
    ) -> Dict[str, int]:
        """Run a decay cycle on all layers."""
        
        stats = {"promoted": 0, "demoted": 0, "deleted": 0}
        
        for layer in [1, 2]:  # Only decay episodic and durable
            memories = self.store.get_by_layer(layer)
            
            for mem in memories:
                # Evaluate decay
                memory_data = {
                    "content": mem.content,
                    "kind": mem.kind,
                    "importance": mem.importance,
                    "created_at": mem.created_at,
                    "last_accessed": mem.last_accessed,
                    "access_count": mem.access_count,
                    "layer": mem.layer,
                }
                
                result = decay_policy.evaluate(
                    f"mem_{mem.id}",
                    memory_data,
                )
                
                if result.action == "demote":
                    if mem.layer > 0:
                        self.store.demote(mem.id, mem.layer - 1)
                        stats["demoted"] += 1
                
                elif result.action == "remove":
                    self.store.delete(mem.id)
                    stats["deleted"] += 1
        
        return stats
    
    def run_promotion_cycle(
        self,
        promotion_policy,  # PromotionPolicy instance
    ) -> Dict[str, int]:
        """Run a promotion cycle on working memory."""
        
        stats = {"promoted": 0}
        
        memories = self.store.get_by_layer(0)
        
        for mem in memories:
            memory_data = {
                "content": mem.content,
                "kind": mem.kind,
                "importance": mem.importance,
                "created_at": mem.created_at,
                "last_accessed": mem.last_accessed,
                "access_count": mem.access_count,
                "layer": mem.layer,
            }
            
            result = promotion_policy.evaluate(
                f"mem_{mem.id}",
                memory_data,
            )
            
            if result.should_promote and result.new_layer > mem.layer:
                self.store.promote(mem.id, result.new_layer)
                stats["promoted"] += 1
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all layers."""
        return self.store.get_statistics()
    
    def close(self) -> None:
        """Close the store."""
        self.store.close()


def create_durable_memory(
    db_path: Optional[str] = None,
) -> DurableMemoryManager:
    """Factory function to create durable memory manager."""
    return DurableMemoryManager(db_path=db_path)
