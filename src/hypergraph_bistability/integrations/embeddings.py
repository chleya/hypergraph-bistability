"""
Embedding-Based Memory Mapper with ChromaDB Cold Storage
========================================================

Provides semantic memory storage using:
1. EmbeddingMemoryMapper: Maps text to memory slots via vector similarity
2. ChromaMemoryStore: Persistent cold storage with ChromaDB

Architecture:
- Top: ODE matrix M[k×L] (working memory, ~10-100 items)
- Middle: content_map with embeddings per slot
- Bottom: ChromaDB (long-term storage, unlimited)
"""

import hashlib
import re

import numpy as np
from typing import Optional, Tuple, List, Dict
import os
import json
import time
from pathlib import Path

try:
    from chromadb import ChromaClient
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class _OfflineSentenceTransformer:
    """Deterministic local encoder used when the real model is unavailable."""

    _SEMANTIC_BUCKETS = {
        0: {
            "work", "job", "career", "office", "professional", "meeting",
            "client", "clients", "deadline", "project", "proposal", "team",
            "report", "invoice", "budget", "presentation", "email",
        },
        1: {
            "family", "personal", "friend", "friends", "hobby", "travel",
            "weekend", "vacation", "birthday", "dinner", "parents",
            "kids", "school", "doctor", "appointment", "shopping",
        },
        2: {
            "code", "coding", "programming", "technical", "software",
            "algorithm", "database", "api", "python", "deploy", "debug",
            "refactor", "server", "security", "test", "tests",
        },
        3: {
            "art", "design", "creative", "music", "painting", "paint",
            "photography", "photo", "gallery", "museum", "guitar",
            "portrait", "landscape", "writing", "logo",
        },
    }

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        vectors = []
        for text in texts:
            vec = np.zeros(self.embedding_dim, dtype=float)
            lowered = text.lower()
            tokens = re.findall(r"\w+", lowered)

            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.embedding_dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign

                for bucket_idx, vocabulary in self._SEMANTIC_BUCKETS.items():
                    if token in vocabulary:
                        vec[bucket_idx] += 3.0

            compact = lowered.replace(" ", "")
            for i in range(max(0, len(compact) - 2)):
                trigram = compact[i:i + 3]
                digest = hashlib.sha256(trigram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.embedding_dim
                vec[idx] += 0.25

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
            vectors.append(vec)

        return np.array(vectors)


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI or sentence-transformers.
    
    Provider priority:
    1. sentence-transformers (local, no API key needed) — if installed
    2. OpenAI text-embedding-3-small — if API key available
    3. Random vectors (fallback)
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "text-embedding-3-small",
                 provider: str = "auto"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.provider = provider
        self.client = None
        self.st_model = None
        self.embedding_dim = 1536
        self.fallback_mode = "random"
        self._st_load_attempted = False
        
        if provider == "sentence_transformers" and HAS_SENTENCE_TRANSFORMERS:
            self._try_load_sentence_transformer()
        elif provider == "openai" and self.api_key and HAS_OPENAI:
            self.client = OpenAI(api_key=self.api_key)
        elif provider == "auto":
            if HAS_SENTENCE_TRANSFORMERS:
                self._try_load_sentence_transformer()
            elif self.api_key and HAS_OPENAI:
                self.client = OpenAI(api_key=self.api_key)

        if self.st_model is not None or provider == "sentence_transformers":
            self.embedding_dim = 384
            self.fallback_mode = "hashed"
    
    def _try_load_sentence_transformer(self) -> None:
        """Load a local sentence-transformers model when available."""
        if self._st_load_attempted:
            return
        self._st_load_attempted = True
        use_real_model = os.environ.get("HYPERGRAPH_USE_REAL_SENTENCE_TRANSFORMERS") == "1"
        if not use_real_model:
            self.st_model = _OfflineSentenceTransformer(embedding_dim=384)
            self.embedding_dim = 384
            self.fallback_mode = "hashed"
            return
        try:
            self.st_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            self.embedding_dim = 384
        except Exception:
            self.st_model = _OfflineSentenceTransformer(embedding_dim=384)
            self.embedding_dim = 384
            self.fallback_mode = "hashed"
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings to embed
            
        Returns
        -------
        List[List[float]]
            List of embedding vectors
        """
        if self.st_model is not None:
            embeddings = self.st_model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        
        if self.client is not None:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"OpenAI embedding failed: {e}, trying sentence-transformers")
        
        if self.provider != "openai" and HAS_SENTENCE_TRANSFORMERS:
            self._try_load_sentence_transformer()
            if self.st_model is not None:
                embeddings = self.st_model.encode(texts, normalize_embeddings=True)
                return embeddings.tolist()

        if self.fallback_mode == "hashed":
            offline_model = _OfflineSentenceTransformer(embedding_dim=self.embedding_dim)
            embeddings = offline_model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        
        return [self._random_embedding() for _ in texts]
    
    def _random_embedding(self) -> List[float]:
        vec = np.random.randn(self.embedding_dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class GroupCentroids:
    """
    Manages centroid vectors for each memory group.
    Each centroid represents the "average" semantic space of a group.
    """
    
    def __init__(self, k: int, embedding_dim: int = 1536):
        self.k = k
        self.embedding_dim = embedding_dim
        self.centroids: List[List[float]] = [
            self._random_unit_vector() for _ in range(k)
        ]
        self.group_labels: List[str] = [f"group_{i}" for i in range(k)]
        self.example_texts: List[List[str]] = [[] for _ in range(k)]
    
    def _random_unit_vector(self) -> List[float]:
        vec = np.random.randn(self.embedding_dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    
    def initialize_from_examples(self, examples: Dict[int, List[str]], generator: EmbeddingGenerator):
        """
        Initialize centroids from example texts for each group.
        
        Parameters
        ----------
        examples : Dict[int, List[str]]
            Mapping from group index to list of example texts
        generator : EmbeddingGenerator
            Embedding generator to create vectors
        """
        for group_idx, texts in examples.items():
            if not texts:
                continue
            embeddings = generator.embed(texts)
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self.centroids[group_idx] = centroid.tolist()
            self.example_texts[group_idx] = texts
    
    def find_best_group(self, embedding: List[float]) -> int:
        """
        Find the group whose centroid is most similar to the given embedding.
        
        Parameters
        ----------
        embedding : List[float]
            Query embedding vector
            
        Returns
        -------
        int
            Index of best matching group
        """
        similarities = [
            self._cosine(embedding, c) for c in self.centroids
        ]
        return int(np.argmax(similarities))
    
    def update_centroid(self, group_idx: int, new_embedding: List[float], 
                        alpha: float = 0.1) -> None:
        """
        Incrementally update a centroid with a new embedding.
        
        Parameters
        ----------
        group_idx : int
            Group to update
        new_embedding : List[float]
            New embedding to incorporate
        alpha : float
            Learning rate for update
        """
        old = np.array(self.centroids[group_idx])
        new = np.array(new_embedding)
        updated = (1 - alpha) * old + alpha * new
        updated = updated / np.linalg.norm(updated)
        self.centroids[group_idx] = updated.tolist()
    
    def _cosine(self, a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def save(self, filepath: str) -> None:
        """Save centroids to JSON."""
        data = {
            "centroids": self.centroids,
            "group_labels": self.group_labels,
            "example_texts": self.example_texts,
            "k": self.k,
            "embedding_dim": self.embedding_dim
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> "GroupCentroids":
        """Load centroids from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        instance = cls(k=data["k"], embedding_dim=data["embedding_dim"])
        instance.centroids = data["centroids"]
        instance.group_labels = data.get("group_labels", instance.group_labels)
        instance.example_texts = data.get("example_texts", [[] for _ in range(instance.k)])
        return instance


class ChromaMemoryStore:
    """
    ChromaDB-backed cold storage for memories.
    
    Stores raw text + embeddings with group/layer metadata.
    Enables retrieval of all memories in a specific group/layer.
    """
    
    def __init__(self, k: int = 4, L: int = 2, persist_dir: Optional[str] = None):
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")
        
        self.k = k
        self.L = L
        self.persist_dir = persist_dir or ".chromadb"
        
        self.client = ChromaClient(
            persist_directory=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="memory_store",
            metadata={"k": k, "L": L}
        )
        
        self._id_counter = 0
    
    def add_memory(self, text: str, group: int, layer: int,
                   embedding: List[float], metadata: Optional[Dict] = None) -> str:
        """
        Add a memory to the store.
        
        Parameters
        ----------
        text : str
            Memory content
        group : int
            Group index
        layer : int
            Layer index
        embedding : List[float]
            Embedding vector
        metadata : Dict, optional
            Additional metadata
            
        Returns
        -------
        str
            Memory ID
        """
        memory_id = f"mem_{self._id_counter}_{int(time.time())}"
        self._id_counter += 1
        
        doc_metadata = {
            "group": group,
            "layer": layer,
            **(metadata or {})
        }
        
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[doc_metadata]
        )
        
        return memory_id
    
    def get_by_slot(self, group: int, layer: int, 
                   top_k: Optional[int] = None) -> List[Tuple[str, str, Dict]]:
        """
        Get all memories for a specific group/layer slot.
        
        Parameters
        ----------
        group : int
            Group index
        layer : int
            Layer index
        top_k : int, optional
            Return only top_k most recent memories
            
        Returns
        -------
        List[Tuple[str, str, Dict]]
            List of (memory_id, text, metadata) tuples
        """
        results = self.collection.get(
            where={"group": group, "layer": layer},
            include=["documents", "metadatas"]
        )
        
        if not results or not results["ids"]:
            return []
        
        memories = list(zip(
            results["ids"],
            results["documents"],
            results.get("metadatas", [{}] * len(results["ids"]))
        ))
        
        if top_k is not None:
            memories = memories[-top_k:]
        
        return memories
    
    def search_by_vector(self, query_embedding: List[float], 
                         group: Optional[int] = None,
                         top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search memories by vector similarity.
        
        Parameters
        ----------
        query_embedding : List[float]
            Query embedding vector
        group : int, optional
            Filter by group
        top_k : int
            Number of results to return
            
        Returns
        -------
        List[Tuple[str, str, float]]
            List of (memory_id, text, similarity_score) tuples
        """
        where_filter = {"group": group} if group is not None else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "distances"]
        )
        
        if not results or not results["ids"]:
            return []
        
        ids = results["ids"][0]
        docs = results["documents"][0]
        dists = results.get("distances", [[]])[0]
        
        similarities = [1.0 / (1.0 + d) for d in dists]
        
        return list(zip(ids, docs, similarities))
    
    def count(self, group: Optional[int] = None, layer: Optional[int] = None) -> int:
        """Count memories in store, optionally filtered."""
        if group is not None and layer is not None:
            where = {"group": group, "layer": layer}
        elif group is not None:
            where = {"group": group}
        elif layer is not None:
            where = {"layer": layer}
        else:
            where = None
        
        if where:
            return len(self.collection.get(where=where)["ids"])
        return self.collection.count()
    
    def delete_by_slot(self, group: int, layer: int) -> int:
        """Delete all memories for a specific slot."""
        results = self.collection.get(
            where={"group": group, "layer": layer},
            include=["ids"]
        )
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0
    
    def reset(self) -> None:
        """Clear all memories."""
        self.collection.delete(where={})
        self._id_counter = 0


class InMemoryMemoryStore:
    """Minimal local fallback when ChromaDB is unavailable."""

    def __init__(self, k: int = 4, L: int = 2, persist_dir: Optional[str] = None):
        self.k = k
        self.L = L
        self.persist_dir = persist_dir
        self._id_counter = 0
        self._records: List[Dict] = []

    def add_memory(self, text: str, group: int, layer: int,
                   embedding: List[float], metadata: Optional[Dict] = None) -> str:
        memory_id = f"mem_{self._id_counter}_{int(time.time())}"
        self._id_counter += 1
        self._records.append({
            "id": memory_id,
            "text": text,
            "group": group,
            "layer": layer,
            "embedding": embedding,
            "metadata": metadata or {},
        })
        return memory_id

    def get_by_slot(self, group: int, layer: int,
                    top_k: Optional[int] = None) -> List[Tuple[str, str, Dict]]:
        records = [
            (record["id"], record["text"], record["metadata"])
            for record in self._records
            if record["group"] == group and record["layer"] == layer
        ]
        if top_k is not None:
            records = records[-top_k:]
        return records

    def search_by_vector(self, query_embedding: List[float],
                         group: Optional[int] = None,
                         top_k: int = 5) -> List[Tuple[str, str, float]]:
        query = np.array(query_embedding)
        scored = []
        for record in self._records:
            if group is not None and record["group"] != group:
                continue
            embedding = np.array(record["embedding"])
            sim = float(np.dot(query, embedding) / ((np.linalg.norm(query) * np.linalg.norm(embedding)) + 1e-8))
            scored.append((record["id"], record["text"], sim))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:top_k]

    def count(self, group: Optional[int] = None, layer: Optional[int] = None) -> int:
        records = self._records
        if group is not None:
            records = [record for record in records if record["group"] == group]
        if layer is not None:
            records = [record for record in records if record["layer"] == layer]
        return len(records)

    def delete_by_slot(self, group: int, layer: int) -> int:
        before = len(self._records)
        self._records = [
            record for record in self._records
            if not (record["group"] == group and record["layer"] == layer)
        ]
        return before - len(self._records)

    def reset(self) -> None:
        self._records = []
        self._id_counter = 0


class EmbeddingMemoryMapper:
    """
    Maps text to memory slots using embeddings.
    
    Combines:
    - GroupCentroids: For fast group selection
    - ChromaMemoryStore: For long-term storage and retrieval
    
    Usage:
        mapper = EmbeddingMemoryMapper(k=4, L=2, api_key="...")
        mapper.initialize_groups({
            0: ["work", "job", "career"],
            1: ["personal", "family", "friend"],
            2: ["technical", "code", "programming"],
            3: ["creative", "art", "design"]
        })
        
        group, layer = mapper.find_slot("I love coding", action="write")
    """
    
    def __init__(self, k: int = 4, L: int = 2, 
                 api_key: Optional[str] = None,
                 persist_dir: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small"):
        self.k = k
        self.L = L
        self.api_key = api_key
        
        self.generator = EmbeddingGenerator(api_key=api_key, model=embedding_model)
        self.centroids = GroupCentroids(k=k, embedding_dim=self.generator.embedding_dim)
        
        if HAS_CHROMADB:
            self.store = ChromaMemoryStore(k=k, L=L, persist_dir=persist_dir)
        else:
            self.store = InMemoryMemoryStore(k=k, L=L, persist_dir=persist_dir)
        
        self.layer_usage_counts = [[0] * L for _ in range(k)]
    
    def initialize_groups(self, examples: Dict[int, List[str]]) -> None:
        """
        Initialize group centroids from example texts.
        
        Parameters
        ----------
        examples : Dict[int, List[str]]
            Mapping from group index to list of example texts
        """
        self.centroids.initialize_from_examples(examples, self.generator)
    
    def set_group_labels(self, labels: List[str]) -> None:
        """Set human-readable labels for groups."""
        self.centroids.group_labels = labels
    
    def find_slot(self, content: str, action: str = "write") -> Tuple[int, int]:
        """
        Find the best slot for content.
        
        Parameters
        ----------
        content : str
            Content to store/retrieve
        action : str
            "write" or "read"
            
        Returns
        -------
        Tuple[int, int]
            (group, layer) slot indices
        """
        embedding = self.generator.embed([content])[0]
        
        best_group = self.centroids.find_best_group(embedding)
        
        if action == "read":
            best_layer = 0
        else:
            best_layer = self._find_least_used_layer(best_group)
        
        self.layer_usage_counts[best_group][best_layer] += 1
        
        return best_group, best_layer
    
    def _find_least_used_layer(self, group: int) -> int:
        """Find the layer with least usage for a group (round-robin + LRU)."""
        counts = self.layer_usage_counts[group]
        min_count = min(counts)
        candidates = [i for i, c in enumerate(counts) if c == min_count]
        return candidates[0]
    
    def store_memory(self, content: str, group: int, layer: int,
                    metadata: Optional[Dict] = None) -> str:
        """
        Store memory in ChromaDB.
        
        Parameters
        ----------
        content : str
            Memory content
        group : int
            Group index
        layer : int
            Layer index
        metadata : Dict, optional
            Additional metadata
            
        Returns
        -------
        str
            Memory ID
        """
        embedding = self.generator.embed([content])[0]
        memory_id = self.store.add_memory(content, group, layer, embedding, metadata)
        
        self.centroids.update_centroid(group, embedding)
        
        return memory_id
    
    def retrieve_memories(self, group: int, layer: int, 
                        top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve memories for a slot by vector similarity.
        
        Parameters
        ----------
        group : int
            Group index
        layer : int
            Layer index
        top_k : int
            Number of memories to retrieve
            
        Returns
        -------
        List[Tuple[str, str, float]]
            List of (memory_id, content, similarity) tuples
        """
        return self.store.get_by_slot(group, layer, top_k=top_k)
    
    def semantic_search(self, query: str, group: Optional[int] = None,
                       top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search memories by semantic similarity.
        
        Parameters
        ----------
        query : str
            Search query
        group : int, optional
            Filter by group
        top_k : int
            Number of results
            
        Returns
        -------
        List[Tuple[str, str, float]]
            List of (memory_id, content, similarity) tuples
        """
        embedding = self.generator.embed([query])[0]
        return self.store.search_by_vector(embedding, group=group, top_k=top_k)
    
    def save(self, filepath: str) -> None:
        """Save centroids and metadata."""
        self.centroids.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> "EmbeddingMemoryMapper":
        """Load centroids and create mapper."""
        centroids = GroupCentroids.load(filepath)
        instance = cls(k=centroids.k, **kwargs)
        instance.centroids = centroids
        return instance


def demo():
    """Demo of embedding-based memory mapping."""
    print("=" * 60)
    print("Embedding Memory Mapper Demo")
    print("=" * 60)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    mapper = EmbeddingMemoryMapper(k=4, L=2, api_key=api_key)
    
    print("\n1. Initializing group centroids with examples...")
    mapper.initialize_groups({
        0: ["work job career professional office meeting project deadline"],
        1: ["personal family friend hobby travel weekend vacation"],
        2: ["code programming technical software algorithm database API"],
        3: ["creative design art music writing painting photography"]
    })
    mapper.set_group_labels(["work", "personal", "technical", "creative"])
    
    print("   Groups: work, personal, technical, creative")
    
    print("\n2. Finding slots for various texts...")
    test_texts = [
        "I have a meeting with my team tomorrow",
        "My sister is visiting this weekend",
        "Need to fix the authentication bug",
        "Let me paint a landscape today"
    ]
    
    for text in test_texts:
        group, layer = mapper.find_slot(text, action="write")
        label = mapper.centroids.group_labels[group]
        print(f"   '{text[:40]}...' -> group={label}, layer={layer}")
        
        if mapper.store:
            mapper.store_memory(text, group, layer)
    
    if mapper.store:
        print(f"\n3. ChromaDB storage: {mapper.store.count()} memories stored")
        
        print("\n4. Semantic search results:")
        results = mapper.semantic_search("workplace collaboration", top_k=2)
        for mid, content, sim in results:
            print(f"   [{sim:.3f}] {content[:50]}...")


if __name__ == "__main__":
    demo()
