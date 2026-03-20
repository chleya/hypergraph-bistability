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

import numpy as np
from typing import Optional, Tuple, List, Dict
import os
import json
import time

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
        
        if provider == "sentence_transformers" and HAS_SENTENCE_TRANSFORMERS:
            self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = 384
        elif provider == "openai" and self.api_key and HAS_OPENAI:
            self.client = OpenAI(api_key=self.api_key)
        elif provider == "auto":
            if HAS_SENTENCE_TRANSFORMERS:
                self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_dim = 384
            elif self.api_key and HAS_OPENAI:
                self.client = OpenAI(api_key=self.api_key)
    
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
        
        if HAS_SENTENCE_TRANSFORMERS:
            self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = 384
            embeddings = self.st_model.encode(texts, normalize_embeddings=True)
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
        with open(filepath, 'w') as f:
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
            self.store = None
        
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
        if self.store is None:
            return ""
        
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
        if self.store is None:
            return []
        
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
        if self.store is None:
            return []
        
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
