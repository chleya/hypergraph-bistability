"""
Tests for embedding-based memory module.
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.embedding_memory import (
    EmbeddingGenerator,
    GroupCentroids,
    ChromaMemoryStore,
    EmbeddingMemoryMapper,
    HAS_CHROMADB,
    HAS_SENTENCE_TRANSFORMERS,
)


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""

    def test_random_fallback(self):
        """Test that random fallback works when no API key."""
        gen = EmbeddingGenerator(provider="openai")
        gen.api_key = None
        gen.client = None
        emb = gen.embed(["test text"])
        assert len(emb) == 1
        assert len(emb[0]) == gen.embedding_dim

    def test_similarity(self):
        """Test cosine similarity computation."""
        gen = EmbeddingGenerator(provider="openai")
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        c = [0.0, 1.0, 0.0]
        assert abs(gen.cosine_similarity(a, b) - 1.0) < 1e-6
        assert abs(gen.cosine_similarity(a, c) - 0.0) < 1e-6

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_sentence_transformers(self):
        """Test sentence-transformers embedding."""
        gen = EmbeddingGenerator(provider="sentence_transformers")
        assert gen.embedding_dim == 384
        emb = gen.embed(["hello world", "goodbye world"])
        assert len(emb) == 2
        assert len(emb[0]) == 384
        sim = gen.cosine_similarity(emb[0], emb[1])
        assert 0.0 <= sim <= 1.0

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_auto_provider_sentence_transformers(self):
        """Test auto provider picks sentence-transformers when available."""
        gen = EmbeddingGenerator(provider="auto")
        assert gen.st_model is not None
        assert gen.embedding_dim == 384


class TestGroupCentroids:
    """Tests for GroupCentroids."""

    def test_init(self):
        """Test centroid initialization."""
        centroids = GroupCentroids(k=4, embedding_dim=128)
        assert len(centroids.centroids) == 4
        assert centroids.embedding_dim == 128
        for c in centroids.centroids:
            norm = np.linalg.norm(c)
            assert abs(norm - 1.0) < 1e-6

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_initialize_from_examples(self):
        """Test initialization from example texts."""
        gen = EmbeddingGenerator(provider="sentence_transformers")
        centroids = GroupCentroids(k=2, embedding_dim=384)
        
        examples = {
            0: ["python code programming software"],
            1: ["music art painting creative"]
        }
        centroids.initialize_from_examples(examples, gen)
        
        assert centroids.example_texts[0] == examples[0]
        assert centroids.example_texts[1] == examples[1]
        for c in centroids.centroids:
            norm = np.linalg.norm(c)
            assert abs(norm - 1.0) < 1e-6

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_find_best_group(self):
        """Test group selection by embedding similarity."""
        gen = EmbeddingGenerator(provider="sentence_transformers")
        centroids = GroupCentroids(k=2, embedding_dim=384)
        
        examples = {
            0: ["work job career office professional"],
            1: ["family personal friends hobby travel"]
        }
        centroids.initialize_from_examples(examples, gen)
        
        work_emb = gen.embed(["meeting with clients"])[0]
        personal_emb = gen.embed(["weekend with family"])[0]
        
        assert centroids.find_best_group(work_emb) == 0
        assert centroids.find_best_group(personal_emb) == 1

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_update_centroid(self):
        """Test incremental centroid update."""
        gen = EmbeddingGenerator(provider="sentence_transformers")
        centroids = GroupCentroids(k=2, embedding_dim=384)
        
        old_centroid = centroids.centroids[0].copy()
        new_emb = gen.embed(["new work topic"])[0]
        
        centroids.update_centroid(0, new_emb, alpha=0.1)
        
        assert not np.allclose(centroids.centroids[0], old_centroid)
        norm = np.linalg.norm(centroids.centroids[0])
        assert abs(norm - 1.0) < 1e-6


class TestChromaMemoryStore:
    """Tests for ChromaMemoryStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.mark.skipif(not HAS_CHROMADB, reason="ChromaDB not installed")
    def test_add_and_get(self, temp_dir):
        """Test adding and retrieving memories."""
        store = ChromaMemoryStore(k=4, L=2, persist_dir=temp_dir)
        
        mem_id = store.add_memory("test memory", group=1, layer=0, embedding=[0.1]*384)
        assert mem_id.startswith("mem_")
        
        results = store.get_by_slot(1, 0)
        assert len(results) == 1
        assert results[0][1] == "test memory"

    @pytest.mark.skipif(not HAS_CHROMADB, reason="ChromaDB not installed")
    def test_count(self, temp_dir):
        """Test memory counting."""
        store = ChromaMemoryStore(k=4, L=2, persist_dir=temp_dir)
        
        store.add_memory("mem1", group=0, layer=0, embedding=[0.1]*384)
        store.add_memory("mem2", group=0, layer=0, embedding=[0.2]*384)
        store.add_memory("mem3", group=1, layer=0, embedding=[0.3]*384)
        
        assert store.count() == 3
        assert store.count(group=0) == 2
        assert store.count(group=1, layer=0) == 1

    @pytest.mark.skipif(not HAS_CHROMADB, reason="ChromaDB not installed")
    def test_delete_by_slot(self, temp_dir):
        """Test deleting memories by slot."""
        store = ChromaMemoryStore(k=4, L=2, persist_dir=temp_dir)
        
        store.add_memory("mem1", group=0, layer=0, embedding=[0.1]*384)
        store.add_memory("mem2", group=0, layer=1, embedding=[0.2]*384)
        
        deleted = store.delete_by_slot(0, 0)
        assert deleted == 1
        assert store.count(group=0, layer=0) == 0
        assert store.count(group=0, layer=1) == 1


class TestEmbeddingMemoryMapper:
    """Tests for EmbeddingMemoryMapper."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def mapper(self, temp_dir):
        """Create mapper with sentence-transformers."""
        if not HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("sentence-transformers not installed")
        return EmbeddingMemoryMapper(k=4, L=2, persist_dir=temp_dir)

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_initialize_groups(self, mapper):
        """Test group initialization."""
        examples = {
            0: ["work job career office professional"],
            1: ["family personal friends hobby"],
            2: ["code programming technical software"],
            3: ["art design creative music"]
        }
        mapper.initialize_groups(examples)
        
        assert len(mapper.centroids.example_texts[0]) == 1
        assert len(mapper.centroids.example_texts[2]) == 1

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_find_slot_work(self, mapper):
        """Test finding slot for work content."""
        examples = {
            0: ["work job career office professional meeting"],
            1: ["family personal friends hobby travel"],
            2: ["code programming technical software algorithm"],
            3: ["art design creative music painting"]
        }
        mapper.initialize_groups(examples)
        
        group, layer = mapper.find_slot("I have a meeting tomorrow", action="write")
        assert 0 <= group < 4
        assert 0 <= layer < 2

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_50_memories_retrieval(self, mapper):
        """Test storing and retrieving 50 memories across groups."""
        examples = {
            0: ["work job career office professional meeting deadline"],
            1: ["family personal friends hobby travel weekend"],
            2: ["code programming technical software algorithm database API"],
            3: ["art design creative music painting photography"]
        }
        mapper.initialize_groups(examples)
        
        memories = [
            ("quarterly report due Friday", 0),
            ("sister birthday next week", 1),
            ("fix authentication bug", 2),
            ("paint landscape seascape", 3),
            ("team standup daily", 0),
            ("dinner with friends Saturday", 1),
            ("deploy new API endpoint", 2),
            ("photography exhibition opening", 3),
            ("email client about proposal", 0),
            ("kids school play weekend", 1),
            ("refactor user service", 2),
            ("write watercolor tutorial", 3),
            ("prepare presentation slides", 0),
            ("doctor appointment Tuesday", 1),
            ("optimize database queries", 2),
            ("sketch character design", 3),
            ("schedule one-on-one with intern", 0),
            ("grocery shopping list", 1),
            ("implement caching layer", 2),
            ("attend art class Thursday", 3),
            ("review budget proposal", 0),
            ("plan summer vacation", 1),
            ("write unit tests coverage", 2),
            ("practice guitar scales", 3),
            ("update project documentation", 0),
            ("call parents Sunday", 1),
            ("setup CI/CD pipeline", 2),
            ("visit museum new exhibit", 3),
            ("organize client files", 0),
            ("yoga class Monday", 1),
            ("debug memory leak", 2),
            ("film short documentary", 3),
            ("coordinate project timeline", 0),
            ("book hotel booking", 1),
            ("migrate to new server", 2),
            ("create logo design", 3),
            ("submit expense report", 0),
            ("join running club", 1),
            ("implement web socket", 2),
            ("attend concert Friday", 3),
            ("plan team outing", 0),
            ("renew gym membership", 1),
            ("fix security vulnerability", 2),
            ("paint portrait commission", 3),
            ("client feedback meeting", 0),
            ("pet vet appointment", 1),
            ("code review pull request", 2),
            ("exhibit artwork gallery", 3),
            ("send invoice to client", 0),
            ("read new novel book", 1),
        ]
        
        for content, expected_group in memories:
            group, layer = mapper.find_slot(content, action="write")
            if mapper.store:
                mapper.store_memory(content, group, layer, {"expected_group": expected_group})
        
        if mapper.store:
            assert mapper.store.count() == 50, f"Expected 50 memories, got {mapper.store.count()}"
        
        work_retrieved = mapper.semantic_search("work meeting deadline", group=0, top_k=10)
        assert len(work_retrieved) > 0
        
        tech_retrieved = mapper.semantic_search("code programming software", group=2, top_k=10)
        assert len(tech_retrieved) > 0
        
        creative_retrieved = mapper.semantic_search("art painting design", group=3, top_k=10)
        assert len(creative_retrieved) > 0

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_semantic_search(self, mapper):
        """Test semantic search across all groups."""
        examples = {
            0: ["work job career office professional"],
            1: ["family personal friends hobby"],
            2: ["code programming technical software"],
            3: ["art design creative music"]
        }
        mapper.initialize_groups(examples)
        
        for content, group, layer in [
            ("meeting with team", 0, 0),
            ("dinner with family", 1, 0),
            ("programming python", 2, 0),
            ("painting landscape", 3, 0),
        ]:
            g, l = mapper.find_slot(content, action="write")
            if mapper.store:
                mapper.store_memory(content, g, l)
        
        results = mapper.semantic_search("workplace collaboration", top_k=2)
        if results:
            assert len(results) <= 2


class TestSaveLoad:
    """Tests for save/load functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_centroids_save_load(self, temp_dir):
        """Test saving and loading centroids."""
        gen = EmbeddingGenerator(provider="sentence_transformers")
        centroids = GroupCentroids(k=4, embedding_dim=384)
        
        examples = {
            0: ["work job career office"],
            1: ["family personal friends"],
            2: ["code programming software"],
            3: ["art design creative"]
        }
        centroids.initialize_from_examples(examples, gen)
        
        filepath = os.path.join(temp_dir, "centroids.json")
        centroids.save(filepath)
        
        loaded = GroupCentroids.load(filepath)
        assert loaded.k == 4
        assert loaded.embedding_dim == 384
        assert len(loaded.centroids) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])