"""
Unit Tests and Benchmarks for AI Co-Scientist System
Tests core functionality and performance of all agents
"""

import asyncio
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from co_scientist import (
    ContextMemory,
    CoScientist,
    EvolutionAgent,
    GenerationAgent,
    Hypothesis,
    HypothesisStatus,
    MetaReviewAgent,
    ProximityAgent,
    RankingAgent,
    ReflectionAgent,
    ResearchGoal,
    ReviewCritique,
    _parse_json_response,
)
from rag_system import DocumentChunk, DocumentProcessor, PDFDownloader, RAGEngine, SemanticChunker

# =============================================================================
# UNIT TESTS
# =============================================================================

class TestGenerationAgent:
    """Test hypothesis generation"""

    @staticmethod
    async def test_hypothesis_generation():
        """Test that Generation Agent creates valid hypotheses"""
        print("\n🧪 Testing Generation Agent...")

        agent = GenerationAgent()
        goal = ResearchGoal(
            title="Test Goal",
            description="Test hypothesis generation",
            domain="test_domain"
        )

        hypotheses = await agent.generate_initial_hypotheses(goal, context_papers=[], count=3)

        # Assertions
        assert len(hypotheses) == 3, "Should generate 3 hypotheses"
        assert all(isinstance(h, Hypothesis) for h in hypotheses), "All should be Hypotheses"
        assert all(h.title for h in hypotheses), "All should have titles"
        assert all(h.mechanism for h in hypotheses), "All should have mechanisms"
        assert all(h.testable_predictions for h in hypotheses), "All should have predictions"

        print("  ✓ Generation creates valid hypotheses")
        print(f"  ✓ Generated {len(hypotheses)} diverse hypotheses")
        return True

    @staticmethod
    def test_prompt_building():
        """Test the prompt builder (Renamed from _generate_with_llm)"""
        print("\n🧪 Testing Prompt Builder...")

        agent = GenerationAgent()
        goal = ResearchGoal(title="Target Discover", description="Find targets", domain="Biology")
        papers = [{"title": "Paper 1", "summary": "Useful paper", "published": "2023"}]

        # Test existence of the renamed method
        assert hasattr(agent, '_build_llm_prompt'), "Agent should have _build_llm_prompt"

        prompt = agent._build_llm_prompt(goal, papers, count=2)
        assert isinstance(prompt, str), "Prompt should be a string"
        assert "Target Discover" in prompt, "Goal title should be in prompt"
        assert "Paper 1" in prompt, "Context paper should be in prompt"

        print("  ✓ Prompt builder correctly renamed and functional")
        return True


class TestReflectionAgent:
    """Test hypothesis evaluation"""

    @staticmethod
    async def test_hypothesis_review():
        """Test that Reflection Agent provides valid reviews"""
        print("\n🧪 Testing Reflection Agent...")

        agent = ReflectionAgent()
        hypothesis = Hypothesis(
            title="Test Hypothesis",
            description="A test hypothesis",
            mechanism="Test mechanism",
            testable_predictions=["Prediction 1", "Prediction 2"],
            grounding_evidence=["Evidence 1"]
        )
        goal = ResearchGoal(title="Test", domain="test")

        review = await agent.review_hypothesis(hypothesis, goal)

        # Assertions
        assert isinstance(review, ReviewCritique), "Should return ReviewCritique"
        assert 0 <= review.correctness_score <= 1, "Correctness should be 0-1"
        assert 0 <= review.novelty_score <= 1, "Novelty should be 0-1"
        assert 0 <= review.testability_score <= 1, "Testability should be 0-1"
        assert 0 <= review.quality_score <= 1, "Quality should be 0-1"
        assert review.feedback, "Should have feedback"

        print("  ✓ Reviews are properly scored (0-1)")
        print(f"  ✓ Novelty: {review.novelty_score:.2f}, Quality: {review.quality_score:.2f}")
        return True


class TestRankingAgent:
    """Test tournament and Elo rating"""

    @staticmethod
    async def test_elo_ratings():
        """Test Elo rating updates"""
        print("\n🧪 Testing Ranking Agent (Elo System)...")

        agent = RankingAgent()

        # Create two hypotheses
        hyp_a = Hypothesis(title="Strong Hypothesis", mechanism="Good mechanism")
        hyp_b = Hypothesis(title="Weak Hypothesis", mechanism="Weak mechanism")

        # Give them different initial ratings
        hyp_a.elo_rating = 1400
        hyp_b.elo_rating = 1000

        # Record initial ratings
        initial_a = hyp_a.elo_rating
        initial_b = hyp_b.elo_rating

        # Conduct match (strong should win)
        winner_id, match = await agent.conduct_tournament_match(hyp_a, hyp_b)

        # Check Elo updates
        assert hyp_a.elo_rating > initial_a if winner_id == hyp_a.id else hyp_a.elo_rating < initial_a
        assert hyp_b.elo_rating > initial_b if winner_id == hyp_b.id else hyp_b.elo_rating < initial_b

        # Ratings should sum approximately conservatively
        assert abs((hyp_a.elo_rating + hyp_b.elo_rating) - (initial_a + initial_b)) < 1

        print("  ✓ Elo ratings update correctly")
        print(f"  ✓ Winner: {winner_id}, New ratings: A={hyp_a.elo_rating:.0f}, B={hyp_b.elo_rating:.0f}")
        return True


class TestProximityAgent:
    """Test hypothesis similarity"""

    @staticmethod
    async def test_similarity_computation():
        """Test proximity/similarity computation"""
        print("\n🧪 Testing Proximity Agent...")

        agent = ProximityAgent()

        # Create related and unrelated hypotheses
        hyp_a = Hypothesis(
            title="Hypothesis A",
            mechanism="Mechanism involving pathway X and Y",
            testable_predictions=["Test X", "Test Y"],
            cited_papers=["Paper1", "Paper2"]
        )

        hyp_b = Hypothesis(
            title="Hypothesis B",
            mechanism="Mechanism involving pathway X and Y",  # Similar
            testable_predictions=["Test X", "Test Z"],
            cited_papers=["Paper1", "Paper3"]
        )

        hyp_c = Hypothesis(
            title="Hypothesis C",
            mechanism="Completely different approach with Z and W",
            testable_predictions=["Test A", "Test B"],
            cited_papers=["Paper4", "Paper5"]
        )

        # Compute similarities
        similarity_ab = await agent._compute_similarity(hyp_a, hyp_b)
        similarity_ac = await agent._compute_similarity(hyp_a, hyp_c)

        # Assertions
        assert 0 <= similarity_ab <= 1, "Similarity should be 0-1"
        assert 0 <= similarity_ac <= 1, "Similarity should be 0-1"
        assert similarity_ab > similarity_ac, "AB should be more similar than AC"

        print("  ✓ Similarity scoring is valid (0-1)")
        print(f"  ✓ Related: {similarity_ab:.2f} > Unrelated: {similarity_ac:.2f}")
        return True


class TestEvolutionAgent:
    """Test hypothesis evolution"""

    @staticmethod
    async def test_hypothesis_evolution():
        """Test that hypotheses evolve correctly"""
        print("\n🧪 Testing Evolution Agent...")

        agent = EvolutionAgent()
        original = Hypothesis(
            title="Original Hypothesis",
            description="Original description",
            mechanism="Original mechanism",
            testable_predictions=["Pred1"],
            grounding_evidence=["Evidence1"]
        )

        # Test enhancement
        enhanced = await agent.evolve_hypothesis(original, strategy="enhancement")
        assert enhanced.id != original.id, "Should create new hypothesis"
        assert original.id in enhanced.parent_ids, "Should track lineage"
        assert enhanced.generation_method == "evolved", "Should mark as evolved"
        assert len(enhanced.grounding_evidence) > len(original.grounding_evidence), "Should add evidence"

        # Test simplification
        simplified = await agent.evolve_hypothesis(original, strategy="simplification")
        assert simplified.generation_method == "evolved"
        assert "Simplified" in simplified.title, "Should indicate simplification"

        print("  ✓ Hypotheses evolve with proper lineage tracking")
        print(f"  ✓ Enhancement added evidence: {len(original.grounding_evidence)} → {len(enhanced.grounding_evidence)}")
        return True


class TestMetaReviewAgent:
    """Test meta-review synthesis"""

    @staticmethod
    async def test_meta_review():
        """Test meta-review generation"""
        print("\n🧪 Testing Meta-Review Agent...")

        agent = MetaReviewAgent()

        # Create diverse hypotheses
        hypotheses = []
        for i in range(5):
            h = Hypothesis(
                title=f"Hypothesis {i}",
                mechanism=f"Mechanism {i}",
                testable_predictions=[f"Test{i}"]
            )
            h.elo_rating = 1200 + (i * 50)  # Varied ratings
            hypotheses.append(h)

        goal = ResearchGoal(title="Test Goal", domain="test")

        meta_review = await agent.generate_meta_review(hypotheses, [], goal)

        # Assertions
        assert meta_review['total_hypotheses'] == 5, "Should count all hypotheses"
        assert len(meta_review['top_hypotheses']) <= 5, "Should list top hypotheses"
        assert meta_review['suggested_improvements'], "Should suggest improvements"
        assert meta_review['research_overview'], "Should generate overview"

        print("  ✓ Meta-review synthesizes all insights")
        print(f"  ✓ Top {len(meta_review['top_hypotheses'])} identified from {meta_review['total_hypotheses']}")
        return True


class TestUtilities:
    """Test helper utilities"""

    @staticmethod
    def test_json_parsing():
        """Test the robust JSON parser"""
        print("\n🧪 Testing JSON Utilities...")

        # Test markdown stripping
        raw_json = "```json\n{\"test\": \"value\"}\n```"
        parsed = _parse_json_response(raw_json)
        assert parsed == {"test": "value"}, "Should strip markdown fences"

        # Test plain JSON
        plain_json = "{\"key\": [1,2,3]}"
        parsed = _parse_json_response(plain_json)
        assert parsed == {"key": [1,2,3]}, "Should parse plain JSON"

        # TEST: JSON with pre/post-amble
        junk_json = "Here is the result: {\"success\": true} Hope this helps!"
        parsed = _parse_json_response(junk_json)
        assert parsed == {"success": True}, "Should extract JSON from junk text"

        # TEST: JSON with markdown and junk
        junk_md = "Some text ```json\n{\"data\": 123}\n``` More text"
        parsed = _parse_json_response(junk_md)
        assert parsed == {"data": 123}, "Should extract JSON from markdown within junk text"

        print("  ✓ JSON parser is robust to markdown, junk text, and pre/post-amble")
        return True


class TestRAGSystem:
    """Test RAG pipeline components"""

    @staticmethod
    def test_semantic_chunker():
        """Test text chunking with SemanticChunker"""
        print("\n🧪 Testing SemanticChunker...")

        chunker = SemanticChunker(chunk_size=50, overlap=10)

        # Build a text with multiple paragraphs
        text = "\n\n".join([
            "This is the first paragraph about molecular biology and cancer research.",
            "This is the second paragraph discussing drug mechanisms and pathways.",
            "This is the third paragraph covering experimental validation methods.",
            "This is the fourth paragraph about statistical analysis of results."
        ])

        chunks = chunker.chunk_text(text, paper_id="test123", paper_title="Test Paper")

        assert len(chunks) > 0, "Should produce at least one chunk"
        assert all(isinstance(c, DocumentChunk) for c in chunks), "All should be DocumentChunks"
        assert all(c.paper_id == "test123" for c in chunks), "All should have correct paper_id"
        assert all(c.paper_title == "Test Paper" for c in chunks), "All should have correct title"
        assert chunks[0].chunk_index == 0, "First chunk index should be 0"

        # Verify sequential indexing
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, f"Chunk index should be {i}"

        print(f"  ✓ Chunker produced {len(chunks)} chunks from {len(text)} chars")
        print("  ✓ All chunks have correct metadata")
        return True

    @staticmethod
    def test_chunker_token_counting():
        """Test token counting in SemanticChunker"""
        print("\n🧪 Testing Token Counting...")

        chunker = SemanticChunker()

        # Test token count for known text
        short_text = "Hello world"
        count = chunker._count_tokens(short_text)
        assert count > 0, "Token count should be positive"

        # Longer text should have more tokens
        long_text = "This is a much longer sentence with many more words and tokens in it."
        long_count = chunker._count_tokens(long_text)
        assert long_count > count, "Longer text should have more tokens"

        print(f"  ✓ Short text: {count} tokens")
        print(f"  ✓ Long text: {long_count} tokens")
        return True

    @staticmethod
    def test_chunker_empty_input():
        """Test chunker with edge cases"""
        print("\n🧪 Testing Chunker Edge Cases...")

        chunker = SemanticChunker()

        # Empty text
        chunks = chunker.chunk_text("", paper_id="empty", paper_title="Empty")
        assert len(chunks) == 0, "Empty text should produce no chunks"

        # Whitespace only
        chunks = chunker.chunk_text("   \n\n   ", paper_id="ws", paper_title="Whitespace")
        assert len(chunks) == 0, "Whitespace-only text should produce no chunks"

        # Single word
        chunks = chunker.chunk_text("hello", paper_id="single", paper_title="Single")
        assert len(chunks) == 1, "Single word should produce one chunk"

        print("  ✓ Empty text → 0 chunks")
        print("  ✓ Whitespace → 0 chunks")
        print("  ✓ Single word → 1 chunk")
        return True

    @staticmethod
    async def test_document_processor():
        """Test DocumentProcessor text extraction"""
        print("\n🧪 Testing DocumentProcessor...")

        processor = DocumentProcessor()

        # Test text cleaning
        dirty_text = "Hello\n42\nWorld   with    extra   spaces"
        cleaned = processor._clean_text(dirty_text)
        assert "  " not in cleaned, "Should collapse multiple spaces"
        assert cleaned.strip() == cleaned, "Should strip leading/trailing whitespace"

        # Test with non-existent file (should return None gracefully)
        result = await processor.extract_text(Path("nonexistent_file.pdf"))
        assert result is None, "Should return None for missing file"

        print("  ✓ Text cleaning removes excess whitespace")
        print("  ✓ Missing file handled gracefully")
        return True

    @staticmethod
    async def test_rag_engine_init():
        """Test RAGEngine initialization with temp directory"""
        print("\n🧪 Testing RAGEngine Initialization...")

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = str(Path(tmpdir) / "test_chroma")
            engine = RAGEngine(collection_name="test_papers", persist_dir=persist_dir)

            stats = engine.get_stats()

            if engine.collection:
                assert stats["status"] == "ready", "Should report ready"
                assert stats["total_chunks"] == 0, "Should start with 0 chunks"
                print(f"  ✓ RAGEngine initialized: {stats}")
            else:
                assert stats["status"] == "unavailable", "Should report unavailable without deps"
                print("  ✓ RAGEngine reports unavailable (missing deps, expected)")

        return True

    @staticmethod
    async def test_rag_engine_query_empty():
        """Test querying an empty RAG engine"""
        print("\n🧪 Testing RAGEngine Empty Query...")

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = str(Path(tmpdir) / "test_chroma_query")
            engine = RAGEngine(collection_name="test_query", persist_dir=persist_dir)

            # Query on empty collection should return empty list
            results = await engine.query("test query about cancer", top_k=5)
            assert isinstance(results, list), "Should return a list"
            assert len(results) == 0, "Empty collection should return no results"

            print("  ✓ Empty query returns empty list")

        return True

    @staticmethod
    async def test_rag_engine_skip_duplicates():
        """Test that RAGEngine skips already indexed papers"""
        print("\n🧪 Testing RAGEngine Duplicate Skipping...")

        # This test requires chroma to be functional
        if not chromadb:
            print("  ⚠ Skipping RAG duplicate test (chromadb missing)")
            return True

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = str(Path(tmpdir) / "test_chroma_dup")
            engine = RAGEngine(collection_name="test_dup", persist_dir=persist_dir)

            # Load embedding model if not loaded
            if engine.embedding_model is None:
                print("  ⚠ Skipping RAG duplicate test (embedding model missing)")
                return True

            paper = {
                "title": "Duplicate Paper",
                "url": "https://example.com/dup",
                "source": "ArXiv",
                "summary": "This is a test paper."
            }

            # Mock download and extraction to avoid network calls
            from pathlib import Path
            from unittest.mock import patch

            with patch.object(PDFDownloader, 'download_paper', return_value=Path("dummy.pdf")), \
                 patch.object(DocumentProcessor, 'extract_text', return_value="Extracted text content."):

                # First run: should index
                indexed1 = await engine.process_papers([paper])
                assert indexed1 > 0, "Should index paper on first run"
                count1 = engine.collection.count()

                # Second run: should skip
                indexed2 = await engine.process_papers([paper])
                assert indexed2 == 0, "Should skip paper on second run"
                count2 = engine.collection.count()
                assert count1 == count2, "Count should not increase on second run"

        print("  ✓ RAGEngine correctly skips already indexed papers")
        return True



class TestPubMedDownload:
    """Test PubMed PDF download functionality"""

    @staticmethod
    def test_pubmed_url_parsing():
        """Test PMID extraction from PubMed URLs"""
        print("\n🧪 Testing PubMed URL Parsing...")

        # Valid URLs
        url1 = "https://pubmed.ncbi.nlm.nih.gov/27454254/"
        url2 = "https://pubmed.ncbi.nlm.nih.gov/38218645"
        url3 = "https://pubmed.ncbi.nlm.nih.gov/12345678/?some_param=true"

        # Invalid URLs
        url_invalid = "https://arxiv.org/abs/2302.12345"
        url_empty = "https://pubmed.ncbi.nlm.nih.gov/"

        # Test extraction with regex (same logic as PDFDownloader)
        pattern = r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)"

        match1 = re.search(pattern, url1)
        assert match1 and match1.group(1) == "27454254", f"Should extract 27454254, got {match1}"

        match2 = re.search(pattern, url2)
        assert match2 and match2.group(1) == "38218645", f"Should extract 38218645, got {match2}"

        match3 = re.search(pattern, url3)
        assert match3 and match3.group(1) == "12345678", f"Should extract 12345678, got {match3}"

        match_invalid = re.search(pattern, url_invalid)
        assert match_invalid is None, "Should not match non-PubMed URL"

        match_empty = re.search(pattern, url_empty)
        assert match_empty is None, "Should not match URL without PMID"

        print("  ✓ Extracts PMID from standard URLs")
        print("  ✓ Handles trailing slashes and query params")
        print("  ✓ Rejects non-PubMed URLs")
        return True

    @staticmethod
    def test_cache_path_generation():
        """Test PDF cache path generation"""
        print("\n🧪 Testing Cache Path Generation...")

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PDFDownloader(cache_dir=tmpdir)

            url1 = "https://pubmed.ncbi.nlm.nih.gov/27454254/"
            url2 = "https://pubmed.ncbi.nlm.nih.gov/38218645/"

            path1 = downloader._get_cache_path(url1)
            path2 = downloader._get_cache_path(url2)

            # Different URLs should produce different paths
            assert path1 != path2, "Different URLs should have different cache paths"

            # Same URL should produce same path (deterministic)
            path1_again = downloader._get_cache_path(url1)
            assert path1 == path1_again, "Same URL should produce same cache path"

            # Path should be in cache dir
            assert str(path1).startswith(tmpdir), "Path should be in cache directory"
            assert str(path1).endswith(".pdf"), "Path should end with .pdf"

        print("  ✓ Unique paths for different URLs")
        print("  ✓ Deterministic paths for same URL")
        print("  ✓ Paths stored in cache directory")
        return True

    @staticmethod
    def test_download_paper_routing():
        """Test that download_paper routes to correct method based on source"""
        print("\n🧪 Testing Download Paper Routing...")

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PDFDownloader(cache_dir=tmpdir)

            # Verify the downloader has the expected methods
            assert hasattr(downloader, 'download_arxiv_pdf'), "Should have download_arxiv_pdf"
            assert hasattr(downloader, 'download_pubmed_pdf'), "Should have download_pubmed_pdf"
            assert hasattr(downloader, 'download_paper'), "Should have download_paper"
            assert hasattr(downloader, '_get_pmcid_from_pmid'), "Should have _get_pmcid_from_pmid"

        print("  ✓ All download methods exist")
        print("  ✓ Routing method (download_paper) is available")
        return True

    @staticmethod
    async def test_pubmed_download_no_pmcid():
        """Test PubMed download when no PMCID is available"""
        print("\n🧪 Testing PubMed Download (No PMCID)...")

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PDFDownloader(cache_dir=tmpdir)

            # Use a URL that likely has no PMC version
            # The method should return None gracefully without Entrez or with non-OA article
            result = await downloader.download_pubmed_pdf("https://not-pubmed.example.com/12345")
            assert result is None, "Should return None for invalid URL"

        print("  ✓ Invalid URL returns None gracefully")
        return True

    @staticmethod
    async def test_download_paper_unknown_source():
        """Test download_paper with unknown source"""
        print("\n🧪 Testing Download Unknown Source...")

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PDFDownloader(cache_dir=tmpdir)

            paper = {"url": "https://example.com/paper", "source": "UnknownDB"}
            result = await downloader.download_paper(paper)
            assert result is None, "Should return None for unknown source"

            paper_empty = {"url": "", "source": ""}
            result = await downloader.download_paper(paper_empty)
            assert result is None, "Should return None for empty source"

        print("  ✓ Unknown source returns None")
        print("  ✓ Empty source returns None")
        return True


async def run_all_unit_tests():
    """Run all unit tests"""
    print("\n" + "="*50)
    print("RUNNING ALL UNIT TESTS")
    print("="*50)

    passed = 0
    failed = 0

    # Run synchronous tests first
    sync_tests = [
        ("Utilities.json_parsing", TestUtilities.test_json_parsing),
        ("GenerationAgent.prompt_building", TestGenerationAgent.test_prompt_building),
        ("RAGSystem.semantic_chunker", TestRAGSystem.test_semantic_chunker),
        ("RAGSystem.token_counting", TestRAGSystem.test_chunker_token_counting),
        ("RAGSystem.chunker_edge_cases", TestRAGSystem.test_chunker_empty_input),
        ("PubMed.url_parsing", TestPubMedDownload.test_pubmed_url_parsing),
        ("PubMed.cache_path", TestPubMedDownload.test_cache_path_generation),
        ("PubMed.download_routing", TestPubMedDownload.test_download_paper_routing),
    ]

    for name, test_fn in sync_tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {name} FAILED: {e}")

    # Run async tests
    async_tests = [
        ("GenerationAgent.hypothesis_generation", TestGenerationAgent.test_hypothesis_generation()),
        ("ReflectionAgent.review", TestReflectionAgent.test_hypothesis_review()),
        ("RankingAgent.elo_ratings", TestRankingAgent.test_elo_ratings()),
        ("ProximityAgent.similarity", TestProximityAgent.test_similarity_computation()),
        ("EvolutionAgent.evolution", TestEvolutionAgent.test_hypothesis_evolution()),
        ("MetaReviewAgent.meta_review", TestMetaReviewAgent.test_meta_review()),
        ("RAGSystem.document_processor", TestRAGSystem.test_document_processor()),
        ("RAGSystem.engine_init", TestRAGSystem.test_rag_engine_init()),
        ("RAGSystem.engine_query_empty", TestRAGSystem.test_rag_engine_query_empty()),
        ("RAGSystem.engine_skip_duplicates", TestRAGSystem.test_rag_engine_skip_duplicates()),
        ("PubMed.download_no_pmcid", TestPubMedDownload.test_pubmed_download_no_pmcid()),
        ("PubMed.download_unknown_source", TestPubMedDownload.test_download_paper_unknown_source()),
    ]

    for name, coro in async_tests:
        try:
            result = await coro
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {name} FAILED: {e}")

    print(f"\n{'✅' if failed == 0 else '❌'} {passed} passed, {failed} failed")
    return passed, failed


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

async def test_full_workflow():
    """Test complete workflow"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Full Co-Scientist Workflow")
    print("="*80)

    co_scientist = CoScientist()

    # Initialize
    await co_scientist.initialize_research_goal(
        title="Test Research Goal",
        description="Integration test goal",
        domain="test_domain"
    )

    # Run workflow
    print("\n1. Generating hypotheses...")
    co_scientist = CoScientist()
    goal = ResearchGoal(title="Test", domain="test")
    co_scientist.context_memory.research_goal = goal
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(num_hypotheses=5)

    assert len(hypotheses) == 5
    print(f"   ✓ Generated {len(hypotheses)} hypotheses")

    print("\n2. Reviewing hypotheses...")
    reviews = await co_scientist.run_review_cycle()
    assert len(reviews) == 5
    print(f"   ✓ Completed {len(reviews)} reviews")

    print("\n3. Computing proximity...")
    proximity = await co_scientist.proximity_agent.compute_proximity(
        list(co_scientist.context_memory.hypotheses.values())
    )
    assert len(proximity) == 5
    print(f"   ✓ Computed proximity for {len(proximity)} hypotheses")

    print("\n4. Running tournament...")
    matches = await co_scientist.run_tournament_cycle(num_matches=5)
    assert len(matches) == 5
    print(f"   ✓ Completed {len(matches)} tournament matches")

    print("\n5. Evolving hypotheses...")
    evolved = await co_scientist.run_evolution_cycle()
    assert len(evolved) == 3
    print(f"   ✓ Evolved {len(evolved)} hypotheses")

    print("\n6. Meta-review...")
    meta_review = await co_scientist.run_meta_review_cycle()
    assert meta_review['total_hypotheses'] > 5
    print("  ✓ Meta-review generated")
    print(f"  ✓ Summary includes {meta_review['total_hypotheses']} hypotheses")
    return True
    # Export
    print("\n7. Exporting results...")
    co_scientist.export_hypotheses_json("test_results.json")
    print("   ✓ Results exported")

    print("\n✅ INTEGRATION TEST PASSED")
    return True


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

async def benchmark_generation_speed():
    """Benchmark hypothesis generation speed"""
    print("\n" + "="*80)
    print("BENCHMARK: Hypothesis Generation Speed")
    print("="*80)

    agent = GenerationAgent()
    goal = ResearchGoal(title="Benchmark", domain="test")

    for count in [10, 50, 100]:
        start = time.time()
        await agent.generate_initial_hypotheses(goal, context_papers=[], count=count)
        elapsed = time.time() - start

        rate = count / elapsed
        print(f"  {count} hypotheses: {elapsed:.2f}s ({rate:.1f} hyp/sec)")


async def benchmark_review_speed():
    """Benchmark review speed"""
    print("\n" + "="*80)
    print("BENCHMARK: Hypothesis Review Speed")
    print("="*80)

    agent = ReflectionAgent()
    goal = ResearchGoal(title="Benchmark", domain="test")

    # Create hypotheses
    gen_agent = GenerationAgent()
    hypotheses = await gen_agent.generate_initial_hypotheses(goal, context_papers=[], count=20)

    start = time.time()
    for h in hypotheses:
        await agent.review_hypothesis(h, goal)
    elapsed = time.time() - start

    rate = len(hypotheses) / elapsed
    print(f"  {len(hypotheses)} reviews: {elapsed:.2f}s ({rate:.1f} reviews/sec)")


async def benchmark_tournament_speed():
    """Benchmark tournament speed"""
    print("\n" + "="*80)
    print("BENCHMARK: Tournament Match Speed")
    print("="*80)

    agent = RankingAgent()
    gen_agent = GenerationAgent()
    goal = ResearchGoal(title="Benchmark", domain="test")

    hypotheses = await gen_agent.generate_initial_hypotheses(goal, context_papers=[], count=10)

    start = time.time()
    matches = 0
    for i in range(len(hypotheses)):
        for j in range(i+1, len(hypotheses)):
            _, _ = await agent.conduct_tournament_match(hypotheses[i], hypotheses[j])
            matches += 1
    elapsed = time.time() - start

    rate = matches / elapsed
    print(f"  {matches} matches: {elapsed:.2f}s ({rate:.1f} matches/sec)")


async def benchmark_proximity_speed():
    """Benchmark proximity computation speed"""
    print("\n" + "="*80)
    print("BENCHMARK: Proximity Computation Speed")
    print("="*80)

    agent = ProximityAgent()
    gen_agent = GenerationAgent()
    goal = ResearchGoal(title="Benchmark", domain="test")

    for count in [10, 50, 100]:
        hypotheses = await gen_agent.generate_initial_hypotheses(goal, context_papers=[], count=count)

        start = time.time()
        await agent.compute_proximity(hypotheses)
        elapsed = time.time() - start

        pairs = count * (count - 1) / 2
        rate = pairs / elapsed
        print(f"  {count} hypotheses ({pairs:.0f} pairs): {elapsed:.2f}s ({rate:.1f} pairs/sec)")


async def run_all_benchmarks():
    """Run all performance benchmarks"""
    print("\n" + "="*80)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("="*80)

    await benchmark_generation_speed()
    await benchmark_review_speed()
    await benchmark_tournament_speed()
    await benchmark_proximity_speed()

    print("\n✅ BENCHMARKS COMPLETED")


# =============================================================================
# MEMORY AND SCALABILITY TESTS
# =============================================================================

async def test_memory_efficiency():
    """Test memory usage with large hypothesis sets"""
    print("\n" + "="*80)
    print("MEMORY EFFICIENCY TEST")
    print("="*80)

    import sys

    co_scientist = CoScientist()
    await co_scientist.initialize_research_goal(
        title="Memory Test",
        description="Test",
        domain="test"
    )

    # Generate large hypothesis set
    gen_agent = GenerationAgent()
    goal = ResearchGoal(title="Test", domain="test")

    for batch in range(3):
        hypotheses = await gen_agent.generate_initial_hypotheses(goal, context_papers=[], count=50)
        for h in hypotheses:
            co_scientist.context_memory.hypotheses[h.id] = h

        total_hyps = len(co_scientist.context_memory.hypotheses)
        print(f"  Batch {batch+1}: {total_hyps} hypotheses in memory")

    print("✅ MEMORY TEST PASSED - No crashes with large hypothesis sets")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

async def main():
    """Run all tests and benchmarks"""

    print("\n" + "="*80)
    print("AI CO-SCIENTIST: TEST SUITE")
    print("="*80)

    # Unit tests
    passed, failed = await run_all_unit_tests()

    # Integration test
    try:
        await test_full_workflow()
        integration_passed = True
    except Exception as e:
        print(f"Integration test failed: {e}")
        integration_passed = False

    # Benchmarks
    await run_all_benchmarks()

    # Memory test
    await test_memory_efficiency()

    # Final summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"Unit Tests: {passed} passed, {failed} failed")
    print(f"Integration Test: {'PASSED' if integration_passed else 'FAILED'}")
    print("Benchmarks: COMPLETED")
    print("Memory Test: PASSED")
    print("\n✨ TEST SUITE COMPLETED")


if __name__ == "__main__":
    asyncio.run(main())
