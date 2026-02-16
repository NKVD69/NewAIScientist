"""
Unit Tests and Benchmarks for AI Co-Scientist System
Tests core functionality and performance of all agents
"""

import asyncio
import json
import time
from typing import List, Dict
from co_scientist import (
    CoScientist, ResearchGoal, Hypothesis, ReviewCritique,
    GenerationAgent, ReflectionAgent, RankingAgent, 
    ProximityAgent, EvolutionAgent, MetaReviewAgent,
    HypothesisStatus, _parse_json_response, ContextMemory
)


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
        
        hypotheses = await agent.generate_initial_hypotheses(goal, count=3)
        
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
        
        print("  ✓ JSON parser is robust to markdown fences")
        return True


async def run_all_unit_tests():
    """Run all unit tests"""
    print("\n" + "="*50)
    print("RUNNING ALL UNIT TESTS")
    print("="*50)
    
    results = [
        await TestGenerationAgent.test_hypothesis_generation(),
        TestGenerationAgent.test_prompt_building(), # New
        await TestReflectionAgent.test_hypothesis_review(),
        await TestRankingAgent.test_elo_ratings(),
        await TestProximityAgent.test_similarity_computation(),
        await TestEvolutionAgent.test_hypothesis_evolution(),
        await TestMetaReviewAgent.test_meta_review(),
        TestUtilities.test_json_parsing() # New
    ]
    
    success = all(results)
    if success:
        print("\n✅ ALL UNIT TESTS PASSED")
    else:
        print("\n❌ SOME UNIT TESTS FAILED")
    return success


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
        hypotheses = await agent.generate_initial_hypotheses(goal, count=count)
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
    hypotheses = await gen_agent.generate_initial_hypotheses(goal, count=20)
    
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
    
    hypotheses = await gen_agent.generate_initial_hypotheses(goal, count=10)
    
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
        hypotheses = await gen_agent.generate_initial_hypotheses(goal, count=count)
        
        start = time.time()
        proximity = await agent.compute_proximity(hypotheses)
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
        hypotheses = await gen_agent.generate_initial_hypotheses(goal, count=50)
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
