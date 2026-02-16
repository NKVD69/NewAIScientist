"""
Example Usage: Advanced Co-Scientist Workflows
Demonstrates specialized use cases and integration patterns
"""

import asyncio
import json
from co_scientist import (
    CoScientist, ResearchGoal, Hypothesis, 
    HypothesisStatus
)


# =============================================================================
# EXAMPLE 1: Drug Repurposing for Cancer
# =============================================================================

async def example_drug_repurposing():
    """
    Use case: Identify existing drugs for new cancer indications
    Similar to the AML case study in the paper
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: DRUG REPURPOSING FOR CANCER")
    print("="*80)
    
    co_scientist = CoScientist()
    
    # Initialize with drug repurposing goal
    await co_scientist.initialize_research_goal(
        title="Drug Repurposing for Triple-Negative Breast Cancer (TNBC)",
        description=(
            "Identify FDA-approved drugs that could be repurposed to treat "
            "triple-negative breast cancer (TNBC). Target drugs that inhibit "
            "cell proliferation and survival pathways in TNBC. Focus on drugs "
            "with known pharmacokinetic profiles and minimal off-target toxicity."
        ),
        domain="Biomedicine/Oncology",
        preferences={
            "focus_on_novelty": True,
            "require_testability": True,
            "prioritize_clinical_relevance": True,
            "mechanism_clarity": "high"
        },
        constraints=[
            "Only FDA-approved drugs with documented safety",
            "Must target known TNBC pathways",
            "IC50 concentration should be clinically achievable",
            "Avoid drugs with documented cardiotoxicity"
        ]
    )
    
    # Run workflow focused on drug repurposing
    print("\n📊 Running Drug Repurposing Workflow...")
    
    # Generate more hypotheses for combinatorial search
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(
        num_hypotheses=8
    )
    
    # Intensive review process for drug candidates
    reviews = await co_scientist.run_review_cycle()
    print(f"✓ Reviewed {len(reviews)} drug hypotheses")
    
    # Tournament with many matches (simulates drug comparison)
    matches = await co_scientist.run_tournament_cycle(num_matches=8)
    print(f"✓ Conducted {len(matches)} head-to-head drug comparisons")
    
    # Final meta-review for clinical prioritization
    meta_review = await co_scientist.run_meta_review_cycle()
    
    # Extract top candidates for wet-lab testing
    top_drugs = meta_review['top_hypotheses'][:3]
    print(f"\n🏆 Top 3 Drug Candidates for In Vitro Testing:")
    for i, drug in enumerate(top_drugs, 1):
        print(f"  {i}. {drug['title']}")
        print(f"     Elo: {drug['elo_rating']:.0f} | Novelty: {drug['novelty']}")
    
    return co_scientist


# =============================================================================
# EXAMPLE 2: Novel Target Discovery for Disease
# =============================================================================

async def example_target_discovery():
    """
    Use case: Identify completely novel therapeutic targets
    More challenging than drug repurposing - larger hypothesis space
    Similar to liver fibrosis case study
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: NOVEL TARGET DISCOVERY FOR FIBROTIC DISEASE")
    print("="*80)
    
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Novel Epigenetic Targets for Pulmonary Fibrosis",
        description=(
            "Propose novel epigenetic regulators (histone modifiers, chromatin remodelers, "
            "DNA methyltransferases) that could serve as therapeutic targets for pulmonary fibrosis. "
            "Focus on mechanisms controlling myofibroblast differentiation and fibrogenesis. "
            "Hypotheses should integrate recent epigenetic discoveries with fibrosis pathobiology."
        ),
        domain="Biomedicine/Pulmonology",
        preferences={
            "focus_on_novelty": True,
            "require_mechanism_understanding": True,
            "prioritize_druggability": True,
            "emphasize_epigenetics": True
        },
        constraints=[
            "Focus on epigenetic regulators only",
            "Must explain myofibroblast formation",
            "Should be testable in cell/organoid systems",
            "Consider existing epigenetic drugs as templates"
        ]
    )
    
    print("\n🔬 Running Target Discovery Workflow...")
    
    # Generate hypotheses (more needed for novel discovery)
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(
        num_hypotheses=10
    )
    print(f"✓ Generated {len(hypotheses)} novel target hypotheses")
    
    # Run multiple review cycles to deeply evaluate novelty
    print("\n📝 Deep Evaluation Phase...")
    for cycle in range(2):
        reviews = await co_scientist.run_review_cycle()
        print(f"  Cycle {cycle+1}: {len(reviews)} reviews completed")
    
    # Compute proximity to identify clusters of similar targets
    print("\n🔗 Hypothesis Clustering...")
    proximity = await co_scientist.proximity_agent.compute_proximity(
        list(co_scientist.context_memory.hypotheses.values())
    )
    
    # Identify clusters
    clusters = {}
    for hyp_id, similarities in proximity.items():
        # Group by similar hypotheses
        if similarities:
            similar_id = similarities[0][0]
            if similar_id not in clusters:
                clusters[similar_id] = []
            clusters[similar_id].append(hyp_id)
    
    print(f"  Found {len(clusters)} clusters of related targets")
    
    # Evolution focused on improving top targets
    print("\n🧬 Target Refinement...")
    evolved = await co_scientist.run_evolution_cycle()
    print(f"  Refined {len(evolved)} top targets")
    
    # Final meta-review
    meta_review = await co_scientist.run_meta_review_cycle()
    
    print(f"\n🎯 Novel Targets Identified:")
    for i, target in enumerate(meta_review['top_hypotheses'][:3], 1):
        print(f"  {i}. {target['title']}")
        print(f"     Mechanism strength: {target['elo_rating']:.0f}")
    
    return co_scientist


# =============================================================================
# EXAMPLE 3: Mechanism Explanation for Biological Phenomena
# =============================================================================

async def example_mechanism_discovery():
    """
    Use case: Explain mechanisms underlying biological phenomena
    Most complex - involves system-level understanding
    Similar to antimicrobial resistance case study
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: MECHANISM DISCOVERY FOR BACTERIAL EVOLUTION")
    print("="*80)
    
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Gene Transfer Mechanisms in CRISPR-Based Horizontal Gene Transfer",
        description=(
            "Explain the mechanisms by which CRISPR-Cas systems facilitate horizontal "
            "gene transfer (HGT) between distantly related bacterial species. Focus on "
            "understanding how phage-inducible chromosomal islands (PICIs) exploit CRISPR "
            "machinery. Integrate molecular, evolutionary, and ecological perspectives."
        ),
        domain="Microbiology/Molecular Evolution",
        preferences={
            "focus_on_mechanistic_understanding": True,
            "require_evolutionary_perspective": True,
            "emphasize_system_level": True,
            "consider_ecological_context": True
        },
        constraints=[
            "Must integrate molecular and evolutionary mechanisms",
            "Should explain inter-species transfer",
            "Consider selective pressures and ecosystem effects",
            "Validate against experimental observations"
        ]
    )
    
    print("\n🔬 Running Mechanism Discovery Workflow...")
    
    # Generate many hypotheses for complex phenomenon
    print("\n1️⃣ HYPOTHESIS GENERATION PHASE")
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(
        num_hypotheses=12
    )
    print(f"✓ Generated {len(hypotheses)} mechanistic hypotheses")
    
    # Intensive multi-round review process
    print("\n2️⃣ DEEP REVIEW PHASE (Multiple Rounds)")
    for round_num in range(3):
        print(f"\n  Round {round_num+1}:")
        reviews = await co_scientist.run_review_cycle()
        print(f"    - Reviewed {len(reviews)} hypotheses")
        
        # Show some review feedback
        if reviews:
            sample_review = reviews[0]
            print(f"    - Sample novelty score: {sample_review.novelty_score:.2f}")
            print(f"    - Sample testability: {sample_review.testability_score:.2f}")
    
    # Extensive tournament
    print("\n3️⃣ TOURNAMENT PHASE (Mechanism Comparison)")
    print("  Running iterative tournament rounds...")
    
    for tournament_round in range(3):
        matches = await co_scientist.run_tournament_cycle(num_matches=6)
        print(f"  Round {tournament_round+1}: {len(matches)} mechanism debates")
    
    # Show tournament insights
    print(f"\n  Tournament Statistics:")
    print(f"  Total matches: {len(co_scientist.context_memory.tournament_history)}")
    
    # Evolution for mechanism refinement
    print("\n4️⃣ MECHANISM REFINEMENT PHASE")
    evolved = await co_scientist.run_evolution_cycle()
    print(f"✓ Evolved {len(evolved)} top mechanisms")
    
    # Meta-review with focus on consistency
    print("\n5️⃣ META-REVIEW & SYNTHESIS PHASE")
    meta_review = await co_scientist.run_meta_review_cycle()
    
    print(f"\n📋 Research Overview Generated")
    print(meta_review['research_overview'])
    
    return co_scientist


# =============================================================================
# EXAMPLE 4: Multi-Domain Integration
# =============================================================================

async def example_multidomain_discovery():
    """
    Use case: Discover solutions spanning multiple domains
    Leverages trans-disciplinary synthesis
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: TRANS-DISCIPLINARY RESEARCH (Optogenetic Drug Delivery)")
    print("="*80)
    
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Light-Activated Drug Delivery Systems for Precision Medicine",
        description=(
            "Develop novel light-activated drug delivery systems integrating optogenetics, "
            "nanoparticle engineering, and pharmacology. Hypotheses should bridge: "
            "(1) optical control mechanisms from neurobiology, "
            "(2) nanoparticle design from materials science, "
            "(3) drug binding and release from pharmaceutical chemistry, "
            "(4) clinical applicability considerations. Target indication: localized cancer therapy."
        ),
        domain="Trans-disciplinary/Bioengineering",
        preferences={
            "require_cross_domain_integration": True,
            "focus_on_novelty": True,
            "emphasize_practical_implementation": True,
            "require_clinical_relevance": True
        },
        constraints=[
            "Must integrate optics + nanoparticles + chemistry + medicine",
            "Should minimize off-target activation",
            "Consider tissue penetration limits",
            "Propose testable mechanisms"
        ]
    )
    
    print("\n🌈 Running Trans-Disciplinary Workflow...")
    
    # Generate with emphasis on interdisciplinary ideas
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(num_hypotheses=7)
    
    # Review from multiple perspective angles
    reviews = await co_scientist.run_review_cycle()
    print(f"✓ {len(reviews)} reviews from trans-disciplinary perspective")
    
    # Tournament highlights integrative solutions
    matches = await co_scientist.run_tournament_cycle(num_matches=5)
    
    # Evolution combines ideas from different domains
    evolved = await co_scientist.run_evolution_cycle()
    
    # Meta-review synthesizes cross-domain insights
    meta_review = await co_scientist.run_meta_review_cycle()
    
    print(f"\n✨ Top Trans-Disciplinary Hypotheses:")
    for hyp in meta_review['top_hypotheses'][:2]:
        print(f"  • {hyp['title']}")
    
    return co_scientist


# =============================================================================
# EXAMPLE 5: Iterative Scientist-in-the-Loop Collaboration
# =============================================================================

async def example_scientist_in_the_loop():
    """
    Use case: Interactive collaboration with scientist feedback
    Demonstrates the scientist-in-the-loop paradigm
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: SCIENTIST-IN-THE-LOOP COLLABORATION")
    print("="*80)
    
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Immunotherapy Resistance Mechanisms",
        description=(
            "Understand mechanisms of resistance to checkpoint inhibitor immunotherapy. "
            "Generate testable hypotheses that can be refined through iterative scientist feedback."
        ),
        domain="Biomedicine/Immuno-oncology",
        preferences={"require_scientist_feedback": True}
    )
    
    print("\n👥 Scientist-AI Collaborative Mode")
    print("="*80)
    
    # Round 1: Initial generation and evaluation
    print("\nRound 1: Initial Hypothesis Generation")
    print("-" * 40)
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(num_hypotheses=5)
    
    for i, hyp in enumerate(hypotheses[:2], 1):
        print(f"\nGenerated Hypothesis {i}: {hyp.title}")
        print(f"  Mechanism: {hyp.mechanism[:80]}...")
    
    # Scientist feedback (simulated)
    print("\n💬 Scientist Feedback:")
    print("  - 'Hypothesis 1 is interesting, focus on tumor microenvironment angle'")
    print("  - 'Consider adding metabolic exhaustion pathway'")
    
    # Review with guidance from feedback
    reviews = await co_scientist.run_review_cycle()
    print(f"\n✓ {len(reviews)} hypotheses evaluated with feedback guidance")
    
    # Round 2: Refinement based on feedback
    print("\nRound 2: Hypothesis Refinement")
    print("-" * 40)
    evolved = await co_scientist.run_evolution_cycle()
    print(f"✓ Evolved {len(evolved)} hypotheses based on feedback")
    
    # Tournament
    matches = await co_scientist.run_tournament_cycle(num_matches=4)
    
    # Round 3: Specialist review
    print("\nRound 3: Specialist Expert Review")
    print("-" * 40)
    reviews = await co_scientist.run_review_cycle()
    
    # Meta-review with recommendations
    meta_review = await co_scientist.run_meta_review_cycle()
    
    print(f"\n📊 Final Recommendations for Scientist:")
    for i, hyp in enumerate(meta_review['top_hypotheses'][:3], 1):
        print(f"\n  {i}. {hyp['title']}")
        print(f"     Score: {hyp['elo_rating']:.0f}")
        print(f"     Ready for: Molecular validation experiments")
    
    return co_scientist


# =============================================================================
# EXAMPLE 6: Batch Processing Multiple Research Goals
# =============================================================================

async def example_batch_processing():
    """
    Process multiple research goals in sequence
    Useful for large-scale screening
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: BATCH PROCESSING (Multiple Research Goals)")
    print("="*80)
    
    goals = [
        {
            "title": "Drug Repurposing for Rare Disease 1",
            "domain": "Biomedicine/Rare Diseases",
            "description": "Find drugs for Rare Disease X"
        },
        {
            "title": "Drug Repurposing for Rare Disease 2",
            "domain": "Biomedicine/Rare Diseases",
            "description": "Find drugs for Rare Disease Y"
        },
        {
            "title": "Target Discovery for Rare Disease 3",
            "domain": "Biomedicine/Rare Diseases",
            "description": "Find targets for Rare Disease Z"
        }
    ]
    
    results = []
    
    for idx, goal in enumerate(goals, 1):
        print(f"\n📌 Processing Goal {idx}/{len(goals)}: {goal['title']}")
        print("-" * 60)
        
        co_scientist = CoScientist()
        await co_scientist.initialize_research_goal(
            title=goal["title"],
            description=goal["description"],
            domain=goal["domain"]
        )
        
        # Quick processing cycle
        hypotheses = await co_scientist.run_hypothesis_generation_cycle(num_hypotheses=5)
        reviews = await co_scientist.run_review_cycle()
        matches = await co_scientist.run_tournament_cycle(num_matches=3)
        meta_review = await co_scientist.run_meta_review_cycle()
        
        # Store best result
        best_hyp = meta_review['top_hypotheses'][0]
        results.append({
            "goal": goal['title'],
            "best_hypothesis": best_hyp['title'],
            "elo_rating": best_hyp['elo_rating'],
            "hypotheses_generated": len(hypotheses)
        })
        
        print(f"✓ Goal {idx} complete: {best_hyp['title'][:50]}...")
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    for result in results:
        print(f"\n{result['goal']}")
        print(f"  Best: {result['best_hypothesis']}")
        print(f"  Elo: {result['elo_rating']:.0f}")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all examples"""
    
    print("\n" + "="*80)
    print("AI CO-SCIENTIST: ADVANCED USAGE EXAMPLES")
    print("="*80)
    
    # Run examples (comment out as needed)
    
    # Example 1: Drug Repurposing
    co1 = await example_drug_repurposing()
    co1.export_hypotheses_json("results_drug_repurposing.json")
    
    # Example 2: Target Discovery
    co2 = await example_target_discovery()
    co2.export_hypotheses_json("results_target_discovery.json")
    
    # Example 3: Mechanism Discovery
    co3 = await example_mechanism_discovery()
    co3.export_hypotheses_json("results_mechanism_discovery.json")
    
    # Example 4: Multi-Domain
    co4 = await example_multidomain_discovery()
    co4.export_hypotheses_json("results_multidomain.json")
    
    # Example 5: Scientist-in-the-loop
    co5 = await example_scientist_in_the_loop()
    co5.export_hypotheses_json("results_collaborative.json")
    
    # Example 6: Batch processing
    batch_results = await example_batch_processing()
    
    print("\n" + "="*80)
    print("✨ ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nResults exported to:")
    print("  - results_drug_repurposing.json")
    print("  - results_target_discovery.json")
    print("  - results_mechanism_discovery.json")
    print("  - results_multidomain.json")
    print("  - results_collaborative.json")


if __name__ == "__main__":
    asyncio.run(main())
