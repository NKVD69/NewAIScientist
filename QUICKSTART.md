# Quick Start Guide: AI Co-Scientist

## 🚀 Installation

### Prerequisites
- Python 3.9+
- `pypdf`, `chromadb` for RAG system
- `openai` and `streamlit` libraries
- Recommended: Virtual Environment (.venv)

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-repo/ai-co-scientist.git
cd ai-co-scientist

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Start your local LLM server (LM Studio/Ollama)
```

### Run the App
```bash
# Use the python executable from your venv
.venv\Scripts\python.exe -m streamlit run app.py
```

## 📖 Basic Usage (5 minutes)

### 1. Using the GUI (Streamlit)

1.  Run `.venv\Scripts\python.exe -m streamlit run app.py`
2.  In the sidebar, verify the LLM settings:
    *   **URL**: `http://127.0.0.1:1234/v1`
    *   **Model**: `openai/gpt-oss-20b`
3.  Adjust **"Max Papiers (ArXiv)"** to control literature search depth.
4.  Enter your research goal (e.g., "New treatments for Alzheimer's")
5.  Click **"Lancer la Recherche"**
6.  Watch the agents work in real-time!

### 2. Simple Hypothesis Generation (Python Script)

```python
import asyncio
from co_scientist import CoScientist

async def main():
    # Initialize the co-scientist
    # Will automatically try to connect to local LLM at http://127.0.0.1:1234/v1
    co_scientist = CoScientist()
    
    # Define your research goal
    await co_scientist.initialize_research_goal(
        title="Find new anti-cancer drugs",
        description="Identify FDA-approved drugs for novel cancer indications",
        domain="Oncology"
    )
    
    # 1. Search Literature (RAG)
    await co_scientist.run_literature_search(max_results=5)
    
    # 2. Generate hypotheses (using context)
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(
        num_hypotheses=5
    )
    
    # Print results
    for i, h in enumerate(hypotheses, 1):
        print(f"{i}. {h.title}")
        if h.cited_papers:
            print(f"   Cites: {', '.join(h.cited_papers)}")

asyncio.run(main())
```

### 3. Full Workflow (10 minutes)

```python
import asyncio
from co_scientist import CoScientist

async def main():
    co_scientist = CoScientist()
    
    # Define research goal
    await co_scientist.initialize_research_goal(
        title="Novel Drug Targets for Diabetes",
        description="Identify new therapeutic targets for type 2 diabetes",
        domain="Endocrinology"
    )
    
    # Run complete workflow (includes literature search)
    await co_scientist.run_full_cycle(num_iterations=3)
    
    # Export results
    co_scientist.export_hypotheses_json("my_results.json")

asyncio.run(main())
```

## 🎯 Common Use Cases

### Use Case 1: Drug Repurposing

```python
async def drug_repurposing():
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Drug Repurposing for Rare Genetic Disease",
        description=(
            "Identify FDA-approved drugs that could treat "
            "hereditary disease X based on pathway analysis"
        ),
        domain="Biomedicine",
        constraints=[
            "Only FDA-approved drugs",
            "Must have documented mechanism of action",
            "Focus on drugs with safety data"
        ]
    )
    
    # Generate many hypotheses for combinatorial search
    await co_scientist.run_hypothesis_generation_cycle(15)
    
    # Intensive review
    await co_scientist.run_review_cycle()
    
    # Many tournament matches to rank
    await co_scientist.run_tournament_cycle(20)
    
    # Export top candidates
    co_scientist.export_hypotheses_json("drug_candidates.json")
```

### Use Case 2: Novel Target Discovery

```python
async def target_discovery():
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Novel Targets for Fibrosis",
        description=(
            "Propose new epigenetic regulators as targets "
            "for treatment of fibrotic diseases"
        ),
        domain="Biomedicine",
        preferences={"focus_on_novelty": True}
    )
    
    # More hypotheses for larger search space
    await co_scientist.run_hypothesis_generation_cycle(20)
    
    # Multiple review rounds for deep evaluation
    for _ in range(3):
        await co_scientist.run_review_cycle()
    
    # Extensive tournament
    for _ in range(5):
        await co_scientist.run_tournament_cycle(10)
    
    # Evolution to refine top targets
    await co_scientist.run_evolution_cycle()
    
    # Get final recommendations
    meta_review = await co_scientist.run_meta_review_cycle()
    
    print("Top Targets:")
    for target in meta_review['top_hypotheses'][:5]:
        print(f"  • {target['title']}")
```

### Use Case 3: Mechanism Understanding

```python
async def mechanism_explanation():
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Mechanism of Pathogen Evolution",
        description=(
            "Explain the mechanisms of horizontal gene "
            "transfer in bacterial species"
        ),
        domain="Microbiology"
    )
    
    # Run multiple iterations for complex systems
    await co_scientist.run_full_cycle(num_iterations=5)
    
    # Get comprehensive overview
    meta_review = await co_scientist.run_meta_review_cycle()
    print(meta_review['research_overview'])
```

## 📊 Reading the Results

### Exported JSON Structure

```json
{
  "research_goal": {
    "title": "Your Research Goal",
    "domain": "Your Domain",
    "description": "..."
  },
  "literature_context": [
    {
      "title": "Paper Title",
      "summary": "Abstract...",
      "url": "http://arxiv.org/abs/..."
    }
  ],
  "hypotheses": [
    {
      "id": "abc12345",
      "title": "Hypothesis Title",
      "description": "Full description",
      "mechanism": "How it works",
      "elo_rating": 1250.5,
      "novelty_level": "high",
      "status": "ranked",
      "testable_predictions": ["Prediction 1", "Prediction 2"],
      "grounding_evidence": ["Evidence 1", "Evidence 2"],
      "cited_papers": ["Paper Title 1"],
      "generation_method": "initial",
      "num_reviews": 3
    }
  ],
  "tournament_matches": 15,
  "statistics": {
    "generation_agent": 25,
    "reflection_agent": 20,
    "ranking_agent": 15,
    "evolution_agent": 5,
    "literature_agent": 5
  }
}
```

### Interpreting Scores

**Elo Rating** (1200 baseline)
- 1200-1350: Good hypotheses
- 1350-1500: Very good hypotheses
- 1500+: Top-tier hypotheses

**Novelty Levels**
- `low` (0-0.35): Incremental improvements
- `medium` (0.35-0.55): Somewhat novel
- `high` (0.55-0.75): Genuinely new
- `very_high` (0.75+): Breakthrough ideas

**Testability Score** (0-1)
- <0.5: Difficult to test
- 0.5-0.7: Testable with effort
- 0.7+: Clearly testable

## 🔧 Customization

### Modify Agent Behavior

```python
from co_scientist import CoScientist

co_scientist = CoScientist()

# Change Elo K-factor (sensitivity to results)
co_scientist.ranking_agent.k_factor = 64

# Change initial Elo
# (regenerate hypotheses with different baseline)

# Adjust number of top hypotheses to track
top_k = 10  # For meta-review
```

### Add Custom Research Goal Preferences

```python
await co_scientist.initialize_research_goal(
    title="My Goal",
    description="...",
    domain="...",
    preferences={
        "focus_on_novelty": True,  # Emphasize novelty
        "require_testability": True,  # Filter non-testable
        "prioritize_mechanism": True,  # Focus on understanding
        "clinical_relevance": "high",  # Medical focus
        "interdisciplinary": True,  # Cross-domain synthesis
    },
    constraints=[
        "Only consider recent literature (2020+)",
        "Must be implementable in 6 months",
        "Limited to 100k budget"
    ]
)
```

## 📈 Monitoring Progress

### Real-time Monitoring

```python
import asyncio
from co_scientist import CoScientist

async def monitor_workflow():
    co_scientist = CoScientist()
    
    await co_scientist.initialize_research_goal(
        title="Test",
        description="Test",
        domain="test"
    )
    
    # Manual control of workflow
    hypotheses = await co_scientist.run_hypothesis_generation_cycle(10)
    print(f"Generated: {len(hypotheses)}")
    
    # Check progress
    memory = co_scientist.context_memory
    print(f"Total hypotheses: {len(memory.hypotheses)}")
    print(f"Tournament matches: {len(memory.tournament_history)}")
    
    # Get top hypothesis
    top = sorted(
        memory.hypotheses.values(),
        key=lambda h: h.elo_rating,
        reverse=True
    )[0]
    print(f"Top hypothesis Elo: {top.elo_rating:.0f}")
    
    # Continue processing
    await co_scientist.run_review_cycle()
    await co_scientist.run_tournament_cycle(5)
```

## 🐛 Troubleshooting

### Issue: No hypotheses generated
```python
# Check if you initialized the research goal
# ✓ Goal must be initialized before generation

await co_scientist.initialize_research_goal(
    title="Your Goal",
    description="...",
    domain="..."
)
```

### Issue: Low Elo ratings for all hypotheses
```python
# This is normal - Elo starts at 1200
# Ratings improve through tournament matches
# Run more tournament cycles

await co_scientist.run_tournament_cycle(10)  # More matches
```

### Issue: Memory error with large hypothesis sets
```python
# Export periodically to avoid memory bloat
co_scientist.export_hypotheses_json("checkpoint.json")

# Can restart from checkpoint (manual implementation needed)
# Or process in batches with separate instances
```

## 📚 Learning Resources

### Understanding the System

1. **Read the README.md** - Full system overview
2. **Review ARCHITECTURE.md** - Technical details
3. **Study examples.py** - Real use cases
4. **Run test_suite.py** - See system in action

### Key Concepts

- **Elo Rating**: Tournament-based hypothesis ranking
- **Meta-Review**: Synthesizing insights from all agents
- **Proximity Graph**: Clustering similar hypotheses
- **Evolution Strategies**: Improving hypotheses iteratively

## 🚀 Next Steps

1. **Try the basic example** (5 min)
   ```bash
   python co_scientist.py
   ```

2. **Run a custom goal** (10 min)
   - Modify the main() function in co_scientist.py
   - Set your own research goal
   - Export your results

3. **Explore use cases** (20 min)
   ```bash
   python examples.py
   ```

4. **Run tests** (5 min)
   ```bash
   python test_suite.py
   ```

5. **Integrate Local LLM** (advanced)
   - Install Ollama or LM Studio
   - Ensure `openai` library is installed
   - Run co_scientist.py

## 📞 Support

### Common Questions

**Q: Can I use this with real research?**
A: Yes! By connecting a local LLM (like Mistral or Llama 3) or using an OpenAI-compatible API, the system can generate genuine scientific hypotheses based on the model's knowledge.

**Q: How do I integrate with my data?**
A: Modify the Generation Agent to query your databases or APIs instead of generating synthetic hypotheses.

**Q: Can I run this on distributed systems?**
A: Current implementation is single-threaded async. For distributed use, modify the Supervisor Agent for multi-machine deployment.

**Q: What's the computational cost?**
A: Minimal - most time is spent in simulated debates. With a local LLM, it depends on your hardware (GPU recommended).

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Ready for use

Happy Researching! 🔬
