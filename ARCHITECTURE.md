# Architecture Documentation: AI Co-Scientist System

## System Overview

The AI Co-Scientist is a sophisticated multi-agent system designed to accelerate scientific discovery through intelligent hypothesis generation, evaluation, and iterative refinement. The architecture mirrors key aspects of the scientific method while leveraging modern AI capabilities.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCIENTIST INTERFACE                       │
│              (Natural Language Research Goals)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                             │
│        (Orchestrates all agents + manages task queue)           │
├─────────────────────────────────────────────────────────────────┤
│  • Task queue management (priority-based)                       │
│  • Worker process assignment                                    │
│  • Resource allocation                                          │
│  • System state monitoring                                      │
└───┬──────────────────────┬──────────────────────┬──────────────┘
    │                      │                      │
    ▼                      ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌────────────────┐
│  LITERATURE   │  │   GENERATION     │  │  REFLECTION    │
│     AGENT     │  │      AGENT       │  │     AGENT      │
├───────────────┤  ├──────────────────┤  ├────────────────┤
│ • ArXiv/PubMed│  │ • Hypothesis     │  │ • Correctness  │
│   search      │  │   creation       │  │   assessment   │
│ • Context     │  │ • Assumption     │  │ • Novelty eval │
│   retrieval   │  │   discovery      │  │ • Testability  │
│               │  │ • Analogy        │  │   review       │
└───────────────┘  └──────────────────┘  └────────────────┘
    │                      │                      │
    ▼                      ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌────────────────┐
│   RANKING     │  │   PROXIMITY      │  │    EVOLUTION   │
│    AGENT      │  │     AGENT        │  │      AGENT     │
├───────────────┤  ├──────────────────┤  ├────────────────┤
│ • Elo Tourney │  │ • Similarity     │  │ • Enhancement  │
│ • Debate sim  │  │   scoring        │  │ • Simplification│
│ • Rating upd  │  │ • Proximity      │  │ • Combination  │
│               │  │   graph          │  │ • Inspiration  │
└───────────────┘  └──────────────────┘  └────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    META-REVIEW AGENT                            │
│ • Pattern synthesis  • Insight extraction  • Feedback gen       │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT MEMORY                               │
│     (Persistent state across iterations + history)              │
├─────────────────────────────────────────────────────────────────┤
│  • Hypotheses database          • Literature Context            │
│  • Tournament history           • Research goal specification   │
│  • Agent performance statistics • Iteration counter             │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Standard Workflow Iteration

```
START
  │
  ├─► LITERATURE SEARCH CYCLE
  │   ├─ Query external databases (ArXiv, PubMed)
  │   └─ Populate Context Memory with relevant papers
  │
  ├─► GENERATION CYCLE
  │   ├─ Generate new hypotheses (using literature context)
  │   └─ Store in Context Memory
  │
  ├─► REVIEW CYCLE
  │   ├─ Reflect agent reviews unreviewed hypotheses
  │   ├─ Assess: correctness, novelty, testability, quality
  │   └─ Update hypothesis with scores and feedback
  │
  ├─► PROXIMITY CYCLE
  │   ├─ Compute similarity between all hypotheses
  │   └─ Build proximity graph for clustering
  │
  ├─► TOURNAMENT CYCLE
  │   ├─ Select hypothesis pairs (similar ones prioritized)
  │   ├─ Simulate scientific debates
  │   ├─ Determine winner and update Elo ratings
  │   └─ Store match in tournament history
  │
  ├─► EVOLUTION CYCLE
  │   ├─ Select top-ranked hypotheses
  │   ├─ Apply evolution strategies (enhance, simplify, combine)
  │   ├─ Create new evolved hypotheses
  │   └─ Return to REVIEW CYCLE
  │
  ├─► META-REVIEW CYCLE
  │   ├─ Synthesize all reviews and debates
  │   ├─ Identify patterns and recurring issues
  │   ├─ Generate research overview
  │   ├─ Suggest improvements and focus areas
  │   └─ Provide feedback to other agents
  │
  └─► DECISION
      ├─ If iterations remaining → Repeat
      └─ Else → FINALIZE
```

## Agent Interactions

### Communication Patterns

```
1. FEEDBACK PROPAGATION
   Meta-Review Agent → All Agents
   (Updated critiques via prompt injection)

2. EVOLUTIONARY IMPROVEMENT
   Evolution Agent → Generation Agent
   (Top hypothesis patterns for new generation)

3. RANKING OPTIMIZATION
   Ranking Agent ← Reflection Agent
   (Review scores used in debate scoring)

4. PROXIMITY-GUIDED MATCHING
   Ranking Agent ← Proximity Agent
   (Similarity graph guides tournament matches)

5. MEMORY PERSISTENCE
   All Agents → Context Memory
   (State maintained across iterations)
```

## Task Queue System

### Priority-Based Execution

```
Task Definition:
├─ Agent name (which agent to execute)
├─ Action (generate, review, tournament, etc.)
├─ Parameters (goal, hypotheses, etc.)
└─ Priority (lower number = higher priority)

Priority Levels:
├─ 1-2:   CRITICAL (reviews of high-Elo hypotheses)
├─ 3-5:   HIGH (tournament matches, evolution)
├─ 6-8:   NORMAL (generation of new hypotheses)
└─ 9-10:  LOW (auxiliary computations)

Queue Management:
├─ Min-heap for O(log n) insertion/extraction
├─ Asynchronous execution (non-blocking)
├─ Task history tracking
└─ Supervisor orchestration
```

## Review System

### Multi-Dimensional Evaluation

```
Hypothesis Quality = f(
    correctness:     0-1  (logical validity, grounding)
    novelty:         0-1  (new vs. existing literature)
    testability:     0-1  (empirical validation feasibility)
    quality:         0-1  (overall integration of above)
)

Review Types:
├─ Initial Review
│  ├─ Quick assessment
│  ├─ No external tools
│  └─ Early filtering

├─ Full Review
│  ├─ Comprehensive evaluation
│  ├─ Literature search
│  └─ Detailed critique

├─ Deep Verification
│  ├─ Assumption decomposition
│  ├─ Sub-assumption validation
│  └─ Error identification

├─ Observation Review
│  ├─ Experimental evidence check
│  ├─ Long-tail observation coverage
│  └─ Explanatory power assessment

├─ Simulation Review
│  ├─ Mechanism simulation
│  ├─ Failure scenario identification
│  └─ Feasibility assessment

└─ Recurrent/Tournament Review
   ├─ Dynamic based on patterns
   ├─ Cumulative learning
   └─ Adaptive criteria
```

## Elo Rating System

### Scientific Hypothesis Tournament

```
Initial Rating: 1200 (standard Elo baseline)

Match Formula:
├─ Expected Probability:
│  P(A wins) = 1 / (1 + 10^((Rating_B - Rating_A)/400))
│  P(B wins) = 1 - P(A wins)
│
├─ Rating Update (K=32):
│  New_Rating = Old_Rating + K × (Result - Expected)
│  │
│  ├─ If Win:  Result = 1,  Gain = K × (1 - Expected)
│  └─ If Loss: Result = 0,  Gain = K × (0 - Expected)
│
└─ Properties:
   ├─ Zero-sum (total ratings conserved)
   ├─ Reward upsets (underdog wins)
   ├─ Converges to true ranking over many matches
   └─ Dynamic (tracks hypothesis quality evolution)

Tournament Structure:
├─ Round-robin or selective matching
├─ Similarity-based pairing (Proximity Agent)
├─ Top-k frequently re-matched
├─ Multi-turn debates for high-Elo hypotheses
└─ Single-turn comparisons for lower-Elo
```

## Hypothesis Lifecycle

```
States:
├─ GENERATED: newly created
├─ UNDER_REVIEW: review in progress
├─ REVIEWED: review complete
├─ IN_TOURNAMENT: actively competing
├─ RANKED: assigned Elo rating
├─ EVOLVED: derived from evolution
└─ COMPLETED: final status

Genealogy Tracking:
├─ Parent IDs: source hypotheses
├─ Generation method: initial/evolved/combined/inspired
├─ Timestamp: creation time
└─ Review history: all reviews applied

Status Progression:
GENERATED
   │
   ▼
UNDER_REVIEW (Reflection Agent active)
   │
   ▼
REVIEWED (Scores assigned)
   │
   ├─► IN_TOURNAMENT (Ranking Agent)
   │      │
   │      ▼
   │    RANKED (Elo updated)
   │      │
   │      ├─► Feedback to Meta-Review
   │      └─► Selected for Evolution
   │
   └─► EVOLVED (Evolution Agent)
          │
          ▼
       GENERATED (cycle continues)
          │
          ▼
       COMPLETED (final outputs)
```

## Memory Architecture

```
ContextMemory:
├─ research_goal: ResearchGoal
│  ├─ title: Research question
│  ├─ description: Detailed specification
│  ├─ domain: Scientific domain
│  ├─ preferences: Dict of constraints
│  └─ constraints: List of requirements
│
├─ hypotheses: Dict[id, Hypothesis]
│  └─ Indexed by unique ID for O(1) access
│
├─ tournament_history: List[TournamentMatch]
│  ├─ Chronological record
│  └─ Used for pattern analysis
│
├─ agent_performance_stats: Dict[agent_name, stats]
│  ├─ Tracks each agent's contributions
│  └─ Enables resource optimization
│
└─ iteration_count: int
   └─ Total iterations completed

Persistence:
├─ In-memory during execution (fast)
├─ Periodic JSON export
├─ Enables restart/recovery
└─ Historical analysis
```

## Integration Points

### For OpenAI / Local LLM Integration

```python
# Supports OpenAI API and Local LLMs (LM Studio, Ollama)

import openai

class GenerationAgent:
    def __init__(self, base_url="http://localhost:1234/v1"):
        self.client = openai.OpenAI(base_url=base_url, api_key="lm-studio")
    
    async def generate_with_llm(self, goal: ResearchGoal):
        response = self.client.chat.completions.create(
            model="local-model",
            messages=[{
                "role": "user",
                "content": goal.description
            }],
            response_format={"type": "json_object"}
        )
        # Parse and return Hypothesis objects
```

### External Tool Integration

```python
class EvolutionAgent:
    async def ground_hypothesis_with_tools(self, hypothesis):
        # Web search for evidence
        web_results = await web_search_api(hypothesis.mechanism)
        
        # Database queries (PubMed, ChemSpider, etc.)
        papers = await pubmed_api(hypothesis.domain)
        
        # Specialized tools (AlphaFold for proteins, etc.)
        structure = await alphafold_api(hypothesis.protein_sequence)
        
        # Update hypothesis with grounding
        return enhanced_hypothesis
```

## Configuration Parameters

```python
# Generation Agent
GENERATION_STRATEGIES = 4  # Literature, assumptions, analogies, divergent
GENERATION_BATCH_SIZE = 5  # Hypotheses per cycle

# Reflection Agent
REVIEW_DIMENSIONS = 4  # Correctness, novelty, testability, quality
SCORE_RANGE = (0.0, 1.0)  # Min, max

# Ranking Agent
ELO_K_FACTOR = 32  # Sensitivity to match results
ELO_INITIAL = 1200  # Starting rating
DEBATE_ROUNDS = [2, 1]  # Multi-turn for top, single for others

# Proximity Agent
SIMILARITY_WEIGHTS = {
    "mechanism": 0.5,
    "predictions": 0.3,
    "citations": 0.2
}

# Evolution Agent
EVOLUTION_STRATEGIES = [
    "enhancement",
    "simplification",
    "out_of_box"
]

# Meta-Review Agent
TOP_K = 5  # Number of top hypotheses to track
SUMMARY_LENGTH = "medium"  # Brief/medium/detailed
```

## Performance Characteristics

```
Time Complexity:
├─ Generation: O(n) where n = batch size
├─ Review: O(n) with external tool overhead
├─ Tournament: O(m) where m = number of matches
├─ Proximity: O(n²) for all-pairs comparison
├─ Evolution: O(k) where k = top hypotheses
└─ Meta-review: O(n + m) for synthesis

Space Complexity:
├─ Hypotheses storage: O(n × h) where h = avg hypothesis size
├─ Proximity graph: O(n²) for full similarity matrix
├─ Tournament history: O(m) matches
└─ Total: O(n² + m) for large-scale runs

Typical Scaling:
├─ 100 hypotheses, 3 iterations: ~30 seconds
├─ 1000 hypotheses, 3 iterations: ~5 minutes
└─ Scales linearly with hypothesis count (with O(n²) proximity)
```

## Error Handling

```
Graceful Degradation:
├─ If review fails: mark hypothesis with partial scores
├─ If tournament fails: maintain previous Elo
├─ If evolution fails: keep original hypothesis
├─ If meta-review fails: generate basic summary
└─ If memory corruption: restore from checkpoint

Validation:
├─ Hypothesis validation (all required fields)
├─ Review score validation (0-1 range)
├─ Elo rating sanity checks
├─ Memory consistency checks
└─ Data integrity verification
```

## Future Extensions

### Multi-Modal Learning
```
├─ Process protein structures (AlphaFold)
├─ Analyze molecular graphs
├─ Incorporate experimental images
└─ Multi-modality fusion
```

### Reinforcement Learning Integration
```
├─ Learn optimal agent weighting
├─ Optimize tournament structure
├─ Improve review criteria
└─ Refine evolution strategies
```

### Distributed Execution
```
├─ Multi-machine deployment
├─ Distributed task queue
├─ Federated agent networks
└─ Scalable hypothesis storage
```

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Stable
