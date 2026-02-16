"""
NewAI Scientist: Multi-agent system for scientific hypothesis generation and refinement
Reproduces the co-scientist workflow from Google DeepMind research

This implementation includes:
- Asynchronous task framework with worker queue
- 7 specialized agents (Generation, Reflection, Ranking, Evolution, Proximity, Meta-review, Literature)
- Tournament-based hypothesis ranking (Elo rating system)
- Natural language interface for scientist-in-the-loop collaboration
- Persistent context memory for iterative reasoning
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import heapq
import math
from collections import defaultdict
import re


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _parse_json_response(content: str) -> Any:
    """Robustly parse JSON from LLM responses, stripping markdown fences."""
    content = content.strip()
    # Remove markdown code fences
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    return json.loads(content)

# Attempt to import openai, but don't fail if not installed
try:
    import openai
except ImportError:
    openai = None

# Attempt to import arxiv
try:
    import arxiv
except ImportError:
    arxiv = None

# Attempt to import biopython for PubMed
try:
    from Bio import Entrez
    Entrez.email = "your.email@example.com"  # Required by NCBI
except ImportError:
    Entrez = None

# Attempt to import RAG system
try:
    from rag_system import RAGEngine
except ImportError:
    RAGEngine = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class HypothesisStatus(Enum):
    """Hypothesis lifecycle states"""
    GENERATED = "generated"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    IN_TOURNAMENT = "in_tournament"
    RANKED = "ranked"
    EVOLVED = "evolved"
    COMPLETED = "completed"


@dataclass
class ReviewCritique:
    """Structure for review feedback"""
    review_type: str  # initial, full, deep_verification, observation, simulation, recurrent
    correctness_score: float  # 0-1
    novelty_score: float  # 0-1
    testability_score: float  # 0-1
    quality_score: float  # 0-1
    feedback: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Hypothesis:
    """Core hypothesis data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    reasoning: str = ""  # New: Logic/papers that led to this hypothesis
    mechanism: str = ""
    testable_predictions: List[str] = field(default_factory=list)
    grounding_evidence: List[str] = field(default_factory=list)
    
    # Quality metrics
    elo_rating: float = 1200.0  # Initial Elo rating
    novelty_level: str = "unknown"  # low, medium, high, very_high
    
    # Lifecycle
    status: HypothesisStatus = HypothesisStatus.GENERATED
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reviews: List[ReviewCritique] = field(default_factory=list)
    
    # Genealogy
    parent_ids: List[str] = field(default_factory=list)
    generation_method: str = "initial"  # initial, evolved, combined, inspired
    
    # Citations
    cited_papers: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


@dataclass
class ResearchGoal:
    """Research goal specification from scientist"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    domain: str = ""  # biomedicine, physics, chemistry, etc.
    preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TournamentMatch:
    """Record of a pairwise hypothesis comparison"""
    match_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_a_id: str = ""
    hypothesis_b_id: str = ""
    winner_id: str = ""
    debate_summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ContextMemory:
    """Persistent memory of system state"""
    research_goal: ResearchGoal = field(default_factory=ResearchGoal)
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)
    tournament_history: List[TournamentMatch] = field(default_factory=list)
    agent_performance_stats: Dict[str, Dict] = field(default_factory=dict)
    iteration_count: int = 0
    literature_context: List[Dict] = field(default_factory=list) # Stores retrieved papers
    meta_reviews: List[Dict] = field(default_factory=list)  # Stores meta-review results
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class LiteratureAgent:
    """Retrieves and analyzes relevant scientific literature"""
    
    def __init__(self, use_local_llm: bool = True, enable_rag: bool = True):
        self.name = "Literature"
        self.papers_retrieved = 0
        self.llm_client = None
        self.rag_engine = None
        self.enable_rag = enable_rag
        self.use_local_llm = use_local_llm
        
        if use_local_llm and openai:
            try:
                self.llm_client = openai.OpenAI(
                    base_url=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1"),
                    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
                )
            except Exception:
                pass
        
        # Initialize RAG system if enabled
        if enable_rag and RAGEngine:
            try:
                self.rag_engine = RAGEngine()
                print("✓ RAG system initialized")
            except Exception as e:
                print(f"⚠ RAG initialization failed: {e}")
                self.rag_engine = None

    async def _generate_search_queries(self, goal: ResearchGoal) -> List[str]:
        """Uses LLM to generate optimized boolean search queries"""
        if not self.llm_client:
            # Fallback: simple keyword extraction
            words = [w for w in goal.title.split() if len(w) > 3]
            return [f"{goal.title}", f"{' AND '.join(words[:3])}"]

        prompt = f"""
        You are an expert at searching scientific databases (ArXiv, PubMed).
        Generate 2 optimized search queries for the following research goal.
        Goal: "{goal.title}"
        Description: "{goal.description}"
        
        Return ONLY a JSON object with a list of strings: {{ "queries": ["query1", "query2"] }}
        Queries should use keywords and boolean operators (AND, OR). Keep them concise.
        """
        try:
            try:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            except Exception:
                # Fallback to text mode if JSON format not supported
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                    messages=[{"role": "user", "content": prompt}],
                )
            data = _parse_json_response(response.choices[0].message.content)
            return data.get("queries", [goal.title])
        except Exception as e:
            print(f"⚠ Query generation failed: {e}")
            return [goal.title]

    async def _refine_query(self, goal: ResearchGoal, current_papers: List[Dict], last_query: str) -> str:
        """Analyze current papers and suggest a refined query to fill gaps"""
        if not self.use_local_llm or not self.llm_client:
            return None
            
        summaries = "\n".join([f"- {p['title']}: {p.get('summary', '')[:100]}..." for p in current_papers[:5]])
        
        prompt = f"""
        Research Goal: {goal.title}
        Current Search Query: "{last_query}"
        
        Papers found so far:
        {summaries}
        
        Analyze what is missing to fully address the research goal. 
        Generate ONE single, precise search query (keywords) to find the missing information.
        Do not explain, just provide the query.
        """
        
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip().replace('"', '')
            # Ensure we don't return a huge explanation if LLM ignores instructions
            if len(content) > 150: 
                return None
            return content
        except Exception as e:
            print(f"⚠ Query refinement failed: {e}")
            return None

    async def search_literature(self, goal: ResearchGoal, max_results: int = 5, sources: List[str] = None, iterations: int = 2) -> List[Dict]:
        """
        Search for relevant papers using specified source APIs with iterative refinement.
        Returns a list of paper dictionaries (title, summary, authors, url).
        """
        if sources is None:
            sources = ["arxiv"]
        all_papers = []
        known_titles = set()
        
        # 1. Initial Search
        current_query = goal.title
        queries = await self._generate_search_queries(goal)
        if queries:
            current_query = queries[0]
            
        print(f"   🔍 Iteration 1: Query = {current_query}")

        for i in range(iterations):
            # Execute search on all sources
            iteration_papers = []
            for source in sources:
                source = source.lower()
                print(f"📚 Searching {source.upper()} (Iter {i+1})...")
                
                if source == "arxiv":
                    papers = await self._search_arxiv(current_query, max_results)
                elif source == "pubmed":
                    papers = await self._search_pubmed(current_query, max_results)
                else:
                    papers = []
                
                iteration_papers.extend(papers)
            
            # Filter duplicates
            new_papers = []
            for p in iteration_papers:
                if p['title'] not in known_titles:
                    new_papers.append(p)
                    known_titles.add(p['title'])
                    all_papers.append(p)
            
            print(f"   Found {len(new_papers)} new papers.")
            
            # If no new papers or last iteration, stop
            if not new_papers or i == iterations - 1:
                break
                
            # Refine query for next iteration based on what we found
            print("   🤔 Analyzing results to refine search...")
            refinement = await self._refine_query(goal, all_papers, current_query)
            if refinement:
                current_query = refinement
                print(f"   🔄 Refined Query: {current_query}")
            else:
                break
        
        return all_papers[:max_results * 2] # Allow more results total for broader context

    async def extract_key_findings(self, papers: List[Dict], goal: ResearchGoal = None) -> str:
        """
        Extract and synthesize key findings from a list of papers for CAG context.
        Returns a formatted markdown string.
        """
        if not papers:
            return "No papers available for context."
            
        context_str = "## Key Findings from Literature (CAG Context)\n\n"
        domain = goal.domain if goal else "domain"
        
        for i, paper in enumerate(papers[:10]): # Limit to top 10 for context window efficiency
            summary = paper.get('summary', 'No summary available').replace('\n', ' ')[:300]
            context_str += f"### {i+1}. {paper['title']} ({paper.get('published', 'N/A')})\n"
            context_str += f"**Summary:** {summary}...\n"
            context_str += f"**Key Insight:** Relevance to {domain}\n\n"
            
        return context_str

    async def _search_arxiv(self, query: str, max_results: int) -> List[Dict]:
        if not arxiv:
            print("⚠ `arxiv` library not found.")
            return []
            
        # Query is now passed directly
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            client = arxiv.Client()
            
            def fetch_results():
                papers = []
                for result in client.results(search):
                    papers.append({
                        "title": result.title,
                        "summary": result.summary.replace("\n", " "),
                        "authors": [a.name for a in result.authors],
                        "published": result.published.strftime("%Y-%m-%d"),
                        "url": result.entry_id,
                        "source": "ArXiv"
                    })
                return papers

            results = await asyncio.to_thread(fetch_results)
            self.papers_retrieved += len(results)
            print(f"✓ Found {len(results)} papers on ArXiv.")
            return results
            
        except Exception as e:
            print(f"⚠ ArXiv search failed: {e}")
            return []

    async def _search_pubmed(self, query: str, max_results: int) -> List[Dict]:
        if not Entrez:
            print("⚠ `biopython` library not found.")
            return []
            
        # Query is now passed directly (assumed cleaned/optimized by LLM or fallback)
        # But we still ensure basic safety for Entrez
        safe_query = re.sub(r'[^\w\s\-\(\)ANDOR]', '', query)
        
        try:
            def fetch_pubmed():
                handle = Entrez.esearch(db="pubmed", term=safe_query, retmax=max_results)
                record = Entrez.read(handle)
                handle.close()
                id_list = record["IdList"]
                
                if not id_list:
                    return []
                
                # 2. Fetch details
                # Parsing MEDLINE format manually or using Bio.Medline is better, 
                # but let's use xml for easier parsing
                handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                
                papers = []
                if 'PubmedArticle' not in records:
                    return []

                for article in records['PubmedArticle']:
                    medline = article['MedlineCitation']['Article']
                    title = medline.get('ArticleTitle', 'No Title')
                    abstract = "No abstract available."
                    if 'Abstract' in medline and 'AbstractText' in medline['Abstract']:
                        # Handle list or string content in AbstractText
                        abstract_content = medline['Abstract']['AbstractText']
                        if isinstance(abstract_content, list):
                            abstract = " ".join(str(x) for x in abstract_content)
                        else:
                            abstract = str(abstract_content)
                    
                    authors = []
                    if 'AuthorList' in medline:
                        for a in medline['AuthorList']:
                            if 'LastName' in a and 'Initials' in a:
                                authors.append(f"{a['LastName']} {a['Initials']}")
                    
                    pub_date = "Unknown"
                    if 'ArticleDate' in medline and medline['ArticleDate']:
                        d = medline['ArticleDate'][0]
                        pub_date = f"{d.get('Year', '')}-{d.get('Month', '')}-{d.get('Day', '')}".strip('-')
                    elif 'Journal' in medline and 'JournalIssue' in medline['Journal'] and 'PubDate' in medline['Journal']['JournalIssue']:
                        # Fallback to print date if electronic date is missing
                        pd = medline['Journal']['JournalIssue']['PubDate']
                        pub_date = f"{pd.get('Year', '')} {pd.get('Month', '')}".strip()
                    
                    pmid = article['MedlineCitation']['PMID']
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    
                    papers.append({
                        "title": title,
                        "summary": abstract,
                        "authors": authors,
                        "published": pub_date,
                        "url": url,
                        "source": "PubMed"
                    })
                return papers

            results = await asyncio.to_thread(fetch_pubmed)
            self.papers_retrieved += len(results)
            print(f"✓ Found {len(results)} papers on PubMed.")
            return results
            
        except Exception as e:
            print(f"⚠ PubMed search failed: {e}")
            return []

    async def process_papers_with_rag(self, papers: List[Dict]) -> int:
        """Download and index papers using RAG system"""
        if not self.rag_engine:
            print("ℹ RAG system not available. Skipping paper processing.")
            return 0
        
        print(f"\n📥 Processing {len(papers)} papers with RAG system...")
        chunks_indexed = await self.rag_engine.process_papers(papers)
        return chunks_indexed
    
    async def query_rag(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query RAG system for relevant paper chunks"""
        if not self.rag_engine:
            return []
        
        return await self.rag_engine.query(query, top_k)
    
    def get_rag_stats(self) -> Dict:
        """Get RAG system statistics"""
        if not self.rag_engine:
            return {"status": "disabled"}
        
        return self.rag_engine.get_stats()



# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class GraphAgent:
    """Agent responsible for building a lightweight knowledge graph from papers"""
    
    def __init__(self, use_local_llm: bool = True):
        self.name = "Graph"
        self.graph = {} # Simple adjacency list: {entity: {relation: [target_entities]}}
        self.llm_client = None
        
        if use_local_llm and openai:
            try:
                self.llm_client = openai.OpenAI(
                    base_url=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1"),
                    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"), 
                )
            except Exception:
                self.llm_client = None

    async def build_graph(self, papers: List[Dict], goal: ResearchGoal = None) -> str:
        """Extract entities and relations, return graph summary"""
        if not self.llm_client or not papers:
            return "Graph construction skipped (no LLM or papers)."
            
        print("🕸️ Building Knowledge Graph from literature...")
        
        # Batch process summaries for efficiency
        combined_text = "\n".join([f"{p['title']}: {p.get('summary', '')[:200]}" for p in papers[:5]])
        
        goal_text = f"Research Goal: {goal.title}\n" if goal else ""
        
        prompt = f"""
        {goal_text}
        Extract key scientific entities (Proteins, Genes, Drugs, Diseases, Concepts) and their relationships from the following text based on the research goal.
        Return a JSON list of triples: [{{"subject": "Entity A", "relation": "interacts_with", "object": "Entity B"}}]
        
        Text:
        {combined_text}
        
        Provide ONLY the JSON list.
        """
        
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                # response_format={"type": "json_object"} # Often causes 400 with some models, better parse manually
            )
            content = response.choices[0].message.content.strip()
            # Clean markdown
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            
            triples = json.loads(content.strip())
            
            # Build graph
            self.graph = {}
            for t in triples:
                subj, rel, obj = t.get("subject"), t.get("relation"), t.get("object")
                if subj and rel and obj:
                    if subj not in self.graph: self.graph[subj] = []
                    self.graph[subj].append(f"{rel} -> {obj}")
            
            # Generate summary
            summary = "## 🕸️ Knowledge Graph Insights (GraphRAG)\n"
            sorted_entities = sorted(self.graph.keys(), key=lambda k: len(self.graph[k]), reverse=True)
            for entity in sorted_entities[:5]:
                summary += f"- **{entity}**: {', '.join(self.graph[entity][:3])}\n"
                
            return summary
            
        except Exception as e:
            print(f"⚠ Graph construction failed: {e}")
            return "Graph construction failed."

class GenerationAgent:
    """Generates initial hypotheses and explores research space"""
    
    def __init__(self, use_local_llm: bool = True):
        self.name = "Generation"
        self.generated_count = 0
        self.llm_client = None
        self.last_error = None
        
        if use_local_llm and openai:
            try:
                # Configure for local LLM server (Ollama, LM Studio)
                self.llm_client = openai.OpenAI(
                    base_url=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1"),
                    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"), 
                )
                print("✓ Generation Agent initialized with local LLM connection.")
            except Exception as e:
                print(f"⚠ Could not connect to local LLM: {e}")
                self.llm_client = None
        elif use_local_llm and not openai:
            print("⚠ `openai` library not found. Falling back to simulated generation.")

    async def generate_initial_hypotheses(self, 
                                        goal: ResearchGoal, 
                                        context_papers: List[Dict],
                                        count: int = 5,
                                        rag_context: List[Dict] = None) -> List[Hypothesis]:
        """Generate initial hypotheses using LLM with Self-Refinement"""
        print(f"💡 Generating {count} initial hypotheses with Self-Refinement...")
        
        if self.llm_client:
            try:
                # 1. Generate Drafts
                print("   ✍️ Generating Drafts...")
                draft_hypotheses = await self._generate_with_llm(goal, context_papers, count, rag_context, mode="draft")
                
                if not draft_hypotheses:
                    self.last_error = "LLM returned empty draft list"
                    return await self._generate_simulated(goal, count)

                # 2. Refine Drafts (Self-Correction)
                print("   🛡️ Critiquing and Refining Drafts...")
                refined_hypotheses = []
                for draft in draft_hypotheses:
                    refined = await self._refine_hypothesis(draft, goal)
                    refined_hypotheses.append(refined)
                
                self.generated_count += len(refined_hypotheses)
                return refined_hypotheses

            except Exception as e:
                import traceback
                self.last_error = f"LLM generation failed: {str(e)}\n{traceback.format_exc()}"
                print(f"⚠ {self.last_error}")

        if not self.last_error:
             self.last_error = "LLM client not initialized (check logs)"
        
        return await self._generate_simulated(goal, count)

    async def _refine_hypothesis(self, draft: Hypothesis, goal: ResearchGoal) -> Hypothesis:
        """Critique and refine a single hypothesis"""
        if not self.llm_client:
            return draft
            
        prompt = f"""
        Critique and refine the following scientific hypothesis to ensure it is rigorous, specific, and testable.
        
        Research Goal: {goal.title}
        
        Draft Hypothesis:
        Title: {draft.title}
        Description: {draft.description}
        Mechanism: {draft.mechanism}
        
        Identify 1 weakness (e.g. vague mechanism, lack of feasibility) and ANY improvements.
        Then rewrite the hypothesis in the SAME JSON format as the input, but improved.
        Output ONLY the JSON object.
        """
        
        try:
            # Try with JSON mode first
            try:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
            except Exception:
                # Fallback to text mode if JSON mode fails (e.g. 400 Bad Request)
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )

            content = response.choices[0].message.content.strip()
            
            # Robust parsing: remove markdown code blocks
            if content.startswith("```json"): content = content[7:]
            if content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            
            # Update draft with refined content
            draft.title = data.get("title", draft.title)
            draft.description = data.get("description", draft.description)
            draft.reasoning = data.get("reasoning", draft.reasoning)
            draft.mechanism = data.get("mechanism", draft.mechanism)
            draft.testable_predictions = data.get("testable_predictions", draft.testable_predictions)
            draft.limitations = data.get("limitations", draft.limitations)
            
            return draft
        except Exception as e:
            print(f"⚠ Refinement failed for '{draft.title}': {e}")
            return draft

    def _build_llm_prompt(self, goal: ResearchGoal, context_papers: List[Dict], count: int, rag_context: List[Dict] = None) -> str:
        """Build the LLM prompt for hypothesis generation"""
        
        literature_context = ""
        if context_papers:
            literature_context = "\n**Relevant Literature Context:**\n"
            for i, paper in enumerate(context_papers, 1):
                # Improvement: Increased context limit from 200 to 800 chars to capture methodology/results
                literature_context += f"{i}. {paper['title']} ({paper['published']}): {paper['summary'][:800]}...\n"
        
        # Add RAG context if available (full-text chunks)
        rag_context_text = ""
        if rag_context:
            rag_context_text = "\n**Deep Literature Analysis (RAG):**\n"
            rag_context_text += "The following are the most relevant passages from full papers on this topic:\n\n"
            for i, chunk in enumerate(rag_context, 1):
                rag_context_text += f"Excerpt {i} from '{chunk['paper_title']}':\n"
                rag_context_text += f"{chunk['text'][:1000]}...\n\n"
        
        # CAG: Inject Key Findings if available (Hybrid Context)
        key_findings = ""
        if hasattr(self, 'cag_context') and self.cag_context:
            key_findings = f"""
**Contexte Scientifique Global (CAG):**
Utilisez ces découvertes clés issues de la littérature pour ancrer vos hypothèses dans la réalité actuelle du domaine :
{self.cag_context}
"""

        return f"""
Vous êtes un assistant de recherche IA expert. Votre tâche est de générer {count} hypothèses scientifiques novatrices et testables basées sur l'objectif de recherche et le contexte littéraire fournis.

**Objectif de Recherche :**
- **Titre :** {goal.title}
- **Domaine :** {goal.domain}
- **Description :** {goal.description}
- **Contraintes :** {', '.join(goal.constraints)}
- **Préférences :** {json.dumps(goal.preferences)}

{key_findings}

{literature_context}

{rag_context_text}

Veuillez générer {count} hypothèses distinctes. Pour chaque hypothèse, fournissez les informations suivantes dans un format JSON valide au sein d'un seul tableau JSON `[]`. **Toute l'argumentation et les descriptions doivent être en français.**

**Structure JSON pour chaque hypothèse :**
{{
  "title": "Un titre précis, technique et descriptif (ex: 'Inhibition de X via la voie Y pour Z').",
  "description": "Une explication approfondie de l'hypothèse. Soyez précis sur les cibles, les processus et les impacts attendus. Évitez les généralités.",
  "reasoning": "Détaillez le raisonnement logique et les données bibliographiques spécifiques qui ont permis de formuler cette hypothèse. Expliquez la connexion entre les preuves existantes et l'idée nouvelle.",
  "mechanism": "Décrivez précisément le mécanisme biochimique ou physique proposé. Comment les différentes composantes interagissent-elles ?",
  "testable_predictions": ["Liste de prédictions techniques et quantifiables.", "Prédiction 2", "..."],
  "grounding_evidence": ["Références précises (Auteurs, Année) ou principes physiques fondamentaux.", "Preuve 2", "..."],
  "limitations": ["Analyse critique des failles potentielles de l'hypothèse.", "Limitation 2", "..."]
}}

Assurez-vous que la sortie entière est un seul tableau JSON contenant {count} objets d'hypothèse. N'incluez aucun autre texte ou explication en dehors du tableau JSON.
"""

    async def _generate_with_llm(self, goal: ResearchGoal, context_papers: List[Dict], count: int, rag_context: List[Dict] = None) -> List[Hypothesis]:
        """Generate hypotheses using a local LLM."""
        prompt = self._build_llm_prompt(goal, context_papers, count, rag_context)
        
        model_name = os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b")
        
        try:
            # First attempt with json_object format
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            # If json_object is not supported (e.g. 400 Error), retry without it
            print(f"⚠ JSON mode failed, retrying with standard text mode: {e}")
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
        
        content = response.choices[0].message.content
        
        try:
            # The response should be a JSON string of a list of hypotheses
            # Sometimes the LLM wraps the list in a dictionary, e.g. {"hypotheses": [...]}, so we handle that.
            data = _parse_json_response(content)
            
            # Handle various JSON structures
            if isinstance(data, list):
                hypotheses_data = data
            elif isinstance(data, dict):
                # Try to find a list in the dictionary values
                list_values = [v for v in data.values() if isinstance(v, list)]
                if list_values:
                    hypotheses_data = list_values[0]
                else:
                    # Maybe the dict itself is a single hypothesis? (unlikely but possible)
                    hypotheses_data = [data]
            else:
                raise ValueError("Unexpected JSON structure")

            hypotheses = []
            for item in hypotheses_data:
                # Force citation of context papers if LLM missed them
                cited = item.get("cited_papers", [])
                grounding = item.get("grounding_evidence", [])
                
                if context_papers and not cited:
                    # Heuristic: Add top 2 papers as citations if none provided
                    cited = [f"{p['title']} ({p['published']})" for p in context_papers[:2]]
                    # Also add to grounding evidence if empty
                    if not grounding:
                        grounding = [f"Supported by: {p['title']}" for p in context_papers[:2]]
                
                hypotheses.append(Hypothesis(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    reasoning=item.get("reasoning", ""),
                    mechanism=item.get("mechanism", ""),
                    testable_predictions=item.get("testable_predictions", []),
                    grounding_evidence=grounding,
                    limitations=item.get("limitations", []),
                    cited_papers=cited,
                    generation_method="llm-generated"
                ))
            return hypotheses
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"⚠ Error parsing LLM response: {e}")
            print(f"  Raw response: {content}")
            return []

    async def _generate_simulated(self, goal: ResearchGoal, count: int) -> List[Hypothesis]:
        """
        Generate initial hypotheses addressing the research goal using simulation.
        """
        hypotheses = []
        
        # Simulate different generation strategies
        strategies = [
            self._generate_from_literature,
            self._generate_from_assumptions,
            self._generate_from_analogies,
            self._generate_unconventional,
        ]
        
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            hypothesis = await strategy(goal, i)
            hypotheses.append(hypothesis)
            self.generated_count += 1
        
        return hypotheses
    
    async def _generate_from_literature(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Simulate literature exploration-based generation"""
        h = Hypothesis(
            title=f"Hypothèse {index+1} (Basée sur la littérature) : Mécanisme de {goal.domain}",
            description=f"Cette hypothèse propose un nouveau mécanisme intégrant les recherches existantes sur {goal.domain}. "
                       f"Elle suggère que l'interaction entre les facteurs environnementaux et la prédisposition génétique "
                       f"crée une synergie unique qui explique la variabilité observée dans les études cliniques récentes.",
            reasoning=f"Le raisonnement repose sur l'observation que les études de Wang et al. (2023) montrent une corrélation "
                      f"mais pas de causalité claire. En croisant ces données avec les travaux de Smith (2024), on peut déduire "
                      f"qu'une variable médiatrice, le facteur X, est nécessaire pour déclencher la réponse biologique.",
            mechanism=f"S'appuyant sur la compréhension actuelle de {goal.domain}, ce mécanisme suggère "
                     f"que la voie clé implique une intégration interdisciplinaire d'effets moléculaires et systémiques. "
                     f"Concrètement, la protéine A se lie au récepteur B, activant la cascade C.",
            generation_method="simulated-literature"
        )
        h.testable_predictions = [
            f"Prédiction 1 : Effet observable dans le contexte de {goal.domain} lors de l'activation du facteur X",
            f"Prédiction 2 : Conséquence mesurable en aval sur la concentration de la protéine C",
            f"Prédiction 3 : Changement de paramètre quantifiable via imagerie moléculaire"
        ]
        h.grounding_evidence = [
            "Wang et al. (2023) - Analyse des voies de soutien montrant une corrélation",
            "Smith & Jones (2024) - Mécanisme moléculaire du récepteur B",
            "Méta-analyse récente - Preuves contextuelles sur la variabilité génétique"
        ]
        h.cited_papers = h.grounding_evidence.copy()
        return h
    
    async def _generate_from_assumptions(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Generate from iterative assumptions"""
        h = Hypothesis(
            title=f"Hypothèse {index+1} (Basée sur les suppositions) : Nouvelle cible dans {goal.domain}",
            description=f"Cette hypothèse identifie le facteur Y comme une nouvelle cible thérapeutique potentielle dans {goal.domain}. "
                       f"Elle remet en cause le dogme actuel qui privilégie la cible Z, en montrant que Y agit plus tôt dans la cascade.",
            reasoning=f"Si l'on suppose que la cible Z est en fait un effet et non une cause (supposition X), alors "
                      f"l'analyse logique pointe vers un régulateur amont. Le facteur Y possède toutes les caractéristiques "
                      f"d'un tel régulateur selon les modèles biologiques fondamentaux.",
            mechanism="Chaîne logique : Si la supposition X tient → le facteur Y régule l'expression de Z → l'effet intermédiaire Y impacte le résultat final Z",
            generation_method="simulated-assumption"
        )
        h.testable_predictions = [
            "Test 1 : Valider la supposition fondamentale via un knockdown de Y",
            "Test 2 : Mesurer l'effet intermédiaire sur l'expression de Z",
            "Test 3 : Confirmer le résultat final sur le phénotype cellulaire"
        ]
        h.grounding_evidence = [
            "Déduction logique à partir des principes de régulation génétique",
            "Cadre analytique inter-domaines appliqué aux systèmes complexes"
        ]
        h.limitations = ["Suppose une progression linéaire des voies qui pourrait être plus complexe en réalité"]
        return h
    
    async def _generate_from_analogies(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Generate using analogical reasoning"""
        h = Hypothesis(
            title=f"Hypothèse {index+1} (Basée sur l'analogie) : Mécanisme inter-domaines pour {goal.domain}",
            description=f"Analogue aux mécanismes dans des domaines liés, proposant une nouvelle application à {goal.domain}",
            mechanism="Le mécanisme découvert dans le domaine A peut s'appliquer au domaine B par similarité structurelle",
            generation_method="simulated-analogy"
        )
        h.testable_predictions = [
            "Prédiction d'analogie 1 : Manifestation spécifique au domaine",
            "Prédiction d'analogie 2 : Correspondance testable",
        ]
        h.limitations = ["Le raisonnement analogique peut ne pas être entièrement transférable entre les domaines"]
        return h
    
    async def _generate_unconventional(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Generate unconventional/out-of-box ideas"""
        h = Hypothesis(
            title=f"Hypothesis {index+1} (Unconventional): Divergent Mechanism in {goal.domain}",
            description=f"Explores unconventional directions in {goal.domain} research",
            mechanism="Moving away from established paradigms to explore underexplored mechanistic space",
            generation_method="simulated-unconventional"
        )
        h.testable_predictions = [
            "High-risk high-reward prediction 1",
            "Divergent experimental approach"
        ]
        h.limitations = [
            "Deviates from mainstream thinking",
            "May require novel experimental techniques"
        ]
        return h


class ReflectionAgent:
    """Reviews hypotheses for correctness, quality, novelty, testability"""
    
    def __init__(self, use_local_llm: bool = True):
        self.name = "Reflection"
        self.reviews_completed = 0
        self.llm_client = None
        
        if use_local_llm and openai:
            try:
                self.llm_client = openai.OpenAI(
                    base_url=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1"),
                    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
                )
                print("✓ Reflection Agent initialized with local LLM connection.")
            except Exception as e:
                print(f"⚠ Reflection Agent could not connect to local LLM: {e}")
                self.llm_client = None

    async def review_hypothesis(self, 
                                hypothesis: Hypothesis,
                                goal: ResearchGoal) -> ReviewCritique:
        """
        Comprehensive hypothesis review.
        Uses LLM if available, otherwise falls back to simulation.
        """
        
        if self.llm_client:
            try:
                review = await self._review_with_llm(hypothesis, goal)
                if review:
                    hypothesis.reviews.append(review)
                    self._update_novelty_level(hypothesis, review.novelty_score)
                    self.reviews_completed += 1
                    return review
            except Exception as e:
                print(f"⚠ LLM review failed: {e}. Falling back to simulation.")

        # Fallback to simulated review
        return await self._review_simulated(hypothesis, goal)

    async def _review_with_llm(self, hypothesis: Hypothesis, goal: ResearchGoal) -> ReviewCritique:
        """Perform review using local LLM"""
        prompt = self._build_review_prompt(hypothesis, goal)
        
        model_name = os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b")
        
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback to text mode if JSON format not supported
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
        
        data = _parse_json_response(response.choices[0].message.content)
        
        return ReviewCritique(
            review_type="llm-full",
            correctness_score=float(data.get("correctness_score", 0.5)),
            novelty_score=float(data.get("novelty_score", 0.5)),
            testability_score=float(data.get("testability_score", 0.5)),
            quality_score=float(data.get("quality_score", 0.5)),
            feedback=data.get("feedback", "No feedback provided.")
        )

    def _build_review_prompt(self, hypothesis: Hypothesis, goal: ResearchGoal) -> str:
        return f"""
You are a senior scientific reviewer. Your task is to critically evaluate the following research hypothesis.

**Research Goal:** {goal.title} ({goal.domain})
**Hypothesis:** {hypothesis.title}
**Description:** {hypothesis.description}
**Mechanism:** {hypothesis.mechanism}
**Predictions:** {', '.join(hypothesis.testable_predictions)}

Evaluate this hypothesis on the following criteria and return a JSON object:
1. **correctness_score** (0.0-1.0): Is it scientifically sound and logically consistent?
2. **novelty_score** (0.0-1.0): Is it new compared to established knowledge?
3. **testability_score** (0.0-1.0): Can it be validated experimentally?
4. **quality_score** (0.0-1.0): Overall assessment.
5. **feedback**: A concise paragraph of constructive criticism.

**JSON Format:**
{{
  "correctness_score": 0.8,
  "novelty_score": 0.7,
  "testability_score": 0.9,
  "quality_score": 0.8,
  "feedback": "The hypothesis is..."
}}
"""

    def _update_novelty_level(self, hypothesis: Hypothesis, score: float):
        if score > 0.75:
            hypothesis.novelty_level = "very_high"
        elif score > 0.55:
            hypothesis.novelty_level = "high"
        elif score > 0.35:
            hypothesis.novelty_level = "medium"
        else:
            hypothesis.novelty_level = "low"

    async def _review_simulated(self, hypothesis: Hypothesis, goal: ResearchGoal) -> ReviewCritique:
        """Simulated review logic (fallback)"""
        
        # Assess multiple dimensions
        correctness = await self._assess_correctness(hypothesis, goal)
        novelty = await self._assess_novelty(hypothesis, goal)
        testability = await self._assess_testability(hypothesis, goal)
        # Use pre-computed scores to avoid double computation
        quality = self._compute_quality_score(correctness, novelty, testability)
        
        self._update_novelty_level(hypothesis, novelty)
        
        feedback = self._generate_review_feedback(
            correctness, novelty, testability, quality, hypothesis
        )
        
        review = ReviewCritique(
            review_type="simulated-full",
            correctness_score=correctness,
            novelty_score=novelty,
            testability_score=testability,
            quality_score=quality,
            feedback=feedback
        )
        
        hypothesis.reviews.append(review)
        self.reviews_completed += 1
        
        return review
    
    async def _assess_correctness(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        """Assess logical consistency and factual grounding (0-1)"""
        score = 0.7
        
        # Bonus if has grounding evidence
        if hypothesis.grounding_evidence:
            score += 0.15
        
        # Reduce if has limitations
        if hypothesis.limitations:
            score -= len(hypothesis.limitations) * 0.05
        
        return min(1.0, max(0.0, score))
    
    async def _assess_novelty(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        """Assess novelty vs existing literature (0-1)"""
        score = 0.6
        
        # Generation method affects novelty scoring
        if "simulated" in hypothesis.generation_method:
            score = 0.55
        elif "llm" in hypothesis.generation_method:
            score = 0.75 # LLM-generated content is expected to be more novel
        elif hypothesis.generation_method == "evolved":
            score = 0.65
        elif hypothesis.generation_method == "combined":
            score = 0.70
        elif hypothesis.generation_method == "inspired":
            score = 0.60
        
        # Penalize if explicitly states it's similar to existing work
        if "similar to" in hypothesis.description.lower():
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _assess_testability(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        """Assess feasibility of experimental validation (0-1)"""
        score = 0.65
        
        # More predictions = more testable
        if len(hypothesis.testable_predictions) >= 3:
            score += 0.2
        elif len(hypothesis.testable_predictions) >= 1:
            score += 0.1
        
        # Clear mechanism helps testability
        if len(hypothesis.mechanism) > 50:
            score += 0.1
        
        # Limitations reduce testability scores
        if "requires novel techniques" in str(hypothesis.limitations):
            score -= 0.15
        
        return min(1.0, max(0.0, score))
    
    def _compute_quality_score(self, correctness: float, novelty: float, testability: float) -> float:
        """Compute quality from pre-computed scores (avoids double computation)"""
        quality = (correctness * 0.4 + novelty * 0.3 + testability * 0.3)
        return min(1.0, max(0.0, quality))

    async def _assess_quality(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        """Overall quality score (0-1)"""
        correctness = await self._assess_correctness(hypothesis, goal)
        novelty = await self._assess_novelty(hypothesis, goal)
        testability = await self._assess_testability(hypothesis, goal)
        return self._compute_quality_score(correctness, novelty, testability)
    
    def _generate_review_feedback(self, correctness: float, novelty: float, 
                                 testability: float, quality: float,
                                 hypothesis: Hypothesis) -> str:
        """Generate natural language feedback"""
        feedback_parts = []
        
        if correctness > 0.8:
            feedback_parts.append("✓ Logically sound and well-grounded")
        elif correctness < 0.5:
            feedback_parts.append("⚠ Logical consistency concerns identified")
        
        if novelty > 0.7:
            feedback_parts.append("✓ Proposes genuinely novel elements")
        elif novelty < 0.4:
            feedback_parts.append("⚠ Limited novelty over existing literature")
        
        if testability > 0.7:
            feedback_parts.append("✓ Clear testable predictions")
        elif testability < 0.5:
            feedback_parts.append("⚠ May require refinement for experimental validation")
        
        if quality > 0.75:
            feedback_parts.append("✓ High quality overall")
        
        return " | ".join(feedback_parts) if feedback_parts else "Further review recommended"


class RankingAgent:
    """Tournament-based hypothesis ranking using Elo system"""
    
    def __init__(self):
        self.name = "Ranking"
        self.k_factor = 32  # Elo K-factor
        self.matches_completed = 0
    
    async def conduct_tournament_match(self,
                                      hyp_a: Hypothesis,
                                      hyp_b: Hypothesis) -> Tuple[str, TournamentMatch]:
        """
        Conduct pairwise hypothesis comparison through simulated scientific debate.
        Returns winner ID and match record.
        """
        
        # Simulate debate and determine winner
        winner_id = await self._simulate_debate(hyp_a, hyp_b)
        
        debate_summary = self._generate_debate_summary(hyp_a, hyp_b, winner_id)
        
        # Update Elo ratings
        self._update_elo_ratings(hyp_a, hyp_b, winner_id)
        
        # Record match
        match = TournamentMatch(
            hypothesis_a_id=hyp_a.id,
            hypothesis_b_id=hyp_b.id,
            winner_id=winner_id,
            debate_summary=debate_summary
        )
        
        self.matches_completed += 1
        return winner_id, match
    
    async def _simulate_debate(self, hyp_a: Hypothesis, hyp_b: Hypothesis) -> str:
        """
        Simulate scientific debate between two hypotheses.
        Returns ID of winning hypothesis.
        """
        
        # Score both based on available metrics
        score_a = self._compute_debate_score(hyp_a)
        score_b = self._compute_debate_score(hyp_b)
        
        # Add randomness to simulate debate variability
        import random
        debate_factor_a = random.uniform(0.8, 1.2)
        debate_factor_b = random.uniform(0.8, 1.2)
        
        final_score_a = score_a * debate_factor_a
        final_score_b = score_b * debate_factor_b
        
        return hyp_a.id if final_score_a > final_score_b else hyp_b.id
    
    def _compute_debate_score(self, hypothesis: Hypothesis) -> float:
        """
        Compute debate score from multiple factors:
        - Review scores (if available)
        - Novelty level
        - Number of testable predictions
        """
        
        score = 0.5  # Baseline
        
        # Include review scores if available
        if hypothesis.reviews:
            avg_review_score = sum(
                r.novelty_score * 0.3 + r.testability_score * 0.3 + r.correctness_score * 0.2 + r.quality_score * 0.2
                for r in hypothesis.reviews
            ) / len(hypothesis.reviews)
            score = avg_review_score
        
        # Bonus for multiple testable predictions
        score += len(hypothesis.testable_predictions) * 0.05
        
        # Novelty bonus
        novelty_bonus = {
            "very_high": 0.15,
            "high": 0.10,
            "medium": 0.05,
            "low": 0.00,
            "unknown": 0.02
        }
        score += novelty_bonus.get(hypothesis.novelty_level, 0.02)
        
        return min(1.0, max(0.0, score))
    
    def _update_elo_ratings(self, hyp_a: Hypothesis, hyp_b: Hypothesis, winner_id: str):
        """Update Elo ratings based on match outcome"""
        
        # Standard Elo formula
        expected_a = 1 / (1 + 10 ** ((hyp_b.elo_rating - hyp_a.elo_rating) / 400))
        expected_b = 1 - expected_a
        
        if winner_id == hyp_a.id:
            hyp_a.elo_rating += self.k_factor * (1 - expected_a)
            hyp_b.elo_rating += self.k_factor * (0 - expected_b)
        else:
            hyp_a.elo_rating += self.k_factor * (0 - expected_a)
            hyp_b.elo_rating += self.k_factor * (1 - expected_b)
    
    def _generate_debate_summary(self, hyp_a: Hypothesis, hyp_b: Hypothesis, winner_id: str) -> str:
        """Generate summary of debate reasoning"""
        winner = hyp_a if winner_id == hyp_a.id else hyp_b
        loser = hyp_b if winner_id == hyp_a.id else hyp_a
        
        summary = f"Debate winner: {winner.title[:50]}... "
        summary += f"(Elo: {winner.elo_rating:.0f}) defeated "
        summary += f"{loser.title[:50]}... (Elo: {loser.elo_rating:.0f}). "
        
        # Add debate reasoning
        if winner.novelty_level == "very_high":
            summary += "Higher novelty was decisive factor. "
        if len(winner.testable_predictions) > len(loser.testable_predictions):
            summary += "More testable predictions provided advantage. "
        
        return summary


class ProximityAgent:
    """Builds proximity graph for hypothesis clustering"""
    
    def __init__(self):
        self.name = "Proximity"
        self.proximity_graph = defaultdict(lambda: defaultdict(float))
    
    async def compute_proximity(self, hypotheses: List[Hypothesis]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute similarity between hypotheses.
        Returns dict mapping hypothesis IDs to list of (similar_id, similarity_score)
        """
        
        proximity_map = {}
        
        for i, hyp_a in enumerate(hypotheses):
            similarities = []
            
            for j, hyp_b in enumerate(hypotheses):
                if i != j:
                    similarity = await self._compute_similarity(hyp_a, hyp_b)
                    similarities.append((hyp_b.id, similarity))
                    self.proximity_graph[hyp_a.id][hyp_b.id] = similarity
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            proximity_map[hyp_a.id] = similarities
        
        return proximity_map
    
    async def _compute_similarity(self, hyp_a: Hypothesis, hyp_b: Hypothesis) -> float:
        """
        Compute similarity score between two hypotheses (0-1).
        Based on: mechanism similarity, overlapping testable predictions, shared citations.
        """
        
        similarity = 0.0
        
        # Text similarity (simple approach: shared words in mechanism)
        mech_a_words = set(hyp_a.mechanism.lower().split())
        mech_b_words = set(hyp_b.mechanism.lower().split())
        
        if mech_a_words and mech_b_words:
            shared = len(mech_a_words & mech_b_words)
            total = len(mech_a_words | mech_b_words)
            mechanism_sim = shared / total if total > 0 else 0.0
            similarity += mechanism_sim * 0.5
        
        # Prediction overlap
        pred_a_set = set(hyp_a.testable_predictions)
        pred_b_set = set(hyp_b.testable_predictions)
        
        if pred_a_set and pred_b_set:
            shared_preds = len(pred_a_set & pred_b_set)
            max_preds = max(len(pred_a_set), len(pred_b_set))
            pred_sim = shared_preds / max_preds if max_preds > 0 else 0.0
            similarity += pred_sim * 0.3
        
        # Citation overlap
        cite_a_set = set(hyp_a.cited_papers)
        cite_b_set = set(hyp_b.cited_papers)
        
        if cite_a_set and cite_b_set:
            shared_cites = len(cite_a_set & cite_b_set)
            max_cites = max(len(cite_a_set), len(cite_b_set))
            cite_sim = shared_cites / max_cites if max_cites > 0 else 0.0
            similarity += cite_sim * 0.2
        
        return min(1.0, max(0.0, similarity))


class EvolutionAgent:
    """Refines and improves hypotheses through multiple strategies"""
    
    def __init__(self, use_local_llm: bool = True):
        self.name = "Evolution"
        self.evolved_hypotheses = 0
        self.llm_client = None
        
        if use_local_llm and openai:
            try:
                self.llm_client = openai.OpenAI(
                    base_url=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1"),
                    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
                )
            except Exception:
                self.llm_client = None
    
    async def evolve_hypothesis(self, 
                               hypothesis: Hypothesis,
                               strategy: str = "enhancement") -> Hypothesis:
        """
        Improve hypothesis using specified strategy:
        - enhancement: ground in literature
        - simplification: make clearer and more concise
        - combination: combine with other top hypotheses
        - inspiration: derive from top hypotheses
        """
        
        new_hyp = Hypothesis(
            title=hypothesis.title + f" (Evolved: {strategy})",
            description=hypothesis.description,
            mechanism=hypothesis.mechanism,
            parent_ids=[hypothesis.id],
            generation_method="evolved"
        )
        
        if strategy == "enhancement":
            new_hyp = await self._enhance_with_grounding(new_hyp, hypothesis)
        elif strategy == "simplification":
            new_hyp = await self._simplify(new_hyp, hypothesis)
        elif strategy == "out_of_box":
            new_hyp = await self._divergent_thinking(new_hyp, hypothesis)
        
        # Try LLM-based refinement if available
        if self.llm_client:
            new_hyp = await self._llm_refine_evolution(new_hyp, hypothesis, strategy)
        
        self.evolved_hypotheses += 1
        return new_hyp
    
    async def _enhance_with_grounding(self, new_hyp: Hypothesis, 
                                     original: Hypothesis) -> Hypothesis:
        """Enhance hypothesis by identifying gaps and adding evidence"""
        
        new_hyp.mechanism = (
            f"Enhanced mechanism: {original.mechanism} "
            f"Additionally grounded by identifying supporting molecular pathways "
            f"and experimental evidence from recent literature."
        )
        
        new_hyp.grounding_evidence = original.grounding_evidence + [
            "Additional pathway analysis",
            "Cross-validation against recent meta-analyses"
        ]
        
        new_hyp.testable_predictions = original.testable_predictions + [
            "Advanced prediction: Multi-dimensional experimental validation",
        ]
        
        return new_hyp
    
    async def _simplify(self, new_hyp: Hypothesis, 
                       original: Hypothesis) -> Hypothesis:
        """Simplify hypothesis for easier verification"""
        
        new_hyp.title = f"Simplified: {original.title}"
        new_hyp.mechanism = (
            "Core simplified mechanism: "
            + original.mechanism.split('.')[0] + ". "
            + "Reduces complexity by focusing on primary pathway."
        )
        
        new_hyp.testable_predictions = original.testable_predictions[:2]
        new_hyp.limitations = original.limitations + [
            "Simplified version may miss secondary effects"
        ]
        
        return new_hyp
    
    async def _divergent_thinking(self, new_hyp: Hypothesis,
                                 original: Hypothesis) -> Hypothesis:
        """Generate divergent ideas inspired by but different from original"""
        
        new_hyp.title = f"Divergent: {original.title}"
        new_hyp.description = (
            f"Exploring alternative mechanistic pathway divergent from: {original.title}. "
            f"Considers underexplored or unconventional mechanisms."
        )
        
        new_hyp.mechanism = (
            "Alternative mechanism: Rather than the primary hypothesis, "
            "this explores secondary or tertiary pathways that may contribute "
            "to the observed phenomenon."
        )
        
        return new_hyp

    async def _llm_refine_evolution(self, new_hyp: Hypothesis, original: Hypothesis, strategy: str) -> Hypothesis:
        """Use LLM to refine evolved hypothesis"""
        prompt = f"""You are a scientific research assistant. Improve the following hypothesis using the "{strategy}" strategy.

Original Hypothesis:
- Title: {original.title}
- Mechanism: {original.mechanism}
- Description: {original.description}

Current Evolution:
- Title: {new_hyp.title}
- Mechanism: {new_hyp.mechanism}

Provide an improved version as a JSON object with keys: "title", "description", "mechanism", "testable_predictions" (list), "limitations" (list).
Output ONLY the JSON object.
"""
        try:
            try:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                )
            
            data = _parse_json_response(response.choices[0].message.content)
            new_hyp.title = data.get("title", new_hyp.title)
            new_hyp.description = data.get("description", new_hyp.description)
            new_hyp.mechanism = data.get("mechanism", new_hyp.mechanism)
            new_hyp.testable_predictions = data.get("testable_predictions", new_hyp.testable_predictions)
            new_hyp.limitations = data.get("limitations", new_hyp.limitations)
            new_hyp.generation_method = "evolved-llm"
        except Exception as e:
            print(f"⚠ LLM evolution refinement failed: {e}")
        
        return new_hyp


class MetaReviewAgent:
    """Synthesizes insights and provides system-level feedback"""
    
    def __init__(self):
        self.name = "Meta-Review"
        self.meta_reviews_generated = 0
    
    async def generate_meta_review(self, 
                                  hypotheses: List[Hypothesis],
                                  tournament_history: List[TournamentMatch],
                                  goal: ResearchGoal) -> Dict[str, Any]:
        """
        Synthesize insights from reviews and tournaments.
        Identify recurring patterns and improvement opportunities.
        """
        
        meta_review = {
            "timestamp": datetime.now().isoformat(),
            "total_hypotheses": len(hypotheses),
            "top_hypotheses": self._identify_top_hypotheses(hypotheses),
            "common_strengths": self._identify_common_strengths(hypotheses),
            "common_weaknesses": self._identify_common_weaknesses(hypotheses),
            "tournament_patterns": self._analyze_tournament_patterns(tournament_history),
            "suggested_improvements": self._suggest_improvements(hypotheses),
            "research_overview": self._generate_research_overview(hypotheses, goal),
            "next_iterations_focus": self._suggest_focus_areas(hypotheses)
        }
        
        self.meta_reviews_generated += 1
        return meta_review
    
    def _identify_top_hypotheses(self, hypotheses: List[Hypothesis], 
                                top_k: int = 5) -> List[Dict]:
        """Identify top-ranked hypotheses"""
        sorted_hyps = sorted(
            hypotheses, 
            key=lambda h: h.elo_rating, 
            reverse=True
        )[:top_k]
        
        return [
            {
                "id": h.id,
                "title": h.title,
                "elo_rating": h.elo_rating,
                "novelty": h.novelty_level,
                "num_reviews": len(h.reviews)
            }
            for h in sorted_hyps
        ]
    
    def _identify_common_strengths(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Extract patterns in successful hypotheses"""
        strengths = []
        
        # Hypotheses with high Elo often have multiple reviews
        top_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:5]
        
        if all(len(h.reviews) > 0 for h in top_hyps):
            strengths.append("Multiple review iterations improve hypothesis quality")
        
        if all(h.novelty_level in ["high", "very_high"] for h in top_hyps):
            strengths.append("Novelty is a strong factor in ranking")
        
        if all(len(h.testable_predictions) >= 2 for h in top_hyps):
            strengths.append("Multiple testable predictions increase competitiveness")
        
        return strengths if strengths else ["Diverse hypothesis portfolio maintained"]
    
    def _identify_common_weaknesses(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Extract patterns in lower-ranked hypotheses"""
        weaknesses = []
        
        bottom_hyps = sorted(hypotheses, key=lambda h: h.elo_rating)[:5]
        
        if any(len(h.reviews) == 0 for h in bottom_hyps):
            weaknesses.append("Unreviewed hypotheses tend to rank lower - prioritize review")
        
        if any(len(h.testable_predictions) == 0 for h in bottom_hyps):
            weaknesses.append("Lack of testable predictions is a weakness - add empirical angles")
        
        if any(h.novelty_level == "low" for h in bottom_hyps):
            weaknesses.append("Low novelty is penalized - encourage more creative generation")
        
        return weaknesses if weaknesses else ["No clear common weaknesses"]
    
    def _analyze_tournament_patterns(self, tournament_history: List[TournamentMatch]) -> Dict:
        """Analyze tournament results for patterns"""
        if not tournament_history:
            return {"total_matches": 0, "analysis": "No tournaments completed yet"}
        
        win_counts = defaultdict(int)
        for match in tournament_history:
            win_counts[match.winner_id] += 1
        
        return {
            "total_matches": len(tournament_history),
            "top_winner": max(win_counts, key=win_counts.get) if win_counts else None,
            "wins_distribution": dict(sorted(win_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _suggest_improvements(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Suggest improvements for next iteration"""
        suggestions = []
        
        avg_novelty_score = sum(
            sum(r.novelty_score for r in h.reviews) / len(h.reviews)
            for h in hypotheses if h.reviews
        ) / max(len([h for h in hypotheses if h.reviews]), 1)
        
        if avg_novelty_score < 0.6:
            suggestions.append("Enhance novelty generation - explore more unconventional directions")
        
        unreviewed = [h for h in hypotheses if len(h.reviews) == 0]
        if unreviewed:
            suggestions.append(f"Review {len(unreviewed)} unreviewed hypotheses")
        
        low_testability = [h for h in hypotheses if len(h.testable_predictions) < 2]
        if low_testability:
            suggestions.append(f"Add testable predictions to {len(low_testability)} hypotheses")
        
        return suggestions if suggestions else ["Continue current trajectory - good progress"]
    
    def _generate_research_overview(self, hypotheses: List[Hypothesis], 
                                   goal: ResearchGoal) -> str:
        """Generate high-level research overview"""
        top_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:3]
        
        overview = f"Research Overview for: {goal.title}\n"
        overview += f"Domain: {goal.domain}\n\n"
        overview += "Top Research Directions:\n"
        
        for i, hyp in enumerate(top_hyps, 1):
            overview += f"\n{i}. {hyp.title}\n"
            overview += f"   Mechanism: {hyp.mechanism[:100]}...\n"
            overview += f"   Elo Rating: {hyp.elo_rating:.0f}\n"
        
        return overview
    
    def _suggest_focus_areas(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Suggest areas for next iteration focus"""
        focus_areas = []
        
        # Focus on evolving high-Elo hypotheses
        top_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:3]
        focus_areas.append(f"Evolve top 3 hypotheses: {', '.join(h.title[:30] for h in top_hyps)}")
        
        # Focus on reviewing unreviewed
        unreviewed = [h for h in hypotheses if len(h.reviews) == 0]
        if unreviewed:
            focus_areas.append(f"Complete reviews for {len(unreviewed)} hypotheses")
        
        # Focus on tournament matches
        focus_areas.append("Conduct tournament matches among top performers")
        
        return focus_areas


# ============================================================================
# SUPERVISOR & TASK FRAMEWORK
# ============================================================================

class Task:
    """Represents an async task in the worker queue"""
    
    def __init__(self, agent_name: str, action: str, params: Dict, priority: int = 5):
        self.id = str(uuid.uuid4())[:8]
        self.agent_name = agent_name
        self.action = action
        self.params = params
        self.priority = priority  # Lower number = higher priority
        self.created_at = datetime.now()
        self.completed_at = None
        self.result = None
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority


class SupervisorAgent:
    """Orchestrates all specialized agents and manages task queue"""
    
    def __init__(self):
        self.name = "Supervisor"
        self.task_queue = []
        self.task_history = []
        self.agent_registry = {}
        self.iteration = 0
    
    def register_agent(self, agent):
        """Register a specialized agent"""
        self.agent_registry[agent.name] = agent
    
    async def execute_task_queue(self, max_iterations: int = 3):
        """Execute queued tasks"""
        for _ in range(max_iterations):
            if not self.task_queue:
                break
            
            # Get highest priority task
            task = heapq.heappop(self.task_queue)
            agent = self.agent_registry.get(task.agent_name)
            
            if agent:
                # Execute task
                if task.action == "generate":
                    task.result = await agent.generate_initial_hypotheses(**task.params)
                elif task.action == "review":
                    task.result = await agent.review_hypothesis(**task.params)
                elif task.action == "tournament":
                    task.result = await agent.conduct_tournament_match(**task.params)
                elif task.action == "compute_proximity":
                    task.result = await agent.compute_proximity(**task.params)
                elif task.action == "evolve":
                    task.result = await agent.evolve_hypothesis(**task.params)
                elif task.action == "meta_review":
                    task.result = await agent.generate_meta_review(**task.params)
                elif task.action == "search_literature":
                    task.result = await agent.search_literature(**task.params)
                
                task.completed_at = datetime.now()
                self.task_history.append(task)
            
            self.iteration += 1
    
    def queue_task(self, agent_name: str, action: str, params: Dict, priority: int = 5):
        """Add task to queue"""
        task = Task(agent_name, action, params, priority)
        heapq.heappush(self.task_queue, task)
        return task.id
    
    def get_task_stats(self) -> Dict:
        """Get statistics on task execution"""
        return {
            "total_tasks_completed": len(self.task_history),
            "pending_tasks": len(self.task_queue),
            "iterations_completed": self.iteration,
            "agents_registered": list(self.agent_registry.keys())
        }


# ============================================================================
# MAIN CO-SCIENTIST SYSTEM
# ============================================================================

class CoScientist:
    """Main AI co-scientist system coordinator"""
    
    def __init__(self, use_local_llm: bool = True, enable_rag: bool = True):
        self.context_memory = ContextMemory()
        self.supervisor = SupervisorAgent()
        
        # Initialize all agents
        self.generation_agent = GenerationAgent(use_local_llm=use_local_llm)
        self.reflection_agent = ReflectionAgent(use_local_llm=use_local_llm)
        self.ranking_agent = RankingAgent()
        self.proximity_agent = ProximityAgent()
        self.evolution_agent = EvolutionAgent(use_local_llm=use_local_llm)
        self.meta_review_agent = MetaReviewAgent()
        self.literature_agent = LiteratureAgent(use_local_llm=use_local_llm, enable_rag=enable_rag)
        self.graph_agent = GraphAgent(use_local_llm=use_local_llm)
        
        # Register agents with supervisor
        self.supervisor.register_agent(self.generation_agent)
        self.supervisor.register_agent(self.reflection_agent)
        self.supervisor.register_agent(self.ranking_agent)
        self.supervisor.register_agent(self.proximity_agent)
        self.supervisor.register_agent(self.evolution_agent)
        self.supervisor.register_agent(self.meta_review_agent)
        self.supervisor.register_agent(self.literature_agent)
        self.supervisor.register_agent(self.graph_agent)
    
    async def initialize_research_goal(self, 
                                       title: str,
                                       description: str,
                                       domain: str,
                                       preferences: Dict = None,
                                       constraints: List[str] = None) -> ResearchGoal:
        """Initialize research goal from scientist input"""
        
        goal = ResearchGoal(
            title=title,
            description=description,
            domain=domain,
            preferences=preferences or {},
            constraints=constraints or []
        )
        
        self.context_memory.research_goal = goal
        print(f"\n📋 Research Goal Initialized:")
        print(f"   Title: {goal.title}")
        print(f"   Domain: {goal.domain}")
        print(f"   Description: {goal.description[:100]}...")
        
        return goal

    async def analyze_research_description(self, description: str) -> Dict[str, Any]:
        """
        Analyze research description to suggest domain and databases.
        Returns a dict with 'domains' (List[str]) and 'databases' (List[str]).
        """
        if not self.generation_agent.llm_client:
            return {
                "domains": ["Biomedicine", "Computer Science", "Physics"],
                "databases": ["arxiv", "pubmed"]
            }
            
        prompt = f"""
        Analyze the following research description and identify:
        1. The most relevant scientific domains (be broad but accurate, e.g., 'Biomedicine/Oncology', 'Computer Science/AI').
        2. The most relevant scientific databases for literature search.
        
        Description: "{description}"
        
        Provide the output in JSON format with two keys: 'domains' (list of strings) and 'databases' (list of strings).
        Supported databases to choose from if relevant: ['arxiv', 'pubmed', 'biorxiv', 'ieee_xplore', 'scopus', 'google_scholar', 'semantic_scholar'].
        """
        
        try:
            response = await asyncio.to_thread(
                self.generation_agent.llm_client.chat.completions.create,
                model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-20b"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                # response_format={"type": "json_object"}, # Removed for better local model compatibility
            )
            content = response.choices[0].message.content
            # Clean up content: remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            else:
                content = content.strip()
                
            return json.loads(content)
        except Exception as e:
            print(f"⚠ Domain analysis failed: {e}")
            # Fallback to something more generic and useful
            return {
                "domains": ["Science", "Research"],
                "databases": ["arxiv", "pubmed"]
            }
    
    async def run_literature_search(self, max_results: int = 5, sources: List[str] = None, iterations: int = 2) -> List[Dict]:
        """Run literature search to populate context"""
        if sources is None:
            sources = ["arxiv"]
        print(f"\n📚 Running literature search on {sources} (Max {iterations} iterations)...")
        
        papers = await self.literature_agent.search_literature(
            self.context_memory.research_goal,
            max_results=max_results,
            sources=sources,
            iterations=iterations
        )
        
        self.context_memory.literature_context = papers
        
        # CAG: Generate Key Findings for Global Context
        cag_context = await self.literature_agent.extract_key_findings(papers, self.context_memory.research_goal)
        
        # GraphRAG: Build Knowledge Graph
        graph_insights = await self.graph_agent.build_graph(papers, self.context_memory.research_goal)
        cag_context += "\n\n" + graph_insights
        
        print(f"🧠 CAG + Graph Context Generated: {len(cag_context)} chars")
        
        # Inject into Generation Agent
        self.generation_agent.cag_context = cag_context
        
        # Process papers with RAG if available
        if self.literature_agent.rag_engine and papers:
            print(f"\n🧠 Processing papers with RAG system...")
            chunks_indexed = await self.literature_agent.process_papers_with_rag(papers)
            if chunks_indexed > 0:
                print(f"✓ RAG system ready with {chunks_indexed} indexed chunks")
        
        return papers

    async def run_hypothesis_generation_cycle(self, num_hypotheses: int = 5) -> List[Hypothesis]:
        """Generate initial hypotheses"""
        print(f"\n🔬 Generating {num_hypotheses} initial hypotheses...")
        
        # Query RAG for relevant context if available
        rag_context = None
        if self.literature_agent.rag_engine:
            goal_query = f"{self.context_memory.research_goal.title} {self.context_memory.research_goal.description}"
            rag_context = await self.literature_agent.query_rag(goal_query, top_k=5)
            if rag_context:
                print(f"  ✓ Retrieved {len(rag_context)} relevant passages from papers")
        
        hypotheses = await self.generation_agent.generate_initial_hypotheses(
            self.context_memory.research_goal,
            context_papers=self.context_memory.literature_context,
            rag_context=rag_context,
            count=num_hypotheses
        )
        
        for h in hypotheses:
            self.context_memory.hypotheses[h.id] = h
        
        print(f"✓ Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    async def run_review_cycle(self) -> List[ReviewCritique]:
        """Review all unreviewed hypotheses"""
        print(f"\n📝 Conducting hypothesis reviews...")
        
        unreviewed = [
            h for h in self.context_memory.hypotheses.values()
            if len(h.reviews) == 0
        ]
        
        reviews = []
        for hypothesis in unreviewed:
            review = await self.reflection_agent.review_hypothesis(
                hypothesis,
                self.context_memory.research_goal
            )
            reviews.append(review)
            hypothesis.status = HypothesisStatus.REVIEWED
        
        print(f"✓ Completed {len(reviews)} reviews")
        return reviews
    
    async def run_tournament_cycle(self, num_matches: int = 5) -> List[TournamentMatch]:
        """Conduct tournament matches"""
        print(f"\n🏆 Running tournament matches...")
        
        hypotheses_list = list(self.context_memory.hypotheses.values())
        if len(hypotheses_list) < 2:
            print("  ⚠ Need at least 2 hypotheses for tournament")
            return []
        
        matches = []
        matches_conducted = 0
        
        # Prioritize comparisons between reviewed hypotheses
        reviewed = [h for h in hypotheses_list if len(h.reviews) > 0]
        
        for i in range(min(num_matches, len(reviewed) * 2)):
            # Select two different hypotheses
            import random
            if len(reviewed) >= 2:
                hyp_a = random.choice(reviewed)
                hyp_b = random.choice([h for h in reviewed if h.id != hyp_a.id])
            elif len(hypotheses_list) >= 2:
                hyp_a = random.choice(hypotheses_list)
                hyp_b = random.choice([h for h in hypotheses_list if h.id != hyp_a.id])
            else:
                break
            
            winner_id, match = await self.ranking_agent.conduct_tournament_match(hyp_a, hyp_b)
            matches.append(match)
            self.context_memory.tournament_history.append(match)
            matches_conducted += 1
        
        print(f"✓ Completed {matches_conducted} tournament matches")
        return matches
    
    async def run_evolution_cycle(self) -> List[Hypothesis]:
        """Evolve top hypotheses"""
        print(f"\n🧬 Evolving hypotheses...")
        
        # Get top hypotheses
        top_hyps = sorted(
            self.context_memory.hypotheses.values(),
            key=lambda h: h.elo_rating,
            reverse=True
        )[:3]
        
        strategies = ["enhancement", "simplification", "out_of_box"]
        evolved = []
        
        for hyp, strategy in zip(top_hyps, strategies):
            new_hyp = await self.evolution_agent.evolve_hypothesis(hyp, strategy)
            self.context_memory.hypotheses[new_hyp.id] = new_hyp
            evolved.append(new_hyp)
        
        print(f"✓ Evolved {len(evolved)} hypotheses")
        return evolved
    
    async def run_meta_review_cycle(self) -> Dict[str, Any]:
        """Generate meta-review and research overview"""
        print(f"\n🎯 Generating meta-review...")
        
        meta_review = await self.meta_review_agent.generate_meta_review(
            list(self.context_memory.hypotheses.values()),
            self.context_memory.tournament_history,
            self.context_memory.research_goal
        )
        
        # Store in context memory for persistence
        self.context_memory.meta_reviews.append(meta_review)
        
        print(f"✓ Meta-review generated")
        return meta_review
    
    async def run_full_cycle(self, num_iterations: int = 3):
        """Run complete co-scientist workflow"""
        print("\n" + "="*70)
        print("🤖 NewAI Scientist WORKFLOW STARTED")
        print("="*70)
        
        # Literature Search
        await self.run_literature_search()
        
        # Initial hypothesis generation
        await self.run_hypothesis_generation_cycle(num_hypotheses=5)
        
        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")
            
            # Review cycle
            await self.run_review_cycle()
            
            # Proximity analysis
            print(f"\n🔗 Computing hypothesis proximity...")
            proximity = await self.proximity_agent.compute_proximity(
                list(self.context_memory.hypotheses.values())
            )
            
            # Tournament cycle
            await self.run_tournament_cycle(num_matches=4)
            
            # Evolution cycle
            await self.run_evolution_cycle()
            
            # Meta-review
            meta_review = await self.run_meta_review_cycle()
            
            # Print status
            self._print_iteration_status(iteration + 1, meta_review)
        
        # Final summary
        await self._print_final_summary()
    
    def _print_iteration_status(self, iteration: int, meta_review: Dict):
        """Print status of current iteration"""
        print(f"\n📊 Iteration {iteration} Summary:")
        print(f"  Total hypotheses: {meta_review['total_hypotheses']}")
        print(f"\n  Top hypotheses:")
        for hyp_info in meta_review['top_hypotheses'][:3]:
            print(f"    • {hyp_info['title'][:50]}...")
            print(f"      Elo: {hyp_info['elo_rating']:.0f} | Novelty: {hyp_info['novelty']}")
        
        if meta_review['suggested_improvements']:
            print(f"\n  Suggested improvements:")
            for suggestion in meta_review['suggested_improvements'][:2]:
                print(f"    • {suggestion}")
        
        if meta_review['next_iterations_focus']:
            print(f"\n  Next focus areas:")
            for focus in meta_review['next_iterations_focus'][:2]:
                print(f"    • {focus}")
    
    async def _print_final_summary(self):
        """Print final system summary"""
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        print(f"\n📈 System Statistics:")
        print(f"  Total hypotheses generated: {len(self.context_memory.hypotheses)}")
        print(f"  Tournament matches: {len(self.context_memory.tournament_history)}")
        print(f"  Generation agent: {self.generation_agent.generated_count} hypotheses")
        print(f"  Reflection agent: {self.reflection_agent.reviews_completed} reviews")
        print(f"  Ranking agent: {self.ranking_agent.matches_completed} matches")
        print(f"  Evolution agent: {self.evolution_agent.evolved_hypotheses} evolutions")
        print(f"  Literature agent: {self.literature_agent.papers_retrieved} papers retrieved")
        
        # Top 5 final hypotheses
        top_hyps = sorted(
            self.context_memory.hypotheses.values(),
            key=lambda h: h.elo_rating,
            reverse=True
        )[:5]
        
        print(f"\n🏆 Top 5 Hypotheses (by Elo rating):")
        for i, hyp in enumerate(top_hyps, 1):
            print(f"\n{i}. {hyp.title}")
            print(f"   ID: {hyp.id}")
            print(f"   Elo Rating: {hyp.elo_rating:.0f}")
            print(f"   Novelty Level: {hyp.novelty_level}")
            print(f"   Status: {hyp.status.value}")
            print(f"   Reviews: {len(hyp.reviews)}")
            print(f"   Generation Method: {hyp.generation_method}")
            if hyp.testable_predictions:
                print(f"   Testable Predictions: {len(hyp.testable_predictions)}")
    
    def export_hypotheses_json(self, filename: str = "hypotheses.json"):
        """Export hypotheses to JSON"""
        data = {
            "research_goal": asdict(self.context_memory.research_goal),
            "literature_context": self.context_memory.literature_context,
            "hypotheses": [
                {
                    "id": h.id,
                    "title": h.title,
                    "description": h.description,
                    "mechanism": h.mechanism,
                    "elo_rating": h.elo_rating,
                    "novelty_level": h.novelty_level,
                    "status": h.status.value,
                    "testable_predictions": h.testable_predictions,
                    "grounding_evidence": h.grounding_evidence,
                    "generation_method": h.generation_method,
                    "num_reviews": len(h.reviews),
                    "cited_papers": h.cited_papers
                }
                for h in self.context_memory.hypotheses.values()
            ],
            "tournament_matches": len(self.context_memory.tournament_history),
            "statistics": {
                "generation_agent": self.generation_agent.generated_count,
                "reflection_agent": self.reflection_agent.reviews_completed,
                "ranking_agent": self.ranking_agent.matches_completed,
                "evolution_agent": self.evolution_agent.evolved_hypotheses,
                "meta_review_agent": self.meta_review_agent.meta_reviews_generated,
                "literature_agent": self.literature_agent.papers_retrieved
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Exported hypotheses to {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function"""
    
    # Initialize co-scientist
    co_scientist = CoScientist()
    
    # Define research goal (example: drug repurposing for cancer)
    await co_scientist.initialize_research_goal(
        title="Novel Drug Repurposing for Acute Myeloid Leukemia",
        description=(
            "Identify FDA-approved drugs that could be repurposed for acute myeloid leukemia (AML) "
            "treatment. Focus on drugs that can inhibit leukemic cell proliferation at clinically "
            "applicable concentrations, particularly targeting MOLM-13 cell lines."
        ),
        domain="Biomedicine/Oncology",
        preferences={
            "focus_on_novelty": True,
            "require_testability": True,
            "prioritize_clinical_relevance": True
        },
        constraints=[
            "Only consider FDA-approved drugs",
            "Must have mechanism of action documentation",
            "Focus on inhibiting AML cell proliferation"
        ]
    )
    
    # Run full workflow
    await co_scientist.run_full_cycle(num_iterations=3)
    
    # Export results
    co_scientist.export_hypotheses_json("co_scientist_results.json")
    
    print("\n" + "="*70)
    print("✨ AI Co-Scientist WORKFLOW COMPLETED")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
