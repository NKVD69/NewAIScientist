"""
agents/literature.py — LiteratureAgent for scientific literature retrieval and analysis.

Responsible for:
- Generating optimized search queries via LLM
- Searching ArXiv and PubMed
- Iterative query refinement
- RAG system processing
- CAG (Context-Augmented Generation) synthesis
- Semantic reranking of RAG chunks
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

import config
from models.hypothesis import ResearchGoal
from utils.llm import get_llm_completion, parse_json_response, ensure_str
from .base import BaseAgent

logger = logging.getLogger(__name__)

# Optional imports
try:
    import arxiv
except ImportError:
    arxiv = None

try:
    from Bio import Entrez
    Entrez.email = config.get_entrez_email()
    ncbi_key = config.get_ncbi_api_key()
    if ncbi_key:
        Entrez.api_key = ncbi_key
except ImportError:
    Entrez = None

try:
    from rag_system import RAGEngine
except ImportError:
    RAGEngine = None


class LiteratureAgent(BaseAgent):
    """Retrieves and analyzes relevant scientific literature"""

    name = "Literature"

    def __init__(self, use_local_llm: bool = True, enable_rag: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.papers_retrieved = 0
        self.rag_engine = None
        self.enable_rag = enable_rag
        self.use_local_llm = use_local_llm

        # Initialize RAG system if enabled
        if enable_rag and RAGEngine:
            try:
                self.rag_engine = RAGEngine()
                logger.info("RAG system initialized")
            except Exception as e:
                logger.warning("RAG initialization failed: %s", e)
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
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                json_mode=True
            )
            data = parse_json_response(response.choices[0].message.content)
            return data.get("queries", [goal.title])
        except Exception as e:
            logger.warning("Query generation failed: %s", e)
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
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=False
            )
            content = response.choices[0].message.content.strip().replace('"', '')
            # Ensure we don't return a huge explanation if LLM ignores instructions
            if len(content) > 150: 
                return None
            return content
        except Exception as e:
            logger.warning("Query refinement failed: %s", e)
            return None

    async def refresh(
        self,
        goal: ResearchGoal,
        existing_papers: List[Dict],
        last_seen: Dict[str, str],
        max_results: int = 10,
        sources: List[str] = None,
    ) -> Dict[str, Any]:
        """Fetch only papers newer than the per-source watermark.

        Parameters
        ----------
        goal
            The research goal driving query generation.
        existing_papers
            The papers already known locally (for dedup against URL/title).
        last_seen
            ``{source: iso_timestamp}`` watermark dict, mutated in place.
        max_results, sources
            Forwarded to :meth:`search_literature`.

        Returns
        -------
        ``{"new_papers": [...], "last_seen": {...}}``
        """
        from utils.literature_refresh import filter_new_papers, update_watermark

        if sources is None:
            sources = ["arxiv"]

        # Run a single iteration; we only want fresh hits.
        fetched = await self.search_literature(
            goal, max_results=max_results, sources=sources, iterations=1,
        )

        # Per-source filtering (most queries hit every source equally; this
        # keeps the watermark separate per source for accurate next-time
        # cutoffs).
        new_papers: List[Dict] = []
        for source in sources:
            src = source.lower()
            from_src = [p for p in fetched if p.get("source", "").lower() == src]
            fresh = filter_new_papers(
                from_src, existing_papers, last_seen=last_seen.get(src),
            )
            new_papers.extend(fresh)
            update_watermark(last_seen, src, fresh)

        logger.info(
            "refresh(): %d fresh papers (out of %d fetched).",
            len(new_papers), len(fetched),
        )
        return {"new_papers": new_papers, "last_seen": last_seen}

    async def search_literature(self, goal: ResearchGoal, max_results: int = 5, sources: List[str] = None, iterations: int = 2) -> List[Dict]:
        """
        Search for relevant papers using specified source APIs with iterative refinement.
        Returns a list of paper dictionaries (title, summary, authors, url).
        """
        if sources is None:
            sources = ["arxiv"]
        all_papers = []
        known_titles = set()
        
        def normalize_title(title: str) -> str:
            """Normalize title for deduplication: lowercase and alphanumeric only."""
            return re.sub(r'[^a-zA-Z0-9]', '', title.lower())

        # 1. Initial Search Queries
        queries = await self._generate_search_queries(goal)
        if not queries:
            queries = [goal.title]
            
        logger.info("Expanded search: %d initial queries generated.", len(queries))

        current_query = queries[0]

        for i in range(iterations):
            active_queries = queries if i == 0 else [current_query]
            
            iteration_papers = []
            for q in active_queries:
                if i == 0:
                    logger.info("  - Query: %s", q)

                for source in sources:
                    source = source.lower()
                    if i > 0:
                        logger.info("Searching %s (iter %d)...", source.upper(), i + 1)
                    
                    if source == "arxiv":
                        papers = await self._search_arxiv(q, max_results)
                    elif source == "pubmed":
                        papers = await self._search_pubmed(q, max_results)
                    else:
                        papers = []
                    
                    iteration_papers.extend(papers)
            
            # Filter duplicates (case-insensitive)
            new_papers = []
            for p in iteration_papers:
                norm_t = normalize_title(p['title'])
                if norm_t not in known_titles:
                    new_papers.append(p)
                    known_titles.add(norm_t)
                    all_papers.append(p)
            
            logger.info("Found %d new unique papers.", len(new_papers))

            if not new_papers or i == iterations - 1:
                break

            logger.info("Analyzing results to refine search...")
            refinement = await self._refine_query(goal, all_papers, active_queries[0])
            if refinement:
                current_query = refinement
                logger.info("Refined query: %s", current_query)
            else:
                break
        
        return all_papers[:max_results * 2]

    def get_rag_stats(self) -> Dict:
        """Get statistics from the RAG engine if enabled."""
        if self.rag_engine:
            return self.rag_engine.get_stats()
        return {"status": "disabled", "total_chunks": 0}

    async def extract_key_findings(self, papers: List[Dict], goal: ResearchGoal = None) -> str:
        """
        Extract and synthesize key findings from a list of papers for CAG context.
        Returns a formatted markdown string.
        """
        if not papers:
            return "No papers available for context."
            
        # Advanced Phase 2 CAG Synthesis
        if self.rag_engine and self.llm_client and goal:
            logger.info("Synthesizing semantic CAG report from vector chunks...")
            rag_query = f"{goal.title} {goal.description}"
            chunks = await self.rag_engine.query(rag_query, top_k=8)
            
            if chunks:
                formatted_chunks = ""
                for i, chunk in enumerate(chunks):
                    formatted_chunks += f"Source: {chunk['paper_title']}\nExcerpt: {chunk['text']}\n\n"
                    
                prompt = f"""
                You are a senior scientific analyst formulating a Domain Context Report.
                Synthesize the following literature excerpts into a cohesive "Global Background Report".
                Focus heavily on mechanisms, known pathways, limitations, and key findings relevant to this goal:
                
                Goal: {goal.title}
                Domain: {goal.domain}
                
                Excerpts:
                {formatted_chunks}
                
                Write a comprehensive, insightful synthesis in Markdown. Do not just list the abstracts. Extract true cross-paper insights.
                """
                
                try:
                    response = await get_llm_completion(
                        self.llm_client,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        json_mode=False
                    )
                    synthesis = response.choices[0].message.content.strip()
                    if synthesis:
                        return f"## Semantic CAG Synthesis\n\n{synthesis}\n\n"
                except Exception as e:
                    logger.warning("Semantic CAG synthesis failed: %s. Falling back to abstract list.", e)
            
        # Fallback to abstract concatenation (Legacy CAG)
        context_str = "## Key Findings from Literature (Abstracts)\n\n"
        domain = goal.domain if goal else "domain"
        
        for i, paper in enumerate(papers[:10]):
            summary = paper.get('summary', 'No summary available').replace('\n', ' ')[:300]
            context_str += f"### {i+1}. {paper['title']} ({paper.get('published', 'N/A')})\n"
            context_str += f"**Summary:** {summary}...\n"
            context_str += f"**Key Insight:** Relevance to {domain}\n\n"
            
        return context_str

    async def _search_arxiv(self, query: str, max_results: int) -> List[Dict]:
        if not arxiv:
            logger.warning("`arxiv` library not found.")
            return []
            
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
            logger.info("Found %d papers on ArXiv.", len(results))
            return results

        except Exception as e:
            logger.warning("ArXiv search failed: %s", e)
            return []

    async def _search_pubmed(self, query: str, max_results: int) -> List[Dict]:
        if not Entrez:
            logger.warning("`biopython` library not found.")
            return []
            
        base_query = query + ' AND "free full text"[Filter]'
        # Preserve brackets for PubMed filters and apostrophes for diseases (e.g. Alzheimer's)
        safe_query = re.sub(r'[^\w\s\-\(\)\[\]"\'’]', '', base_query)
        
        try:
            def fetch_pubmed():
                Entrez.email = config.get_entrez_email()
                ncbi_key = config.get_ncbi_api_key()
                if ncbi_key:
                    Entrez.api_key = ncbi_key
                    
                handle = Entrez.esearch(db="pubmed", term=safe_query, retmax=max_results)
                record = Entrez.read(handle)
                handle.close()
                id_list = record["IdList"]
                
                if not id_list:
                    return []
                
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
                        pd_obj = medline['Journal']['JournalIssue']['PubDate']
                        pub_date = f"{pd_obj.get('Year', '')} {pd_obj.get('Month', '')}".strip()
                    
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
            logger.info("Found %d papers on PubMed.", len(results))
            return results

        except Exception as e:
            logger.warning("PubMed search failed: %s", e)
            return []

    async def process_papers_with_rag(self, papers: List[Dict]) -> int:
        """Download and index papers using RAG system"""
        if not self.rag_engine:
            logger.info("RAG system not available. Skipping paper processing.")
            return 0

        logger.info("Processing %d papers with RAG system...", len(papers))
        chunks_indexed = await self.rag_engine.process_papers(papers)
        return chunks_indexed
    
    async def query_rag(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query RAG system for relevant paper chunks, with LLM Semantic Reranking"""
        if not self.rag_engine:
            return []
        
        initial_k = top_k * 3
        chunks = await self.rag_engine.query(query, initial_k)
        
        if not chunks or not self.llm_client:
            return chunks[:top_k]
            
        logger.info("Reranking %d chunks with LLM to find top %d...", len(chunks), top_k)
        
        formatted_chunks = ""
        for i, chunk in enumerate(chunks):
            formatted_chunks += f"--- Chunk {i} ---\nPaper: {chunk['paper_title']}\nText: {chunk['text'][:800]}...\n\n"
            
        prompt = f"""
        You are an expert scientific evaluator. Rate the relevance of the following text chunks to the query: "{query}"
        
        {formatted_chunks}
        
        Output a JSON object with a list of indices of the most relevant chunks, ordered from most to least relevant (max {top_k}).
        Example: {{ "relevant_chunks": [3, 0, 4, 1, 2] }}
        """
        
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                json_mode=True
            )
            data = parse_json_response(response.choices[0].message.content)
            indices = data.get("relevant_chunks", [])
            
            reranked = []
            seen = set()
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                    reranked.append(chunks[idx])
                    seen.add(idx)
                    
            for i, chunk in enumerate(chunks):
                if len(reranked) >= top_k: break
                if i not in seen:
                    reranked.append(chunk)
                    seen.add(i)
                    
            return reranked[:top_k]
            
        except Exception as e:
            logger.warning("Reranking failed: %s. Falling back to default ordering.", e)
            return chunks[:top_k]


__all__ = ["LiteratureAgent"]
