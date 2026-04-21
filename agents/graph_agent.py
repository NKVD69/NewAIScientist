"""
agents/graph_agent.py — GraphAgent for knowledge graph construction.

Responsible for:
- Extracting entities and relationships from papers
- Building a lightweight knowledge graph (adjacency list)
- Cross-domain link synthesis for novel insights
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import config
from models.hypothesis import ResearchGoal
from utils.llm import get_llm_completion, parse_json_response, ensure_str

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None


class GraphAgent:
    """Agent responsible for building a lightweight knowledge graph from papers"""
    
    def __init__(self, use_local_llm: bool = True):
        self.name = "Graph"
        self.graph = {}  # Simple adjacency list: {entity: [relation -> target]}
        self.llm_client = None
        
        if use_local_llm and openai:
            try:
                self.llm_client = config.get_openai_client()
            except Exception:
                self.llm_client = None

    async def build_graph(self, papers: List[Dict], goal: ResearchGoal = None) -> str:
        """Extract entities and relations, return graph summary"""
        if not self.llm_client or not papers:
            return "Graph construction skipped (no LLM or papers)."
            
        print("🕸️ Building Knowledge Graph from literature...")
        
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
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True
            )
            content = response.choices[0].message.content.strip()
            
            triples = parse_json_response(content)
            
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
                
            # Phase 3: Cross-Domain Link Synthesis
            cross_domain_insight = await self._synthesize_cross_domain_links(goal)
            if cross_domain_insight:
                summary += f"\n### 🌉 Cross-Domain Synthesis\n{cross_domain_insight}\n"
                
            return summary
            
        except Exception as e:
            print(f"⚠ Graph construction failed: {e}")
            return "Graph construction failed."

    async def _synthesize_cross_domain_links(self, goal: ResearchGoal) -> str:
        """Analyze the current graph to propose a novel link bridging disconnected domains."""
        if not self.llm_client or not self.graph or len(self.graph) < 3:
            return ""
            
        print("   🌉 Synthesizing Cross-Domain Links...")
        
        graph_text = ""
        for subj, links in list(self.graph.items())[:10]:
            graph_text += f"{subj}: {', '.join(links)}\n"
            
        prompt = f"""
        You are an AI specialized in revealing hidden cross-domain insights.
        Analyze the following extracted knowledge graph representing the current literature:
        
        {graph_text}
        
        Goal: {goal.title if goal else "Unknown"}
        
        Identify two entities that are currently unconnected in this graph but, if linked by a novel hypothetical mechanism, could lead to a breakthrough. 
        Describe this single, highly specific "bridging" mechanism in one concise paragraph.
        Return ONLY a JSON object with this format:
        {{"bridging_insight": "Your concise paragraph..."}}
        """
        
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                json_mode=True
            )
            data = parse_json_response(response.choices[0].message.content)
            return data.get("bridging_insight", "")
        except Exception as e:
            print(f"   ⚠ Cross-domain synthesis failed: {e}")
            return ""


__all__ = ["GraphAgent"]
