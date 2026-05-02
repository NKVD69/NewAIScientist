import pytest
import asyncio
import os
import json
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from agents.analysis import AnalysisAgent
from agents.evolution import EvolutionAgent
from agents.generation import GenerationAgent
from agents.literature import LiteratureAgent
from models.hypothesis import Hypothesis, ResearchGoal, AnalysisPlan, StatisticalResult, DatasetInfo, StateOfArt
from rag_system import RAGEngine, PDFDownloader, SemanticChunker, DocumentProcessor
from co_scientist import CoScientist
import hashlib

@pytest.mark.asyncio
class TestAnalysisAgentExtended:
    @pytest.fixture
    def agent(self):
        with patch("config.get_openai_client"):
            return AnalysisAgent(use_local_llm=True)

    async def test_load_csv(self, agent, tmp_path):
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df.to_csv(csv_file, index=False)
        
        info = await agent.load_csv(str(csv_file))
        assert info.name == "test.csv"
        assert info.num_rows == 2
        assert info.num_columns == 2
        assert "A" in info.column_names

    async def test_fetch_public_database_info(self, agent):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            {"name": "Dataset 1", "source": "GEO", "description": "Desc 1", "num_rows_est": 100}
        ])
        
        agent.llm_client = MagicMock()
        with patch("agents.analysis.get_llm_completion", return_value=mock_response):
            results = await agent.fetch_public_database_info("cancer", "GEO")
            assert len(results) == 1
            assert results[0].name == "Dataset 1"

    async def test_run_exploratory_analysis(self, agent):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        agent.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Exploratory report"
        
        with patch("agents.analysis.get_llm_completion", return_value=mock_response):
            report = await agent.run_exploratory_analysis(df)
            assert report == "Exploratory report"

    async def test_run_statistical_tests(self, agent):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        plan = AnalysisPlan(primary_analysis="Test A vs B")
        agent.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "t-test": {"statistic": 2.5, "p_value": 0.04, "significant": True, "interpretation": "Sig"}
        })
        
        with patch("agents.analysis.get_llm_completion", return_value=mock_response):
            results = await agent.run_statistical_tests(df, plan)
            assert len(results) == 1
            assert results[0].test_name == "t-test"
            assert results[0].significant is True

    async def test_interpret_results(self, agent):
        results = [StatisticalResult(test_name="t-test", statistic_value=2.5, p_value=0.04, significant=True)]
        hyp = Hypothesis(title="Test Hyp", mechanism="Mech")
        agent.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Conclusion: Supported"
        
        with patch("agents.analysis.get_llm_completion", return_value=mock_response):
            conclusion = await agent.interpret_results(results, hyp)
            assert "Supported" in conclusion

@pytest.mark.asyncio
class TestEvolutionAgentExtended:
    @pytest.fixture
    def agent(self):
        with patch("config.get_openai_client"):
            return EvolutionAgent(use_local_llm=True)

    async def test_evolve_strategies(self, agent):
        hyp = Hypothesis(title="Original", description="Desc", mechanism="Mech")
        
        # Test enhancement
        enhanced = await agent.evolve_hypothesis(hyp, strategy="enhancement")
        assert "Evolved: enhancement" in enhanced.title
        assert "grounded" in enhanced.mechanism.lower()
        
        # Test simplification
        simplified = await agent.evolve_hypothesis(hyp, strategy="simplification")
        assert "Simplified" in simplified.title
        
        # Test divergent
        divergent = await agent.evolve_hypothesis(hyp, strategy="out_of_box")
        assert "Divergent" in divergent.title

    async def test_llm_refine_evolution(self, agent):
        hyp = Hypothesis(title="Original", description="Desc", mechanism="Mech")
        agent.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "title": "Refined Title",
            "description": "Refined Desc",
            "mechanism": "Refined Mech",
            "testable_predictions": ["P1"],
            "limitations": ["L1"]
        })
        
        with patch("agents.evolution.get_llm_completion", return_value=mock_response):
            refined = await agent.evolve_hypothesis(hyp, strategy="enhancement")
            assert refined.title == "Refined Title"
            assert refined.generation_method == "evolved-llm"

@pytest.mark.asyncio
class TestGenerationAgentExtended:
    @pytest.fixture
    def agent(self):
        with patch("config.get_openai_client"):
            return GenerationAgent(use_local_llm=True)

    async def test_generate_initial_hypotheses_llm(self, agent):
        goal = ResearchGoal(title="Goal", description="Desc", domain="Domain")
        papers = [{"title": "Paper 1", "url": "url1", "summary": "Sum 1"}]
        agent.llm_client = MagicMock()
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([{
            "title": "Hyp 1", "description": "Desc 1", "reasoning": "Reas 1", 
            "mechanism": "Mech 1", "testable_predictions": ["P1"], 
            "cited_papers": ["url1"], "grounding_evidence": ["E1"], "limitations": ["L1"]
        }])
        
        with patch("agents.generation.get_llm_completion", return_value=mock_response):
            # Also mock refinement
            with patch.object(agent, "_refine_hypothesis", side_effect=lambda x, y: x):
                hyps = await agent.generate_initial_hypotheses(goal, papers, count=1)
                assert len(hyps) == 1
                assert hyps[0].title == "Hyp 1"

    async def test_generate_simulated_fallback(self, agent):
        goal = ResearchGoal(title="Goal", description="Desc", domain="Domain")
        agent.llm_client = None
        hyps = await agent.generate_initial_hypotheses(goal, [], count=2)
        assert len(hyps) == 2
        assert "Simulated" in hyps[0].generation_method or "simulated" in hyps[0].generation_method

@pytest.mark.asyncio
class TestLiteratureAgentExtended:
    @pytest.fixture
    def agent(self):
        with patch("config.get_openai_client"):
            return LiteratureAgent(use_local_llm=True, enable_rag=False)

    async def test_generate_search_queries(self, agent):
        goal = ResearchGoal(title="COVID treatment", description="Desc")
        agent.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"queries": ["covid therapy", "sars-cov-2 drug"]})
        
        with patch("agents.literature.get_llm_completion", return_value=mock_response):
            queries = await agent._generate_search_queries(goal)
            assert "covid therapy" in queries

    async def test_search_literature_deduplication(self, agent):
        goal = ResearchGoal(title="Goal")
        mock_papers = [
            {"title": "Unique Paper", "url": "url1", "summary": "S1"},
            {"title": "UNIQUE PAPER", "url": "url2", "summary": "S2"}, # Duplicate
        ]
        
        with patch.object(agent, "_generate_search_queries", return_value=["query"]), \
             patch.object(agent, "_search_arxiv", return_value=mock_papers), \
             patch.object(agent, "_search_pubmed", return_value=[]):
            papers = await agent.search_literature(goal, max_results=5, iterations=1)
            assert len(papers) == 1
            assert papers[0]["title"] == "Unique Paper"

    async def test_extract_key_findings_fallback(self, agent):
        papers = [{"title": "P1", "summary": "S1", "published": "2023"}]
        findings = await agent.extract_key_findings(papers)
        assert "P1" in findings
        assert "S1" in findings

class TestRAGExtended:
    def test_semantic_chunker(self):
        chunker = SemanticChunker(chunk_size=10, overlap=2) # Very small chunk size
        text = "This is a long sentence. It has multiple parts. We want it to be split. Testing the chunker now."
        chunks = chunker.chunk_text(text, "id1", "title1")
        assert len(chunks) > 1
        assert chunks[0].paper_id == "id1"

    @pytest.mark.asyncio
    async def test_pdf_downloader_arxiv_logic(self):
        downloader = PDFDownloader(cache_dir="./test_papers")
        url = "http://arxiv.org/abs/2103.12345"
        # Mocking the actual download
        with patch("urllib.request.urlretrieve"), patch("pathlib.Path.exists", return_value=False):
            path = await downloader.download_arxiv_pdf(url)
            assert "2103.12345.pdf" in str(path) or hashlib.md5((url.replace("/abs/", "/pdf/") + ".pdf").encode()).hexdigest() in str(path)
        
        if os.path.exists("./test_papers"):
            import shutil
            shutil.rmtree("./test_papers")

@pytest.mark.asyncio
class TestCoScientistExtended:
    @pytest.fixture
    def co_scientist(self):
        with patch("co_scientist.GenerationAgent"), \
             patch("co_scientist.LiteratureAgent"), \
             patch("co_scientist.ScopingAgent"), \
             patch("co_scientist.ProtocolAgent"), \
             patch("co_scientist.AnalysisAgent"), \
             patch("co_scientist.WritingAgent"), \
             patch("co_scientist.SupervisorAgent"):
            cs = CoScientist(use_local_llm=False)
            cs.generation_agent.generated_count = 0
            cs.reflection_agent.reviews_completed = 0
            cs.ranking_agent.matches_completed = 0
            cs.evolution_agent.evolved_hypotheses = 0
            cs.meta_review_agent.meta_reviews_generated = 0
            cs.literature_agent.papers_retrieved = 0
            cs.experiment_agent.experiments_run = 0
            return cs

    async def test_analyze_research_description(self, co_scientist):
        co_scientist.generation_agent.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"domains": ["Biology"], "databases": ["pubmed"]})
        
        with patch("co_scientist._get_llm_completion", return_value=mock_response):
            result = await co_scientist.analyze_research_description("cancer research")
            assert "Biology" in result["domains"]

    async def test_run_literature_search_integration(self, co_scientist):
        co_scientist.context_memory.research_goal = ResearchGoal(title="Test")
        co_scientist.literature_agent.search_literature = AsyncMock(return_value=[{"title": "P1"}])
        co_scientist.literature_agent.extract_key_findings = AsyncMock(return_value="Findings")
        co_scientist.graph_agent.build_graph = AsyncMock(return_value="Graph")
        
        papers = await co_scientist.run_literature_search(iterations=1)
        assert len(papers) == 1
        assert co_scientist.generation_agent.cag_context == "Findings\n\nGraph"

    async def test_export_hypotheses_json(self, co_scientist, tmp_path):
        co_scientist.context_memory.research_goal = ResearchGoal(title="Test")
        co_scientist.context_memory.hypotheses = {"h1": Hypothesis(title="H1")}
        
        export_file = tmp_path / "hypotheses.json"
        co_scientist.export_hypotheses_json(str(export_file))
        
        assert export_file.exists()
        with open(export_file, "r") as f:
            data = json.load(f)
            assert data["research_goal"]["title"] == "Test"
            assert len(data["hypotheses"]) == 1
