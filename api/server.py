"""
api/server.py — FastAPI backend for NewAI Scientist v3.0

Provides REST endpoints for all 6 workflow phases + session management.
Designed to be consumed by the React frontend.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict
import os
import json
import uvicorn
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from co_scientist import CoScientist
from models import (
    ResearchGoal,
    Hypothesis,
    StudyPhase,
    HypothesisStatus,
)

app = FastAPI(
    title="NewAI Scientist API",
    description="Multi-agent scientific research system API",
    version="3.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instance (session-based in production)
scientist = CoScientist()

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "sessions")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


# --------------------------------------------------------------------------
# Pydantic Request Models
# --------------------------------------------------------------------------

class GoalInput(BaseModel):
    title: str
    description: str
    domain: str
    preferences: Optional[Dict] = {}
    constraints: Optional[List[str]] = []


class LiteratureInput(BaseModel):
    max_results: int = 5
    sources: List[str] = ["arxiv"]
    iterations: int = 2


class HypothesisGenerationInput(BaseModel):
    count: int = 5


class TournamentInput(BaseModel):
    num_matches: int = 5


class AnalysisInput(BaseModel):
    file_path: Optional[str] = None


# --------------------------------------------------------------------------
# Health & Status
# --------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "online", "version": "3.0.0", "timestamp": datetime.now().isoformat()}


@app.get("/session/state")
async def get_state():
    """Returns the full current context memory state."""
    ctx = scientist.context_memory
    return {
        "phase": ctx.current_phase,
        "goal": asdict(ctx.research_goal) if ctx.research_goal.title else None,
        "num_hypotheses": len(ctx.hypotheses),
        "num_papers": len(ctx.literature_context),
        "num_questions": len(ctx.research_questions),
        "num_protocols": len(ctx.experimental_protocols),
        "num_datasets": len(ctx.datasets),
        "iteration": ctx.iteration_count,
        "has_manuscript": ctx.manuscript is not None,
    }


@app.get("/session/hypotheses")
async def get_hypotheses():
    """Returns all hypotheses sorted by Elo."""
    hyps = sorted(
        scientist.context_memory.hypotheses.values(),
        key=lambda h: h.elo_rating,
        reverse=True,
    )
    return [
        {
            "id": h.id,
            "title": h.title,
            "description": h.description,
            "mechanism": h.mechanism,
            "elo_rating": h.elo_rating,
            "novelty_level": h.novelty_level,
            "status": h.status.value,
            "testable_predictions": h.testable_predictions,
            "num_reviews": len(h.reviews),
            "generation_method": h.generation_method,
            "cited_papers": h.cited_papers,
        }
        for h in hyps
    ]


@app.get("/session/papers")
async def get_papers():
    """Returns all retrieved papers."""
    return scientist.context_memory.literature_context


@app.get("/session/questions")
async def get_research_questions():
    """Returns research questions from scoping phase."""
    questions = scientist.context_memory.research_questions
    if not questions:
        return []
    return [asdict(q) if hasattr(q, '__dataclass_fields__') else q for q in questions]


@app.get("/session/scoping")
async def get_scoping_results():
    """Returns state of art, questions, and conceptual framework."""
    return {
        "state_of_art": scientist.context_memory.state_of_art,
        "research_questions": scientist.context_memory.research_questions,
        "conceptual_framework": scientist.context_memory.conceptual_framework,
    }


# --------------------------------------------------------------------------
# Workflow Phase Endpoints
# --------------------------------------------------------------------------

@app.post("/goal/initialize")
async def init_goal(input: GoalInput):
    goal = await scientist.initialize_research_goal(
        title=input.title,
        description=input.description,
        domain=input.domain,
        preferences=input.preferences,
        constraints=input.constraints,
    )
    return asdict(goal)


@app.post("/goal/analyze")
async def analyze_description(description: str):
    """Auto-detect domain and databases from research description."""
    result = await scientist.analyze_research_description(description)
    return result


@app.post("/workflow/literature")
async def run_literature(input: LiteratureInput):
    papers = await scientist.run_literature_search(
        max_results=input.max_results,
        sources=input.sources,
        iterations=input.iterations,
    )
    return {"count": len(papers), "papers": papers}


@app.post("/workflow/scoping")
async def run_scoping():
    result = await scientist.run_scoping_cycle()
    # Convert StateOfArt dataclass to dict
    return {
        "state_of_art": asdict(result["soa"]) if hasattr(result["soa"], '__dataclass_fields__') else result["soa"],
        "questions": [asdict(q) if hasattr(q, '__dataclass_fields__') else q for q in result.get("questions", [])],
        "framework": result.get("framework", {}),
    }


@app.post("/workflow/hypotheses")
async def generate_hypotheses(input: HypothesisGenerationInput):
    hypotheses = await scientist.run_hypothesis_generation_cycle(num_hypotheses=input.count)
    return {
        "count": len(hypotheses),
        "hypotheses": [
            {
                "id": h.id,
                "title": h.title,
                "description": h.description,
                "elo_rating": h.elo_rating,
            }
            for h in hypotheses
        ],
    }


@app.post("/workflow/review")
async def run_review():
    reviews = await scientist.run_review_cycle()
    return {"count": len(reviews), "reviews": [asdict(r) for r in reviews]}


@app.post("/workflow/tournament")
async def run_tournament(input: TournamentInput):
    matches = await scientist.run_tournament_cycle(num_matches=input.num_matches)
    return {"count": len(matches), "matches": [asdict(m) for m in matches]}


@app.post("/workflow/evolution")
async def run_evolution():
    evolved = await scientist.run_evolution_cycle()
    return {
        "count": len(evolved),
        "evolved": [{"id": h.id, "title": h.title, "elo_rating": h.elo_rating} for h in evolved],
    }


@app.post("/workflow/meta-review")
async def run_meta_review():
    meta = await scientist.run_meta_review_cycle()
    return meta


@app.post("/workflow/protocol/{hypothesis_id}")
async def generate_protocol(hypothesis_id: str):
    protocol = await scientist.run_protocol_cycle(hypothesis_id=hypothesis_id)
    if not protocol:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return asdict(protocol)


@app.post("/workflow/analysis/{hypothesis_id}")
async def run_analysis(hypothesis_id: str, input: AnalysisInput):
    result = await scientist.run_analysis_cycle(
        hypothesis_id=hypothesis_id, file_path=input.file_path,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return result

@app.patch("/hypothesis/{hypothesis_id}/notes")
async def update_notes(hypothesis_id: str, notes: str):
    hyp = await scientist.update_hypothesis(hypothesis_id, {"scientist_notes": notes})
    return hyp

@app.post("/workflow/writing")
async def run_writing():
    manuscript = await scientist.run_writing_cycle()
    return asdict(manuscript)


# --------------------------------------------------------------------------
# File Upload
# --------------------------------------------------------------------------

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Uploads a CSV for analysis phase."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "path": os.path.abspath(file_path)}


# --------------------------------------------------------------------------
# Session Persistence
# --------------------------------------------------------------------------

@app.post("/session/save")
async def save_session(name: str = "default"):
    """Save current session state to disk."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = os.path.join(SESSIONS_DIR, f"{name}.json")
    try:
        data = {
            "research_goal": asdict(scientist.context_memory.research_goal),
            "current_phase": scientist.context_memory.current_phase,
            "literature_context": scientist.context_memory.literature_context,
            "state_of_art": scientist.context_memory.state_of_art,
            "conceptual_framework": scientist.context_memory.conceptual_framework,
            "iteration_count": scientist.context_memory.iteration_count,
            "num_hypotheses": len(scientist.context_memory.hypotheses),
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return {"status": "saved", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/list")
async def list_sessions():
    """List all saved sessions."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                sessions.append({
                    "name": fname.replace(".json", ""),
                    "saved_at": data.get("saved_at", "unknown"),
                    "phase": data.get("current_phase", "unknown"),
                })
            except Exception:
                sessions.append({"name": fname.replace(".json", ""), "saved_at": "error"})
    return sessions


@app.get("/session/export")
async def export_json():
    """Export all hypotheses as JSON."""
    scientist.export_hypotheses_json("export.json")
    with open("export.json", "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
