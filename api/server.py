"""
api/server.py — FastAPI backend for NewAI Scientist v3.0

Provides REST endpoints for all 6 workflow phases + session management.
Designed to be consumed by the React frontend.

Security:
  - Every state-mutating endpoint requires an ``X-API-Key`` header (see
    ``api/security.py``). Set the ``API_KEYS`` env var to enable auth, or
    ``ALLOW_UNAUTHENTICATED=1`` for local dev only.
  - CORS origins come from the ``CORS_ALLOWED_ORIGINS`` env var
    (comma-separated); wildcard ``*`` is refused with credentials.
  - Uploaded filenames are sanitised; user-supplied file paths must
    resolve inside ``UPLOAD_DIR``.
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.security import (
    get_cors_origins,
    require_api_key,
    safe_path_within,
    sanitise_filename,
)
from co_scientist import CoScientist

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NewAI Scientist API",
    description="Multi-agent scientific research system API",
    version="3.0.0",
)

# CORS — explicit allowlist driven by env var; never the wildcard "*"
# combined with credentials (forbidden by the CORS spec).
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# Global instance (session-based in production)
scientist = CoScientist()

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "sessions")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")

# Whitelisted file types for analysis uploads.
ALLOWED_DATASET_EXTENSIONS = (".csv", ".tsv")


# --------------------------------------------------------------------------
# Pydantic Request Models
# --------------------------------------------------------------------------

class GoalInput(BaseModel):
    title: str
    description: str
    domain: str
    preferences: dict | None = {}
    constraints: list[str] | None = []


class LiteratureInput(BaseModel):
    max_results: int = 5
    sources: list[str] = ["arxiv"]
    iterations: int = 2


class HypothesisGenerationInput(BaseModel):
    count: int = 5


class TournamentInput(BaseModel):
    num_matches: int = 5


class AnalysisInput(BaseModel):
    file_path: str | None = None


# --------------------------------------------------------------------------
# Health & Status
# --------------------------------------------------------------------------

@app.get("/")
def read_root():
    """Health check — intentionally unauthenticated."""
    return {"status": "online", "version": "3.0.0", "timestamp": datetime.now().isoformat()}


@app.get("/session/state", dependencies=[Depends(require_api_key)])
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


@app.get("/session/hypotheses", dependencies=[Depends(require_api_key)])
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


@app.get("/session/papers", dependencies=[Depends(require_api_key)])
async def get_papers():
    """Returns all retrieved papers."""
    return scientist.context_memory.literature_context


@app.get("/session/questions", dependencies=[Depends(require_api_key)])
async def get_research_questions():
    """Returns research questions from scoping phase."""
    questions = scientist.context_memory.research_questions
    if not questions:
        return []
    return [asdict(q) if hasattr(q, '__dataclass_fields__') else q for q in questions]


@app.get("/session/scoping", dependencies=[Depends(require_api_key)])
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

@app.post("/goal/initialize", dependencies=[Depends(require_api_key)])
async def init_goal(input: GoalInput):
    goal = await scientist.initialize_research_goal(
        title=input.title,
        description=input.description,
        domain=input.domain,
        preferences=input.preferences,
        constraints=input.constraints,
    )
    return asdict(goal)


@app.post("/goal/analyze", dependencies=[Depends(require_api_key)])
async def analyze_description(description: str):
    """Auto-detect domain and databases from research description."""
    result = await scientist.analyze_research_description(description)
    return result


@app.post("/workflow/literature", dependencies=[Depends(require_api_key)])
async def run_literature(input: LiteratureInput):
    papers = await scientist.run_literature_search(
        max_results=input.max_results,
        sources=input.sources,
        iterations=input.iterations,
    )
    return {"count": len(papers), "papers": papers}


@app.post("/workflow/scoping", dependencies=[Depends(require_api_key)])
async def run_scoping():
    result = await scientist.run_scoping_cycle()
    # Convert StateOfArt dataclass to dict
    return {
        "state_of_art": asdict(result["soa"]) if hasattr(result["soa"], '__dataclass_fields__') else result["soa"],
        "questions": [asdict(q) if hasattr(q, '__dataclass_fields__') else q for q in result.get("questions", [])],
        "framework": result.get("framework", {}),
    }


@app.post("/workflow/hypotheses", dependencies=[Depends(require_api_key)])
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


@app.post("/workflow/review", dependencies=[Depends(require_api_key)])
async def run_review():
    reviews = await scientist.run_review_cycle()
    return {"count": len(reviews), "reviews": [asdict(r) for r in reviews]}


@app.post("/workflow/tournament", dependencies=[Depends(require_api_key)])
async def run_tournament(input: TournamentInput):
    matches = await scientist.run_tournament_cycle(num_matches=input.num_matches)
    return {"count": len(matches), "matches": [asdict(m) for m in matches]}


@app.post("/workflow/evolution", dependencies=[Depends(require_api_key)])
async def run_evolution():
    evolved = await scientist.run_evolution_cycle()
    return {
        "count": len(evolved),
        "evolved": [{"id": h.id, "title": h.title, "elo_rating": h.elo_rating} for h in evolved],
    }


@app.post("/workflow/meta-review", dependencies=[Depends(require_api_key)])
async def run_meta_review():
    meta = await scientist.run_meta_review_cycle()
    return meta


@app.post("/workflow/protocol/{hypothesis_id}", dependencies=[Depends(require_api_key)])
async def generate_protocol(hypothesis_id: str):
    protocol = await scientist.run_protocol_cycle(hypothesis_id=hypothesis_id)
    if not protocol:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return asdict(protocol)


@app.post("/workflow/analysis/{hypothesis_id}", dependencies=[Depends(require_api_key)])
async def run_analysis(hypothesis_id: str, input: AnalysisInput):
    # Path-traversal defence: if the caller passed a file_path, it must
    # resolve inside UPLOAD_DIR. Empty / None ⇒ pass through (orchestrator
    # falls back to "no dataset" mode).
    resolved_path: str | None = None
    if input.file_path:
        try:
            resolved = safe_path_within(input.file_path, UPLOAD_DIR)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not resolved.is_file():
            raise HTTPException(
                status_code=404, detail=f"dataset not found: {resolved.name}",
            )
        resolved_path = str(resolved)

    result = await scientist.run_analysis_cycle(
        hypothesis_id=hypothesis_id, file_path=resolved_path,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return result


@app.patch("/hypothesis/{hypothesis_id}/notes", dependencies=[Depends(require_api_key)])
async def update_notes(hypothesis_id: str, notes: str):
    """Attach free-text scientist notes to a hypothesis."""
    hyp = scientist.update_hypothesis(hypothesis_id, {"scientist_notes": notes})
    if hyp is None:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return {
        "id": hyp.id,
        "title": hyp.title,
        "scientist_notes": hyp.scientist_notes,
    }


@app.post("/workflow/writing", dependencies=[Depends(require_api_key)])
async def run_writing():
    manuscript = await scientist.run_writing_cycle()
    return asdict(manuscript)


# --------------------------------------------------------------------------
# File Upload
# --------------------------------------------------------------------------

@app.post("/upload/csv", dependencies=[Depends(require_api_key)])
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV/TSV dataset for the analysis phase.

    Filename is stripped to its basename and validated against an
    extension allowlist; the resolved path is enforced to lie inside
    UPLOAD_DIR. Both defences combined prevent path-traversal attacks
    (``../../etc/passwd``) and arbitrary file overwrites.
    """
    try:
        safe_name = sanitise_filename(
            file.filename or "", ALLOWED_DATASET_EXTENSIONS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        target = safe_path_within(os.path.join(UPLOAD_DIR, safe_name), UPLOAD_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with open(target, "wb") as f:
        f.write(await file.read())
    return {"filename": safe_name, "path": str(target)}


# --------------------------------------------------------------------------
# Session Persistence
# --------------------------------------------------------------------------

@app.post("/session/save", dependencies=[Depends(require_api_key)])
async def save_session(name: str = "default"):
    """Save current session state to disk.

    The session ``name`` is sanitised to a single safe filename so
    that ``name=../../etc/passwd`` cannot escape SESSIONS_DIR.
    """
    try:
        safe_name = sanitise_filename(f"{name}.json", (".json",))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        path = safe_path_within(os.path.join(SESSIONS_DIR, safe_name), SESSIONS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/session/list", dependencies=[Depends(require_api_key)])
async def list_sessions():
    """List all saved sessions."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                sessions.append({
                    "name": fname.replace(".json", ""),
                    "saved_at": data.get("saved_at", "unknown"),
                    "phase": data.get("current_phase", "unknown"),
                })
            except Exception:
                sessions.append({"name": fname.replace(".json", ""), "saved_at": "error"})
    return sessions


@app.get("/session/export", dependencies=[Depends(require_api_key)])
async def export_json():
    """Export all hypotheses as JSON.

    Writes a fixed-name file inside SESSIONS_DIR (not the process CWD),
    then reads it back. Both paths are inside the configured root so
    nothing escapes.
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    export_path = safe_path_within(
        os.path.join(SESSIONS_DIR, "export.json"), SESSIONS_DIR,
    )
    scientist.export_hypotheses_json(str(export_path))
    with open(export_path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Default to loopback only; flip API_HOST=0.0.0.0 in env to expose
    # externally (alongside API_KEYS to enforce authentication).
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=True)
