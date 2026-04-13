from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import uvicorn
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from co_scientist import CoScientist
from models import ResearchGoal, Hypothesis, StudyPhase

app = FastAPI(title="NewAI Scientist API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (in a real app, use a dependency or session management)
scientist = CoScientist()

# --------------------------------------------------------------------------
# Models for API
# --------------------------------------------------------------------------

class GoalInput(BaseModel):
    title: str
    description: str
    domain: str
    preferences: Optional[Dict] = {}
    constraints: Optional[List[str]] = []

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "online", "version": "3.0"}

@app.get("/session/state")
async def get_state():
    """Returns the current context memory state."""
    # Convert context memory to dict (handling non-serializable objects if any)
    # This is a bit simplified; real implementation might need a proper serializer
    return {
        "phase": scientist.context_memory.current_phase,
        "goal": scientist.context_memory.research_goal,
        "num_hypotheses": len(scientist.context_memory.hypotheses),
        "num_papers": len(scientist.context_memory.literature_context),
        "iteration": scientist.context_memory.iteration_count
    }

@app.post("/goal/initialize")
async def init_goal(input: GoalInput):
    goal = await scientist.initialize_research_goal(
        title=input.title,
        description=input.description,
        domain=input.domain,
        preferences=input.preferences,
        constraints=input.constraints
    )
    return goal

@app.post("/workflow/scoping")
async def run_scoping():
    result = await scientist.run_scoping_cycle()
    return result

@app.post("/workflow/literature")
async def run_literature(max_results: int = 5, sources: List[str] = ["arxiv"]):
    papers = await scientist.run_literature_search(max_results=max_results, sources=sources)
    return papers

@app.post("/workflow/hypotheses")
async def generate_hypotheses(count: int = 5):
    result = await scientist.run_hypothesis_generation_cycle(num_hypotheses=count)
    return result

@app.post("/workflow/protocol/{hypothesis_id}")
async def generate_protocol(hypothesis_id: str):
    protocol = await scientist.run_protocol_cycle(hypothesis_id=hypothesis_id)
    if not protocol:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return protocol

@app.post("/workflow/analysis/{hypothesis_id}")
async def run_analysis(hypothesis_id: str, file_path: str = None):
    result = await scientist.run_analysis_cycle(hypothesis_id=hypothesis_id, file_path=file_path)
    return result

@app.patch("/hypothesis/{hypothesis_id}/notes")
async def update_notes(hypothesis_id: str, notes: str):
    hyp = await scientist.update_hypothesis(hypothesis_id, {"scientist_notes": notes})
    return hyp

@app.post("/workflow/writing")
async def run_writing():
    manuscript = await scientist.run_writing_cycle()
    return manuscript

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Uploads a CSV to the server and returns the local path."""
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    return {"filename": file.filename, "path": os.path.abspath(file_path)}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
