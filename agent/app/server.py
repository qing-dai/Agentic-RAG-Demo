# server.py
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict
from pprint import pprint
from agent.app.graph.build import build_app

app = FastAPI(title="Agentic RAG Demo")
graph = build_app()  # compile LangGraph once

# Static dir next to this file: agent/app/static
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# serve the static frontend (index.html lives in ./static)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    """
    POST /chat
    body: {"question": "..."}
    returns: {"answer": "...", "state": {...optional...}}
    """
    last = {}
    # Run the compiled graph synchronously (LangGraph stream is sync)
    for chunk in graph.stream({"question": req.question}):
        for node, state in chunk.items():
            print(f"Node '{node}':")
            pprint(state, indent=2, width=100)
            last = state

    answer = last.get("generation")
    return {"answer": answer}


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))