"""
api.py - FastAPI service exposing the RAG endpoints.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /ingest              -> (re)build the vector index from documents/
    POST /ask                 -> {question} -> grounded answer with citations
    POST /contradict          -> {doc_id_1, doc_id_2} -> conflict analysis
    GET  /documents           -> list of indexed doc_ids
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag_core

app = FastAPI(title="Document Q&A RAG API")


class AskRequest(BaseModel):
    question: str
    top_k: int = rag_core.TOP_K_FINAL


class ContradictRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str


@app.post("/ingest")
def ingest():
    result = rag_core.build_index()
    return result


@app.get("/documents")
def documents():
    try:
        return {"doc_ids": rag_core.list_doc_ids()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return rag_core.answer_question(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/contradict")
def contradict(req: ContradictRequest):
    try:
        result = rag_core.contradiction_check(req.doc_id_1, req.doc_id_2)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/ask", "/contradict", "/documents", "/ingest"]}
