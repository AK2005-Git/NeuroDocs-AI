from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from rag_chat import ask_rag

# ============================================================
# PDF Upload Router
# ============================================================

from api.upload import router as upload_router

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Advanced RAG API",
    version="3.0.0",
    description=(
        "Advanced Hybrid RAG System with "
        "Groq, FAISS, BM25, Reranking, "
        "Chat Memory and PDF Upload"
    )
)

# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)

# ============================================================
# REGISTER ROUTERS
# ============================================================

app.include_router(
    upload_router,
    tags=["PDF Upload"]
)

# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    retrieval_mode: str = "hybrid"        # hybrid | faiss | bm25
    top_k_retrieval: int | None = None
    top_k_final: int | None = None
    rrf_k: int | None = None
    use_reranker: bool = True
    enable_query_rewrite: bool = True
    enable_groundedness_check: bool = True

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Advanced RAG API is active",
        "version": "3.0.0"
    }

# ============================================================
# WEB UI
# ============================================================

@app.get("/ui")
def ui():

    return FileResponse(
        "frontend/index.html"
    )

# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat_endpoint(
    request: QuestionRequest
):

    result = ask_rag(
        request.question,
        retrieval_mode=request.retrieval_mode,
        top_k_retrieval=request.top_k_retrieval,
        top_k_final=request.top_k_final,
        rrf_k=request.rrf_k,
        use_reranker=request.use_reranker,
        enable_query_rewrite=request.enable_query_rewrite,
        enable_groundedness_check=request.enable_groundedness_check
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "response_time": result["response_time"],
        "retrieval_time": result.get("retrieval_time", 0),
        "generation_time": result.get("generation_time", 0),
        "routing": result.get("routing", "normal"),
        "suggestions": result.get("suggestions", []),
        "search_query": result.get("search_query", request.question),
        "was_rewritten": result.get("was_rewritten", False),
        "grounded": result.get("grounded", True),
        "groundedness_explanation": result.get("groundedness_explanation"),
        "total_queries": result.get(
            "total_queries",
            0
        )
    }

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "version": "3.0.0"
    }

# ============================================================
# PROJECT INFO
# ============================================================

@app.get("/info")
def info():

    return {
        "project": "Advanced RAG System",
        "version": "3.0.0",
        "features": [
            "PDF Upload",
            "Hybrid Retrieval",
            "FAISS Search",
            "BM25 Search",
            "RRF Fusion",
            "Cross Encoder Reranking",
            "Groq LLM Integration",
            "Query Rewriting",
            "Hallucination Detection",
            "Chat Memory",
            "Analytics",
            "FastAPI Backend",
            "Web UI"
        ]
    }

# ============================================================
# API STATUS
# ============================================================

@app.get("/status")
def status():

    return {
        "server": "online",
        "api": "working",
        "chat_endpoint": "/chat",
        "upload_endpoint": "/upload",
        "ui": "/ui",
        "swagger_docs": "/docs"
    }

# ============================================================
# EVALUATION DASHBOARD
# ============================================================

@app.get("/evaluation/results")
def get_evaluation_results():
    """
    Returns the most recently saved evaluation run
    (evaluation/last_run_results.json), produced by running:
        python evaluation/run_eval.py
    """

    import json
    from pathlib import Path

    results_path = Path("evaluation/last_run_results.json")

    if not results_path.exists():

        return {
            "status": "no_results",
            "message": (
                "No evaluation has been run yet. Run "
                "'python evaluation/run_eval.py' from the project "
                "root to generate results."
            )
        }

    with open(results_path, "r", encoding="utf-8") as f:

        data = json.load(f)

    return {
        "status": "success",
        **data
    }


@app.post("/evaluation/run")
def run_evaluation():
    """
    Runs the evaluation suite live (against evaluation/qa_pairs.json)
    and returns fresh results. This re-runs retrieval for every
    labeled question, so it can take a while with many test cases.
    """

    from evaluation.run_eval import load_qa_pairs, evaluate
    from pathlib import Path
    import json

    dataset_path = Path("evaluation/qa_pairs.json")

    if not dataset_path.exists():

        return {
            "status": "error",
            "message": "evaluation/qa_pairs.json not found"
        }

    qa_pairs = load_qa_pairs(dataset_path)

    summary, results = evaluate(qa_pairs, top_k=5)

    output = {
        "summary": summary,
        "results": results
    }

    output_path = Path("evaluation/last_run_results.json")

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(output, f, indent=2)

    return {
        "status": "success",
        **output
    }

# ============================================================
# STARTUP CHECK
# ============================================================

@app.on_event("startup")
async def startup_event():

    import os

    print("\n" + "=" * 60)
    print("Advanced RAG API Started")
    print("Swagger : http://127.0.0.1:8000/docs")
    print("UI      : http://127.0.0.1:8000/ui")

    if os.environ.get("GROQ_API_KEY"):

        print("Groq API key: loaded OK")

    else:

        print(
            "WARNING: GROQ_API_KEY not found in environment. "
            "Check your .env file exists and is in the project root."
        )

    print("=" * 60)