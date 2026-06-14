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
        "Ollama, FAISS, BM25, Reranking, "
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
        request.question
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "response_time": result["response_time"],
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
            "Ollama Integration",
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
# STARTUP CHECK
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("\n" + "=" * 60)
    print("Advanced RAG API Started")
    print("Swagger : http://127.0.0.1:8000/docs")
    print("UI      : http://127.0.0.1:8000/ui")
    print("=" * 60)