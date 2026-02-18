import os
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from .rag import RAGService, GuardConfig

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SmartCampus Agentic RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGService()
guard = GuardConfig(
    k=int(os.getenv("TOP_K", "12")),
    min_conf=float(os.getenv("MIN_CONF", "0.20")),
    min_cov=float(os.getenv("MIN_COV", "0.20")),
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/query")
async def query(
    question: str = Form(...),
    mode: str = Form("qa"),
    pdf: UploadFile = File(...),
):
    # save upload
    os.makedirs("data/uploads", exist_ok=True)
    pdf_path = os.path.join("data/uploads", pdf.filename)
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    vs = rag.build_or_load_vectorstore(pdf_path)
    result = rag.answer_guarded(vs, question=question, guard=guard, mode=mode)
    return result
