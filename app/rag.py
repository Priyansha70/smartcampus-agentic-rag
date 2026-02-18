import os
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

STOPWORDS = {
    "the","a","an","and","or","to","of","in","for","is","are","was","were","be","been",
    "this","that","it","as","on","with","by","at","from","what","which","when","where",
    "how","do","does","did","best"
}

def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return [w for w in words if w not in STOPWORDS]

def keyword_coverage(question: str, context: str) -> float:
    q_words = set(extract_keywords(question))
    if not q_words:
        return 0.0
    ctx = context.lower()
    hits = sum(1 for w in q_words if w in ctx)
    return hits / len(q_words)

def score_to_confidence(score: float) -> float:
    # distance -> confidence proxy
    return 1.0 / (1.0 + float(score))

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class GuardConfig:
    k: int = 12
    min_conf: float = 0.20
    min_cov: float = 0.20

class RAGService:
    def __init__(self, model: str = "gpt-4o-mini", emb_model: str = "text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Put it in your environment or .env file.")

        self.llm = ChatOpenAI(model=model, temperature=0.2)
        self.embeddings = OpenAIEmbeddings(model=emb_model)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    def _index_dir(self, pdf_hash: str) -> str:
        return os.path.join("data", "indexes", pdf_hash)

    def build_or_load_vectorstore(self, pdf_path: str) -> FAISS:
        pdf_hash = sha256_file(pdf_path)
        idx_dir = self._index_dir(pdf_hash)
        os.makedirs(idx_dir, exist_ok=True)

        faiss_path = os.path.join(idx_dir, "faiss_index")
        meta_path = os.path.join(idx_dir, "meta.json")

        # If already cached, load
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            logger.info(f"Loading cached index: {idx_dir}")
            return FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)

        logger.info(f"Building new index for: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)

        vs = FAISS.from_documents(chunks, self.embeddings)
        vs.save_local(faiss_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"pdf_path": pdf_path, "pdf_hash": pdf_hash, "pages": len(docs), "chunks": len(chunks)}, f, indent=2)

        return vs

    def answer_guarded(self, vectorstore: FAISS, question: str, guard: GuardConfig, mode: str = "qa") -> Dict[str, Any]:
        # Retrieval with scores
        retrieved: List[Tuple[Any, float]] = vectorstore.similarity_search_with_score(question, k=guard.k)
        if not retrieved:
            return {"mode": mode, "answer": "No relevant context retrieved.", "confidence": 0.0, "coverage": 0.0, "pages": [], "top_matches": []}

        top_matches = []
        pages_set = set()
        context_blocks = []

        for doc, score in retrieved:
            page = doc.metadata.get("page", "?")
            pages_set.add(page)
            conf = score_to_confidence(score)
            top_matches.append({
                "page": page,
                "score": float(score),
                "confidence": conf,
                "preview": doc.page_content[:180].replace("\n", " ")
            })
            context_blocks.append(f"[page {page}]\n{doc.page_content}")

        overall_conf = sum(m["confidence"] for m in top_matches[:3]) / min(3, len(top_matches))
        context = "\n\n---\n\n".join(context_blocks)
        cov = keyword_coverage(question, context)
        allowed_pages = sorted(pages_set, key=lambda x: str(x))
        allowed_pages_str = ", ".join(map(str, allowed_pages))

        logger.info(f"Guard: conf={overall_conf:.2f} cov={cov:.2f} pages={allowed_pages}")

        if overall_conf < guard.min_conf or cov < guard.min_cov:
            return {
                "mode": "refused",
                "answer": (
                    "I’m not confident the PDF supports that request.\n"
                    f"- retrieval confidence={overall_conf:.2f}\n"
                    f"- keyword coverage={cov:.2f}\n\n"
                    "Try using exact terms from the PDF or ask a more specific question."
                ),
                "confidence": overall_conf,
                "coverage": cov,
                "pages": allowed_pages,
                "top_matches": top_matches,
            }

        # Mode transforms
        prompt = question
        if mode == "summary":
            prompt = "Summarize (exam-style bullets): " + question

        msgs = [
            SystemMessage(content=
                "Use ONLY the provided context.\n"
                f"You may ONLY cite these pages: {allowed_pages_str}.\n"
                "For each concrete claim, add (page X).\n"
                "If not stated, say: 'Not stated in the provided pages.'"
            ),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}")
        ]

        # Retry + fallback
        tries = 2
        for t in range(tries):
            try:
                ans = self.llm.invoke(msgs).content
                return {"mode": mode, "answer": ans, "confidence": overall_conf, "coverage": cov, "pages": allowed_pages, "top_matches": top_matches}
            except Exception as e:
                logger.exception(f"LLM call failed (try {t+1}/{tries}): {e}")

        return {"mode": "refused", "answer": "LLM failed repeatedly; refusing to answer safely.", "confidence": overall_conf, "coverage": cov, "pages": allowed_pages, "top_matches": top_matches}
