# 🚀 SmartCampus Agentic RAG

A production-style Retrieval-Augmented Generation (RAG) system with guardrails, refusal logic, and API deployment.

This project transforms a basic notebook RAG implementation into a deployable AI service using FastAPI, retrieval confidence scoring, keyword coverage validation, and hallucination prevention.

---

## 🔥 Key Features

- ✅ FastAPI backend for real API deployment  
- ✅ Swagger auto-generated API documentation  
- ✅ FAISS vector search for semantic retrieval  
- ✅ Embedding caching (prevents re-indexing repeated PDFs)  
- ✅ Retrieval confidence scoring  
- ✅ Keyword coverage validation  
- ✅ Hallucination refusal logic  
- ✅ Retry + fallback mechanism  
- ✅ Environment-based configuration  
- ✅ Production-ready project structure  

---

## 🧠 Problem Statement

Large Language Models can hallucinate when asked questions not supported by context.

This system enforces:

1. Retrieval grounding
2. Confidence scoring
3. Keyword coverage validation
4. Strict page-based citation enforcement
5. Automatic refusal when unsafe

Result:  
A safer, more production-ready RAG architecture.

---

## 🏗 System Architecture

1. User uploads PDF
2. PDF is chunked via `RecursiveCharacterTextSplitter`
3. Chunks are embedded using OpenAI embeddings
4. FAISS performs semantic similarity search
5. Guard layer computes:
   - Retrieval confidence
   - Keyword coverage
6. If safe → LLM answers using ONLY retrieved context
7. If unsafe → system refuses with explanation

---

## 🛡 Guard Layer

The guard prevents hallucination using:

- Top-K retrieval confidence scoring
- Keyword overlap validation
- Page-level citation enforcement
- Strict refusal thresholding

Example refusal:

