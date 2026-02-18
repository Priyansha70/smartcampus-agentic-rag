import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="SmartCampus Agentic RAG", layout="wide")
st.title("SmartCampus Agentic RAG")
st.caption("PDF QA / Summary / Quiz with a hallucination guard (confidence + coverage + refusal).")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])
mode = st.selectbox("Mode", ["qa", "summary", "quiz"])
question = st.text_input("Ask a question")

if st.button("Run") and pdf and question:
    with st.spinner("Querying..."):
        files = {"pdf": (pdf.name, pdf.getvalue(), "application/pdf")}
        data = {"question": question, "mode": mode}
        r = requests.post(API_URL, data=data, files=files, timeout=120)
        r.raise_for_status()
        res = r.json()

    st.subheader("Answer")
    st.write(res["answer"])

    st.subheader("Guard")
    st.json({k: res.get(k) for k in ["mode", "confidence", "coverage", "pages"]})

    st.subheader("Top Evidence")
    st.json(res.get("top_matches", [])[:5])
