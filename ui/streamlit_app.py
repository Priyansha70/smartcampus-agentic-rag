import requests
import streamlit as st

st.set_page_config(page_title="SmartCampus Agentic RAG", layout="wide")

st.title("SmartCampus Agentic RAG")
st.caption("PDF QA / Summary / Quiz with guardrails (confidence + coverage + refusal).")

API_URL = st.text_input("Backend API URL", value="http://127.0.0.1:8000/query")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])
mode = st.selectbox("Mode", ["qa", "summary", "quiz"])
question = st.text_input("Question")

col1, col2 = st.columns([1, 1])

if st.button("Run") and pdf and question:
    with st.spinner("Querying..."):
        files = {"pdf": (pdf.name, pdf.getvalue(), "application/pdf")}
        data = {"question": question, "mode": mode}

        r = requests.post(API_URL, data=data, files=files, timeout=180)
        r.raise_for_status()
        res = r.json()

    with col1:
        st.subheader("Answer")
        st.write(res.get("answer", ""))

    with col2:
        st.subheader("Guard")
        st.json({
            "mode": res.get("mode"),
            "confidence": res.get("confidence"),
            "coverage": res.get("coverage"),
            "pages": res.get("pages"),
        })

    st.subheader("Top Evidence (previews)")
    st.json(res.get("top_matches", [])[:5])
