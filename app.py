"""
app.py - Streamlit UI for the Document Q&A RAG system.

Run with:
    streamlit run app.py

This talks to the FastAPI backend over HTTP, so start that first:
    uvicorn api:app --reload --port 8000
"""

import requests
import streamlit as st

st.set_page_config(page_title="Document Q&A RAG", layout="wide")

st.sidebar.title("Settings")
base_url = st.sidebar.text_input("API base URL", value="http://localhost:8000")

if st.sidebar.button("(Re)build index from documents/"):
    with st.spinner("Ingesting, chunking, embedding..."):
        try:
            r = requests.post(f"{base_url}/ingest", timeout=120)
            r.raise_for_status()
            st.sidebar.success(r.json())
        except Exception as e:
            st.sidebar.error(f"Ingest failed: {e}")

try:
    doc_ids = requests.get(f"{base_url}/documents", timeout=10).json().get("doc_ids", [])
except Exception:
    doc_ids = []

st.sidebar.markdown("### Indexed documents")
if doc_ids:
    for d in doc_ids:
        st.sidebar.write(f"- {d}")
else:
    st.sidebar.info("No documents indexed yet. Click the button above.")

st.title("Document Q&A with Citations")

tab_ask, tab_contradict = st.tabs(["Ask a question", "Check for contradictions"])

with tab_ask:
    st.write("Ask a question in any language. If the documents don't cover it, you'll be told explicitly.")
    question = st.text_area("Your question", height=80, placeholder="e.g. How many PTO days do employees get per year?")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=8, value=4)

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating grounded answer..."):
                try:
                    r = requests.post(f"{base_url}/ask", json={"question": question, "top_k": top_k}, timeout=60)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    result = None
                    st.error(f"Request failed: {e}")

            if result:
                st.markdown(f"**Detected language:** {result['detected_language']}")

                if result["needs_human_review"]:
                    st.warning(
                        f"⚠️ Confidence is {result['confidence']} (below threshold). "
                        "This answer is flagged for human review."
                    )

                if result["insufficient_context"]:
                    st.info(result["answer"])
                else:
                    st.markdown("### Answer")
                    st.write(result["answer"])
                    st.markdown(f"**Confidence:** {result['confidence']}")

                if result["citations"]:
                    st.markdown("### Citations")
                    for c in result["citations"]:
                        with st.expander(f"{c['source_file']} — {c['chunk_reference']}"):
                            st.write(c["snippet"])

with tab_contradict:
    st.write("Pick two documents and check whether they conflict on any topic.")
    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.selectbox("Document 1", options=doc_ids, key="doc1")
    with col2:
        doc2 = st.selectbox("Document 2", options=doc_ids, index=min(1, max(len(doc_ids) - 1, 0)), key="doc2")

    if st.button("Check for contradictions"):
        if not doc1 or not doc2:
            st.warning("No documents indexed yet.")
        elif doc1 == doc2:
            st.warning("Please choose two different documents.")
        else:
            with st.spinner("Comparing documents..."):
                try:
                    r = requests.post(
                        f"{base_url}/contradict",
                        json={"doc_id_1": doc1, "doc_id_2": doc2},
                        timeout=60,
                    )
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    result = None
                    st.error(f"Request failed: {e}")

            if result:
                if result["conflict"]:
                    st.error(f"⚠️ Conflict found on topic: **{result['topic']}**")
                else:
                    st.success(f"No conflict found. (Topic compared: {result['topic'] or 'n/a'})")
                st.markdown("**Reasoning:**")
                st.write(result["explanation"])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Evidence — {result['source_file_1']}**")
                    st.write(result["evidence_doc1"] or "n/a")
                with c2:
                    st.markdown(f"**Evidence — {result['source_file_2']}**")
                    st.write(result["evidence_doc2"] or "n/a")
