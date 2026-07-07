"""
rag_core.py
Core logic for the Document Q&A RAG system:
  - chunking
  - embedding + vector store (Chroma)
  - retrieval (+ optional cross-encoder reranking)
  - LLM calls (Groq, OpenAI-compatible /chat/completions API)
  - multilingual detect/translate boundary
  - grounded answer generation with citations + confidence
  - contradiction check between two documents

Nothing in here talks to Streamlit or FastAPI directly - both api.py and
app.py import functions from this module.
"""

import os
import re
import json
import glob
import requests
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "docs"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 1000        # target characters per chunk
CHUNK_OVERLAP = 200      # characters of overlap between consecutive chunks

TOP_K_RETRIEVE = 8       # candidates pulled from the vector store
TOP_K_FINAL = 4          # chunks actually shown to the LLM / user after (optional) rerank
CONFIDENCE_THRESHOLD = 0.5   # below this -> flag for human review

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"

# --------------------------------------------------------------------------
# Lazy singletons (loaded on first use so importing this module is cheap)
# --------------------------------------------------------------------------
_embed_model = None
_reranker = None
_chroma_client = None
_collection = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_reranker():
    global _reranker
    if _reranker is None and USE_RERANKER:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Sliding-window character chunker that snaps chunk boundaries to the
    nearest sentence/paragraph end within a small lookahead window, so
    chunks don't cut off mid-sentence. Overlap keeps context continuous
    across chunk boundaries so an answer that straddles two chunks is
    still retrievable. Returns a list of {text, start, end}.
    """
    text = text.strip()
    chunks = []
    length = len(text)
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window_end = min(end + 200, length)
            boundary = -1
            for sep in [".\n", "\n\n", ". ", "\n"]:
                idx = text.rfind(sep, start, window_end)
                if idx > boundary:
                    boundary = idx + len(sep)
            if boundary > start:
                end = boundary
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append({"text": chunk_str, "start": start, "end": end})
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def build_index(docs_dir=DOCS_DIR, reset=True):
    """
    Reads every .txt file in docs_dir, chunks it, embeds the chunks, and
    (re)builds the Chroma collection. doc_id is the filename without
    extension - this is what /contradict takes as input.
    """
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    model = get_embed_model()

    all_ids, all_texts, all_metas, all_embeds = [], [], [], []
    file_paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    for path in file_paths:
        filename = os.path.basename(path)
        doc_id = os.path.splitext(filename)[0]
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        chunks = chunk_text(raw)
        for i, c in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk_{i}"
            all_ids.append(chunk_id)
            all_texts.append(c["text"])
            all_metas.append({
                "doc_id": doc_id,
                "source_file": filename,
                "chunk_index": i,
                "num_chunks": len(chunks),
                "char_start": c["start"],
                "char_end": c["end"],
            })

    if not all_texts:
        return {"status": "no documents found", "files_indexed": 0, "chunks_indexed": 0}

    embeddings = model.encode(all_texts, show_progress_bar=False).tolist()
    collection.add(ids=all_ids, documents=all_texts, metadatas=all_metas, embeddings=embeddings)

    global _collection, _chroma_client
    _chroma_client = client
    _collection = collection

    return {
        "status": "ok",
        "files_indexed": len(file_paths),
        "chunks_indexed": len(all_ids),
        "doc_ids": sorted({m["doc_id"] for m in all_metas}),
    }


def list_doc_ids():
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    return sorted({m["doc_id"] for m in data["metadatas"]}) if data["metadatas"] else []


# --------------------------------------------------------------------------
# LLM helper
# --------------------------------------------------------------------------
def call_llm(messages, json_mode=False, temperature=0.0, max_tokens=1200):
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file (see .env.example)."
        )
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _safe_json_parse(raw, fallback):
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return fallback


# --------------------------------------------------------------------------
# Multilingual boundary
# --------------------------------------------------------------------------
def detect_and_translate_to_english(query):
    """
    Single LLM call: detect the query's language and translate to English.
    Returns dict {language_name, language_code, translated_query}.
    If the query is already English, translated_query == query.
    """
    prompt = (
        "Identify the natural language of the USER_QUERY below and translate it "
        "into English. Respond ONLY with a JSON object with keys "
        "\"language_name\", \"language_code\" (ISO 639-1 if possible), and "
        "\"translated_query\" (the English translation; if the query is already "
        "English, repeat it unchanged).\n\n"
        f"USER_QUERY: {query}"
    )
    raw = call_llm([{"role": "user", "content": prompt}], json_mode=True)
    result = _safe_json_parse(raw, {
        "language_name": "English", "language_code": "en", "translated_query": query
    })
    result.setdefault("translated_query", query)
    result.setdefault("language_name", "English")
    result.setdefault("language_code", "en")
    return result


def translate_from_english(text, target_language_name):
    if target_language_name.strip().lower() == "english":
        return text
    prompt = (
        f"Translate the following text into {target_language_name}. "
        "Respond with ONLY the translation, no preamble.\n\n"
        f"TEXT:\n{text}"
    )
    return call_llm([{"role": "user", "content": prompt}], json_mode=False).strip()


# --------------------------------------------------------------------------
# Retrieval (+ optional rerank)
# --------------------------------------------------------------------------
def retrieve(query_en, top_k_retrieve=TOP_K_RETRIEVE, top_k_final=TOP_K_FINAL, doc_id_filter=None):
    collection = get_collection()
    model = get_embed_model()
    query_embedding = model.encode([query_en]).tolist()

    where = {"doc_id": {"$in": doc_id_filter}} if doc_id_filter else None
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k_retrieve,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    if results["documents"]:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            candidates.append({
                "text": doc,
                "metadata": meta,
                "vector_score": 1 - dist,  # cosine distance -> similarity
            })

    if not candidates:
        return []

    reranker = get_reranker() if USE_RERANKER else None
    if reranker is not None:
        pairs = [[query_en, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    else:
        candidates.sort(key=lambda c: c["vector_score"], reverse=True)

    return candidates[:top_k_final]


# --------------------------------------------------------------------------
# Answer generation (grounded, with citations + confidence)
# --------------------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = """You are a careful document Q&A assistant. You must answer
ONLY using the numbered CONTEXT chunks provided. Rules:
1. If the context does not contain enough information to answer the question,
   set "insufficient_context" to true and leave "answer" as a short statement
   that the documents do not cover this.
2. Never use outside knowledge. Never guess or invent facts not present in the
   context.
3. Every factual sentence in "answer" must be traceable to one or more of the
   numbered context chunks. Reference them by number in "cited_chunks".
4. "confidence" is a number from 0 to 1 reflecting how directly and completely
   the context supports the answer (1 = fully and explicitly supported,
   0 = essentially no support).
Respond ONLY with a JSON object with keys:
"answer" (string), "insufficient_context" (boolean), "cited_chunks" (array of
integers referring to the CONTEXT chunk numbers actually used),
"confidence" (number between 0 and 1)."""


def answer_question(query, top_k=TOP_K_FINAL):
    lang = detect_and_translate_to_english(query)
    query_en = lang["translated_query"]

    chunks = retrieve(query_en, top_k_final=top_k)

    if not chunks:
        answer_en = "The documents do not contain information to answer this question."
        result = {
            "answer": translate_from_english(answer_en, lang["language_name"]),
            "insufficient_context": True,
            "confidence": 0.0,
            "needs_human_review": True,
            "detected_language": lang["language_name"],
            "citations": [],
        }
        return result

    context_block = "\n\n".join(
        f"[{i+1}] (source: {c['metadata']['source_file']}, chunk {c['metadata']['chunk_index']}/"
        f"{c['metadata']['num_chunks']-1}):\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {query_en}"
    raw = call_llm(
        [{"role": "system", "content": ANSWER_SYSTEM_PROMPT},
         {"role": "user", "content": user_prompt}],
        json_mode=True,
    )
    parsed = _safe_json_parse(raw, {
        "answer": "The documents do not contain information to answer this question.",
        "insufficient_context": True,
        "cited_chunks": [],
        "confidence": 0.0,
    })

    cited_indices = [i for i in parsed.get("cited_chunks", []) if isinstance(i, int) and 1 <= i <= len(chunks)]
    citations = []
    for i in cited_indices:
        c = chunks[i - 1]
        snippet = c["text"][:280] + ("..." if len(c["text"]) > 280 else "")
        citations.append({
            "source_file": c["metadata"]["source_file"],
            "doc_id": c["metadata"]["doc_id"],
            "chunk_reference": f"chunk {c['metadata']['chunk_index']} of {c['metadata']['num_chunks']} "
                               f"(chars {c['metadata']['char_start']}-{c['metadata']['char_end']})",
            "snippet": snippet,
        })

    insufficient = bool(parsed.get("insufficient_context", False))
    confidence = float(parsed.get("confidence", 0.0))
    answer_en = parsed.get("answer", "").strip() or "The documents do not contain information to answer this question."

    # Extra safety net: if the model says insufficient, don't let stray citations imply otherwise.
    if insufficient:
        citations = []
        confidence = min(confidence, 0.2)

    final_answer = translate_from_english(answer_en, lang["language_name"])

    return {
        "answer": final_answer,
        "insufficient_context": insufficient,
        "confidence": round(confidence, 2),
        "needs_human_review": confidence < CONFIDENCE_THRESHOLD or insufficient,
        "detected_language": lang["language_name"],
        "citations": citations,
    }


# --------------------------------------------------------------------------
# Contradiction check between two documents
# --------------------------------------------------------------------------
CONTRADICT_SYSTEM_PROMPT = """You compare two policy/document excerpts and determine
whether they conflict with each other on any topic they both address. Be precise:
only flag a real substantive conflict (e.g. different numeric limits, opposite
rules, incompatible requirements), not superficial wording differences or topics
only one document covers. Respond ONLY with a JSON object with keys:
"conflict" (boolean), "topic" (short string describing the topic being compared,
or empty string if there is no overlap at all), "explanation" (string explaining
the reasoning), "evidence_doc1" (short quote or paraphrase from document 1
supporting your conclusion, or empty string), "evidence_doc2" (same, for
document 2)."""


def _get_full_document_text(doc_id):
    collection = get_collection()
    data = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    if not data["documents"]:
        return None, []
    pairs = sorted(zip(data["metadatas"], data["documents"]), key=lambda p: p[0]["chunk_index"])
    metas = [p[0] for p in pairs]
    full_text = "\n".join(p[1] for p in pairs)
    return full_text, metas


def contradiction_check(doc_id_1, doc_id_2, max_chars_per_doc=6000):
    text1, metas1 = _get_full_document_text(doc_id_1)
    text2, metas2 = _get_full_document_text(doc_id_2)

    if text1 is None or text2 is None:
        missing = [d for d, t in [(doc_id_1, text1), (doc_id_2, text2)] if t is None]
        return {
            "error": f"Unknown document id(s): {', '.join(missing)}. "
                     f"Call /documents (or list_doc_ids()) for valid ids."
        }

    text1 = text1[:max_chars_per_doc]
    text2 = text2[:max_chars_per_doc]

    source1 = metas1[0]["source_file"] if metas1 else doc_id_1
    source2 = metas2[0]["source_file"] if metas2 else doc_id_2

    prompt = (
        f"DOCUMENT 1 (id: {doc_id_1}, file: {source1}):\n{text1}\n\n"
        f"DOCUMENT 2 (id: {doc_id_2}, file: {source2}):\n{text2}\n\n"
        "Compare these two documents and determine if they conflict."
    )
    raw = call_llm(
        [{"role": "system", "content": CONTRADICT_SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        json_mode=True,
    )
    parsed = _safe_json_parse(raw, {
        "conflict": False, "topic": "", "explanation": "Could not parse model output.",
        "evidence_doc1": "", "evidence_doc2": "",
    })

    return {
        "doc_id_1": doc_id_1,
        "doc_id_2": doc_id_2,
        "source_file_1": source1,
        "source_file_2": source2,
        "conflict": bool(parsed.get("conflict", False)),
        "topic": parsed.get("topic", ""),
        "explanation": parsed.get("explanation", ""),
        "evidence_doc1": parsed.get("evidence_doc1", ""),
        "evidence_doc2": parsed.get("evidence_doc2", ""),
    }
