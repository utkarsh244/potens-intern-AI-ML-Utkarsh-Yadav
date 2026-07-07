# Document Q&A with Citations (RAG)

A no-hallucination RAG system over 5 sample HR policy docs — local embeddings + Chroma, Groq LLM (free tier), FastAPI backend, Streamlit UI. Answers are grounded strictly in retrieved chunks, with citations, a confidence score, and a human-review flag when the docs don't support a confident answer.

## Stack
`FastAPI` · `Streamlit` · `ChromaDB` · `sentence-transformers (all-MiniLM-L6-v2)` · `cross-encoder reranker` · `Groq (llama-3.3-70b)`

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY (free at console.groq.com/keys)

uvicorn api:app --reload --port 8000     # terminal 1 — API
curl -X POST http://localhost:8000/ingest # build the index (or use the UI button)
streamlit run app.py                      # terminal 2 — UI
```
Open the Streamlit URL it prints, ask a question, or try `/contradict` on `remote_work_policy_2023` vs `remote_work_policy_2024_addendum` (written to genuinely conflict).

## What it does

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Chunk + embed all `.txt` files in `documents/`, (re)build the Chroma index |
| `POST /ask` | `{question}` → grounded answer, citations, confidence, `needs_human_review` |
| `POST /contradict` | `{doc_id_1, doc_id_2}` → does the model find a substantive conflict between them |
| `GET /documents` | List indexed doc IDs |

Works in any language — `/ask` detects and translates the query to English, retrieves/answers in English, then translates the answer back.

## Key design decisions

- **Chunking:** 1000-char sliding window, 200-char overlap, snapped to the nearest sentence/paragraph boundary so chunks stay readable and boundary-straddling facts stay retrievable.
- **Retrieval:** top 8 by cosine similarity → reranked by a cross-encoder → top 4 sent to the LLM (`USE_RERANKER=false` to skip reranking).
- **Grounding:** the LLM must self-report `insufficient_context` and `confidence`; the API also enforces this in code — if context is insufficient, citations are stripped and confidence is force-capped, so a shaky answer can never look authoritative. Confidence `< 0.5` triggers a visible "needs human review" banner in the UI.

## Known limitations / not done

- No automated tests; checked manually via the UI and curl.
- `/ingest` always fully rebuilds the index — no incremental re-indexing.
- Chunk-boundary sentence splitting is a punctuation heuristic, not a real tokenizer.
- `/contradict` truncates each doc to 6000 chars — could miss conflicts past that point.
- No auth/rate-limiting — fine for local demo, not production-ready.

## Next up

Real sentence tokenization, incremental ingestion, chunk-level retrieval for `/contradict` on longer docs, basic API auth, and a test suite around chunking/retrieval/the JSON-parsing guardrails.

## AI Use Log

| Tool | Approx. usage | Used for |
|---|---|---|
| _(e.g. Claude)_ | _(e.g. ~15 messages)_ | _(e.g. scaffolding chunking logic, debugging Chroma filters)_ |
| _(e.g. ChatGPT)_ | _(e.g. ~5 messages)_ | _(e.g. drafting the contradiction-check prompt)_ |
