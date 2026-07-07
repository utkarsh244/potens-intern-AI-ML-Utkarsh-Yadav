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
| _( Claude)_ | _( ~15 messages)_ | _( scaffolding chunking logic, debugging Chroma filters)_ |
| _( ChatGPT)_ | _( ~5 messages)_ | _( drafting the contradiction-check prompt)_ |


##screenshots 
1.showing the multilingual outputs with the chunks to retrieve,confidence level and citation.
<img width="1912" height="1023" alt="Screenshot 2026-07-07 201523" src="https://github.com/user-attachments/assets/25299205-b5e3-42ec-b885-0dbd9ed81e5b" />
<img width="1915" height="1013" alt="Screenshot 2026-07-07 201352" src="https://github.com/user-attachments/assets/a90fb0da-7d2c-4c6c-af2f-85653ae09f72" />
2.If the docs do not cover the related question, the system  say so explicitly.
<img width="1906" height="973" alt="Screenshot 2026-07-07 201722" src="https://github.com/user-attachments/assets/f1a38cf7-c2d3-4696-8985-66dbb2205bba" />
3./contradict endpoint that takes two document IDs and returns whether they conflict on a topic, with reasoning
<img width="1917" height="1017" alt="Screenshot 2026-07-07 201544" src="https://github.com/user-attachments/assets/44ec0850-3590-4ad8-8675-9819ce14fa21" />
<img width="1916" height="1026" alt="Screenshot 2026-07-07 201613" src="https://github.com/user-attachments/assets/b7aa5461-177e-4ec7-b1f2-1a6d4fc69524" />

