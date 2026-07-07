# Document Q&A with Citations (RAG)

A small, no-hallucination RAG system over 5 sample policy documents. Built to
spec for a 24-hour build: local embeddings + Chroma vector store, Groq for the
LLM (free tier), FastAPI for the endpoints, Streamlit for the UI.

## What's included

- `documents/` — 5 sample `.txt` policy documents (see "Sample documents" below)
- `rag_core.py` — chunking, embedding, retrieval, reranking, LLM calls, translation, grounded answer generation, contradiction check
- `api.py` — FastAPI app: `/ask`, `/contradict`, `/documents`, `/ingest`
- `app.py` — Streamlit UI that talks to the API
- `eval.py` + `eval_set.json` — retrieval@k eval over 10 Q&A pairs (stretch goal)
- `.env.example` — copy to `.env` and add your Groq API key

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com/keys (no credit card
needed for the free tier). The first run will also download two small
open-source models from Hugging Face (embedding model + reranker), so you'll
need normal internet access the first time.

## Running it

1. Start the API:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
2. Build the index (either curl it once, or use the "Rebuild index" button in
   the Streamlit sidebar):
   ```bash
   curl -X POST http://localhost:8000/ingest
   ```
3. Start the UI:
   ```bash
   streamlit run app.py
   ```

## Chunking strategy

Each document is chunked with a **character-based sliding window with
sentence-boundary snapping and overlap**:

- Target chunk size: **1000 characters**, overlap: **200 characters**.
- The window doesn't cut off mid-sentence: when a chunk boundary would land
  inside a sentence, it looks ahead up to 200 extra characters for the
  nearest sentence or paragraph end (`. `, `.\n`, `\n\n`) and snaps to that
  instead. This keeps each chunk a coherent, readable unit rather than an
  arbitrary character slice.
- The 200-character overlap means an answer that straddles a chunk boundary
  (e.g. a rule stated in one sentence, its exception in the next) is still
  fully retrievable from at least one chunk.
- These policy documents are short (a few hundred to ~2000 words each), so
  1000 characters is roughly one policy section per chunk — small enough for
  precise citations, large enough to keep clauses and their caveats together.
- Every chunk keeps metadata: `source_file`, `doc_id`, `chunk_index`,
  `num_chunks`, and the original `char_start`/`char_end` offsets, which is
  what powers the citation fields returned by `/ask`.

Chunks are embedded with a local, free sentence-transformer
(`all-MiniLM-L6-v2`) and stored in a persistent Chroma collection — no paid
embedding API required.

## Retrieval

1. Query is embedded and the top 8 candidate chunks are pulled from Chroma
   (cosine similarity).
2. Those 8 are reranked with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) and
   the top 4 are kept. This is the optional reranker stretch goal — set
   `USE_RERANKER=false` in `.env` to disable it and fall back to plain vector
   similarity ranking.

## No silent hallucination

The answer-generation prompt requires the model to only use the numbered
context chunks it's given, to flag `insufficient_context: true` when the
docs don't cover the question, and to report a `confidence` score. The API
enforces this in code, too: if the model marks a question as
under-supported, citations are stripped and confidence is capped low — so a
low-confidence, unsupported answer can never sneak out looking authoritative.

- If confidence is below the threshold (default **0.5**) or context is
  insufficient, the response sets `"needs_human_review": true`. This is the
  human-in-the-loop gate (stretch goal) — the Streamlit UI surfaces it as a
  visible warning banner rather than silently showing an answer.

## Multilingual flow

`/ask` runs a translation step at the boundary:
1. One LLM call detects the query's language and translates it to English.
2. Retrieval and answer generation happen in English (documents are English).
3. If the detected language wasn't English, the final answer is translated
   back into that language before being returned.

The detected language name is included in the response so the UI can show
what was detected.

## `/contradict`

Takes two `doc_id`s (the filename without extension, e.g.
`remote_work_policy_2023`). Pulls the full text of both documents from the
store and asks the LLM to judge whether they substantively conflict on a
shared topic, returning `conflict` (bool), `topic`, `explanation`, and a short
evidence quote from each document. Use `/documents` to see valid ids.

Try `remote_work_policy_2023` vs `remote_work_policy_2024_addendum` — these
two sample documents are written to genuinely conflict (in-office attendance
requirement vs. unrestricted fully-remote work), so this pair is a good
demo of a detected conflict. Any other pair of sample docs should come back
as no conflict.

## Sample documents

Five short HR-style policy documents (fictional, for demo purposes):

1. `remote_work_policy_2023.txt` — original office-attendance policy
2. `remote_work_policy_2024_addendum.txt` — later policy that contradicts #1
3. `expense_reimbursement_policy.txt`
4. `data_privacy_policy.txt`
5. `leave_pto_policy.txt`

You can drop your own `.txt` files into `documents/` and re-run `/ingest` —
nothing else needs to change.

## Eval (stretch goal)

```bash
python eval.py
```

Runs the 10 questions in `eval_set.json` through retrieval only, and reports
whether the expected source document appears among the top-k retrieved
chunks (retrieval@k accuracy).

## Notes / limitations (honest, 24-hour-build scope)

- LLM: Groq (`llama-3.3-70b-versatile` by default) — swap `GROQ_MODEL` or
  point `call_llm()` in `rag_core.py` at a different OpenAI-compatible
  endpoint if you'd rather use OpenAI or Gemini's OpenAI-compatible endpoint.
- Contradiction check sends full document text (truncated to 6000 chars per
  doc) rather than doing a separate topic-alignment retrieval step first —
  fine for short policy docs, would need chunk-level topic matching first
  for much longer documents.
- Sentence-splitting for chunk boundaries is a simple punctuation-based
  heuristic, not a full sentence tokenizer — good enough for plain-prose
  policy text.
