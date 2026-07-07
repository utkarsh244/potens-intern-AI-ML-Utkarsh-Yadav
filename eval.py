"""
eval.py - Retrieval@k evaluation against a small ground-truth Q&A set.

Run with:
    python eval.py

For each question in eval_set.json, checks whether the expected_doc_id
appears among the doc_ids of the top-k retrieved chunks. Prints per-question
hit/miss and an overall retrieval@k accuracy. Requires the index to already
be built (run build_index() / hit the /ingest endpoint first).
"""

import json
import rag_core


def main(top_k=rag_core.TOP_K_FINAL):
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    hits = 0
    for item in eval_set:
        question = item["question"]
        expected = item["expected_doc_id"]

        lang = rag_core.detect_and_translate_to_english(question)
        chunks = rag_core.retrieve(lang["translated_query"], top_k_final=top_k)
        retrieved_doc_ids = {c["metadata"]["doc_id"] for c in chunks}

        hit = expected in retrieved_doc_ids
        hits += int(hit)
        print(f"[{'HIT ' if hit else 'MISS'}] {question}")
        print(f"        expected={expected}  retrieved={sorted(retrieved_doc_ids)}")

    accuracy = hits / len(eval_set)
    print(f"\nRetrieval@{top_k} accuracy: {hits}/{len(eval_set)} = {accuracy:.2f}")


if __name__ == "__main__":
    main()
