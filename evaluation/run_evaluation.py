"""
evaluation/run_evaluation.py

Runs all questions in evaluation/questions.json against the live RAG system.
Records retrieval correctness and refusal behavior.
Writes evaluation/evaluation_results.csv.

REQUIRES:
- Ollama running with llama3.2:3b and nomic-embed-text
- Documents indexed in ChromaDB
- Human evaluators to fill in the score columns afterwards
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_service import ask_question
from config import UNKNOWN_ANSWER

QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "questions.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    questions = load_questions()
    rows = []

    for q in questions:
        print(f"[{q['id']}/50] {q['question']}")
        try:
            result = ask_question(q["question"])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = {"answer": f"ERROR: {exc}", "sources": [], "refused": False}

        retrieved_sources = [s["source"] for s in result["sources"]]
        expected = q.get("expected_source")

        if q["answerable"]:
            retrieval_correct = expected in retrieved_sources if expected else False
            correct_refusal = ""
        else:
            retrieval_correct = ""
            correct_refusal = (result["answer"].strip() == UNKNOWN_ANSWER)

        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_source": expected or "",
            "retrieved_sources": "; ".join(retrieved_sources),
            "retrieval_correct": retrieval_correct,
            "answer": result["answer"],
            "refused": result["refused"],
            "correct_refusal": correct_refusal,
            "score_correctness": "",
            "score_relevance": "",
            "score_groundedness": "",
            "score_clarity": "",
            "score_completeness": "",
        })

    # Write CSV
    fieldnames = [
        "id", "category", "question", "expected_source", "retrieved_sources",
        "retrieval_correct", "answer", "refused", "correct_refusal",
        "score_correctness", "score_relevance", "score_groundedness",
        "score_clarity", "score_completeness",
    ]
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Compute metrics
    answerable = [r for r in rows if r["category"] in ("direct", "paraphrased")]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]

    correct_retrievals = sum(1 for r in answerable if r["retrieval_correct"] is True)
    retrieval_accuracy = (correct_retrievals / len(answerable)) * 100 if answerable else 0

    correct_refusals = sum(1 for r in unanswerable if r["correct_refusal"] is True)
    safe_refusal_rate = (correct_refusals / len(unanswerable)) * 100 if unanswerable else 0

    print("\n=== Evaluation Summary ===")
    print(f"Total questions: {len(rows)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Unanswerable questions: {len(unanswerable)}")
    print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}%")
    print(f"Safe Refusal Rate: {safe_refusal_rate:.2f}%")
    print(f"\nResults written to: {RESULTS_FILE}")
    print("\nNEXT STEP: Have human evaluators fill in the score_* columns (1-5).")


if __name__ == "__main__":
    main()