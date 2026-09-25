"""
SourceGuard RAG — retrieval-only 20-question evaluation.

This version deliberately DOES NOT call the Hugging Face LLM.

Why:
The Hugging Face account currently returns HTTP 402 because the
monthly included inference credits are exhausted. The assignment's
retrieval/abstention behavior can still be measured independently.

Run:
    python evaluate.py

This produces:
    evaluation_results.json

It also performs a threshold sweep so 0.55 is not treated as a
final threshold without measurement.
"""

import json
import statistics
from pathlib import Path
from datetime import datetime

from query2 import run_retrieval


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "questions.json"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"

THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
]


def load_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def unique_urls(result):
    urls = []
    for source in result.get("sources") or []:
        url = source.get("url")
        if url and url not in urls:
            urls.append(url)
    return urls


def expected_sources_found(question, result):
    expected = set(question.get("expected_sources") or [])
    if not expected:
        return 0

    retrieved = set(unique_urls(result))
    return len(expected.intersection(retrieved))


def classify_at_threshold(score, threshold):
    if score is None:
        return True  # abstain
    return score < threshold


def evaluate():
    questions = load_questions()

    print("=" * 72)
    print("SOURCEGUARD RAG — RETRIEVAL-ONLY 20 QUESTION EVALUATION")
    print("=" * 72)
    print(f"Questions: {len(questions)}")
    print("LLM calls: 0")
    print("Hugging Face inference credits used: 0")
    print()

    records = []

    for question in questions:
        qid = question["id"]
        group = question["group"]
        text = question["question"]

        print(f"[{qid:02d}/20] {group}")
        print(f"Question: {text}")

        try:
            result = run_retrieval(text)

            score = result.get("retrieval_score")
            urls = unique_urls(result)
            sources_found = expected_sources_found(question, result)

            record = {
                "id": qid,
                "group": group,
                "question": text,
                "expected_behavior": question["expected_behavior"],
                "expected_sources": question.get("expected_sources", []),
                "top_score": score,
                "retrieved_source_urls": urls,
                "expected_source_urls_found": sources_found,
                "num_expected_sources": len(question.get("expected_sources", [])),
                "sources": result.get("sources", []),
                "retrieval_error": None,
            }

            records.append(record)

            print(
                "Top score:",
                f"{score:.4f}" if score is not None else "N/A"
            )
            print("Expected:", question["expected_behavior"])
            print("Expected sources found:", sources_found)
            print("-" * 72)

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            print("-" * 72)

            records.append(
                {
                    "id": qid,
                    "group": group,
                    "question": text,
                    "expected_behavior": question["expected_behavior"],
                    "expected_sources": question.get("expected_sources", []),
                    "top_score": None,
                    "retrieved_source_urls": [],
                    "expected_source_urls_found": 0,
                    "num_expected_sources": len(
                        question.get("expected_sources", [])
                    ),
                    "sources": [],
                    "retrieval_error": f"{type(exc).__name__}: {exc}",
                }
            )

    print_threshold_sweep(records)

    payload = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(records),
        "evaluation_mode": "retrieval_only",
        "llm_calls": 0,
        "notes": [
            "LLM generation was intentionally skipped.",
            "Hugging Face inference credits were exhausted.",
            "Threshold must be selected from measured retrieval results.",
            "Answer quality for answerable questions still requires LLM/manual review."
        ],
        "threshold_sweep": threshold_sweep(records),
        "results": records,
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print()
    print(f"Detailed results saved to: {RESULTS_FILE}")


def threshold_sweep(records):
    output = []

    for threshold in THRESHOLDS:
        valid = [r for r in records if r.get("retrieval_error") is None]

        answerable = [
            r for r in valid
            if r["expected_behavior"] == "answer"
        ]

        abstain_expected = [
            r for r in valid
            if r["expected_behavior"] == "abstain"
        ]

        answerable_pass = sum(
            1 for r in answerable
            if r["top_score"] is not None
            and r["top_score"] >= threshold
        )

        abstain_correct = sum(
            1 for r in abstain_expected
            if r["top_score"] is None
            or r["top_score"] < threshold
        )

        answerable_recall = (
            answerable_pass / len(answerable)
            if answerable else None
        )

        abstain_specificity = (
            abstain_correct / len(abstain_expected)
            if abstain_expected else None
        )

        balanced = (
            (answerable_recall + abstain_specificity) / 2
            if answerable_recall is not None
            and abstain_specificity is not None
            else None
        )

        output.append(
            {
                "threshold": threshold,
                "answerable_pass": answerable_pass,
                "answerable_total": len(answerable),
                "answerable_recall": answerable_recall,
                "abstain_correct": abstain_correct,
                "abstain_total": len(abstain_expected),
                "abstain_specificity": abstain_specificity,
                "balanced_retrieval_score": balanced,
            }
        )

    return output


def print_threshold_sweep(records):
    print()
    print("=" * 72)
    print("THRESHOLD SWEEP")
    print("=" * 72)
    print(
        "Threshold | Answerable kept | Answerable recall | "
        "Absent rejected | Absent specificity | Balanced"
    )

    for row in threshold_sweep(records):
        print(
            f"{row['threshold']:9.2f} | "
            f"{row['answerable_pass']:15}/{row['answerable_total']:<3} | "
            f"{row['answerable_recall']:.3f}             | "
            f"{row['abstain_correct']:14}/{row['abstain_total']:<3} | "
            f"{row['abstain_specificity']:.3f}              | "
            f"{row['balanced_retrieval_score']:.3f}"
        )

    print()
    print(
        "Do not blindly choose the highest balanced value. "
        "Inspect the individual questions and retrieved passages before "
        "declaring the final threshold."
    )


if __name__ == "__main__":
    evaluate()
