"""
Evaluation script for the Advanced Hybrid RAG System.

Computes real retrieval-quality metrics — Recall@K and Mean
Reciprocal Rank (MRR) — against a small labeled test set
(evaluation/qa_pairs.json). This is a genuine offline evaluation
of retrieval quality, not just "it looks right to me."

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --dataset evaluation/qa_pairs.json
    python evaluation/run_eval.py --top-k 5
"""

import json
import argparse
from pathlib import Path

from retrieval.advanced_search import advanced_search, metadata


def load_qa_pairs(path):

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


def evaluate(qa_pairs, top_k=5):
    """
    For each labeled question, run retrieval and check whether
    any of the top-K returned chunks come from one of the
    question's known-relevant source files.

    Returns per-question results plus aggregate Recall@K and MRR.
    """

    results = []

    hits = 0

    reciprocal_ranks = []

    for pair in qa_pairs:

        question = pair["question"]

        relevant_files = set(
            pair.get("relevant_source_files", [])
        )

        if not relevant_files:

            # Skip unlabeled/placeholder entries rather than
            # silently counting them as failures
            continue

        retrieved = advanced_search(question, top_k_final=top_k)

        retrieved_files = [
            metadata[doc_id]["source_file"]
            for doc_id, _ in retrieved
        ]

        # Find rank of first relevant hit (1-indexed) for MRR
        rank_of_first_hit = None

        for i, source_file in enumerate(retrieved_files, start=1):

            if source_file in relevant_files:

                rank_of_first_hit = i

                break

        is_hit = rank_of_first_hit is not None

        if is_hit:

            hits += 1

            reciprocal_ranks.append(1.0 / rank_of_first_hit)

        else:

            reciprocal_ranks.append(0.0)

        results.append(
            {
                "question": question,
                "expected_sources": list(relevant_files),
                "retrieved_sources": retrieved_files,
                "hit": is_hit,
                "rank_of_first_hit": rank_of_first_hit
            }
        )

    total = len(results)

    recall_at_k = (hits / total) if total > 0 else 0.0

    mrr = (
        sum(reciprocal_ranks) / total
        if total > 0 else 0.0
    )

    summary = {
        "total_questions": total,
        "top_k": top_k,
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "hits": hits,
        "misses": total - hits
    }

    return summary, results


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality"
    )

    parser.add_argument(
        "--dataset",
        default="evaluation/qa_pairs.json",
        help="Path to labeled QA pairs JSON file"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question"
    )

    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.dataset)

    print("\n" + "=" * 60)
    print("RUNNING RAG EVALUATION")
    print("=" * 60)
    print(f"Dataset : {args.dataset}")
    print(f"Top-K   : {args.top_k}")
    print(f"Questions: {len(qa_pairs)}")
    print("=" * 60 + "\n")

    summary, results = evaluate(qa_pairs, top_k=args.top_k)

    for r in results:

        status = "HIT " if r["hit"] else "MISS"

        print(f"[{status}] {r['question']}")

        print(f"   Expected : {r['expected_sources']}")
        print(f"   Retrieved: {r['retrieved_sources']}")

        if r["hit"]:

            print(f"   Rank of first hit: {r['rank_of_first_hit']}")

        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Questions : {summary['total_questions']}")
    print(f"Recall@{summary['top_k']}       : {summary['recall_at_k']}")
    print(f"MRR             : {summary['mrr']}")
    print(f"Hits / Misses   : {summary['hits']} / {summary['misses']}")
    print("=" * 60)

    # Save results to disk so the dashboard API can read them
    # without re-running retrieval every time someone opens it
    output_path = Path("evaluation/last_run_results.json")

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            {
                "summary": summary,
                "results": results
            },
            f,
            indent=2
        )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":

    main()