import pickle
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

# =====================================================
# CONFIG
# =====================================================

TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 5
RRF_K = 60

# =====================================================
# PATHS
# =====================================================

INDEX_DIR = Path("data/indexes")

FAISS_PATH = INDEX_DIR / "rag_index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
BM25_PATH = INDEX_DIR / "bm25.pkl"

# =====================================================
# LOAD MODELS
# =====================================================

print("=" * 60)
print("Loading BGE-M3 Embedding Model...")
print("=" * 60)

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("\nLoading BGE Reranker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("\nLoading FAISS Index...")

faiss_index = faiss.read_index(
    str(FAISS_PATH)
)

print("\nLoading Metadata...")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("\nLoading BM25 Index...")

with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

print("\nSystem Ready!")
print(f"Chunks Loaded: {len(metadata)}")
print("=" * 60)

# =====================================================
# FAISS SEARCH
# =====================================================

def faiss_search(query, top_k=TOP_K_RETRIEVAL):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    return indices[0].tolist()

# =====================================================
# BM25 SEARCH
# =====================================================

def bm25_search(query, top_k=TOP_K_RETRIEVAL):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return top_indices.tolist()

# =====================================================
# RRF FUSION
# =====================================================

def reciprocal_rank_fusion(result_lists, rrf_k=RRF_K):

    scores = defaultdict(float)

    for result_list in result_lists:

        for rank, doc_id in enumerate(result_list):

            scores[doc_id] += 1.0 / (
                rrf_k + rank + 1
            )

    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results

# =====================================================
# RERANKING
# =====================================================

def rerank_results(query, candidate_ids, top_k_final=TOP_K_FINAL):

    candidate_texts = []

    for doc_id in candidate_ids:

        candidate_texts.append(
            metadata[doc_id]["chunk_text"]
        )

    pairs = [
        [query, text]
        for text in candidate_texts
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    results = []

    for doc_id, score in zip(
        candidate_ids,
        rerank_scores
    ):

        results.append(
            (
                doc_id,
                float(score)
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results[:top_k_final]

# =====================================================
# ADVANCED SEARCH PIPELINE
# =====================================================

def advanced_search(
    query,
    mode="hybrid",
    top_k_retrieval=None,
    top_k_final=None,
    rrf_k=None,
    use_reranker=True
):
    """
    mode: "hybrid" (FAISS+BM25+RRF), "faiss" (dense only),
          or "bm25" (sparse only)
    top_k_retrieval: candidates considered before reranking
    top_k_final: final number of chunks returned
    rrf_k: RRF damping constant (only used in hybrid mode)
    use_reranker: if False, skips cross-encoder reranking and
                  returns retrieval-stage ranking directly
                  (results will use retrieval rank, not a
                  rerank score, since rerank is skipped)
    """

    retrieval_k = top_k_retrieval or TOP_K_RETRIEVAL
    final_k = top_k_final or TOP_K_FINAL
    fusion_k = rrf_k or RRF_K

    if mode == "faiss":

        print("\nRunning FAISS-only Search...")

        candidate_ids = faiss_search(
            query,
            top_k=retrieval_k
        )

    elif mode == "bm25":

        print("\nRunning BM25-only Search...")

        candidate_ids = bm25_search(
            query,
            top_k=retrieval_k
        )

    else:

        print("\nRunning FAISS Search...")

        faiss_results = faiss_search(
            query,
            top_k=retrieval_k
        )

        print("Running BM25 Search...")

        bm25_results = bm25_search(
            query,
            top_k=retrieval_k
        )

        print("Running RRF Fusion...")

        fused_results = reciprocal_rank_fusion(
            [
                faiss_results,
                bm25_results
            ],
            rrf_k=fusion_k
        )

        candidate_ids = [
            doc_id
            for doc_id, _
            in fused_results[:retrieval_k]
        ]

    if not candidate_ids:
        return []

    if use_reranker:

        print("Running Cross-Encoder Reranking...")

        final_results = rerank_results(
            query,
            candidate_ids,
            top_k_final=final_k
        )

    else:

        # No reranking — synthesize a descending pseudo-score
        # from retrieval rank so downstream confidence logic
        # (which expects (doc_id, score) tuples) still works.
        final_results = [
            (doc_id, round(1.0 - (i / max(len(candidate_ids), 1)), 4))
            for i, doc_id in enumerate(candidate_ids[:final_k])
        ]

    return final_results

# =====================================================
# DISPLAY RESULTS
# =====================================================

def display_results(results):

    print("\n")
    print("=" * 120)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 120)

    for rank, (doc_id, score) in enumerate(
        results,
        start=1
    ):

        chunk = metadata[doc_id]

        print(f"\nRank         : {rank}")
        print(f"Rerank Score : {score:.4f}")
        print(f"Source File  : {chunk['source_file']}")
        print(f"Chunk ID     : {chunk['chunk_id']}")
        print(f"Chunk Length : {chunk['chunk_length']}")

        print("\nPreview")
        print("-" * 120)

        preview = chunk["chunk_text"][:1000]

        print(preview)

        print("\n" + "=" * 120)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    greetings = {
        "hi",
        "hello",
        "hey",
        "good",
        "good morning",
        "good afternoon",
        "good evening"
    }

    print("\nAdvanced Hybrid Retrieval Ready")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Ask a question: ").strip()

        # Exit
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Empty query
        if not query:
            print("\nPlease enter a question.\n")
            continue

        # Greeting handling
        if query.lower() in greetings:
            print("\nHello! I am your Advanced RAG Assistant.")
            print("Ask me a question about the documents you uploaded.\n")
            continue

        # Very short query handling
        if len(query.split()) < 2:
            print(
                "\nPlease enter a more specific question.\n"
                "Example: 'What is cyber security?' or "
                "'Explain food preservation.'\n"
            )
            continue

        try:

            results = advanced_search(query)

            # No results
            if not results:
                print(
                    "\nNo relevant information found in the indexed documents.\n"
                )
                continue

            # Confidence threshold
            best_score = results[0][1]

            if best_score < 0.10:
                print(
                    "\nNo strong match found in the documents.\n"
                )
                continue

            display_results(results)

        except Exception as e:

            print("\nError occurred:")
            print(e)