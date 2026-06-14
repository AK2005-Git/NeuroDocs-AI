from ollama import chat

from retrieval.advanced_search import (
    advanced_search,
    metadata
)

from memory.chat_memory import ChatMemory
from analytics.stats import Analytics

import time

# ============================================================
# CONFIG
# ============================================================

MAX_CONTEXT_CHUNKS = 5
MIN_RELEVANCE_SCORE = 0.10

# ============================================================
# GLOBALS
# ============================================================

memory = ChatMemory()
analytics = Analytics()

# ============================================================
# MEMORY
# ============================================================

def build_history():

    recent_messages = memory.get_recent(5)

    history_text = ""

    for msg in recent_messages:

        history_text += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    return history_text

# ============================================================
# CONTEXT
# ============================================================

def build_context(results):

    context_parts = []
    sources = []

    count = 0

    for doc_id, score in results:

        if score < MIN_RELEVANCE_SCORE:
            continue

        chunk = metadata[doc_id]

        context_parts.append(
            chunk["chunk_text"]
        )

        sources.append(
            {
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 4)
            }
        )

        count += 1

        if count >= MAX_CONTEXT_CHUNKS:
            break

    context = "\n\n".join(
        context_parts
    )

    return context, sources

# ============================================================
# MAIN RAG
# ============================================================

def ask_rag(question):

    if not question.strip():

        return {
            "answer": "Please enter a valid question.",
            "sources": [],
            "response_time": 0
        }

    analytics.increment()

    start_time = time.time()

    memory.add_user(question)

    retrieved_results = advanced_search(
        question
    )

    if not retrieved_results:

        return {
            "answer": "No relevant information found.",
            "sources": [],
            "response_time": 0
        }

    context, sources = build_context(
        retrieved_results
    )

    if not context:

        return {
            "answer": "No sufficiently relevant information found.",
            "sources": [],
            "response_time": 0
        }

    history = build_history()

    prompt = f"""
You are an expert Retrieval Augmented Generation assistant.

Rules:

1. Answer ONLY using the provided context.
2. Do not invent facts.
3. If the answer is not available in the context, reply:
   "I could not find the answer in the documents."
4. Use conversation history when needed.
5. Keep answers concise and accurate.
6. Use bullet points when appropriate.

Conversation History:

{history}

Context:

{context}

Question:

{question}

Answer:
"""

    try:

        response = chat(
            model="mistral",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        memory.add_assistant(
            answer
        )

        response_time = round(
            time.time() - start_time,
            2
        )

        return {
            "answer": answer,
            "sources": sources,
            "response_time": response_time,
            "total_queries":
                analytics.get_total_queries()
        }

    except Exception as e:

        return {
            "answer": f"Ollama error: {str(e)}",
            "sources": [],
            "response_time": 0
        }

# ============================================================
# DISPLAY
# ============================================================

def display_sources(sources):

    if not sources:
        return

    print("\nSources:")
    print("-" * 60)

    seen = set()

    for source in sources:

        key = (
            source["source_file"],
            source["chunk_id"]
        )

        if key in seen:
            continue

        seen.add(key)

        print(
            f"- {source['source_file']} "
            f"(Chunk {source['chunk_id']}) "
            f"[Score: {source['score']}]"
        )

def display_confidence(sources):

    if not sources:
        return

    best_score = sources[0]["score"]

    if best_score >= 0.80:

        confidence = "High"

    elif best_score >= 0.40:

        confidence = "Medium"

    else:

        confidence = "Low"

    print(
        f"\nConfidence: {confidence} "
        f" ({best_score:.4f})"
    )

# ============================================================
# TERMINAL
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("Advanced RAG Assistant Ready")
    print("Type 'exit' to quit")
    print("=" * 70)

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":

            print("\nGoodbye!")
            break

        try:

            result = ask_rag(
                question
            )

            print("\nAnswer:")
            print("-" * 60)
            print(result["answer"])

            print(
                f"\nResponse Time: "
                f"{result['response_time']} sec"
            )

            print(
                f"Total Queries: "
                f"{result.get('total_queries', 0)}"
            )

            display_confidence(
                result["sources"]
            )

            display_sources(
                result["sources"]
            )

            print(
                "\n" + "=" * 100
            )

        except Exception as e:

            print("\nERROR:")
            print(str(e))