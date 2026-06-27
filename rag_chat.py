import os

from groq import Groq

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

# Confidence routing thresholds — below this, warn the user
# instead of silently presenting a low-confidence answer as fact
LOW_CONFIDENCE_THRESHOLD = 0.40

# ============================================================
# GROQ CLIENT
# ============================================================
# The API key is read from the GROQ_API_KEY environment variable —
# never hardcoded here. Set it locally with:
#   export GROQ_API_KEY=your_key_here       (Mac/Linux)
#   $env:GROQ_API_KEY="your_key_here"        (Windows PowerShell)
# On your hosting platform (Render, Railway, etc.), set it in the
# dashboard's "Environment Variables" section instead.

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# Free-tier, fast, strong instruction-following — good default
# for RAG-style answering and the rewrite/groundedness checks below.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def call_groq(messages):
    """
    Thin wrapper around the Groq chat completion call so the
    rest of this file doesn't need to know about Groq's specific
    response shape. Mirrors the (role, content) message format
    used everywhere else in this file.
    """

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages
    )

    return response.choices[0].message.content

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
# QUERY REWRITING
# ============================================================
# Rephrases vague/contextual questions into clearer, standalone
# search queries BEFORE retrieval runs. Skips rewriting for
# already-clear, multi-word, first-turn questions to avoid
# unnecessary latency/risk on queries that don't need it.

MIN_WORDS_TO_SKIP_REWRITE = 6

def needs_rewriting(question, history_text):

    word_count = len(question.split())

    # Long, detailed questions are usually already search-ready
    if word_count >= MIN_WORDS_TO_SKIP_REWRITE and not history_text:

        return False

    return True


def rewrite_query(question, history_text):

    if not needs_rewriting(question, history_text):

        return question, False

    rewrite_prompt = f"""You rewrite chat questions into clear,
standalone search queries for a document retrieval system.

Rules:
- Resolve pronouns ("it", "that", "this") using the conversation
  history below.
- Fix obvious typos and expand vague phrasing into a specific
  question.
- Keep the same intent and meaning — do not add new topics.
- Output ONLY the rewritten query, nothing else. No quotes,
  no explanation, no prefix.
- If the question is already clear and standalone, output it
  unchanged.

Conversation History:
{history_text if history_text else "(none — this is the first message)"}

Original Question: {question}

Rewritten Query:"""

    try:

        rewritten = call_groq(
            [
                {
                    "role": "user",
                    "content": rewrite_prompt
                }
            ]
        ).strip()

        # Safety net: if the model returns something empty or
        # absurdly long (sign it went off the rails), fall back
        # to the original question rather than risk a broken query.
        if not rewritten or len(rewritten) > 300:

            return question, False

        return rewritten, (rewritten.lower() != question.lower())

    except Exception as e:

        print(f"\nQuery rewrite failed, using original: {e}")

        return question, False

# ============================================================
# CONTEXT
# ============================================================

def score_to_confidence(score):

    if score >= 0.80:
        return "High"

    if score >= 0.40:
        return "Medium"

    return "Low"


def suggest_rephrasing(question):

    suggestions = []

    word_count = len(question.split())

    if word_count <= 2:

        suggestions.append(
            "Try asking a more complete question — "
            "for example, instead of a couple of words, "
            "describe what you want to know in a full sentence."
        )

    suggestions.append(
        "Double-check that a PDF covering this topic has "
        "actually been uploaded to the Document Library."
    )

    suggestions.append(
        "Try rephrasing using different keywords — for "
        "example, a synonym or a more specific term from "
        "the document."
    )

    return suggestions


# ============================================================
# HALLUCINATION / GROUNDEDNESS CHECK
# ============================================================
# After the answer is generated, ask the LLM a second, narrower
# question: is this answer actually supported by the context?
# This is a real, lightweight self-consistency check — not
# perfect (the same model checking its own work has known
# limits), but it catches clear unsupported claims and gives
# the user an honest signal instead of presenting every answer
# as equally trustworthy.

def check_groundedness(answer, context):

    check_prompt = f"""You are a strict fact-checker. Determine
whether the ANSWER below is fully supported by the CONTEXT.

Respond with EXACTLY one word on the first line: GROUNDED or
UNSUPPORTED. Then, only if UNSUPPORTED, add one short sentence
on the next line explaining what part isn't supported. Do not
add anything else.

CONTEXT:
{context}

ANSWER:
{answer}

Verdict:"""

    try:

        verdict_text = call_groq(
            [
                {
                    "role": "user",
                    "content": check_prompt
                }
            ]
        ).strip()

        lines = verdict_text.split("\n")

        verdict_line = lines[0].strip().upper()

        is_grounded = "UNSUPPORTED" not in verdict_line

        explanation = (
            lines[1].strip()
            if len(lines) > 1 and not is_grounded
            else None
        )

        return {
            "grounded": is_grounded,
            "explanation": explanation
        }

    except Exception as e:

        print(f"\nGroundedness check failed: {e}")

        # Fail open — don't block the answer if the check itself
        # errors out, just skip the warning rather than guessing
        return {
            "grounded": True,
            "explanation": None
        }


def build_context(results):

    context_parts = []
    sources = []

    count = 0

    for doc_id, score in results:

        if score < MIN_RELEVANCE_SCORE:
            continue

        chunk = metadata[doc_id]

        rank = count + 1

        # Numbered so the LLM can cite this exact chunk as [rank]
        context_parts.append(
            f"[{rank}] {chunk['chunk_text']}"
        )

        sources.append(
            {
                "rank": rank,
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 4),
                "confidence": score_to_confidence(score),
                # Preview only — keeps payload small while still proving
                # retrieval actually happened (full chunk_text omitted on purpose)
                "chunk_preview": chunk["chunk_text"][:280].strip() + (
                    "..." if len(chunk["chunk_text"]) > 280 else ""
                )
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

def ask_rag(
    question,
    retrieval_mode="hybrid",
    top_k_retrieval=None,
    top_k_final=None,
    rrf_k=None,
    use_reranker=True,
    enable_query_rewrite=True,
    enable_groundedness_check=True
):

    if not question.strip():

        return {
            "answer": "Please enter a valid question.",
            "sources": [],
            "response_time": 0
        }

    analytics.increment()

    start_time = time.time()

    memory.add_user(question)

    history = build_history()

    # ------------------------------------------------
    # QUERY REWRITING
    # ------------------------------------------------

    search_query = question
    was_rewritten = False

    if enable_query_rewrite:

        search_query, was_rewritten = rewrite_query(
            question,
            history
        )

    retrieval_start = time.time()

    retrieved_results = advanced_search(
        search_query,
        mode=retrieval_mode,
        top_k_retrieval=top_k_retrieval,
        top_k_final=top_k_final,
        rrf_k=rrf_k,
        use_reranker=use_reranker
    )

    retrieval_time = round(
        time.time() - retrieval_start,
        2
    )

    if not retrieved_results:

        return {
            "answer": (
                "I couldn't find anything relevant to that "
                "question in the uploaded documents."
            ),
            "sources": [],
            "response_time": 0,
            "routing": "no_results",
            "suggestions": suggest_rephrasing(question),
            "search_query": search_query,
            "was_rewritten": was_rewritten
        }

    context, sources = build_context(
        retrieved_results
    )

    if not context:

        return {
            "answer": (
                "I found some related content, but none of it "
                "was a strong enough match to confidently answer "
                "this question."
            ),
            "sources": [],
            "response_time": 0,
            "routing": "no_results",
            "suggestions": suggest_rephrasing(question),
            "search_query": search_query,
            "was_rewritten": was_rewritten
        }

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
7. The context below is split into numbered sources like [1], [2], [3].
   After any sentence or bullet point that uses information from a
   specific source, add its number in square brackets, e.g. "...is
   absorbed in the small intestine [2]." Use the number exactly as
   shown before each source. If a sentence draws on multiple sources,
   cite all of them, e.g. [1][3]. Do not cite a source you did not
   actually use.

Conversation History:

{history}

Context:

{context}

Question:

{question}

Answer:
"""

    try:

        generation_start = time.time()

        answer = call_groq(
            [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        generation_time = round(
            time.time() - generation_start,
            2
        )

        memory.add_assistant(
            answer
        )

        # ------------------------------------------------
        # HALLUCINATION / GROUNDEDNESS CHECK
        # ------------------------------------------------

        groundedness = {"grounded": True, "explanation": None}

        if enable_groundedness_check:

            groundedness = check_groundedness(
                answer,
                context
            )

        response_time = round(
            time.time() - start_time,
            2
        )

        top_confidence = sources[0]["confidence"]
        top_score = sources[0]["score"]

        if not groundedness["grounded"]:

            # Groundedness failure takes priority — an answer
            # that isn't supported by the context matters more
            # than retrieval confidence alone.
            routing = "unsupported"
            suggestions = [
                "This answer may include claims not directly "
                "supported by the uploaded documents. Treat it "
                "as a starting point, not a final answer.",
                "Check the retrieved chunks below to verify the "
                "claim yourself."
            ]

        elif top_score < LOW_CONFIDENCE_THRESHOLD:

            routing = "low_confidence"
            suggestions = suggest_rephrasing(question)

        else:

            routing = "normal"
            suggestions = []

        return {
            "answer": answer,
            "sources": sources,
            "response_time": response_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "routing": routing,
            "suggestions": suggestions,
            "search_query": search_query,
            "was_rewritten": was_rewritten,
            "grounded": groundedness["grounded"],
            "groundedness_explanation": groundedness["explanation"],
            "total_queries":
                analytics.get_total_queries()
        }

    except Exception as e:

        return {
            "answer": f"Groq API error: {str(e)}",
            "sources": [],
            "response_time": 0,
            "routing": "error",
            "suggestions": []
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
    print("Advanced RAG Assistant Ready (Groq backend)")
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