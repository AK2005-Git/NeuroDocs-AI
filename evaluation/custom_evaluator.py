import sys
import os
import time
import pandas as pd

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

# ============================================================
# IMPORTS
# ============================================================

from ollama import chat
from rag_chat import ask_rag

# ============================================================
# JUDGE ANSWER
# ============================================================

def judge_answer(
    question,
    ground_truth,
    generated_answer
):

    prompt = f"""
You are an AI evaluator.

Question:
{question}

Ground Truth:
{ground_truth}

Generated Answer:
{generated_answer}

Evaluate the generated answer.

Return ONLY a score between 0 and 10.

Scoring:

10 = Perfect answer
8 = Mostly correct
6 = Partially correct
4 = Weak answer
2 = Mostly incorrect
0 = Completely wrong

Output only the number.
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

        score_text = (
            response["message"]["content"]
            .strip()
        )

        try:

            score = float(
                score_text.split()[0]
            )

        except:

            score = 0.0

        return score

    except Exception as e:

        print(
            f"Evaluation error: {e}"
        )

        return 0.0


# ============================================================
# LOAD QUESTIONS
# ============================================================

print("\nLoading test dataset...")

df = pd.read_excel(
    "evaluation/test_questions.xlsx"
)

print(
    f"Questions Loaded: {len(df)}"
)

# ============================================================
# EVALUATION LOOP
# ============================================================

results = []

total_score = 0
total_time = 0

for index, row in df.iterrows():

    question = str(
        row["question"]
    )

    ground_truth = str(
        row["ground_truth"]
    )

    print(
        f"\n[{index+1}/{len(df)}] "
        f"{question}"
    )

    start_time = time.time()

    result = ask_rag(
        question
    )

    response_time = round(
        time.time() - start_time,
        2
    )

    answer = result["answer"]

    sources = result["sources"]

    score = judge_answer(
        question,
        ground_truth,
        answer
    )

    total_score += score
    total_time += response_time

    source_names = []

    for source in sources:

        source_names.append(
            source["source_file"]
        )

    results.append(
        {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "score": score,
            "response_time": response_time,
            "sources": str(source_names)
        }
    )

    print(
        f"Score: {score}/10"
    )

# ============================================================
# METRICS
# ============================================================

avg_score = round(
    total_score / len(results),
    2
)

avg_time = round(
    total_time / len(results),
    2
)

# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "evaluation/evaluation_results.csv",
    index=False
)

# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    "evaluation/evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ADVANCED RAG EVALUATION REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Questions Evaluated : {len(results)}\n"
    )

    f.write(
        f"Average Score       : {avg_score}/10\n"
    )

    f.write(
        f"Average Response Time: {avg_time} sec\n"
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Questions Evaluated : {len(results)}"
)

print(
    f"Average Score       : {avg_score}/10"
)

print(
    f"Average Response Time: {avg_time} sec"
)

print("\nSaved Files:")

print(
    "evaluation/evaluation_results.csv"
)

print(
    "evaluation/evaluation_report.txt"
)