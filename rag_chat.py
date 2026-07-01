import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_rag(question: str, **kwargs):
    response = client.chat.completions.create(
        model="llama3-8b-8192",   # Example Groq model
        messages=[{"role": "user", "content": question}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [],
        "response_time": 0,
    }
