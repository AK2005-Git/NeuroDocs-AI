from ollama import chat

response = chat(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "What is cyber security?"
        }
    ]
)

print(response["message"]["content"])