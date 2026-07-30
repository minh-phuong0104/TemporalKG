from ollama import chat

response = chat(
    model="gemma3:latest",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ]
)

print(response["message"]["content"])