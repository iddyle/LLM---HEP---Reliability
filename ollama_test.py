import ollama

response = ollama.chat(
    model="qwen2.5-coder:7b",
    messages=[
        {
            "role": "user",
            "content": "What is quark-gluon discrimination in particle physics? Explain briefly."
        }
    ]
)

print(response["message"]["content"])