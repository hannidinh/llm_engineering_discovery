import ollama

MODEL = "llama3.2"
messages = [
    {
        "role": "user",
        "content": "Describe the affect of AI to software developer job market in the US"
    }
]

response = ollama.chat(model=MODEL, messages=messages)
print(response['message']['content'])