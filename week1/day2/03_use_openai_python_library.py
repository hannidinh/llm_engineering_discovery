# Alternative approach - using OpenAI python library to connect to Ollama
from openai import OpenAI
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

MODEL = "llama3.2"
messages = [
    {
        "role": "user",
        "content":"Describe RAG in AI context"
    }
]

response = ollama_via_openai.chat.completions.create(
    model=MODEL,
    messages=messages
)

print(response.choices[0].message.content)