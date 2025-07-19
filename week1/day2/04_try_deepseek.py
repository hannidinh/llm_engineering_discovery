#!ollama pull deepseek-r1:1.5b

from openai import OpenAI
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

MODEL = "deepseek-r1:1.5b"

response = ollama_via_openai.chat.completions.create(
    model = MODEL,
    messages=[
        {
            "role": "user",
            "content": "Please give definitions of some core concepts behind LLMs: a neural network, attention and the transformer" 
        }
    ]
)

print(response.choices[0].message.content)