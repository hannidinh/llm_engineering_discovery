import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Constants
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

# Create a messages list using the same format that we used for OpenAI
messages = [
    {
        "role": "user",
        "content": "Describe some of the business applications of Generative AI"
    },
    {
        "role":"user",
        "content": "Describe the meaning of falling in love from men's perspective"
    },
    {
        "role":"user",
        "content": "Describe the meaning of falling in love from women's perspective"
    }
]

payload = {
    "model": MODEL,
    "messages": messages,
    "stream": False
}

# Under the hood, it's making the same call to the ollama server running at localhost:11434
response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
print(response.json()['message']['content'])