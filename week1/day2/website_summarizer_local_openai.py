#!ollama pull llama3.2

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# Initialize Ollama client using OpenAI library
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Some websites need you to use proper headers when fetching them:
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\n The contents of this website is as follows; \
        please provide a short summary of this website in markdown. \
            If it includes news or announcements, then summarize these too. \n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url):
    website = Website(url)
    MODEL = "llama3.2"
    
    # Using OpenAI library with Ollama backend
    response = ollama_via_openai.chat.completions.create(
        model=MODEL,
        messages=messages_for(website)
    )
    return response.choices[0].message.content

def display_summary(url):
    summary = summarize(url)
    print(f"Summary of {url}:")
    print("=" * 50)
    print(summary)
    print("=" * 50)

# Example usage
if __name__ == "__main__":
    display_summary("https://anthropic.com") 