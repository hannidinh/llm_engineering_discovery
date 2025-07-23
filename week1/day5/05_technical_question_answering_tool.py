# Technical question answering tool
import os
import requests
import json
from dotenv import load_dotenv
import time
from openai import OpenAI

# constants
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2'

# set up environment
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
    print("OpenAI API key looks good")
else:
    print("Please check your OpenAI API key")

# Initialize OpenAI client
openai_client = OpenAI()

# here is the question; type over this to ask something new
question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

def get_openai_response(question: str, stream: bool = True):
    """
    Get response from OpenAI API with streaming
    """
    system_prompt = """You are an expert programming instructor and technical mentor. 
    Explain code and technical concepts clearly and concisely. 
    Break down complex concepts into understandable parts.
    Use examples when helpful.
    Focus on both what the code does and why it's written that way."""
    
    try:
        if stream:
            print("🤖 OpenAI GPT-4o-mini response:")
            print("-" * 40)
            
            stream_response = openai_client.chat.completions.create(
                model=MODEL_GPT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                stream=True,
                temperature=0.7
            )
            
            full_response = ""
            for chunk in stream_response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end='', flush=True)
                    full_response += content
                    time.sleep(0.01)  # Small delay for readability
            
            print("\n")
            return full_response
        else:
            response = openai_client.chat.completions.create(
                model=MODEL_GPT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None
    

def get_ollama_response(question: str, stream: bool = True):
    """
    Get response from Ollama (local Llama model)
    """
    system_prompt = """You are an expert programming instructor and technical mentor.
    Explain code and technical concepts clearly and concisely.
    Break down complex concepts into understandable parts. Use examples when helpful.
    Focus on both what the code does and why it's written that way.
    """
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": MODEL_LLAMA,
            "prompt": f"System: {system_prompt}\n\nUser: {question}",
            "stream": stream
        }

        if stream:
            print("Ollama Llama 3.2 response:")
            print("-" * 40)

            response = requests.post(url, json=payload, stream=True)
            full_response = ""

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                content = chunk['response']
                                print(content, end='', flush=True)
                                full_response += content
                                time.sleep(0.01)
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                print("\n")
                return full_response
            else:
                print(f"Error: HTTP {response.status_code}")
                return None
        else:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Error: HTTP {response.status_code}")
                return None
    
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama. Make sure Ollama is running locally.")
        print("To start Ollama: 'ollama serve' in terminal")
        print(f"To install the model: 'ollama pull {MODEL_LLAMA}")
        return None
    except Exception as e:
        print(f"Error with Ollama: {e}")
        return None
    
def compare_responses(questions: str):
    """
    Get responses from both APIs and compare them
    """
    print("=" * 60)
    print("TECHNICAL QUESTION ANSWERING TOOL")
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    # Get OpenAI response
    openai_response = get_openai_response(question, stream=True)

    print("=" * 60)

    # Get Ollama response
    ollama_response = get_ollama_response(question, stream=True)

    print("=" * 60)
    print("COMPARISION COMPLETE")
    print("=" * 60)

    return {
        "openai": openai_response,
        "ollama": ollama_response
    }

def interactive_mode():
    """
    Interactive mode for asking questions
    """
    print("Technical Question Answering Tool - Interactive Mode")
    print("Type 'quit' to exit, 'compare' to get both responses")
    print("-" * 50)

    while True:
        user_question = input("\n Enter your technical question: ").strip()

        if user_question.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_question.lower() == 'compare':
            question_to_ask = input("Enter question for comparison: ").strip()
            compare_responses(question_to_ask)
            continue
        elif not user_question:
            print("Please enter a question.")
            continue
        
        # Choose which model to use
        choice = input("Choose model (1=OpenAI, 2=Ollama, 3=Both): ").strip()

        if choice == '1':
            get_openai_response(user_question, stream=True)
        elif choice == '2':
            get_ollama_response(user_question, stream=True)

def analyze_code_snippet():
    """
    Analyze the specific code snippet from the exercise
    """
    code_question = """
    Please explain what this Python code does and why:
    
    yield from {book.get("author") for book in books if book.get("author")}
    
    Break down each part and explain the concepts involved.
    """
    return compare_responses(code_question)

if __name__ == "__main__":
    print("Technical Question Answering Tool")
    print("1. Analyze the given code snippet")
    print("2. Interactive mode")
    print("3. Compare responses for custom question")
    
    choice = input("\nChoose an option (1-3): ").strip()
    
    if choice == '1':
        analyze_code_snippet()
    elif choice == '2':
        interactive_mode()
    elif choice == '3':
        custom_question = input("Enter your question: ").strip()
        compare_responses(custom_question)
    else:
        # Default: analyze the code snippet
        print("Analyzing the code snippet...")
        analyze_code_snippet()

# Additional utility functions

def save_responses(question: str, responses: dict, filename: str = None):
    """
    Save responses to a file for later reference
    """
    if filename is None:
        timestamp = int(time.time())
        filename = f"technical_qa_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Technical Question & Answers\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(f"## OpenAI Response\n\n{responses.get('openai', 'No response')}\n\n")
        f.write(f"## Ollama Response\n\n{responses.get('ollama', 'No response')}\n\n")
    
    print(f"Responses saved to {filename}")

def get_quick_explanation(code_snippet: str):
    """
    Quick explanation function for code snippets
    """
    question = f"Quickly explain this code:\n\n{code_snippet}"
    return get_openai_response(question, stream=False)

# Example usage and testing
def run_examples():
    """
    Run some example questions to test the tool
    """
    examples = [
        "What is the difference between yield and return in Python?",
        "Explain list comprehension vs generator expression",
        "What does the .get() method do on dictionaries?",
        "When should I use yield from in Python?"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*20} Example {i} {'='*20}")
        print(f"Question: {example}")
        print("-" * 50)
        get_openai_response(example, stream=True)

# Uncomment to run examples:
run_examples()