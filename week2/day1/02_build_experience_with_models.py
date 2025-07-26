import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai

# Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

# Initialize clients
openai_client = OpenAI()
claude_client = anthropic.Anthropic()
google.generativeai.configure()

if deepseek_api_key:
    deepseek_client = OpenAI(
        api_key=deepseek_api_key, 
        base_url="https://api.deepseek.com"
    )

# Test questions
questions = {
    "word_count": {
        "prompt": "How many words are there in your answer to this prompt",
        "description": "Tests self-awareness and counting ability"
    },
    "creative": {
        "prompt": "In 3 sentences, describe the color Blue to someone who's never been able to see",
        "description": "Tests creativity and ability to explain abstract concepts"
    },
    "riddle": {
        "prompt": "On a bookshelf, two volumes of Pushkin stand side by side: the first and the second. The pages of each volume together have a thickness of 2 cm, and each cover is 2 mm thick. A worm gnawed (perpendicular to the pages) from the first page of the first volume to the last page of the second volume. What distance did it gnaw through?",
        "description": "Tests logical reasoning and spatial thinking (the answer is 4mm - just the two covers!)"
    }
}

def test_model(model_name, model_function, question_key, question_data):
    """Test a specific model with a question and measure response time"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"Question: {question_data['description']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        response = model_function(question_data['prompt'])
        end_time = time.time()
        
        print(f"Response ({end_time - start_time:.2f}s):")
        print(response)
        
        # For word count question, let's actually count the words
        if question_key == "word_count":
            actual_word_count = len(response.split())
            print(f"\nActual word count: {actual_word_count}")
            
    except Exception as e:
        print(f"Error: {e}")

def gpt_4o_mini(prompt):
    """GPT-4o-mini (Fast and efficient)"""
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def gpt_4_1(prompt):
    """GPT-4.1 (More capable reasoning)"""
    response = openai_client.chat.completions.create(
        model='gpt-4.1',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def o3_mini(prompt):
    """o3-mini (Reasoning model - should think through responses)"""
    response = openai_client.chat.completions.create(
        model='o3-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def claude_sonnet(prompt):
    """Claude 3.7 Sonnet"""
    message = claude_client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=500,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def gemini_flash(prompt):
    """Gemini 2.0 Flash"""
    gemini = google.generativeai.GenerativeModel(model_name='gemini-2.0-flash')
    response = gemini.generate_content(prompt)
    return response.text

def deepseek_chat(prompt):
    """DeepSeek Chat"""
    if not deepseek_api_key:
        return "DeepSeek API key not available"
    
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def deepseek_reasoner(prompt):
    """DeepSeek Reasoner (Thinking model)"""
    if not deepseek_api_key:
        return "DeepSeek API key not available"
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}]
        )
        
        reasoning = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        
        result = f"REASONING:\n{reasoning}\n\nFINAL ANSWER:\n{content}"
        return result
    except Exception as e:
        return f"DeepSeek Reasoner currently unavailable: {e}"

# Models to test
models = {
    "GPT-4o-mini": gpt_4o_mini,
    "GPT-4.1": gpt_4_1,
    "o3-mini (Reasoning)": o3_mini,
    "Claude 3.7 Sonnet": claude_sonnet,
    "Gemini 2.0 Flash": gemini_flash,
    "DeepSeek Chat": deepseek_chat,
    "DeepSeek Reasoner": deepseek_reasoner
}

def main():
    print("LLM Model Experience Builder")
    print("Testing different models with various types of questions")
    print("\nWhat to observe:")
    print("- Reasoning models vs Chat models (how they approach problems)")
    print("- Creativity and explanation abilities")
    print("- Problem-solving accuracy")
    print("- Response speed")
    print("- Different approaches to the same question")
    
    for question_key, question_data in questions.items():
        print(f"\n\n{'#'*80}")
        print(f"QUESTION SET: {question_data['description'].upper()}")
        print(f"{'#'*80}")
        
        for model_name, model_function in models.items():
            test_model(model_name, model_function, question_key, question_data)
            time.sleep(1)  # Small delay to be respectful to APIs

if __name__ == "__main__":
    main() 