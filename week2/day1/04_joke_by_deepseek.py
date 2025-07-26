import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize DeepSeek client
deepseek_client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)

def generate_joke(joke_topic, joke_style, audience):
    """Generate a joke using DeepSeek with high creativity."""
    
    # Create system message based on selected parameters
    system_message = f"""You are a creative and witty comedian. Generate {joke_style} jokes that are appropriate for {audience}. 
    Be creative, original, and entertaining while keeping the content appropriate."""
    
    # Create user prompt
    if joke_topic.strip():
        user_prompt = f"Tell me a {joke_style} joke about {joke_topic}"
    else:
        user_prompt = f"Tell me a {joke_style} joke"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Use high temperature for maximum creativity
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.9,  # High creativity level
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating joke: {str(e)}"

def generate_joke_stream(joke_topic, joke_style, audience):
    """Generate a joke with streaming for better user experience."""
    
    system_message = f"""You are a creative and witty comedian. Generate {joke_style} jokes that are appropriate for {audience}. 
    Be creative, original, and entertaining while keeping the content appropriate."""
    
    if joke_topic.strip():
        user_prompt = f"Tell me a {joke_style} joke about {joke_topic}"
    else:
        user_prompt = f"Tell me a {joke_style} joke"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        stream = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.9,  # High creativity level
            max_tokens=300,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            content_chunk = chunk.choices[0].delta.content or ''
            full_response += content_chunk
            yield full_response
            
    except Exception as e:
        yield f"Error generating joke: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="DeepSeek Joke Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 DeepSeek Joke Generator
        
        Generate creative jokes using DeepSeek AI with high creativity settings!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            joke_topic = gr.Textbox(
                label="Joke Topic (optional)",
                placeholder="e.g., programming, cats, coffee...",
                lines=1
            )
            
            joke_style = gr.Dropdown(
                choices=[
                    "funny", "witty", "clever", "pun-filled", "dad joke style", 
                    "one-liner", "observational", "absurd", "light-hearted"
                ],
                label="Joke Style",
                value="funny"
            )
            
            audience = gr.Dropdown(
                choices=[
                    "general audience", "software engineers", "students", 
                    "office workers", "friends", "family-friendly"
                ],
                label="Target Audience",
                value="general audience"
            )
            
            with gr.Row():
                generate_btn = gr.Button("🎲 Generate Joke", variant="primary")
                stream_btn = gr.Button("📺 Generate with Streaming", variant="secondary")
        
        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Generated Joke",
                lines=10,
                placeholder="Your joke will appear here..."
            )
    
    # Examples section
    gr.Markdown("## Example Topics")
    gr.Examples(
        examples=[
            ["programming", "witty", "software engineers"],
            ["coffee", "observational", "office workers"],
            ["cats", "dad joke style", "family-friendly"],
            ["", "clever", "general audience"],
            ["artificial intelligence", "absurd", "students"]
        ],
        inputs=[joke_topic, joke_style, audience],
        outputs=output,
        fn=generate_joke,
        cache_examples=False
    )
    
    # Event handlers
    generate_btn.click(
        fn=generate_joke,
        inputs=[joke_topic, joke_style, audience],
        outputs=output
    )
    
    stream_btn.click(
        fn=generate_joke_stream,
        inputs=[joke_topic, joke_style, audience],
        outputs=output
    )

if __name__ == "__main__":
    print("Starting DeepSeek Joke Generator...")
    print("Make sure your DEEPSEEK_API_KEY is set in your .env file!")
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860,
        share=False,
        show_error=True
    ) 