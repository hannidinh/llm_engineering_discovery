import ollama
import gradio as gr

# Model configuration
MODEL = "llama3.2"  # Using Llama 3.2 as the open source model

def generate_joke(joke_topic, joke_style, audience):
    """Generate a joke using Ollama with an open source model."""
    
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
        # Generate joke using Ollama
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": 0.9,  # High creativity level
                "num_predict": 300   # Max tokens equivalent
            }
        )
        
        return response['message']['content']
        
    except Exception as e:
        return f"Error generating joke: {str(e)}\n\nMake sure Ollama is running and the {MODEL} model is installed.\nRun: ollama pull {MODEL}"

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
        # Stream response from Ollama
        stream = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.9,  # High creativity level
                "num_predict": 300   # Max tokens equivalent
            }
        )
        
        full_response = ""
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content_chunk = chunk['message']['content']
                full_response += content_chunk
                yield full_response
            
    except Exception as e:
        yield f"Error generating joke: {str(e)}\n\nMake sure Ollama is running and the {MODEL} model is installed.\nRun: ollama pull {MODEL}"

# Create Gradio interface
with gr.Blocks(title="Ollama Joke Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🎭 Ollama Joke Generator
        
        Generate creative jokes using **{MODEL}** running locally with Ollama!
        
        **Requirements:** Make sure Ollama is running and you have the {MODEL} model installed:
        ```bash
        ollama pull {MODEL}
        ollama serve  # if not already running
        ```
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
    
    # Model info section
    gr.Markdown(
        f"""
        ## ℹ️ Model Information
        
        - **Model:** {MODEL} (Open Source)
        - **Running:** Locally via Ollama
        - **Temperature:** 0.9 (High creativity)
        - **Privacy:** Complete - no data sent to external servers
        """
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
    print(f"Starting Ollama Joke Generator with {MODEL}...")
    print("Make sure Ollama is running and the model is installed!")
    print(f"Run: ollama pull {MODEL}")
    print("Run: ollama serve")
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7861,  # Different port from DeepSeek version
        share=False,
        show_error=True
    ) 