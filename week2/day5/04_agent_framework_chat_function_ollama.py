# imports
import os
import json
from dotenv import load_dotenv
import ollama
import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import base64

# Try to import text-to-speech libraries
try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("Text-to-speech available via pyttsx3")
except ImportError:
    TTS_AVAILABLE = False
    print("Text-to-speech not available. Install with: pip install pyttsx3")

# Load environment variables
load_dotenv(override=True)

def list_available_voices():
    """List all available TTS voices and test them"""
    if not TTS_AVAILABLE:
        print("Text-to-speech not available. Install pyttsx3: pip install pyttsx3")
        return
    
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if not voices:
            print("No voices available")
            return
        
        print(f"\nFound {len(voices)} available voices:")
        print("-" * 50)
        
        for i, voice in enumerate(voices):
            print(f"\nVoice {i}:")
            print(f"  Name: {voice.name}")
            print(f"  ID: {voice.id}")
            print(f"  Languages: {getattr(voice, 'languages', 'N/A')}")
            print(f"  Gender: {getattr(voice, 'gender', 'N/A')}")
            print(f"  Age: {getattr(voice, 'age', 'N/A')}")
            
            # Test the voice
            engine.setProperty('voice', voice.id)
            print(f"  Testing voice {i}...")
            engine.say(f"Hello, I am voice number {i}. This is how I sound.")
            engine.runAndWait()
            
        print(f"\nTo use a specific voice, change VOICE_INDEX to the number you prefer (0 to {len(voices)-1})")
        
    except Exception as e:
        print(f"Error listing voices: {e}")

# Check if Ollama is available (optional - for debugging)
try:
    models = ollama.list()
    print(f"Ollama is available with {len(models['models'])} models")
    if models['models']:
        print("Available models:")
        for model in models['models']:
            print(f"  - {model['name']}")
except Exception as e:
    print(f"Ollama connection issue: {e}")
    print("Make sure Ollama is installed and running (ollama serve)")

# List available voices if TTS is available
if TTS_AVAILABLE:
    print("\n" + "="*50)
    print("VOICE SETUP")
    print("="*50)
    list_available_voices()
    
# Configuration
MODEL = "llama3.2"  # You can change this to any model you have installed
# Popular alternatives: "llama3.1", "phi3", "mistral", "codellama"

# Voice configuration for TTS
VOICE_INDEX = 0  # Change this to select different voices (0, 1, 2, etc.)

# System message
system_message = "You are a helpful AI assistant that can provide weather information and create images of cities."

# Function definitions for tools
def get_weather(city):
    """Mock function to get weather for a city"""
    # In a real application, you would call a weather API here
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperatures = ["72°F", "68°F", "75°F", "65°F", "70°F"]
    import random
    condition = random.choice(weather_conditions)
    temp = random.choice(temperatures)
    return f"The weather in {city} is {condition} and {temp}"

def talker(message, voice_index=None):
    """Generate speech from text using local TTS"""
    if not TTS_AVAILABLE:
        print("Text-to-speech not available. Install pyttsx3: pip install pyttsx3")
        return
    
    try:
        engine = pyttsx3.init()
        
        # Set speech rate (slower)
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 50)
        
        # Get available voices
        voices = engine.getProperty('voices')
        if voices:
            print(f"Available voices ({len(voices)}):")
            for i, voice in enumerate(voices):
                print(f"  {i}: {voice.name} ({voice.id})")
            
            # Set voice based on index or default to first available
            if voice_index is not None and 0 <= voice_index < len(voices):
                engine.setProperty('voice', voices[voice_index].id)
                print(f"Using voice {voice_index}: {voices[voice_index].name}")
            else:
                # Default to first voice if no index specified
                engine.setProperty('voice', voices[0].id)
                print(f"Using default voice: {voices[0].name}")
        
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def generate_image_placeholder(city):
    """Generate a simple placeholder image since we don't have DALL-E"""
    try:
        # Create a simple colored rectangle with text as placeholder
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image
        width, height = 400, 300
        colors = [(135, 206, 235), (144, 238, 144), (255, 182, 193), (255, 218, 185)]
        import random
        color = random.choice(colors)
        
        img = Image.new('RGB', (width, height), color=color)
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            # Try to use a better font
            font = ImageFont.truetype("Arial.ttf", 24)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        text = f"🏙️ {city}\n📍 Beautiful City View"
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill='white', font=font, align='center')
        
        # Save to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img
        
    except Exception as e:
        print(f"Error generating placeholder image: {e}")
        return None

def artist(city):
    """Generate an image for a city - placeholder implementation"""
    # Option 1: Generate a simple placeholder (implemented above)
    placeholder_img = generate_image_placeholder(city)
    if placeholder_img:
        return placeholder_img
    
    # Option 2: You could integrate with other image generation APIs here
    # For example: Stability AI, Hugging Face, or local Stable Diffusion
    # 
    # Example with Hugging Face (commented out):
    # try:
    #     API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    #     headers = {"Authorization": f"Bearer {your_hf_token}"}
    #     payload = {"inputs": f"A beautiful landscape view of {city}"}
    #     response = requests.post(API_URL, headers=headers, json=payload)
    #     if response.status_code == 200:
    #         return Image.open(BytesIO(response.content))
    # except Exception as e:
    #     print(f"Error with Hugging Face API: {e}")
    
    return None

# Tool definition for Ollama
weather_function = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for"
                }
            },
            "required": ["city"],
            "additionalProperties": False
        }
    }
}

def handle_tool_call(tool_calls):
    """Handle tool calls from the AI model"""
    results = []
    for tool_call in tool_calls:
        if tool_call['function']['name'] == 'get_weather':
            arguments = tool_call['function']['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            city = arguments.get('city')
            weather = get_weather(city)
            results.append({
                "role": "tool",
                "content": json.dumps({"city": city, "weather": weather})
            })
    return results

def chat(message, history):
    conversation = [{"role": "system", "content": system_message}]
    for human, assistant in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": assistant})
    conversation.append({"role": "user", "content": message})
    
    image = None
    city_mentioned = None
    
    try:
        # First attempt: try with tools
        response = ollama.chat(
            model=MODEL,
            messages=conversation,
            tools=[weather_function]
        )
        
        # Check if the model wants to use tools
        if 'tool_calls' in response['message'] and response['message']['tool_calls']:
            # Handle tool calls
            tool_responses = handle_tool_call(response['message']['tool_calls'])
            conversation.append(response['message'])
            conversation.extend(tool_responses)
            
            # Extract city from tool call for image generation
            for tool_call in response['message']['tool_calls']:
                if tool_call['function']['name'] == 'get_weather':
                    arguments = tool_call['function']['arguments']
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    city_mentioned = arguments.get('city')
                    break
            
            # Get final response after tool use
            response = ollama.chat(model=MODEL, messages=conversation)
        
        reply = response['message']['content']
        
        # Generate image if a city was mentioned
        if city_mentioned:
            image = artist(city_mentioned)
        
        # Generate speech
        if TTS_AVAILABLE:
            talker(reply, VOICE_INDEX)
        
        return reply, image
        
    except Exception as e:
        print(f"Error in chat with tools: {e}")
        # Fallback: try without tools if the model doesn't support them
        try:
            response = ollama.chat(
                model=MODEL,
                messages=conversation
            )
            reply = response['message']['content']
            
            # Simple keyword-based weather detection as fallback
            if 'weather' in message.lower():
                # Try to extract city name from the message
                words = message.split()
                for i, word in enumerate(words):
                    if word.lower() in ['in', 'for', 'at']:
                        if i + 1 < len(words):
                            city = words[i + 1].strip('.,!?')
                            weather_info = get_weather(city)
                            reply += f"\n\n{weather_info}"
                            city_mentioned = city
                            break
            
            # Generate image if a city was mentioned
            if city_mentioned:
                image = artist(city_mentioned)
            
            # Generate speech
            if TTS_AVAILABLE:
                talker(reply, VOICE_INDEX)
            
            return reply, image
            
        except Exception as e2:
            print(f"Error in fallback chat: {e2}")
            return f"Sorry, I encountered an error: {e2}. Please make sure Ollama is running and the model '{MODEL}' is available.", None

# Gradio interface
with gr.Blocks(title="Ollama Chat Assistant") as ui:
    gr.Markdown("# Chat Assistant powered by Ollama")
    gr.Markdown(f"Currently using model: **{MODEL}**")
    
    if TTS_AVAILABLE:
        gr.Markdown("🔊 **Text-to-speech enabled**")
    else:
        gr.Markdown("🔇 Text-to-speech disabled (install pyttsx3 to enable)")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500, 
                type="messages",
                label="Conversation",
                show_label=True
            )
        with gr.Column(scale=1):
            image_output = gr.Image(
                height=500,
                label="Generated Image",
                show_label=True
            )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Chat with our AI Assistant (powered by Ollama):",
            placeholder="Ask me about the weather in any city!",
            container=False,
            scale=4
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")
    
    with gr.Row():
        clear = gr.Button("Clear Chat", variant="secondary")
        
    gr.Markdown("### Examples:")
    gr.Examples(
        examples=[
            "What's the weather in New York?",
            "Tell me about the weather in Tokyo",
            "How's the weather in London today?",
            "What's your favorite color?",
            "Tell me a joke"
        ],
        inputs=msg
    )

    def user(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history):
        if not history:
            return history, None
            
        user_message = history[-1]["content"]
        # Convert history to the format expected by chat function
        chat_history = []
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history) - 1:
                chat_history.append([history[i]["content"], history[i + 1]["content"]])
        
        bot_message, image = chat(user_message, chat_history)
        history.append({"role": "assistant", "content": bot_message})
        return history, image

    # Event handlers
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, image_output]
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, image_output]
    )
    clear.click(lambda: ([], None), None, [chatbot, image_output], queue=False)

if __name__ == "__main__":
    print(f"\nStarting Ollama Chat Assistant with model: {MODEL}")
    print("Make sure Ollama is running (ollama serve) and the model is installed")
    print(f"To install the model, run: ollama pull {MODEL}")
    ui.launch(
        share=False,
        debug=True,
        show_error=True
    )
