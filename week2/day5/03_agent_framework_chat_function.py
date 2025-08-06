# imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

# Load environment variables
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
# Configuration
MODEL = "gpt-4o-mini"
openai = OpenAI()

# System message
system_message = "You are a helpful AI assistant that can provide weather information and create images of cities."

# Function definitions for tools
def get_weather(city):
    """Mock function to get weather for a city"""
    return f"The weather in {city} is sunny and 75°F"

# Tool definition
weather_function = {
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

tools = [{"type": "function", "function": weather_function}]

def handle_tool_call(message):
    """Handle tool calls from the AI model"""
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('city')
    weather = get_weather(city)
    response = {
        "role": "tool",
        "content": json.dumps({"city": city, "weather": weather}),
        "tool_call_id": tool_call.id
    }
    return response, city

def artist(city):
    """Mock function to generate an image of a city"""
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=f"A beautiful landscape view of {city}",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def talker(message, voice="alloy", model="tts-1", speed=1.0):
    """Generate speech from text using OpenAI's TTS API"""
    try:
        response = openai.audio.speech.create(
            model=model,
            voice=voice,
            input=message,
            speed=speed
        )
        
        # Convert to audio and play
        audio_bytes = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_bytes, format="mp3")
        play(audio)
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def chat(message, history):
    image = None
    conversation = [{"role": "system", "content": system_message}]
    for human, assistant in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": assistant})
    conversation.append({"role": "user", "content": message})
    response = openai.chat.completions.create(model=MODEL, messages=conversation, tools=tools)
    
    if response.choices[0].finish_reason == "tool_calls":
        tool_call_message = response.choices[0].message
        tool_response, city = handle_tool_call(tool_call_message)
        conversation.append(tool_call_message)
        conversation.append(tool_response)
        image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=conversation)
    
    reply = response.choices[0].message.content
    talker(reply)
    return reply, image

# More involved Gradio code as we're not using the preset Chat interface
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        msg = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history):
        user_message = history[-1]["content"]
        # Convert history to the format expected by chat function
        chat_history = []
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history) - 1:
                chat_history.append([history[i]["content"], history[i + 1]["content"]])
        
        bot_message, image = chat(user_message, chat_history)
        history.append({"role": "assistant", "content": bot_message})
        return history, image

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, image_output]
    )
    clear.click(lambda: ([], None), None, [chatbot, image_output], queue=False)

ui.launch()