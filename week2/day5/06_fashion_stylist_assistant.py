# Fashion Stylist AI Assistant
# Building on the patterns from day5 examples

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import random
from datetime import datetime

# Try to import audio libraries
try:
    import pygame
    AUDIO_AVAILABLE = True
    print("Audio playback available via pygame")
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio playback not available. Install with: pip install pygame")

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
MODEL = "gpt-4o-mini"  # You can change this to gpt-4, gpt-3.5-turbo, etc.
VOICE = "nova"  # OpenAI TTS voices: alloy, echo, fable, onyx, nova, shimmer

# Fashion-focused system message
system_message = """You are FashionGenie, an expert personal stylist and fashion consultant. You help people with:
- Outfit recommendations based on occasion, weather, and personal style
- Color coordination and matching advice
- Fashion trends and seasonal suggestions
- Wardrobe planning and capsule wardrobes
- Styling tips for different body types
- Accessory recommendations
- Shopping suggestions within budget

Be friendly, encouraging, and give specific, actionable advice. Consider factors like:
- Weather conditions
- Event type (casual, business, formal, date, etc.)
- Personal style preferences
- Budget constraints
- Color preferences
- Body type considerations

Always be inclusive and body-positive in your recommendations."""

def get_weather(city):
    """Get weather information for outfit planning"""
    # Mock weather data - in production, use a real weather API
    weather_conditions = [
        {"condition": "sunny", "temp": "75°F", "humidity": "40%", "advice": "Light, breathable fabrics recommended"},
        {"condition": "cloudy", "temp": "68°F", "humidity": "60%", "advice": "Perfect for layering pieces"},
        {"condition": "rainy", "temp": "62°F", "humidity": "80%", "advice": "Waterproof materials and darker colors"},
        {"condition": "snowy", "temp": "35°F", "humidity": "70%", "advice": "Warm layers and winter accessories essential"},
        {"condition": "windy", "temp": "70°F", "humidity": "45%", "advice": "Avoid flowing fabrics, choose structured pieces"}
    ]
    
    weather = random.choice(weather_conditions)
    return {
        "city": city,
        "condition": weather["condition"],
        "temperature": weather["temp"],
        "humidity": weather["humidity"],
        "fashion_advice": weather["advice"]
    }

def get_fashion_trends(season=None):
    """Get current fashion trends"""
    current_season = season or get_current_season()
    
    trends_data = {
        "spring": {
            "colors": ["Sage Green", "Coral Pink", "Lavender", "Butter Yellow", "Sky Blue"],
            "styles": ["Midi Dresses", "Wide-leg Trousers", "Cropped Blazers", "Statement Sleeves", "Floral Prints"],
            "accessories": ["Pearl Jewelry", "Woven Bags", "Block Heels", "Silk Scarves", "Cat-eye Sunglasses"]
        },
        "summer": {
            "colors": ["Bright White", "Ocean Blue", "Sunset Orange", "Lime Green", "Hot Pink"],
            "styles": ["Maxi Dresses", "Linen Sets", "High-waisted Shorts", "Off-shoulder Tops", "Wrap Styles"],
            "accessories": ["Statement Earrings", "Straw Hats", "Canvas Sneakers", "Beach Bags", "Layered Necklaces"]
        },
        "fall": {
            "colors": ["Burnt Orange", "Deep Burgundy", "Forest Green", "Camel", "Plum Purple"],
            "styles": ["Oversized Blazers", "Knee-high Boots", "Turtlenecks", "Plaid Patterns", "Leather Jackets"],
            "accessories": ["Chain Bags", "Ankle Boots", "Felt Hats", "Layered Rings", "Crossbody Bags"]
        },
        "winter": {
            "colors": ["Rich Navy", "Emerald Green", "Crimson Red", "Charcoal Gray", "Royal Purple"],
            "styles": ["Long Coats", "Chunky Sweaters", "Thermal Layers", "Wool Pants", "Statement Coats"],
            "accessories": ["Wool Scarves", "Leather Gloves", "Winter Boots", "Beanies", "Statement Jewelry"]
        }
    }
    
    return trends_data.get(current_season.lower(), trends_data["spring"])

def get_current_season():
    """Determine current season based on date"""
    month = datetime.now().month
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

def create_outfit_visualization(outfit_description, color_scheme=None):
    """Create a visual representation of the outfit recommendation"""
    try:
        # Create a stylish outfit card
        width, height = 500, 700
        
        # Always create a gradient background for visual appeal
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Choose colors based on color_scheme or use default elegant gradient
        if color_scheme and 'blue' in color_scheme.lower():
            # Blue gradient
            for i in range(height):
                fade = int(255 * (i / height))
                color = (240 - fade//6, 245 - fade//8, 255 - fade//4)
                draw.line([(0, i), (width, i)], fill=color)
        elif color_scheme and ('pink' in color_scheme.lower() or 'rose' in color_scheme.lower()):
            # Pink gradient
            for i in range(height):
                fade = int(255 * (i / height))
                color = (255 - fade//4, 240 - fade//6, 245 - fade//8)
                draw.line([(0, i), (width, i)], fill=color)
        elif color_scheme and 'green' in color_scheme.lower():
            # Green gradient
            for i in range(height):
                fade = int(255 * (i / height))
                color = (240 - fade//6, 255 - fade//4, 245 - fade//8)
                draw.line([(0, i), (width, i)], fill=color)
        else:
            # Default elegant purple gradient
            for i in range(height):
                fade = int(255 * (i / height))
                color = (250 - fade//5, 245 - fade//6, 255 - fade//3)
                draw.line([(0, i), (width, i)], fill=color)
        
        draw = ImageDraw.Draw(img)
        
        # Add decorative border with rounded corners effect
        border_color = (180, 180, 200)
        for i in range(8):
            draw.rectangle([i, i, width-1-i, height-1-i], outline=border_color, width=1)
        
        # Add some decorative corner elements
        corner_size = 30
        corner_color = (200, 190, 220)
        # Top-left corner decoration
        draw.arc([10, 10, 10+corner_size, 10+corner_size], 180, 270, fill=corner_color, width=3)
        # Top-right corner decoration  
        draw.arc([width-10-corner_size, 10, width-10, 10+corner_size], 270, 360, fill=corner_color, width=3)
        # Bottom-left corner decoration
        draw.arc([10, height-10-corner_size, 10+corner_size, height-10], 90, 180, fill=corner_color, width=3)
        # Bottom-right corner decoration
        draw.arc([width-10-corner_size, height-10-corner_size, width-10, height-10], 0, 90, fill=corner_color, width=3)
        
        # Title
        try:
            title_font = ImageFont.truetype("Arial.ttf", 28)
            subtitle_font = ImageFont.truetype("Arial.ttf", 18)
            text_font = ImageFont.truetype("Arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Add title
        title = "✨ Your Outfit Recommendation ✨"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 30), title, fill=(60, 60, 100), font=title_font)
        
        # Add outfit description (wrapped text)
        lines = wrap_text(outfit_description, 45)  # Wrap text to fit
        y_offset = 100
        
        for line in lines[:15]:  # Limit to 15 lines
            line_bbox = draw.textbbox((0, 0), line, font=text_font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = (width - line_width) // 2 if len(line) < 30 else 40
            draw.text((line_x, y_offset), line, fill=(40, 40, 80), font=text_font)
            y_offset += 25
        
        # Add fashion emoji decorations and visual elements
        emojis = ["👗", "👔", "👠", "👜", "💍", "🧥", "👓", "🎀"]
        for i, emoji in enumerate(emojis[:4]):
            x_pos = 60 + i * 90
            draw.text((x_pos, height-100), emoji, font=title_font)
        
        # Add some decorative lines
        line_color = (180, 170, 200)
        draw.line([(50, 80), (width-50, 80)], fill=line_color, width=2)
        draw.line([(50, height-120), (width-50, height-120)], fill=line_color, width=2)
        
        # Add small decorative dots
        dot_color = (200, 180, 220)
        for i in range(5):
            x = 70 + i * 80
            draw.ellipse([x-3, 85-3, x+3, 85+3], fill=dot_color)
            draw.ellipse([x-3, height-115-3, x+3, height-115+3], fill=dot_color)
        
        # Add current date
        date_str = datetime.now().strftime("%B %d, %Y")
        draw.text((40, height-40), f"Styled on: {date_str}", fill=(100, 100, 120), font=subtitle_font)
        
        return img
        
    except Exception as e:
        print(f"Error creating outfit visualization: {e}")
        return None

def wrap_text(text, width):
    """Wrap text to specified width"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + len(current_line) <= width:
            current_line.append(word)
            current_length += len(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def talker(message):
    """Generate speech from text using OpenAI TTS API"""
    if not AUDIO_AVAILABLE:
        print("Audio playback not available")
        return
    
    try:
        # Use OpenAI's text-to-speech API
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice=VOICE,    # Use the global VOICE setting
            input=message
        )
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Save the audio to a temporary file and play it
        audio_file = "temp_speech.mp3"
        response.stream_to_file(audio_file)
        
        # Play the audio file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        # Clean up
        pygame.mixer.quit()
        
        # Remove temporary file
        import os
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Tool definitions for the fashion assistant
weather_function = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information for outfit planning",
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

trends_function = {
    "type": "function",
    "function": {
        "name": "get_fashion_trends",
        "description": "Get current fashion trends for the season",
        "parameters": {
            "type": "object",
            "properties": {
                "season": {
                    "type": "string",
                    "description": "The season to get trends for (spring, summer, fall, winter)",
                    "enum": ["spring", "summer", "fall", "winter"]
                }
            },
            "additionalProperties": False
        }
    }
}


def fashion_chat(message, history, user_preferences=None):
    """Main chat function for fashion advice"""
    # Build conversation context
    conversation = [{"role": "system", "content": system_message}]
    
    # Add user preferences to context if provided
    if user_preferences:
        pref_context = f"User preferences: {user_preferences}"
        conversation.append({"role": "system", "content": pref_context})
    
    # Add conversation history
    for human, assistant in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": assistant})
    
    conversation.append({"role": "user", "content": message})
    
    outfit_image = None
    
    try:
        # Use OpenAI's function calling with tools
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation,
            tools=[weather_function, trends_function],
            tool_choice="auto"
        )
        
        # Handle tool calls if present
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            # Add the assistant's response with tool calls to conversation
            conversation.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response_message.tool_calls
                ]
            })
            
            # Process each tool call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == 'get_weather':
                    weather_data = get_weather(function_args.get('city'))
                    tool_response = json.dumps(weather_data)
                elif function_name == 'get_fashion_trends':
                    trends_data = get_fashion_trends(function_args.get('season'))
                    tool_response = json.dumps(trends_data)
                else:
                    tool_response = "Function not found"
                
                # Add tool response to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response
                })
            
            # Get final response after tool use
            final_response = client.chat.completions.create(
                model=MODEL,
                messages=conversation
            )
            reply = final_response.choices[0].message.content
        else:
            reply = response_message.content
        
        # Generate outfit visualization if this looks like outfit advice
        outfit_keywords = ['outfit', 'wear', 'dress', 'style', 'fashion', 'clothes', 'look']
        if any(keyword in message.lower() or keyword in reply.lower() for keyword in outfit_keywords):
            # Extract color scheme from the reply for better visualization
            detected_colors = None
            color_words = ['blue', 'pink', 'green', 'red', 'purple', 'yellow', 'orange', 'black', 'white', 'navy', 'coral', 'sage', 'lavender']
            for color in color_words:
                if color in reply.lower():
                    detected_colors = color
                    break
            
            outfit_image = create_outfit_visualization(reply, detected_colors)
        
        # Generate speech
        if AUDIO_AVAILABLE:
            # Create a shorter version for speech
            speech_text = reply
            if len(speech_text) > 300:
                speech_text = speech_text[:300] + "... Check the full advice on screen!"
            talker(speech_text)
        
        return reply, outfit_image
        
    except Exception as e:
        print(f"Error in fashion chat: {e}")
        return f"Sorry, I encountered an error: {e}. Please make sure your OpenAI API key is set correctly.", None

# Gradio interface
with gr.Blocks(title="Fashion Stylist AI", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 👗 Fashion Stylist AI Assistant")
    gr.Markdown("Your personal AI stylist powered by OpenAI - get outfit advice, fashion trends, and styling tips!")
    
    # User preferences section
    with gr.Accordion("👤 Your Style Preferences (Optional)", open=False):
        with gr.Row():
            style_preference = gr.Dropdown(
                choices=["Casual", "Business", "Formal", "Bohemian", "Minimalist", "Trendy", "Classic", "Edgy"],
                label="Style Preference",
                value="Casual"
            )
            color_preference = gr.Textbox(
                label="Favorite Colors",
                placeholder="e.g., blue, black, earth tones",
                value=""
            )
        with gr.Row():
            budget_range = gr.Dropdown(
                choices=["Budget-friendly", "Mid-range", "Luxury", "No preference"],
                label="Budget Range",
                value="No preference"
            )
            body_type = gr.Dropdown(
                choices=["Prefer not to specify", "Petite", "Tall", "Curvy", "Athletic", "Plus-size"],
                label="Body Type (for personalized advice)",
                value="Prefer not to specify"
            )
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                label="Fashion Consultation",
                show_label=True,
                avatar_images=("👤", "👗")
            )
        with gr.Column(scale=1):
            outfit_image = gr.Image(
                height=500,
                label="Outfit Visualization",
                show_label=True
            )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Ask your fashion stylist:",
            placeholder="What should I wear for a dinner date? or What are the trending colors this season?",
            container=False,
            scale=4
        )
        submit_btn = gr.Button("Get Styling Advice", scale=1, variant="primary")
    
    with gr.Row():
        clear = gr.Button("Clear Chat", variant="secondary")
        
    gr.Markdown("### 💡 Example Questions:")
    gr.Examples(
        examples=[
            "What should I wear for a job interview?",
            "I have a dinner date tonight, outfit suggestions?",
            "What are the trending colors for this season?",
            "Help me build a capsule wardrobe",
            "What should I wear in New York today?",
            "I need a casual weekend outfit",
            "How do I style a little black dress?",
            "What accessories go with a navy blazer?",
            "Summer wedding guest outfit ideas",
            "Professional but stylish work outfits"
        ],
        inputs=msg
    )

    def compile_preferences():
        """Compile user preferences into a string"""
        prefs = []
        if style_preference.value and style_preference.value != "":
            prefs.append(f"Style: {style_preference.value}")
        if color_preference.value and color_preference.value.strip():
            prefs.append(f"Colors: {color_preference.value}")
        if budget_range.value and budget_range.value != "No preference":
            prefs.append(f"Budget: {budget_range.value}")
        if body_type.value and body_type.value != "Prefer not to specify":
            prefs.append(f"Body type: {body_type.value}")
        
        return " | ".join(prefs) if prefs else None

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
        
        # Get user preferences
        user_prefs = compile_preferences()
        
        bot_message, image = fashion_chat(user_message, chat_history, user_prefs)
        history.append({"role": "assistant", "content": bot_message})
        return history, image

    # Event handlers
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, outfit_image]
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, outfit_image]
    )
    clear.click(lambda: ([], None), None, [chatbot, outfit_image], queue=False)

if __name__ == "__main__":
    print(f"\n👗 Starting Fashion Stylist AI with model: {MODEL}")
    print("Your personal AI fashion consultant is ready!")
    if AUDIO_AVAILABLE:
        print("🔊 Voice styling advice enabled")
    print("Make sure your OpenAI API key is set in your environment variables")
    
    ui.launch(
        share=False,
        debug=True,
        show_error=True
    )
