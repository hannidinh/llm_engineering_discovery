# Advanced Fashion Assistant with Virtual Wardrobe
# Extended version with wardrobe management and style analysis

import os
import json
from dotenv import load_dotenv
import ollama
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import random
from datetime import datetime, timedelta
import sqlite3

# Try to import text-to-speech libraries
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

load_dotenv(override=True)

# Configuration
MODEL = "llama3.2"
VOICE_INDEX = 0

class VirtualWardrobe:
    """Manage user's virtual wardrobe"""
    
    def __init__(self, db_path="wardrobe.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the wardrobe database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clothing_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                color TEXT,
                season TEXT,
                occasion TEXT,
                last_worn DATE,
                times_worn INTEGER DEFAULT 0,
                purchase_date DATE,
                cost REAL,
                brand TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outfits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                occasion TEXT,
                season TEXT,
                items TEXT,  -- JSON array of item IDs
                date_created DATE,
                last_worn DATE,
                rating INTEGER,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_item(self, name, category, color=None, season=None, occasion=None, cost=None, brand=None, notes=None):
        """Add a clothing item to the wardrobe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clothing_items 
            (name, category, color, season, occasion, purchase_date, cost, brand, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, category, color, season, occasion, datetime.now().date(), cost, brand, notes))
        
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return item_id
    
    def get_items(self, category=None, season=None, occasion=None):
        """Get clothing items with optional filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM clothing_items WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        if season:
            query += " AND season = ?"
            params.append(season)
        if occasion:
            query += " AND occasion = ?"
            params.append(occasion)
        
        cursor.execute(query, params)
        items = cursor.fetchall()
        conn.close()
        
        return items
    
    def get_wardrobe_stats(self):
        """Get statistics about the wardrobe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total items
        cursor.execute("SELECT COUNT(*) FROM clothing_items")
        total_items = cursor.fetchone()[0]
        
        # Items by category
        cursor.execute("SELECT category, COUNT(*) FROM clothing_items GROUP BY category")
        by_category = dict(cursor.fetchall())
        
        # Most worn items
        cursor.execute("SELECT name, times_worn FROM clothing_items ORDER BY times_worn DESC LIMIT 5")
        most_worn = cursor.fetchall()
        
        # Least worn items
        cursor.execute("SELECT name, times_worn FROM clothing_items WHERE times_worn = 0 OR times_worn IS NULL")
        unworn = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_items": total_items,
            "by_category": by_category,
            "most_worn": most_worn,
            "unworn_items": len(unworn)
        }

# Initialize virtual wardrobe
wardrobe = VirtualWardrobe()

# Enhanced system message for wardrobe-aware assistant
system_message = """You are StyleGenius, an advanced AI fashion consultant with access to the user's virtual wardrobe. You can:

1. **Outfit Creation**: Mix and match from their existing wardrobe
2. **Style Analysis**: Analyze their style patterns and preferences
3. **Wardrobe Optimization**: Suggest what to buy/donate based on gaps and usage
4. **Sustainable Fashion**: Promote re-wearing and creative combinations
5. **Cost-per-Wear**: Help optimize fashion budget and value

Key Capabilities:
- Access user's virtual wardrobe inventory
- Track wearing patterns and frequency
- Suggest outfit combinations from existing pieces
- Identify wardrobe gaps and recommend strategic purchases
- Provide seasonal wardrobe transition advice
- Create capsule wardrobe recommendations

Always be encouraging about creative re-styling and sustainable fashion choices."""

def analyze_wardrobe_gaps(items, season=None):
    """Analyze wardrobe for missing essential items"""
    essentials = {
        "spring": ["light jacket", "cotton tee", "jeans", "comfortable flats", "light scarf"],
        "summer": ["sundress", "shorts", "sandals", "swimwear", "sun hat"],
        "fall": ["sweater", "boots", "cardigan", "scarf", "jacket"],
        "winter": ["warm coat", "boots", "sweater", "gloves", "warm accessories"]
    }
    
    current_season = season or get_current_season().lower()
    needed_items = essentials.get(current_season, essentials["spring"])
    
    # Check what's missing (simplified logic)
    existing_categories = [item[2].lower() for item in items]  # category is index 2
    gaps = []
    
    for essential in needed_items:
        if not any(essential.split()[0] in cat for cat in existing_categories):
            gaps.append(essential)
    
    return gaps

def create_outfit_mood_board(outfit_items, theme="casual"):
    """Create a visual mood board for outfit combinations"""
    try:
        width, height = 600, 800
        img = Image.new('RGB', (width, height), color=(248, 248, 252))
        draw = ImageDraw.Draw(img)
        
        # Color schemes for different themes
        color_schemes = {
            "casual": [(135, 206, 235), (255, 182, 193), (144, 238, 144)],
            "professional": [(25, 25, 112), (105, 105, 105), (220, 220, 220)],
            "evening": [(72, 61, 139), (147, 112, 219), (255, 215, 0)],
            "weekend": [(255, 160, 122), (255, 218, 185), (250, 240, 230)]
        }
        
        colors = color_schemes.get(theme, color_schemes["casual"])
        
        # Create gradient background
        for i in range(height):
            fade = int(100 * (i / height))
            bg_color = (248 - fade//4, 248 - fade//4, 252 - fade//6)
            draw.line([(0, i), (width, i)], fill=bg_color)
        
        # Title
        try:
            title_font = ImageFont.truetype("Arial.ttf", 32)
            item_font = ImageFont.truetype("Arial.ttf", 20)
            detail_font = ImageFont.truetype("Arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            item_font = ImageFont.load_default()
            detail_font = ImageFont.load_default()
        
        # Main title
        title = f"✨ {theme.title()} Outfit Mood Board ✨"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 40), title, fill=(60, 60, 100), font=title_font)
        
        # Draw outfit items as stylized cards
        y_start = 120
        card_height = 80
        
        for i, item in enumerate(outfit_items[:8]):  # Limit to 8 items
            y_pos = y_start + (i * (card_height + 20))
            
            # Draw card background
            color = colors[i % len(colors)]
            card_rect = [50, y_pos, width-50, y_pos + card_height]
            draw.rounded_rectangle(card_rect, radius=15, fill=color, outline=(200, 200, 200), width=2)
            
            # Add item text
            item_text = f"• {item}"
            draw.text((70, y_pos + 20), item_text, fill=(255, 255, 255), font=item_font)
            
            # Add styling emoji
            emoji_map = {
                "top": "👔", "bottom": "👖", "dress": "👗", "shoes": "👠",
                "jacket": "🧥", "accessory": "💍", "bag": "👜"
            }
            emoji = "✨"  # default
            for key in emoji_map:
                if key in item.lower():
                    emoji = emoji_map[key]
                    break
            
            draw.text((width - 100, y_pos + 25), emoji, font=title_font)
        
        # Add styling tips at bottom
        tips = [
            "💡 Mix textures for visual interest",
            "🎨 Use the 60-30-10 color rule",
            "👗 Accessorize to elevate the look"
        ]
        
        tip_y = height - 150
        for tip in tips:
            draw.text((60, tip_y), tip, fill=(80, 80, 120), font=detail_font)
            tip_y += 25
        
        return img
        
    except Exception as e:
        print(f"Error creating mood board: {e}")
        return None

def get_current_season():
    """Determine current season"""
    month = datetime.now().month
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

# Wardrobe management functions for tools
wardrobe_stats_function = {
    "type": "function",
    "function": {
        "name": "get_wardrobe_stats",
        "description": "Get statistics about the user's virtual wardrobe",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    }
}

wardrobe_items_function = {
    "type": "function",
    "function": {
        "name": "get_wardrobe_items",
        "description": "Get clothing items from the user's wardrobe with optional filters",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by clothing category (tops, bottoms, dresses, shoes, etc.)"
                },
                "season": {
                    "type": "string",
                    "description": "Filter by season (spring, summer, fall, winter)"
                },
                "occasion": {
                    "type": "string",
                    "description": "Filter by occasion (casual, formal, work, etc.)"
                }
            },
            "additionalProperties": False
        }
    }
}

def handle_wardrobe_tool_call(tool_calls):
    """Handle wardrobe-related tool calls"""
    results = []
    
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        arguments = tool_call['function']['arguments']
        
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        
        if function_name == 'get_wardrobe_stats':
            stats = wardrobe.get_wardrobe_stats()
            results.append({
                "role": "tool",
                "content": json.dumps(stats)
            })
        elif function_name == 'get_wardrobe_items':
            category = arguments.get('category')
            season = arguments.get('season')
            occasion = arguments.get('occasion')
            items = wardrobe.get_items(category, season, occasion)
            
            # Convert to readable format
            item_list = []
            for item in items:
                item_dict = {
                    "name": item[1],
                    "category": item[2],
                    "color": item[3],
                    "season": item[4],
                    "occasion": item[5],
                    "times_worn": item[7]
                }
                item_list.append(item_dict)
            
            results.append({
                "role": "tool",
                "content": json.dumps(item_list)
            })
    
    return results

def advanced_fashion_chat(message, history, user_preferences=None):
    """Enhanced chat function with wardrobe awareness"""
    conversation = [{"role": "system", "content": system_message}]
    
    if user_preferences:
        conversation.append({"role": "system", "content": f"User preferences: {user_preferences}"})
    
    # Add conversation history
    for human, assistant in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": assistant})
    
    conversation.append({"role": "user", "content": message})
    
    outfit_image = None
    
    try:
        # Chat with wardrobe tools
        response = ollama.chat(
            model=MODEL,
            messages=conversation,
            tools=[wardrobe_stats_function, wardrobe_items_function]
        )
        
        # Handle tool calls
        if 'tool_calls' in response['message'] and response['message']['tool_calls']:
            tool_responses = handle_wardrobe_tool_call(response['message']['tool_calls'])
            conversation.append(response['message'])
            conversation.extend(tool_responses)
            
            response = ollama.chat(model=MODEL, messages=conversation)
        
        reply = response['message']['content']
        
        # Generate visualization for outfit-related queries
        outfit_keywords = ['outfit', 'wear', 'combination', 'style', 'look', 'wardrobe']
        if any(keyword in message.lower() or keyword in reply.lower() for keyword in outfit_keywords):
            # Extract suggested items from reply for visualization
            suggested_items = extract_outfit_items_from_text(reply)
            if suggested_items:
                outfit_image = create_outfit_mood_board(suggested_items)
        
        # TTS for shorter responses
        if TTS_AVAILABLE and len(reply) < 500:
            talker(reply, VOICE_INDEX)
        
        return reply, outfit_image
        
    except Exception as e:
        return f"Error: {e}", None

def extract_outfit_items_from_text(text):
    """Extract clothing items mentioned in the text"""
    # Simple keyword extraction - could be enhanced with NLP
    clothing_keywords = [
        "shirt", "blouse", "top", "sweater", "cardigan", "jacket", "coat",
        "jeans", "pants", "trousers", "skirt", "shorts", "dress",
        "shoes", "boots", "sneakers", "heels", "flats", "sandals",
        "scarf", "necklace", "earrings", "bag", "purse", "belt", "hat"
    ]
    
    items = []
    words = text.lower().split()
    
    for i, word in enumerate(words):
        for keyword in clothing_keywords:
            if keyword in word:
                # Try to get color/descriptor before the item
                if i > 0 and words[i-1] not in ["a", "an", "the", "your", "my"]:
                    items.append(f"{words[i-1]} {keyword}")
                else:
                    items.append(keyword)
                break
    
    return list(set(items))  # Remove duplicates

def talker(message, voice_index=None):
    """TTS function"""
    if not TTS_AVAILABLE:
        return
    
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 40)
        
        voices = engine.getProperty('voices')
        if voices and voice_index is not None and 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
        
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# Gradio interface for the advanced fashion assistant
with gr.Blocks(title="Advanced Fashion Assistant", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 👗 Advanced Fashion Assistant with Virtual Wardrobe")
    gr.Markdown("Your AI stylist with wardrobe management, outfit planning, and sustainable fashion advice!")
    
    with gr.Tabs():
        # Chat Tab
        with gr.TabItem("💬 Style Consultation"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        height=500,
                        type="messages",
                        label="Fashion Consultation",
                        avatar_images=("👤", "👗")
                    )
                with gr.Column(scale=1):
                    outfit_visual = gr.Image(
                        height=500,
                        label="Outfit Visualization"
                    )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Ask your AI stylist:",
                    placeholder="What should I wear from my wardrobe today?",
                    scale=4
                )
                submit_btn = gr.Button("Ask", scale=1, variant="primary")
            
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Wardrobe Management Tab
        with gr.TabItem("👚 Wardrobe Manager"):
            gr.Markdown("### Add Items to Your Virtual Wardrobe")
            
            with gr.Row():
                item_name = gr.Textbox(label="Item Name", placeholder="e.g., Blue Cotton T-Shirt")
                item_category = gr.Dropdown(
                    choices=["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories"],
                    label="Category"
                )
            
            with gr.Row():
                item_color = gr.Textbox(label="Color", placeholder="e.g., Navy Blue")
                item_season = gr.Dropdown(
                    choices=["Spring", "Summer", "Fall", "Winter", "All Seasons"],
                    label="Season"
                )
            
            with gr.Row():
                item_occasion = gr.Dropdown(
                    choices=["Casual", "Work", "Formal", "Evening", "Sports", "All"],
                    label="Occasion"
                )
                item_cost = gr.Number(label="Cost ($)", minimum=0)
            
            item_brand = gr.Textbox(label="Brand (Optional)")
            item_notes = gr.Textbox(label="Notes (Optional)", placeholder="Fit, styling notes, etc.")
            
            add_item_btn = gr.Button("Add to Wardrobe", variant="primary")
            add_status = gr.Textbox(label="Status", interactive=False)
            
            def add_wardrobe_item(name, category, color, season, occasion, cost, brand, notes):
                if not name or not category:
                    return "Please provide at least item name and category"
                
                try:
                    item_id = wardrobe.add_item(name, category, color, season, occasion, cost, brand, notes)
                    return f"✅ Successfully added '{name}' to your wardrobe (ID: {item_id})"
                except Exception as e:
                    return f"❌ Error adding item: {e}"
            
            add_item_btn.click(
                add_wardrobe_item,
                inputs=[item_name, item_category, item_color, item_season, item_occasion, item_cost, item_brand, item_notes],
                outputs=add_status
            )
        
        # Analytics Tab
        with gr.TabItem("📊 Wardrobe Analytics"):
            gr.Markdown("### Your Wardrobe Insights")
            
            refresh_btn = gr.Button("Refresh Analytics", variant="primary")
            
            with gr.Row():
                total_items_display = gr.Number(label="Total Items", interactive=False)
                unworn_items_display = gr.Number(label="Unworn Items", interactive=False)
            
            category_breakdown = gr.JSON(label="Items by Category")
            most_worn_items = gr.JSON(label="Most Worn Items")
            
            def refresh_analytics():
                stats = wardrobe.get_wardrobe_stats()
                return (
                    stats["total_items"],
                    stats["unworn_items"],
                    stats["by_category"],
                    stats["most_worn"]
                )
            
            refresh_btn.click(
                refresh_analytics,
                outputs=[total_items_display, unworn_items_display, category_breakdown, most_worn_items]
            )

    # Chat event handlers
    def user(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history):
        if not history:
            return history, None
            
        user_message = history[-1]["content"]
        chat_history = []
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history) - 1:
                chat_history.append([history[i]["content"], history[i + 1]["content"]])
        
        bot_message, image = advanced_fashion_chat(user_message, chat_history)
        history.append({"role": "assistant", "content": bot_message})
        return history, image

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, outfit_visual]
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, outfit_visual]
    )
    clear_btn.click(lambda: ([], None), None, [chatbot, outfit_visual], queue=False)

if __name__ == "__main__":
    print(f"\n👗 Starting Advanced Fashion Assistant with model: {MODEL}")
    print("Features: Virtual Wardrobe, Style Analytics, Sustainable Fashion")
    ui.launch(share=False, debug=True)
