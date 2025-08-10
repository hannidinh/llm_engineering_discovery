# Fashion Challenge & Style Game
# Gamified fashion learning and styling challenges

import os
import json
from dotenv import load_dotenv
import ollama
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import random
from datetime import datetime
import sqlite3

# TTS import
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

load_dotenv(override=True)

MODEL = "llama3.2"
VOICE_INDEX = 0

class FashionChallengeGame:
    """Gamified fashion challenges and learning"""
    
    def __init__(self, db_path="fashion_game.db"):
        self.db_path = db_path
        self.init_database()
        self.challenges = self.load_challenges()
    
    def init_database(self):
        """Initialize game database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id TEXT,
                completed_date DATE,
                score INTEGER,
                feedback TEXT,
                time_taken INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS style_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                style_element TEXT,
                preference_score INTEGER,
                last_updated DATE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_challenges(self):
        """Load fashion challenges"""
        return {
            "color_coordination": {
                "title": "Color Master Challenge",
                "description": "Create outfits using perfect color combinations",
                "difficulty": "Beginner",
                "scenarios": [
                    {"theme": "Monochromatic Magic", "colors": ["navy", "light blue", "white"], "occasion": "office"},
                    {"theme": "Complementary Contrast", "colors": ["blue", "orange"], "occasion": "casual"},
                    {"theme": "Analogous Harmony", "colors": ["red", "orange", "yellow"], "occasion": "party"}
                ]
            },
            "seasonal_styling": {
                "title": "Seasonal Style Transition",
                "description": "Adapt outfits for changing seasons",
                "difficulty": "Intermediate",
                "scenarios": [
                    {"theme": "Summer to Fall", "pieces": ["sundress", "cardigan", "boots"], "challenge": "layer appropriately"},
                    {"theme": "Winter to Spring", "pieces": ["sweater", "light jacket", "scarf"], "challenge": "brighten the look"}
                ]
            },
            "occasion_dressing": {
                "title": "Dress for Success",
                "description": "Perfect outfits for specific occasions",
                "difficulty": "Advanced",
                "scenarios": [
                    {"occasion": "job interview", "style": "professional", "budget": "moderate"},
                    {"occasion": "first date", "style": "casual-chic", "budget": "low"},
                    {"occasion": "wedding guest", "style": "elegant", "budget": "flexible"}
                ]
            },
            "sustainable_fashion": {
                "title": "Eco-Style Champion",
                "description": "Create stylish looks with sustainable principles",
                "difficulty": "Expert",
                "scenarios": [
                    {"theme": "Capsule Wardrobe", "items": 10, "outfits": 15, "challenge": "maximize versatility"},
                    {"theme": "Thrift Store Chic", "budget": 50, "challenge": "create 3 complete outfits"},
                    {"theme": "Upcycle Challenge", "items": ["old jeans", "basic tee", "scarf"], "challenge": "transform into trendy look"}
                ]
            }
        }
    
    def get_random_challenge(self, difficulty=None):
        """Get a random challenge"""
        available_challenges = list(self.challenges.keys())
        if difficulty:
            available_challenges = [c for c in available_challenges 
                                  if self.challenges[c]["difficulty"].lower() == difficulty.lower()]
        
        challenge_key = random.choice(available_challenges)
        challenge = self.challenges[challenge_key].copy()
        challenge["id"] = challenge_key
        challenge["scenario"] = random.choice(challenge["scenarios"])
        return challenge
    
    def evaluate_outfit(self, challenge, user_response):
        """Evaluate user's outfit solution"""
        # Simplified scoring system
        score = 0
        feedback = []
        
        # Color coordination check
        if challenge["id"] == "color_coordination":
            required_colors = challenge["scenario"]["colors"]
            mentioned_colors = [color for color in required_colors 
                              if color.lower() in user_response.lower()]
            score += len(mentioned_colors) * 20
            if len(mentioned_colors) == len(required_colors):
                feedback.append("Excellent color coordination!")
            else:
                feedback.append(f"Try incorporating {', '.join(required_colors)}")
        
        # Occasion appropriateness
        if "occasion" in challenge.get("scenario", {}):
            occasion = challenge["scenario"]["occasion"]
            if occasion.lower() in user_response.lower():
                score += 25
                feedback.append("Great occasion awareness!")
        
        # Creativity and detail
        if len(user_response.split()) > 20:
            score += 15
            feedback.append("Love the detailed description!")
        
        # Random bonus for encouragement
        if random.random() > 0.5:
            score += 10
            feedback.append("Bonus points for style creativity!")
        
        return min(score, 100), feedback

# Fashion education content
FASHION_TIPS = {
    "color_theory": [
        "The color wheel is your best friend - complementary colors create dynamic looks",
        "Monochromatic outfits are foolproof and sophisticated",
        "Use the 60-30-10 rule: 60% dominant color, 30% secondary, 10% accent",
        "Neutral colors (black, white, gray, beige) pair with everything"
    ],
    "body_styling": [
        "Dress for your body, not the trends",
        "Create visual balance with proportions",
        "Highlight your favorite features",
        "Fit is more important than brand or price"
    ],
    "sustainable_fashion": [
        "Quality over quantity - invest in timeless pieces",
        "Learn to mix and match for maximum outfit combinations",
        "Consider cost-per-wear when shopping",
        "Donate or sell items you no longer wear"
    ],
    "trend_integration": [
        "Choose one trend element per outfit to avoid looking costume-y",
        "Incorporate trends through accessories for easy updates",
        "Classic pieces + trendy accessories = perfect balance",
        "Not every trend needs to be followed - choose what suits you"
    ]
}

def create_challenge_visual(challenge):
    """Create a visual representation of the challenge"""
    try:
        width, height = 600, 500
        
        # Challenge-specific color schemes
        color_schemes = {
            "color_coordination": [(255, 182, 193), (135, 206, 235), (255, 255, 224)],
            "seasonal_styling": [(210, 180, 140), (255, 160, 122), (144, 238, 144)],
            "occasion_dressing": [(230, 230, 250), (255, 240, 245), (240, 248, 255)],
            "sustainable_fashion": [(152, 251, 152), (240, 230, 140), (221, 160, 221)]
        }
        
        colors = color_schemes.get(challenge["id"], [(245, 245, 250)])
        bg_color = colors[0] if colors else (245, 245, 250)
        
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Fonts
        try:
            title_font = ImageFont.truetype("Arial.ttf", 24)
            subtitle_font = ImageFont.truetype("Arial.ttf", 18)
            text_font = ImageFont.truetype("Arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Title
        title = f"🎯 {challenge['title']}"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 30), title, fill=(60, 60, 100), font=title_font)
        
        # Difficulty badge
        difficulty = f"Level: {challenge['difficulty']}"
        draw.text((30, 80), difficulty, fill=(100, 100, 150), font=subtitle_font)
        
        # Description
        desc_lines = wrap_text(challenge['description'], 50)
        y_pos = 120
        for line in desc_lines:
            draw.text((30, y_pos), line, fill=(80, 80, 120), font=text_font)
            y_pos += 20
        
        # Scenario details
        scenario = challenge.get('scenario', {})
        y_pos += 20
        
        if 'theme' in scenario:
            theme_text = f"Theme: {scenario['theme']}"
            draw.text((30, y_pos), theme_text, fill=(120, 80, 120), font=subtitle_font)
            y_pos += 30
        
        if 'colors' in scenario:
            colors_text = f"Required Colors: {', '.join(scenario['colors'])}"
            draw.text((30, y_pos), colors_text, fill=(80, 120, 80), font=text_font)
            y_pos += 25
        
        if 'occasion' in scenario:
            occasion_text = f"Occasion: {scenario['occasion']}"
            draw.text((30, y_pos), occasion_text, fill=(120, 80, 80), font=text_font)
            y_pos += 25
        
        if 'challenge' in scenario:
            challenge_text = f"Challenge: {scenario['challenge']}"
            draw.text((30, y_pos), challenge_text, fill=(80, 80, 120), font=text_font)
        
        # Decorative elements
        for i in range(5):
            x = random.randint(450, 580)
            y = random.randint(350, 480)
            draw.ellipse([x, y, x+20, y+20], fill=(200, 200, 220, 100))
        
        return img
        
    except Exception as e:
        print(f"Error creating challenge visual: {e}")
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

def get_fashion_tip():
    """Get a random fashion tip"""
    category = random.choice(list(FASHION_TIPS.keys()))
    tip = random.choice(FASHION_TIPS[category])
    return f"💡 {category.replace('_', ' ').title()}: {tip}"

# Initialize game
game = FashionChallengeGame()

# Game system message
game_system_message = """You are StyleCoach, a fun and encouraging fashion game instructor. You help users learn fashion through interactive challenges and games.

Your role:
1. Present fashion challenges clearly and enthusiastically
2. Evaluate outfit solutions with constructive feedback
3. Provide fashion education and tips
4. Encourage creativity and personal style
5. Make learning fashion fun and accessible

Be supportive, educational, and celebrate user creativity. Provide specific feedback on color coordination, occasion appropriateness, styling techniques, and overall creativity."""

def fashion_game_chat(message, history, current_challenge=None):
    """Chat function for the fashion game"""
    conversation = [{"role": "system", "content": game_system_message}]
    
    if current_challenge:
        challenge_context = f"Current challenge: {json.dumps(current_challenge)}"
        conversation.append({"role": "system", "content": challenge_context})
    
    for human, assistant in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": assistant})
    
    conversation.append({"role": "user", "content": message})
    
    try:
        response = ollama.chat(model=MODEL, messages=conversation)
        reply = response['message']['content']
        
        # Add fashion tip occasionally
        if random.random() > 0.7:  # 30% chance
            tip = get_fashion_tip()
            reply += f"\n\n{tip}"
        
        if TTS_AVAILABLE and len(reply) < 300:
            talker(reply, VOICE_INDEX)
        
        return reply
        
    except Exception as e:
        return f"Error: {e}"

def talker(message, voice_index=None):
    """TTS function"""
    if not TTS_AVAILABLE:
        return
    
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 50)
        
        voices = engine.getProperty('voices')
        if voices and voice_index is not None and 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
        
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# Gradio interface for the fashion game
with gr.Blocks(title="Fashion Challenge Game", theme=gr.themes.Base()) as ui:
    gr.Markdown("# 🎮 Fashion Challenge & Style Game")
    gr.Markdown("Learn fashion through fun challenges and interactive styling games!")
    
    # Game state
    current_challenge = gr.State(None)
    
    with gr.Tabs():
        # Challenge Tab
        with gr.TabItem("🎯 Style Challenges"):
            with gr.Row():
                with gr.Column(scale=1):
                    challenge_image = gr.Image(
                        height=400,
                        label="Current Challenge",
                        show_label=True
                    )
                    
                    with gr.Row():
                        new_challenge_btn = gr.Button("New Challenge", variant="primary")
                        difficulty_filter = gr.Dropdown(
                            choices=["Any", "Beginner", "Intermediate", "Advanced", "Expert"],
                            value="Any",
                            label="Difficulty"
                        )
                
                with gr.Column(scale=1):
                    challenge_chat = gr.Chatbot(
                        height=400,
                        label="Challenge Coach",
                        type="messages",
                        avatar_images=("👤", "👨‍🏫")
                    )
                    
                    solution_input = gr.Textbox(
                        label="Your Outfit Solution:",
                        placeholder="Describe your outfit idea for this challenge...",
                        lines=3
                    )
                    
                    submit_solution_btn = gr.Button("Submit Solution", variant="secondary")
            
            # Score display
            with gr.Row():
                score_display = gr.Number(label="Challenge Score", interactive=False)
                feedback_display = gr.Textbox(label="Feedback", interactive=False, lines=2)
        
        # Learning Center Tab
        with gr.TabItem("📚 Fashion Learning"):
            gr.Markdown("### Fashion Education Center")
            
            with gr.Row():
                with gr.Column():
                    topic_selector = gr.Dropdown(
                        choices=list(FASHION_TIPS.keys()),
                        label="Learning Topic",
                        value="color_theory"
                    )
                    get_tip_btn = gr.Button("Get Fashion Tip", variant="primary")
                    
                with gr.Column():
                    tip_display = gr.Textbox(
                        label="Fashion Tip",
                        lines=3,
                        interactive=False
                    )
            
            gr.Markdown("### Quick Style Consultation")
            style_question = gr.Textbox(
                label="Ask about fashion, styling, or trends:",
                placeholder="How do I style a blazer casually? What colors go with navy blue?"
            )
            ask_btn = gr.Button("Ask StyleCoach", variant="secondary")
            style_answer = gr.Textbox(label="Answer", lines=4, interactive=False)
        
        # Progress Tab
        with gr.TabItem("🏆 Your Progress"):
            gr.Markdown("### Style Journey")
            
            refresh_progress_btn = gr.Button("Refresh Progress", variant="primary")
            
            with gr.Row():
                total_challenges = gr.Number(label="Challenges Completed", interactive=False)
                avg_score = gr.Number(label="Average Score", interactive=False)
            
            recent_challenges = gr.JSON(label="Recent Challenge History")

    # Event handlers
    def generate_new_challenge(difficulty):
        """Generate a new challenge"""
        diff_filter = None if difficulty == "Any" else difficulty
        challenge = game.get_random_challenge(diff_filter)
        
        # Create welcome message
        welcome_msg = f"🎯 **{challenge['title']}**\n\n"
        welcome_msg += f"*Difficulty: {challenge['difficulty']}*\n\n"
        welcome_msg += f"{challenge['description']}\n\n"
        
        scenario = challenge.get('scenario', {})
        if 'theme' in scenario:
            welcome_msg += f"**Theme:** {scenario['theme']}\n"
        if 'colors' in scenario:
            welcome_msg += f"**Colors to use:** {', '.join(scenario['colors'])}\n"
        if 'occasion' in scenario:
            welcome_msg += f"**Occasion:** {scenario['occasion']}\n"
        if 'challenge' in scenario:
            welcome_msg += f"**Challenge:** {scenario['challenge']}\n"
        
        welcome_msg += "\nDescribe your outfit solution!"
        
        challenge_visual = create_challenge_visual(challenge)
        initial_chat = [{"role": "assistant", "content": welcome_msg}]
        
        return challenge, challenge_visual, initial_chat, "", ""
    
    def submit_outfit_solution(challenge, solution, chat_history):
        """Evaluate outfit solution"""
        if not challenge or not solution:
            return chat_history, 0, "Please provide a solution first!"
        
        score, feedback = game.evaluate_outfit(challenge, solution)
        
        # Create feedback message
        feedback_msg = f"**Your Score: {score}/100** ⭐\n\n"
        feedback_msg += "**Feedback:**\n"
        feedback_msg += "\n".join([f"• {fb}" for fb in feedback])
        
        if score >= 80:
            feedback_msg += "\n\n🎉 Excellent work! You're a styling star!"
        elif score >= 60:
            feedback_msg += "\n\n👍 Good job! Keep practicing to improve!"
        else:
            feedback_msg += "\n\n💪 Keep trying! Fashion is all about experimentation!"
        
        # Add to chat history
        chat_history.append({"role": "user", "content": f"My solution: {solution}"})
        chat_history.append({"role": "assistant", "content": feedback_msg})
        
        return chat_history, score, "; ".join(feedback)
    
    def get_random_tip(topic):
        """Get a tip for the selected topic"""
        if topic in FASHION_TIPS:
            tip = random.choice(FASHION_TIPS[topic])
            return f"💡 {topic.replace('_', ' ').title()}: {tip}"
        return "Select a topic to get fashion tips!"
    
    def ask_style_question(question, chat_history):
        """Answer style questions"""
        if not question:
            return "Please ask a question first!"
        
        answer = fashion_game_chat(question, [])
        return answer

    # Wire up the interface
    new_challenge_btn.click(
        generate_new_challenge,
        inputs=[difficulty_filter],
        outputs=[current_challenge, challenge_image, challenge_chat, score_display, feedback_display]
    )
    
    submit_solution_btn.click(
        submit_outfit_solution,
        inputs=[current_challenge, solution_input, challenge_chat],
        outputs=[challenge_chat, score_display, feedback_display]
    )
    
    get_tip_btn.click(
        get_random_tip,
        inputs=[topic_selector],
        outputs=[tip_display]
    )
    
    ask_btn.click(
        ask_style_question,
        inputs=[style_question, challenge_chat],
        outputs=[style_answer]
    )

if __name__ == "__main__":
    print(f"\n🎮 Starting Fashion Challenge Game with model: {MODEL}")
    print("Learn fashion through fun, interactive challenges!")
    ui.launch(share=False, debug=True)
