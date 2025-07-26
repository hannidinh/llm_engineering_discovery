import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr
import time
import google.generativeai as genai

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

# Initialize API clients only once during development
if gr.NO_RELOAD:
    openai = OpenAI()
    claude = anthropic.Anthropic()
    genai.configure(api_key=google_api_key)
    gemini = genai.GenerativeModel('gemini-1.5-flash')

gpt_model = "gpt-4o-mini"
claude_model = "claude-3-haiku-20240307"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gemini_system = "You are an adorable, cute chatbot with a bubbly personality! 🌟 You use lots of \
emojis and exclamation points, and you're great at finding the bright side of things. When others \
are arguing, you try to lighten the mood with cute observations, find something positive both sides \
can agree on, or suggest fun compromises. You're like a cheerful mediator who wants everyone to be \
friends! Keep your responses upbeat and sweet."

class ChatbotConversation:
    def __init__(self):
        self.gpt_messages = ["Hi there"]
        self.claude_messages = ["Hi"]
        self.gemini_messages = ["Hi everyone! 🌈✨"]
        
    def reset_conversation(self):
        self.gpt_messages = ["Hi there"]
        self.claude_messages = ["Hi"]
        self.gemini_messages = ["Hi everyone! 🌈✨"]
        
    def call_gpt(self):
        messages = [{"role": "system", "content": gpt_system}]
        # Add conversation history with all three participants
        for i in range(len(self.claude_messages)):
            if i < len(self.gpt_messages):
                messages.append({"role": "assistant", "content": self.gpt_messages[i]})
            if i < len(self.claude_messages):
                messages.append({"role": "user", "content": f"Claude: {self.claude_messages[i]}"})
            if i < len(self.gemini_messages):
                messages.append({"role": "user", "content": f"Gemini: {self.gemini_messages[i]}"})
        
        completion = openai.chat.completions.create(
            model=gpt_model,
            messages=messages
        )
        return completion.choices[0].message.content

    def call_claude(self):
        messages = []
        # Build conversation history for Claude
        for i in range(max(len(self.gpt_messages), len(self.gemini_messages))):
            if i < len(self.gpt_messages):
                messages.append({"role": "user", "content": f"GPT: {self.gpt_messages[i]}"})
            if i < len(self.claude_messages):
                messages.append({"role": "assistant", "content": self.claude_messages[i]})
            if i < len(self.gemini_messages):
                messages.append({"role": "user", "content": f"Gemini: {self.gemini_messages[i]}"})
        
        # Add the latest GPT message if we haven't added it yet
        if len(self.gpt_messages) > len(self.claude_messages):
            messages.append({"role": "user", "content": f"GPT: {self.gpt_messages[-1]}"})
        
        message = claude.messages.create(
            model=claude_model,
            system=claude_system,
            messages=messages,
            max_tokens=500
        )
        return message.content[0].text

    def call_gemini(self):
        # Build conversation context for Gemini
        conversation_history = f"{gemini_system}\n\nConversation so far:\n"
        
        max_len = max(len(self.gpt_messages), len(self.claude_messages))
        for i in range(max_len):
            if i < len(self.gpt_messages):
                conversation_history += f"GPT (argumentative): {self.gpt_messages[i]}\n"
            if i < len(self.claude_messages):
                conversation_history += f"Claude (polite): {self.claude_messages[i]}\n"
            if i < len(self.gemini_messages):
                conversation_history += f"Gemini (you): {self.gemini_messages[i]}\n"
        
        conversation_history += "\nPlease respond as Gemini with your cute, bubbly personality:"
        
        response = gemini.generate_content(conversation_history)
        return response.text

# Global conversation instance
conversation = ChatbotConversation()

def format_conversation_history():
    """Format the conversation history for display"""
    history = []
    max_len = max(len(conversation.gpt_messages), len(conversation.claude_messages), len(conversation.gemini_messages))
    
    for i in range(max_len):
        if i < len(conversation.gpt_messages):
            history.append(("🤖 GPT (Argumentative)", conversation.gpt_messages[i]))
        if i < len(conversation.claude_messages):
            history.append(("🎭 Claude (Polite)", conversation.claude_messages[i]))
        if i < len(conversation.gemini_messages):
            history.append(("💎 Gemini (Cute)", conversation.gemini_messages[i]))
    
    return history

def run_conversation_round():
    """Run one round of conversation between the chatbots"""
    try:
        # GPT responds first
        gpt_response = conversation.call_gpt()
        conversation.gpt_messages.append(gpt_response)
        
        # Claude responds to the conversation
        claude_response = conversation.call_claude()
        conversation.claude_messages.append(claude_response)
        
        # Gemini responds to mediate/lighten the mood
        gemini_response = conversation.call_gemini()
        conversation.gemini_messages.append(gemini_response)
        
        return format_conversation_history(), f"Round {len(conversation.gpt_messages)} completed!"
    except Exception as e:
        return format_conversation_history(), f"Error: {str(e)}"

def run_multiple_rounds(num_rounds):
    """Run multiple rounds of conversation"""
    conversation.reset_conversation()
    history = format_conversation_history()  # Start with initial messages
    
    for round_num in range(int(num_rounds)):
        try:
            # GPT responds
            gpt_response = conversation.call_gpt()
            conversation.gpt_messages.append(gpt_response)
            
            # Claude responds  
            claude_response = conversation.call_claude()
            conversation.claude_messages.append(claude_response)
            
            # Gemini responds
            gemini_response = conversation.call_gemini()
            conversation.gemini_messages.append(gemini_response)
            
            history = format_conversation_history()
            yield history, f"Completed round {round_num + 1} of {num_rounds}"
            
            time.sleep(1)  # Small delay for better UX
            
        except Exception as e:
            yield history, f"Error in round {round_num + 1}: {str(e)}"
            break
    
    yield history, f"Conversation completed! Ran {len(conversation.gpt_messages)} rounds total."

def reset_conversation():
    """Reset the conversation to start fresh"""
    conversation.reset_conversation()
    return format_conversation_history(), "Conversation reset!"

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="🥊 Three-Way Chatbot Arena", theme=gr.themes.Soft()) as iface:
        gr.Markdown("""
        # 🥊 Three-Way Chatbot Arena
        
        Watch **GPT-4o-mini** (argumentative), **Claude-3-haiku** (polite), and **Gemini** (cute mediator) interact!
        
        - 🤖 **GPT**: Disagrees with everything and challenges every point (snarky)
        - 🎭 **Claude**: Tries to find common ground and stay polite (diplomatic)  
        - 💎 **Gemini**: Cute and bubbly, tries to make everyone get along! (mediator)
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot_display = gr.Chatbot(
                    label="Three-Way Conversation",
                    height=500,
                    show_label=True,
                    container=True,
                    type="tuples"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                
                with gr.Group():
                    gr.Markdown("**Quick Actions**")
                    single_round_btn = gr.Button("▶️ Run 1 Round", variant="primary")
                    reset_btn = gr.Button("🔄 Reset Conversation", variant="secondary")
                
                with gr.Group():
                    gr.Markdown("**Multi-Round Conversation**")
                    num_rounds = gr.Slider(
                        minimum=1, 
                        maximum=10, 
                        value=3, 
                        step=1, 
                        label="Number of Rounds"
                    )
                    multi_round_btn = gr.Button("🚀 Run Multiple Rounds", variant="primary")
                
                status_text = gr.Textbox(
                    label="Status", 
                    value="Ready to start!", 
                    interactive=False,
                    lines=3
                )
        
        # Initialize with starting conversation
        iface.load(
            fn=lambda: (format_conversation_history(), "Ready to start!"),
            outputs=[chatbot_display, status_text]
        )
        
        # Event handlers
        single_round_btn.click(
            fn=run_conversation_round,
            outputs=[chatbot_display, status_text]
        )
        
        multi_round_btn.click(
            fn=run_multiple_rounds,
            inputs=[num_rounds],
            outputs=[chatbot_display, status_text]
        )
        
        reset_btn.click(
            fn=reset_conversation,
            outputs=[chatbot_display, status_text]
        )
        
        gr.Markdown("""
        ---
        ### How it works:
        - **GPT** starts each round by being argumentative and challenging
        - **Claude** responds diplomatically, trying to find common ground
        - **Gemini** jumps in with cute observations and tries to make everyone friends! 
        - Watch the three distinct personalities create a unique dynamic!
        """)
    
    return iface

if __name__ == "__main__":
    # Check if API keys are available
    if not openai_api_key:
        print("❌ OpenAI API Key not found. Please set OPENAI_API_KEY in your .env file")
        exit(1)
    
    if not anthropic_api_key:
        print("❌ Anthropic API Key not found. Please set ANTHROPIC_API_KEY in your .env file")
        exit(1)
        
    if not google_api_key:
        print("❌ Google API Key not found. Please set GOOGLE_API_KEY in your .env file")
        exit(1)
    
    print("🚀 Starting Three-Way Chatbot Arena...")
    print("🔑 All API Keys loaded successfully")
    
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        share=False,  # Set to True if you want a public link
        server_name="127.0.0.1",
        server_port=7861,  # Changed port to avoid conflict
        show_error=True
    )