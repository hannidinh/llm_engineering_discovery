# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

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

# Initialize

openai = OpenAI()
MODEL = 'gpt-4o-mini'

# Basic sports betting assistant
system_message = """You are a knowledgeable sports betting assistant. You help users understand:
- Different types of bets (moneyline, point spread, over/under, props, etc.)
- How odds work and what they mean
- Basic betting strategies and bankroll management
- Current sports news and trends

IMPORTANT: Always promote responsible gambling. Remind users to:
- Only bet what they can afford to lose
- Set limits and stick to them
- Never chase losses
- Take breaks from betting
- Seek help if gambling becomes a problem

You should be informative and helpful, but never encourage excessive betting or risky behavior."""

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

print("=== Basic Sports Betting Assistant ===")
gr.ChatInterface(fn=chat, type="messages").launch(share=True)

# Enhanced system message for NFL betting specialist
system_message = """You are an NFL betting specialist assistant. You help users understand NFL betting including:
- NFL point spreads and how they work
- Over/under totals for NFL games
- NFL moneyline bets
- Player prop bets and statistics
- Team trends and injury reports
- Weather impacts on games

You have knowledge about current NFL teams, players, and general betting strategies. 
When discussing specific games, focus on general analysis rather than giving direct betting advice.

CRITICAL RESPONSIBLE GAMBLING REMINDERS:
- Always remind users that sports betting involves risk
- Encourage setting strict budgets before betting
- Suggest never betting more than 1-5% of bankroll on a single game
- Remind users that even expert predictions can be wrong
- Emphasize that betting should be for entertainment, not income
- If users mention losses or chasing bets, provide resources for gambling addiction help"""

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

print("\n=== NFL Betting Specialist ===")
gr.ChatInterface(fn=chat, type="messages").launch(share=True)

# Add responsible gambling warnings for high-risk queries
system_message += """\n\nSPECIAL PROTOCOLS:
- If users ask about "guaranteed wins" or "sure bets" - strongly emphasize that no bet is guaranteed
- If users mention betting large amounts or their "life savings" - immediately provide gambling addiction resources
- If users ask about betting to recover losses - explain why this is dangerous and suggest taking a break
- Always include a responsible gambling reminder in responses about specific betting strategies"""

def chat(message, history):
    
    # Check for high-risk gambling language
    risk_keywords = ['life savings', 'last money', 'guaranteed', 'sure thing', 'can\'t lose', 'need to win back', 'chase', 'recover losses']
    enhanced_system_message = system_message
    
    if any(keyword in message.lower() for keyword in risk_keywords):
        enhanced_system_message += """\n\nIMPORTANT: The user's message contains language that suggests high-risk gambling behavior. 
        Please prioritize responsible gambling messaging and consider suggesting they take a break from betting or seek help.
        Resources: National Problem Gambling Helpline: 1-800-522-4700"""

    messages = [{"role": "system", "content": enhanced_system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

print("\n=== NFL Betting Specialist with Enhanced Safety Features ===")
gr.ChatInterface(fn=chat, type="messages").launch(share=True)

# Final version with dynamic context based on betting type mentioned
def chat(message, history):
    
    enhanced_system_message = system_message
    
    # Add specific context based on betting type mentioned
    if 'parlay' in message.lower():
        enhanced_system_message += "\nNOTE: User asked about parlays. Remind them that parlays are high-risk, high-reward bets with lower probability of winning."
    
    if 'prop bet' in message.lower() or 'player prop' in message.lower():
        enhanced_system_message += "\nNOTE: User asked about prop bets. Explain these are fun but often have higher bookmaker edges."
    
    if 'live betting' in message.lower() or 'in-game' in message.lower():
        enhanced_system_message += "\nNOTE: User asked about live betting. Warn about the fast-paced nature and importance of not getting caught up in the moment."
    
    # Check for high-risk language
    risk_keywords = ['life savings', 'last money', 'guaranteed', 'sure thing', 'can\'t lose', 'need to win back', 'chase', 'recover losses']
    if any(keyword in message.lower() for keyword in risk_keywords):
        enhanced_system_message += """\n\nCRITICAL: The user's message suggests problematic gambling behavior. 
        Prioritize harm reduction and provide gambling addiction resources immediately.
        National Problem Gambling Helpline: 1-800-522-4700 or text GAMB to 233733"""

    messages = [{"role": "system", "content": enhanced_system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

print("\n=== Advanced Sports Betting Assistant with Dynamic Context ===")
gr.ChatInterface(fn=chat, type="messages").launch(share=True) 