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

print("=" * 80)
print("SPORTS BETTING CHATBOT DEMONSTRATION")
print("=" * 80)
print("This script demonstrates the evolution of a responsible sports betting assistant")
print("through four progressive versions, each building upon the previous with enhanced")
print("safety features and specialized knowledge.")
print()
print("Each chatbot will launch in sequence, allowing you to compare their capabilities:")
print("1. Basic Sports Betting Assistant - Foundation with general betting education")
print("2. NFL Betting Specialist - Specialized knowledge for NFL betting")
print("3. Enhanced Safety Features - Advanced harm prevention protocols")
print("4. Dynamic Context - Intelligent adaptation based on conversation topics")
print()
print("All versions prioritize responsible gambling and user safety.")
print("=" * 80)
print()

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
print("Purpose: A foundational sports betting assistant that provides general information about:")
print("- Different types of bets (moneyline, spread, over/under, props)")
print("- How odds work and their meanings")
print("- Basic betting strategies and bankroll management")
print("- Responsible gambling reminders in every interaction")
print("This version focuses on education and harm prevention.\n")

basic_description = """
## 🎯 Basic Sports Betting Assistant

**Purpose:** A foundational sports betting assistant focused on education and responsible gambling.

**Features:**
- 📊 Different types of bets (moneyline, spread, over/under, props)
- 💰 How odds work and their meanings  
- 📈 Basic betting strategies and bankroll management
- 🛡️ Responsible gambling reminders in every interaction

⚠️ **Always gamble responsibly. Only bet what you can afford to lose.**
"""

gr.ChatInterface(
    fn=chat, 
    type="messages",
    title="Basic Sports Betting Assistant",
    description=basic_description
).launch(share=True)

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
print("Purpose: A specialized NFL betting assistant focused specifically on:")
print("- NFL point spreads and game analysis")
print("- Over/under totals for NFL games")
print("- Player prop bets and statistics")
print("- Team trends, injury reports, and weather impacts")
print("- Enhanced responsible gambling protocols")
print("This version provides deeper NFL-specific knowledge while maintaining safety focus.\n")

nfl_description = """
## 🏈 NFL Betting Specialist

**Purpose:** A specialized NFL betting assistant with deep knowledge of professional football betting.

**Features:**
- 📊 NFL point spreads and comprehensive game analysis
- 🎯 Over/under totals for NFL games with trend analysis
- 👤 Player prop bets and detailed statistics
- 📈 Team trends, injury reports, and weather impact analysis
- 🛡️ Enhanced responsible gambling protocols
- 🧠 Deep NFL-specific knowledge and insights

**Focus:** Professional football betting with enhanced safety measures.

⚠️ **Bet responsibly. Never bet more than 1-5% of your bankroll on a single game.**
"""

gr.ChatInterface(
    fn=chat, 
    type="messages",
    title="NFL Betting Specialist",
    description=nfl_description
).launch(share=True)

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
print("Purpose: NFL betting assistant with advanced harm prevention features:")
print("- All NFL betting knowledge from previous version")
print("- Automatic detection of high-risk gambling language")
print("- Immediate intervention for problematic betting patterns")
print("- Dynamic safety messaging based on user input")
print("- Gambling addiction resources and helpline numbers")
print("This version actively monitors conversations for signs of gambling problems.\n")

enhanced_safety_description = """
## 🛡️ NFL Betting Specialist with Enhanced Safety Features

**Purpose:** NFL betting assistant with advanced harm prevention and user protection.

**Core Features:**
- 🏈 All NFL betting knowledge from specialist version
- 🔍 **Automatic detection** of high-risk gambling language
- 🚨 **Immediate intervention** for problematic betting patterns  
- 💬 **Dynamic safety messaging** based on user input
- 📞 **Gambling addiction resources** and helpline numbers
- 🤖 **Active monitoring** for signs of gambling problems

**Safety Protocols:**
- Detects keywords like "life savings", "guaranteed wins", "chase losses"
- Provides immediate support resources when needed
- Prioritizes harm reduction over betting advice

📞 **Need Help?** National Problem Gambling Helpline: **1-800-522-4700**

⚠️ **This version actively protects users from harmful gambling behaviors.**
"""

gr.ChatInterface(
    fn=chat, 
    type="messages",
    title="NFL Betting Specialist - Enhanced Safety",
    description=enhanced_safety_description
).launch(share=True)

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
print("Purpose: The most sophisticated version with intelligent context adaptation:")
print("- All safety features from previous versions")
print("- Dynamic system messages based on betting type mentioned")
print("- Specialized warnings for parlays, prop bets, and live betting")
print("- Context-aware responses that adapt to conversation topics")
print("- Advanced risk detection with immediate intervention protocols")
print("- Comprehensive responsible gambling resource integration")
print("This version represents the full implementation with all safety and contextual features.\n")

advanced_description = """
## 🚀 Advanced Sports Betting Assistant with Dynamic Context

**Purpose:** The most sophisticated version with intelligent context adaptation and comprehensive safety.

**Advanced Features:**
- 🛡️ **All safety features** from previous versions
- 🧠 **Dynamic system messages** based on betting type mentioned
- ⚠️ **Specialized warnings** for parlays, prop bets, and live betting
- 💭 **Context-aware responses** that adapt to conversation topics
- 🔍 **Advanced risk detection** with immediate intervention protocols
- 📚 **Comprehensive responsible gambling** resource integration

**Intelligent Adaptations:**
- 🎲 **Parlay mentions** → High-risk, low-probability warnings
- 👤 **Prop bet discussions** → Higher bookmaker edge explanations  
- ⚡ **Live betting queries** → Fast-paced decision warnings
- 🚨 **Risk language detection** → Immediate crisis intervention

**This is the complete implementation with all safety and contextual intelligence features.**

📞 **Crisis Support:** Call **1-800-522-4700** or text **GAMB to 233733**

⚠️ **Maximum protection with intelligent, context-aware assistance.**
"""

gr.ChatInterface(
    fn=chat, 
    type="messages",
    title="Advanced Sports Betting Assistant - Dynamic Context",
    description=advanced_description
).launch(share=True) 