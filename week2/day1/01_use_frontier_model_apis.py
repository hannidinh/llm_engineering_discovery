import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

import google.generativeai

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

# Connect to OpenAI, Anthropic

openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Software Engineer"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

# GPT-4o-mini
completion = openai.chat.completions.create(
    model='gpt-4o-mini', 
    messages=prompts,
    temperature=0.7  # Add temperature for variation
)
print(completion.choices[0].message.content)

# GPT-4.1-mini
# Temperature setting controls creativity
completion = openai.chat.completions.create(
    model='gpt-4.1-mini',
    messages=prompts,
    temperature=0.7
)
print(completion.choices[0].message.content)

# GPT-4.1-nano - extremely fast and cheap
completion = openai.chat.completions.create(
    model='gpt-4.1-nano',
    messages=prompts,
    temperature=0.7  # Add temperature for variation
)
print(completion.choices[0].message.content)

# GPT-4.1
completion = openai.chat.completions.create(
    model='gpt-4.1',
    messages=prompts,
    temperature=0.4
)
print(completion.choices[0].message.content)

# If you have access to this, here is the reasoning model o3-mini
# This is trained to think through its response before replying
# So it will take longer but the answer should be more reasoned - not that this helps..
completion = openai.chat.completions.create(
    model='o3-mini',
    messages=prompts,
    temperature=0.7  # Add temperature for variation
)
print(completion.choices[0].message.content)

# Claude 3.7 Sonnet
# API needs system message provided separately from user prompt
# Also adding max_tokens
message = claude.messages.create(
    model="claude-3-7-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

print(message.content[0].text)

# Claude 3.7 Sonnet again
# Now let's add in streaming back results
# If the streaming looks strange, then please see the note below this cell!

result = claude.messages.stream(
    model="claude-3-7-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

with result as stream:
    for text in stream.text_stream:
            clean_text = text.replace("\n", " ").replace("\r", " ")
            print(clean_text, end="", flush=True)

# The API for Gemini has a slightly different structure.
# I've heard that on some PCs, this Gemini code causes the Kernel to crash.
# If that happens to you, please skip this cell and use the next cell instead - an alternative approach.

gemini = google.generativeai.GenerativeModel(
    model_name='gemini-2.0-flash',
    system_instruction=system_message
)
response = gemini.generate_content(
    user_prompt,
    generation_config=google.generativeai.types.GenerationConfig(
        temperature=0.7  # Add temperature for variation
    )
)
print(response.text)

# As an alternative way to use Gemini that bypasses Google's python API library,
# Google released endpoints that means you can use Gemini via the client libraries for OpenAI!
# We're also trying Gemini's latest reasoning/thinking model

# gemini_via_openai_client = OpenAI(
#     api_key=google_api_key, 
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# response = gemini_via_openai_client.chat.completions.create(
#     model="gemini-2.5-flash-preview-04-17",
#     messages=prompts
# )
# print(response.choices[0].message.content)

# Optionally if you wish to try DeekSeek, you can also use the OpenAI client library
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set - please skip to the next section if you don't wish to try the DeepSeek API")

# Using DeepSeek Chat

deepseek_via_openai_client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)

response = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-chat",
    messages=prompts,
    temperature=0.7
)

print(response.choices[0].message.content)

challenge = [{"role": "system", "content": "You are a helpful assistant"},
             {"role": "user", "content": "How many words are there in your answer to this prompt"}]

# Using DeepSeek Chat with a harder question! And streaming results

stream = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-chat",
    messages=challenge,
    temperature=0.7,  # Add temperature for variation
    stream=True
)

reply = ""
for chunk in stream:
    content_chunk = chunk.choices[0].delta.content or ''
    reply += content_chunk
    # Print each chunk as it arrives for streaming effect
    print(content_chunk, end='', flush=True)

print()  # Add newline after streaming
print("Number of words:", len(reply.split(" ")))

# Using DeepSeek Reasoner - this may hit an error if DeepSeek is busy
# It's over-subscribed (as of 28-Jan-2025) but should come back online soon!
# If this fails, come back to this in a few days..
response = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-reasoner",
    messages=challenge,
    temperature=0.7  # Add temperature for variation
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(reasoning_content)
print(content)
print("Number of words:", len(content.split(" ")))