# Enhanced Brochure Generator - Improved Version
# Create a product that builds a Brochure for a company to be used for prospective clients, investors and potential recruits
# Input: company name and website
# 
# IMPROVEMENTS:
# - Multiple audience-specific templates (investor, customer, talent, general, humorous)
# - Auto-adaptive template selection based on company analysis
# - Enhanced link classification with priority and reasoning
# - Multi-stage generation for better quality
# - Streaming support with template selection
# - Comprehensive usage examples

import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time
import sys

# For this terminal script, we'll use simple print-based output
# IPython display is only for Jupyter notebooks
HAS_IPYTHON = False

class Markdown:
    def __init__(self, content):
        self.data = content

class DisplayHandle:
    def __init__(self):
        self.display_id = "fallback"

def display(content, display_id=False):
    return DisplayHandle()

def update_display(content, display_id=None):
    pass  # No-op for terminal

from openai import OpenAI

# Initialize and contants

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please check again")

MODEL = 'gpt-4o-mini'
openai = OpenAI()

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    """
        A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"

        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
    
PAYLOAD_CMS = "https://payloadcms.com"
payloadcms = Website(PAYLOAD_CMS)
payloadcms.links

# ENHANCED: Improved link classification with priority and reasoning
link_system_prompt = """You are provided with a list of links found on a webpage. 
You are able to decide which of the links would be most relevant to include in a brochure about the company.

Prioritize links in this order:
1. About/Company pages - core company information
2. Products/Services pages - what the company offers
3. Team/Leadership pages - key people and culture
4. Careers/Jobs pages - employment opportunities
5. Case Studies/Portfolio - client success stories
6. News/Blog - recent developments

You should response in JSON as in this example:"""
link_system_prompt += """
    {
        "links": [
            {
                "type": "about page", 
                "url": "https://full.url/goes/here/about",
                "priority": "high",
                "reason": "Contains core company information"
            },
            {
                "type": "careers page", 
                "url":"https://another.full.url/careers",
                "priority": "medium", 
                "reason": "Shows company culture and opportunities"
            }
        ]
    }
"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} -"
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
        ],
        response_format={"type":"json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

# Second step: make the brochure
def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

# ENHANCED: Multiple audience-specific brochure templates
BROCHURE_TEMPLATES = {
    "investor": """You are an assistant that creates professional investment-focused brochures. 
    Structure your response with these sections:
    
    ## Executive Summary
    ## Market Opportunity  
    ## Product/Service Overview
    ## Business Model
    ## Team & Leadership
    ## Financial Highlights (if available)
    ## Growth Strategy
    ## Investment Highlights
    
    Focus on metrics, scalability, and competitive advantages. Use professional tone.""",
    
    "customer": """You are an assistant that creates customer-focused marketing brochures.
    Structure your response with these sections:
    
    ## What We Do
    ## Why Choose Us
    ## Our Solutions
    ## Success Stories
    ## Our Team
    ## Get Started
    
    Focus on benefits, solutions, and customer value. Use engaging, benefit-driven language.""",
    
    "talent": """You are an assistant that creates recruitment-focused brochures.
    Structure your response with these sections:
    
    ## About Our Company
    ## Our Mission & Values
    ## Why Work With Us
    ## Career Opportunities
    ## Team Culture
    ## Benefits & Perks
    ## Apply Now
    
    Focus on company culture, growth opportunities, and employee value proposition.""",
    
    "general": """You are an assistant that creates comprehensive company brochures.
    Structure your response with these sections:
    
    ## Company Overview
    ## What We Offer
    ## Our Story
    ## Meet The Team
    ## Why Choose Us
    ## Join Our Journey
    
    Balance information for all audiences - customers, investors, and potential employees.""",
    
    "humorous": """You are an assistant that creates humorous, entertaining company brochures.
    Structure your response with these sections:
    
    ## The Company (But Make It Fun)
    ## What We Actually Do
    ## Meet The Humans
    ## Why We're Awesome (Allegedly)
    ## Work With Us (We Have Snacks)
    
    Use wit, humor, and personality while still conveying important company information."""
}

# Default template selection
system_prompt = BROCHURE_TEMPLATES["humorous"]  # You can change this to any template

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

def create_brochure(company_name, url):
    """Original brochure creation function"""
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
    )
    result = response.choices[0].message.content
    print(result)
    return result

def stream_brochure(company_name, url):
    """Original streaming brochure function"""
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
        stream=True
    )
    response = ""
    
    if HAS_IPYTHON:
        # Use IPython display for Jupyter notebooks
        display_handle = display(Markdown(""), display_id=True)
        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            response_clean = response.replace("```", "").replace("markdown", "")
            update_display(Markdown(response_clean), display_id=display_handle.display_id)
    else:
        # Use terminal-friendly streaming for regular Python
        print("Generating brochure...\n")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                time.sleep(0.01)  # Small delay for typewriter effect
        print("\n")  # Final newline

# =====================================================================
# ENHANCED FUNCTIONS - NEW INTELLIGENT FEATURES
# =====================================================================

def analyze_company_type(company_name, url):
    """Analyze the company to determine the best brochure template"""
    analysis_prompt = """Analyze this company and determine what type of brochure would be most appropriate.
    
    Consider:
    - Is this a B2B or B2C company?
    - What stage is the company in (startup, growth, established)?
    - What's their primary goal (funding, customers, talent)?
    - What industry are they in?
    
    Respond with JSON:
    {
        "primary_audience": "investor|customer|talent|general",
        "company_stage": "startup|growth|established",
        "industry": "tech|saas|ecommerce|consulting|other",
        "tone": "professional|casual|humorous|technical",
        "reasoning": "Brief explanation of your recommendation"
    }
    """
    
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": f"Analyze {company_name} based on their website content:\n{get_all_details(url)[:3000]}"}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def create_adaptive_brochure(company_name, url, template_type=None):
    """Create a brochure with adaptive template selection"""
    if template_type is None:
        # Auto-select template based on company analysis
        analysis = analyze_company_type(company_name, url)
        template_type = analysis["primary_audience"]
        print(f"Auto-selected '{template_type}' template based on analysis: {analysis['reasoning']}")
    
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": BROCHURE_TEMPLATES[template_type]},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
    )
    result = response.choices[0].message.content
    return result

def create_multi_stage_brochure(company_name, url):
    """Create brochure using multi-stage approach for better quality"""
    
    # Stage 1: Extract key information
    extraction_prompt = """Extract key information from this company's website content.
    
    Return JSON with:
    {
        "company_name": "...",
        "tagline": "...",
        "main_products": ["...", "..."],
        "target_customers": "...",
        "key_differentiators": ["...", "..."],
        "company_size": "...",
        "founding_info": "...",
        "culture_values": ["...", "..."]
    }
    """
    
    extraction_response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": get_all_details(url)[:4000]}
        ],
        response_format={"type": "json_object"}
    )
    
    key_info = json.loads(extraction_response.choices[0].message.content)
    print("Extracted key information:", key_info)
    
    # Stage 2: Generate brochure using extracted information
    structured_prompt = f"""Create a compelling brochure using this structured information:
    
    Company Data: {json.dumps(key_info, indent=2)}
    
    Create a professional yet engaging brochure with:
    - Compelling headline
    - Clear value proposition  
    - Key benefits and features
    - Social proof (if available)
    - Strong call-to-action
    
    Format in markdown with clear sections."""
    
    brochure_response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional marketing copywriter creating compelling brochures."},
            {"role": "user", "content": structured_prompt}
        ],
    )
    
    return brochure_response.choices[0].message.content

def stream_adaptive_brochure(company_name, url, template_type=None):
    """Stream brochure with adaptive template selection"""
    if template_type is None:
        # Quick analysis for template selection
        print("Analyzing company to select best template...")
        analysis = analyze_company_type(company_name, url)
        template_type = analysis["primary_audience"]
        print(f"Selected '{template_type}' template: {analysis['reasoning']}\n")
    
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": BROCHURE_TEMPLATES[template_type]},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
        stream=True
    )
    
    response = ""
    print("Generating brochure...\n")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            time.sleep(0.01)
    print("\n")

def demo_all_templates(company_name, url):
    """Generate brochures using all templates for comparison"""
    for template_name in BROCHURE_TEMPLATES.keys():
        print(f"\n{'='*50}")
        print(f"BROCHURE TYPE: {template_name.upper()}")
        print(f"{'='*50}\n")
        result = create_adaptive_brochure(company_name, url, template_name)
        print(result)
        print(f"\n{'='*50}\n")

# =====================================================================
# DEMONSTRATION - RUN THE ORIGINAL EXAMPLE
# =====================================================================

# Original example with humorous template
print("ORIGINAL EXAMPLE (Humorous Template):")
print("="*50)
stream_brochure("PayloadCMS", PAYLOAD_CMS)

# =================================================================
# ENHANCED PROMPTING DEMONSTRATIONS
# =================================================================
# Uncomment any of these to test different approaches:

# Approach 1: Auto-adaptive template selection
# print("\n1. AUTO-ADAPTIVE BROCHURE (AI selects best template):")
# print("-" * 50)
# result = create_adaptive_brochure("PayloadCMS", PAYLOAD_CMS)
# print(result)

# Approach 2: Multi-stage generation
# print("\n2. MULTI-STAGE BROCHURE (Extract then generate):")  
# print("-" * 50)
# multi_stage_result = create_multi_stage_brochure("PayloadCMS", PAYLOAD_CMS)
# print(multi_stage_result)

# Approach 3: Specific template (e.g., investor-focused)
# print("\n3. INVESTOR-FOCUSED BROCHURE:")
# print("-" * 50)
# investor_brochure = create_adaptive_brochure("PayloadCMS", PAYLOAD_CMS, "investor")
# print(investor_brochure)

# Approach 4: Customer-focused with streaming
# print("\n4. CUSTOMER-FOCUSED BROCHURE (Streaming):")
# print("-" * 50)
# stream_adaptive_brochure("PayloadCMS", PAYLOAD_CMS, "customer")

# Approach 5: Compare all templates
# print("\n5. ALL TEMPLATE VARIATIONS:")
# demo_all_templates("PayloadCMS", PAYLOAD_CMS)

# =====================================================================
# USAGE GUIDE AND EXAMPLES
# =====================================================================

"""
ENHANCED BROCHURE GENERATOR - USAGE GUIDE

This improved version offers multiple ways to generate brochures:

=== BASIC USAGE (Original Functions) ===
1. create_brochure("Company Name", "https://website.com")
2. stream_brochure("Company Name", "https://website.com")

=== ENHANCED FEATURES ===

🎯 AUTO-ADAPTIVE TEMPLATE SELECTION:
   create_adaptive_brochure("Company Name", "https://website.com")
   # AI analyzes the company and selects the best template automatically

🎨 SPECIFIC TEMPLATE SELECTION:
   create_adaptive_brochure("Company Name", "https://website.com", "investor")
   create_adaptive_brochure("Company Name", "https://website.com", "customer")
   create_adaptive_brochure("Company Name", "https://website.com", "talent")
   
🏭 MULTI-STAGE GENERATION:
   create_multi_stage_brochure("Company Name", "https://website.com")
   # First extracts key info, then generates brochure from structured data

📺 STREAMING WITH AUTO-SELECTION:
   stream_adaptive_brochure("Company Name", "https://website.com")
   stream_adaptive_brochure("Company Name", "https://website.com", "customer")

🔄 COMPARE ALL TEMPLATES:
   demo_all_templates("Company Name", "https://website.com")
   # Generates brochures using all 5 templates for comparison

=== AVAILABLE TEMPLATES ===
- "investor"  - Professional, metrics-focused, for funding/investment
- "customer"  - Benefits-driven, for marketing to potential clients
- "talent"    - Culture-focused, for recruiting employees
- "general"   - Balanced approach for all audiences
- "humorous"  - Fun and entertaining while informative

=== IMPROVEMENTS OVER ORIGINAL ===
✅ Enhanced link classification with priority and reasoning
✅ Multiple audience-specific templates with structured sections
✅ AI-powered template selection based on company analysis
✅ Multi-stage generation for higher quality results
✅ Flexible usage patterns (auto vs manual selection)
✅ Comprehensive comparison tools

=== PROMPTING TECHNIQUES USED ===
1. Structured JSON responses with reasoning
2. Priority-based classification
3. Multi-stage processing (extract → generate)
4. Template-driven consistency
5. Adaptive system prompts based on context
6. Audience-specific language and focus

Try different approaches to see which works best for your use case!
""" 