# Create a product that finds top 5 courses from an educational website and provides detailed recommendations
# Input: website URL (like Udemy, Coursera, etc.)

import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time
import sys
import re

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

# Initialize and constants
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

class CourseWebsite:
    """
    A utility class to represent a Course Website that we have scraped
    """

    def __init__(self, url):
        self.url = url
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            self.body = response.content
            soup = BeautifulSoup(self.body, 'html.parser')
            self.title = soup.title.string if soup.title else "No title found"

            if soup.body:
                for irrelevant in soup.body(["script", "style", "img", "input", "svg"]):
                    irrelevant.decompose()
                self.text = soup.body.get_text(separator="\n", strip=True)
            else:
                self.text = ""
            
            # Extract course-related links
            links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and ('course' in href.lower() or 'class' in href.lower() or 'learn' in href.lower()):
                    links.append(href)
            
            self.links = [link for link in links if link][:50]  # Limit to 50 most relevant links
            
        except Exception as e:
            print(f"Error scraping website: {e}")
            self.text = ""
            self.links = []

    def get_contents(self):
        return f"Website Title:\n{self.title}\nWebsite Contents:\n{self.text}\n\n"

# First step: Have GPT extract course information from the website
course_extraction_prompt = "You are an expert at analyzing educational websites and extracting course information. \
You will be provided with website content and need to identify courses, programs, or learning materials available.\n"
course_extraction_prompt += "Extract information about courses including titles, descriptions, topics, difficulty levels, ratings, prices, and instructor names if available.\n"
course_extraction_prompt += "You should respond in JSON format as in this example:"
course_extraction_prompt += """
{
    "courses": [
        {
            "title": "Complete Python Bootcamp",
            "description": "Learn Python programming from scratch with hands-on projects",
            "topic": "Programming",
            "difficulty": "Beginner",
            "rating": "4.5/5",
            "price": "$89.99",
            "instructor": "John Doe",
            "key_features": ["Hands-on projects", "Lifetime access", "Certificate"]
        },
        {
            "title": "Advanced Data Science",
            "description": "Master machine learning and data analysis techniques",
            "topic": "Data Science",
            "difficulty": "Advanced",
            "rating": "4.7/5",
            "price": "$129.99",
            "instructor": "Jane Smith",
            "key_features": ["Real-world datasets", "Industry projects", "Job support"]
        }
    ]
}
"""

def get_course_extraction_prompt(website):
    user_prompt = f"Here is the content from the educational website {website.url}:\n"
    user_prompt += "Please extract all available course information and format it as JSON. "
    user_prompt += "If some information is not available, use 'N/A' or make reasonable inferences based on the content.\n\n"
    user_prompt += website.get_contents()
    return user_prompt

def extract_courses(url):
    website = CourseWebsite(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": course_extraction_prompt},
            {"role": "user", "content": get_course_extraction_prompt(website)[:4000]}  # Truncate to fit token limits
        ],
        response_format={"type":"json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

# Second step: Analyze and rank courses
ranking_system_prompt = "You are an expert educational consultant who helps people choose the best courses for their career development. \
You analyze course information and rank them based on value, market demand, learning outcomes, and career impact.\n"
ranking_system_prompt += "You will be given a list of courses and need to select the TOP 5 most valuable courses, ranking them from 1 to 5.\n"
ranking_system_prompt += "For each top course, provide detailed reasoning why someone should invest in it, including:\n"
ranking_system_prompt += "- Career benefits and job market demand\n"
ranking_system_prompt += "- Skill development value\n"
ranking_system_prompt += "- Return on investment\n"
ranking_system_prompt += "- Target audience\n"
ranking_system_prompt += "- Unique advantages\n"

def get_ranking_prompt(courses_data):
    user_prompt = "Here are the available courses:\n\n"
    for i, course in enumerate(courses_data.get("courses", []), 1):
        user_prompt += f"Course {i}:\n"
        user_prompt += f"Title: {course.get('title', 'N/A')}\n"
        user_prompt += f"Description: {course.get('description', 'N/A')}\n"
        user_prompt += f"Topic: {course.get('topic', 'N/A')}\n"
        user_prompt += f"Difficulty: {course.get('difficulty', 'N/A')}\n"
        user_prompt += f"Rating: {course.get('rating', 'N/A')}\n"
        user_prompt += f"Price: {course.get('price', 'N/A')}\n"
        user_prompt += f"Instructor: {course.get('instructor', 'N/A')}\n"
        user_prompt += f"Key Features: {', '.join(course.get('key_features', []))}\n\n"
    
    user_prompt += "\nPlease rank the TOP 5 courses and provide detailed explanations for why customers should invest in each one. "
    user_prompt += "Format your response in markdown with clear headings and bullet points."
    
    return user_prompt

def analyze_and_rank_courses(courses_data):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ranking_system_prompt},
            {"role": "user", "content": get_ranking_prompt(courses_data)}
        ]
    )
    result = response.choices[0].message.content
    return result

# Third step: Stream the final recommendations
def stream_course_recommendations(url):
    print(f"🔍 Analyzing courses from: {url}")
    print("=" * 60)
    
    # Extract courses
    print("📚 Extracting course information...")
    courses_data = extract_courses(url)
    
    if not courses_data.get("courses"):
        print("❌ No courses found on this website.")
        return
    
    print(f"✅ Found {len(courses_data['courses'])} courses")
    print("\n🎯 Analyzing and ranking courses...")
    print("=" * 60)
    
    # Create streaming response for recommendations
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ranking_system_prompt},
            {"role": "user", "content": get_ranking_prompt(courses_data)}
        ],
        stream=True
    )
    
    if HAS_IPYTHON:
        # Use IPython display for Jupyter notebooks
        display_handle = display(Markdown(""), display_id=True)
        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            update_display(Markdown(response), display_id=display_handle.display_id)
    else:
        # Use terminal-friendly streaming for regular Python
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                time.sleep(0.01)  # Small delay for typewriter effect
        print("\n")

# Function to create comprehensive course comparison report
def create_course_report(url):
    print(f"\n📊 Creating comprehensive course report for: {url}")
    print("=" * 60)
    
    courses_data = extract_courses(url)
    recommendations = analyze_and_rank_courses(courses_data)
    
    report = f"""# Course Comparison Report

## Website Analyzed: {url}

## Total Courses Found: {len(courses_data.get('courses', []))}

## Top 5 Course Recommendations

{recommendations}

## Analysis Summary
- This report was generated using AI analysis of course content
- Rankings are based on career value, market demand, and learning outcomes
- Prices and availability may change - please verify on the website
- Consider your personal goals and current skill level when choosing

---
*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save report to file
    filename = f"course_report_{int(time.time())}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {filename}")
    return report

# Example usage and testing
if __name__ == "__main__":
    # Test with educational websites
    test_urls = [
        "https://www.udemy.com",
        "https://www.coursera.org",
        "https://www.edx.org"
    ]
    
    print("🚀 Course Comparison Tool")
    print("=" * 60)
    
    # You can test with any educational website
    test_url = "https://www.udemy.com"
    
    try:
        stream_course_recommendations(test_url)
    except Exception as e:
        print(f"❌ Error analyzing courses: {e}")
        print("💡 Try with a different URL or check your internet connection")

# Uncomment the line below to generate a full report
create_course_report("https://www.udemy.com") 