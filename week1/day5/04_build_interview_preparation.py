# Interview Preparation Generator
# Input: resume text and job description
# Output: list of questions and answers to prepare for interview

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import time
import sys

# For this terminal script, we'll use simple print-based output
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

class InterviewPrep:
    """
    A utility class to generate interview preparation materials
    """
    
    def __init__(self, resume_text: str, job_description: str):
        self.resume_text = resume_text
        self.job_description = job_description
        
    def analyze_fit(self) -> Dict:
        """
        Analyze how well the resume matches the job description
        """
        system_prompt = """You are an expert career coach and recruiter. 
        Analyze how well a candidate's resume matches a job description. 
        Identify strengths, gaps, and key talking points for an interview.
        Respond in JSON format."""
        
        system_prompt += """
        {
            "strengths": [
                "List of candidate's strengths that match the job"
            ],
            "gaps": [
                "Areas where candidate might be weak or lacking experience"
            ],
            "key_experiences": [
                "Most relevant experiences from resume for this role"
            ],
            "talking_points": [
                "Important points candidate should emphasize in interview"
            ]
        }
        """
        
        user_prompt = f"""
        Job Description:
        {self.job_description}
        
        Resume:
        {self.resume_text}
        
        Please analyze how well this resume matches the job description.
        """
        
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        return json.loads(result)

def generate_questions(resume_text: str, job_description: str) -> Dict:
    """
    Generate interview questions based on resume and job description
    """
    system_prompt = """You are an expert interviewer and career coach. 
    Generate relevant interview questions based on a candidate's resume and the job description.
    Include a mix of behavioral, technical, and situational questions.
    Provide sample answers that highlight the candidate's experience.
    Respond in JSON format."""
    
    system_prompt += """
    {
        "behavioral_questions": [
            {
                "question": "Tell me about a time when...",
                "suggested_answer": "Based on your resume, you could discuss...",
                "why_asked": "This tests your ability to..."
            }
        ],
        "technical_questions": [
            {
                "question": "Technical question based on job requirements",
                "suggested_answer": "Key points to cover...",
                "why_asked": "This assesses your technical knowledge in..."
            }
        ],
        "situational_questions": [
            {
                "question": "How would you handle...",
                "suggested_answer": "You should mention...",
                "why_asked": "This evaluates your problem-solving approach..."
            }
        ],
        "company_fit_questions": [
            {
                "question": "Why are you interested in this role?",
                "suggested_answer": "Connect your experience to the role requirements...",
                "why_asked": "This tests your genuine interest and research..."
            }
        ]
    }
    """
    
    user_prompt = f"""
    Job Description:
    {job_description}
    
    Candidate's Resume:
    {resume_text}
    
    Generate 3-4 questions for each category that are specifically tailored to this candidate and role.
    Base the suggested answers on the actual experience shown in the resume.
    """
    
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    return json.loads(result)

def create_interview_prep(resume_text: str, job_description: str, company_name: str = ""):
    """
    Create a comprehensive interview preparation guide
    """
    system_prompt = """You are an expert career coach specializing in interview preparation.
    Create a comprehensive interview preparation guide based on a candidate's resume and job description.
    Include analysis of fit, potential questions with answers, and strategic advice.
    Respond in markdown format with clear sections."""
    
    user_prompt = f"""
    Company: {company_name}
    
    Job Description:
    {job_description}
    
    Candidate's Resume:
    {resume_text}
    
    Create a comprehensive interview preparation guide that includes:
    1. Analysis of how well the candidate fits the role
    2. Likely interview questions with suggested answers based on the resume
    3. Questions the candidate should ask
    4. Tips for highlighting relevant experience
    5. Potential concerns to address
    
    Make the answers specific to the candidate's actual experience from their resume.
    """
    
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    result = response.choices[0].message.content
    print(result)
    return result

def stream_interview_prep(resume_text: str, job_description: str, company_name: str = ""):
    """
    Stream the interview preparation guide with typewriter effect
    """
    system_prompt = """You are an expert career coach specializing in interview preparation.
    Create a comprehensive interview preparation guide based on a candidate's resume and job description.
    Include analysis of fit, potential questions with answers, and strategic advice.
    Respond in markdown format with clear sections."""
    
    user_prompt = f"""
    Company: {company_name}
    
    Job Description:
    {job_description}
    
    Candidate's Resume:
    {resume_text}
    
    Create a comprehensive interview preparation guide that includes:
    1. Analysis of how well the candidate fits the role
    2. Likely interview questions with suggested answers based on the resume
    3. Questions the candidate should ask the interviewer
    4. Tips for highlighting relevant experience
    5. Potential concerns to address proactively
    
    Make the answers specific to the candidate's actual experience from their resume.
    Focus on behavioral questions using the STAR method (Situation, Task, Action, Result).
    """
    
    # Truncate inputs if too long to stay within token limits
    user_prompt = user_prompt[:8000]
    
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
        print("Generating interview preparation guide...\n")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                time.sleep(0.01)  # Small delay for typewriter effect
        print("\n")  # Final newline

# Utility functions for file input
def read_file(file_path: str) -> str:
    """Read text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def main():
    """
    Main function to run the interview prep generator
    """
    print("Interview Preparation Generator")
    print("=" * 40)
    
    # Get input method preference
    print("How would you like to provide the resume and job description?")
    print("1. Type/paste directly")
    print("2. Read from files")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        resume_file = input("Enter path to resume file: ").strip()
        job_file = input("Enter path to job description file: ").strip()
        
        resume_text = read_file(resume_file)
        job_description = read_file(job_file)
        
        if not resume_text or not job_description:
            print("Error reading files. Exiting.")
            return
    else:
        print("\nPaste your resume (press Ctrl+D or Ctrl+Z when done):")
        resume_lines = []
        try:
            while True:
                line = input()
                resume_lines.append(line)
        except EOFError:
            pass
        resume_text = "\n".join(resume_lines)
        
        print("\nPaste the job description (press Ctrl+D or Ctrl+Z when done):")
        job_lines = []
        try:
            while True:
                line = input()
                job_lines.append(line)
        except EOFError:
            pass
        job_description = "\n".join(job_lines)
    
    company_name = input("\nCompany name (optional): ").strip()
    
    # Generate the interview preparation guide
    print("\n" + "=" * 50)
    stream_interview_prep(resume_text, job_description, company_name)

if __name__ == "__main__":
    main()

# Example usage:
# Uncomment the lines below to test with sample data


sample_resume = '''
John Doe
Software Engineer
Email: john.doe@email.com

EXPERIENCE:
Senior Software Engineer - TechCorp (2020-2023)
- Led development of microservices architecture using Python and Docker
- Managed team of 4 junior developers
- Implemented CI/CD pipelines reducing deployment time by 50%

Software Engineer - StartupXYZ (2018-2020)
- Built REST APIs using Django and PostgreSQL
- Collaborated with product team on feature requirements
- Wrote comprehensive unit tests achieving 90% code coverage

EDUCATION:
B.S. Computer Science - State University (2018)

SKILLS:
Python, JavaScript, Docker, AWS, PostgreSQL, Git, Agile
'''

sample_job_description = '''
Senior Software Engineer - CloudTech Solutions

We are seeking a Senior Software Engineer to join our growing team. 
The ideal candidate will have 3+ years of experience in backend development 
and a strong background in cloud technologies.

Requirements:
- 3+ years of software development experience
- Proficiency in Python and modern web frameworks
- Experience with cloud platforms (AWS, Azure, or GCP)
- Knowledge of containerization (Docker, Kubernetes)
- Experience with database design and optimization
- Strong communication and leadership skills

Responsibilities:
- Design and implement scalable backend services
- Mentor junior developers
- Collaborate with cross-functional teams
- Participate in code reviews and technical discussions
- Contribute to architectural decisions

Nice to have:
- Experience with microservices architecture
- DevOps experience with CI/CD pipelines
- Previous startup experience
'''

# Uncomment to test:
stream_interview_prep(sample_resume, sample_job_description, "CloudTech Solutions")
