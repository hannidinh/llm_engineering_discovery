import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Email:
    def __init__(self, content, sender=None, recipient=None):
        """
        Create an Email object from the given content
        """
        self.content = content
        self.sender = sender
        self.recipient = recipient
        self.cleaned_content = self._clean_content()
    
    def _clean_content(self):
        """
        Clean the email content by removing common email artifacts
        """
        # Remove email signatures
        signature_patterns = [
            r'\n\s*(?:Best regards?|Sincerely|Thanks?|Cheers|Best)\s*,?\s*\n.*',
            r'\n\s*--\s*\n.*',  # Common signature separator
            r'\n\s*Sent from my .*',  # Mobile signatures
        ]

        cleaned = self.content
        for pattern in signature_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip();

# System prompt for generating subject lines
system_prompt = """ You are an AI assistant that generates concise, professional email subject lines.
Your task is to analyze email content and create an appropriate subject line that:
- Captures the main purpose or request of the email
- Is clear and actionable
- Is between 5 - 8 words when possible
- Uses professional language
- Includes urgency indicators when appropriate (e.g., "Urgent:", "Action Required:")
- Avoids spam-like language
Respond with only the subject line, no additional text or formatting.
"""

def user_prompt_for(email):
    """Generate the user prompt for the email"""
    prompt = "Please generate an appropriate subject line for this email content:\n\n"

    if email.sender:
        prompt += f"From: {email.sender}\n"
    if email.recipient:
        prompt += f"To: {email.recipient}\n"
    
    prompt += f"Email Content:\n{email.cleaned_content}"

    return prompt

def messages_for(email): 
    """
    Create the messages array for the OpenAI API
    """
    return [
        {
            "role": "system", "content": system_prompt
        },
        {
            "role": "user", "content": user_prompt_for(email)
        }
    ]

def generate_subject_line(email_content, sender=None, recipient=None):
    """
    Generate a subject line for the given email content
    """
    email = Email(email_content, sender, recipient)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for(email),
        temperature=0.3,
        max_tokens=50
    )

    return response.choices[0].message.content.strip()

def suggest_subject_line(email_content, sender = None, recipient = None):
    """
    Display the suggested subject line for an email
    """
    subject_line = generate_subject_line(email_content, sender, recipient)
    print(f"Suggested Subject Line: {subject_line}")
    return subject_line

if __name__ == "__main__":
    # Test case 1: Meeting request
    recruiter_email = """
Hi Hien,

Thanks for applying to our Senior Full-Stack Engineer (Java) opening (job description below).  Wanted to follow up on your resume submittal for the role.

Could you please answer the following questions and let me know your availability for a potential phone call?

Please review the JD below and confirm interest. 
Why are you looking for a new opportunity? (Main motivation for finding a new position?)
What are you targeting from a compensation standpoint and what is your ideal situation?
Are you currently within commuting distance to Piscataway, NJ?
Are you open to working on-site 3x/week?
Do you have work authorization within the US? With you ever require authorization or sponsorship?
Please briefly outline your experience with Java and Spring/Spring Boot. Include total years of experience, and at least 1 project
Please briefly outline your experience with Angular. Include total years of experience, and at least 1 project

Thanks,
Sean
    """
    
    print("Example 1 - Meeting Request:")
    suggest_subject_line(recruiter_email, "Sean", "Hien")
    print()