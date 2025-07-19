"""
Website Summarizer Demo - Using Local Llama 3.2

This script demonstrates two approaches to summarize websites using local Llama 3.2:
1. Direct Ollama package approach
2. OpenAI library with Ollama backend approach

Requirements:
- Ollama must be installed and running
- Llama 3.2 model must be pulled: ollama pull llama3.2
"""

import sys
import traceback

def test_ollama_approach():
    """Test the direct Ollama package approach"""
    print("Testing Ollama Package Approach")
    print("=" * 40)
    
    try:
        from website_summarizer_local_ollama import display_summary
        display_summary("https://www.python.org")
        return True
    except Exception as e:
        print(f"Error with Ollama approach: {e}")
        traceback.print_exc()
        return False

def test_openai_approach():
    """Test the OpenAI library with Ollama backend approach"""
    print("\n\nTesting OpenAI Library + Ollama Approach")
    print("=" * 40)
    
    try:
        from website_summarizer_local_openai import display_summary
        display_summary("https://www.python.org")
        return True
    except Exception as e:
        print(f"Error with OpenAI + Ollama approach: {e}")
        traceback.print_exc()
        return False

def main():
    print("Website Summarizer Demo")
    print("Using Local Llama 3.2 via Ollama\n")
    
    # Check if user wants to test a specific approach
    if len(sys.argv) > 1:
        approach = sys.argv[1].lower()
        if approach == "ollama":
            test_ollama_approach()
        elif approach == "openai":
            test_openai_approach()
        else:
            print("Usage: python demo_website_summarizer.py [ollama|openai]")
            print("Or run without arguments to test both approaches")
    else:
        # Test both approaches
        print("Testing both approaches...\n")
        
        # Test Ollama approach
        ollama_success = test_ollama_approach()
        
        # Test OpenAI approach
        openai_success = test_openai_approach()
        
        print("\n\nSummary:")
        print(f"Ollama Package Approach: {'✅ Success' if ollama_success else '❌ Failed'}")
        print(f"OpenAI + Ollama Approach: {'✅ Success' if openai_success else '❌ Failed'}")
        
        if not (ollama_success or openai_success):
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running: ollama serve")
            print("2. Make sure Llama 3.2 is pulled: ollama pull llama3.2")
            print("3. Check that port 11434 is available")

if __name__ == "__main__":
    main() 