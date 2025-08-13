# Tokenizer Comparison: Instruct Models and Chat Templates
# Extracted from the notebook showing different model tokenizers

from transformers import AutoTokenizer

# =======================================================
# 1. INSTRUCT VARIANTS AND CHAT TEMPLATES
# =======================================================

# Load Llama 3.1 Instruct tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Meta-Llama-3.1-8B-Instruct', 
    trust_remote_code=True
)

# Define a conversation in standard messages format
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]

# Apply chat template to format the conversation
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,  # Return string, not tokens
    add_generation_prompt=True  # Add prompt for model to continue
)
print("Llama 3.1 Chat Template Output:")
print(prompt)
print()

# =======================================================
# 2. MODEL DEFINITIONS FOR COMPARISON
# =======================================================

# Define model names for different architectures
PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
QWEN2_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
STARCODER2_MODEL_NAME = "bigcode/starcoder2-3b"

# =======================================================
# 3. PHI3 TOKENIZER COMPARISON
# =======================================================

# Load Phi3 tokenizer
phi3_tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME)

# Compare tokenization of the same text
text = "I am excited to show Tokenizers in action to my LLM engineers"

# Tokenize with original Llama tokenizer
llama_tokens = tokenizer.encode(text)
print("Llama 3.1 tokens:")
print(llama_tokens)
print(f"Token count: {len(llama_tokens)}")
print()

# Tokenize with Phi3 tokenizer
phi3_tokens = phi3_tokenizer.encode(text)
print("Phi3 tokens:")
print(phi3_tokens)
print(f"Token count: {len(phi3_tokens)}")
print()

# Compare chat templates
print("Llama 3.1 chat template:")
llama_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(llama_chat)
print()

print("Phi3 chat template:")
phi3_chat = phi3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(phi3_chat)
print()

# =======================================================
# 4. QWEN2 TOKENIZER COMPARISON
# =======================================================

# Load Qwen2 tokenizer
qwen2_tokenizer = AutoTokenizer.from_pretrained(QWEN2_MODEL_NAME)

# Compare all three tokenizations
print("=== TOKENIZATION COMPARISON ===")
print(f"Original text: '{text}'")
print()

print("Llama 3.1 tokens:")
print(llama_tokens)
print(f"Count: {len(llama_tokens)}")
print()

print("Phi3 tokens:")
print(phi3_tokens)
print(f"Count: {len(phi3_tokens)}")
print()

qwen2_tokens = qwen2_tokenizer.encode(text)
print("Qwen2 tokens:")
print(qwen2_tokens)
print(f"Count: {len(qwen2_tokens)}")
print()

# Compare chat templates
print("=== CHAT TEMPLATE COMPARISON ===")
print("Llama 3.1:")
print(repr(llama_chat))
print()

print("Phi3:")
print(repr(phi3_chat))
print()

print("Qwen2:")
qwen2_chat = qwen2_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(qwen2_chat))
print()

# =======================================================
# 5. STARCODER2 FOR CODE TOKENIZATION
# =======================================================

# Load StarCoder2 tokenizer (specialized for code)
starcoder2_tokenizer = AutoTokenizer.from_pretrained(
    STARCODER2_MODEL_NAME, 
    trust_remote_code=True
)

# Test with code snippet
code = '''
def hello_world(person):
    print("Hello", person)
'''

print("=== CODE TOKENIZATION COMPARISON ===")
print(f"Code snippet:\n{code}")
print()

# Compare how different tokenizers handle code
code_tokens_starcoder = starcoder2_tokenizer.encode(code)
print("StarCoder2 tokens:")
print(code_tokens_starcoder)
print(f"Count: {len(code_tokens_starcoder)}")
print()

# Compare with general-purpose tokenizers
code_tokens_llama = tokenizer.encode(code)
print("Llama 3.1 tokens:")
print(code_tokens_llama)
print(f"Count: {len(code_tokens_llama)}")
print()

code_tokens_phi3 = phi3_tokenizer.encode(code)
print("Phi3 tokens:")
print(code_tokens_phi3)
print(f"Count: {len(code_tokens_phi3)}")
print()

# =======================================================
# 6. DETAILED ANALYSIS FUNCTIONS
# =======================================================

def analyze_tokenizer_efficiency(tokenizer, name, texts):
    """Analyze tokenizer efficiency across different text types"""
    print(f"\n=== {name} EFFICIENCY ANALYSIS ===")
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        efficiency = len(text) / len(tokens) if len(tokens) > 0 else 0
        
        print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Characters: {len(text)}, Tokens: {len(tokens)}")
        print(f"  Efficiency: {efficiency:.2f} chars/token")
        print()

def compare_special_tokens(tokenizers_dict):
    """Compare special tokens across different tokenizers"""
    print("\n=== SPECIAL TOKENS COMPARISON ===")
    
    for name, tok in tokenizers_dict.items():
        print(f"{name}:")
        print(f"  BOS: {tok.bos_token} (ID: {tok.bos_token_id})")
        print(f"  EOS: {tok.eos_token} (ID: {tok.eos_token_id})")
        print(f"  PAD: {tok.pad_token} (ID: {tok.pad_token_id})")
        print(f"  UNK: {tok.unk_token} (ID: {tok.unk_token_id})")
        print(f"  Vocab size: {tok.vocab_size}")
        print()

# Example usage of analysis functions
test_texts = [
    "Hello world!",
    "机器学习很有趣",  # Chinese text
    "print('Hello, world!')",  # Code
    "The quick brown fox jumps over the lazy dog.",  # English prose
    "αβγδε ελληνικά",  # Greek text
]

tokenizers_dict = {
    "Llama 3.1": tokenizer,
    "Phi3": phi3_tokenizer,
    "Qwen2": qwen2_tokenizer,
    "StarCoder2": starcoder2_tokenizer
}

# Run analyses
for name, tok in tokenizers_dict.items():
    analyze_tokenizer_efficiency(tok, name, test_texts)

compare_special_tokens(tokenizers_dict)

# =======================================================
# 7. CHAT TEMPLATE DETAILS
# =======================================================

def show_chat_template_details(tokenizer, name):
    """Show detailed information about a tokenizer's chat template"""
    print(f"\n=== {name} CHAT TEMPLATE DETAILS ===")
    
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("Has chat template: Yes")
        print("Template preview:")
        print(tokenizer.chat_template[:200] + "..." if len(tokenizer.chat_template) > 200 else tokenizer.chat_template)
        
        # Test with simple conversation
        test_messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = tokenizer.apply_chat_template(
            test_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"\nFormatted conversation:\n{formatted}")
    else:
        print("Has chat template: No")
    print()

# Show chat template details for each tokenizer
for name, tok in tokenizers_dict.items():
    show_chat_template_details(tok, name)