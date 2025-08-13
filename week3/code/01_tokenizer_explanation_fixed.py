# Tokenizer Examples and Explanations
# Modified for local execution (non-Colab) - Fixed version

# 1. Setup and Imports
import os
from transformers import AutoTokenizer

# 2. Authentication Setup (Optional - for gated models)
# For Llama models, you might need a HuggingFace token
# Set your HF_TOKEN environment variable or uncomment and set the token directly:
# os.environ['HF_TOKEN'] = 'your_token_here'

# Alternative: Use a public model that doesn't require authentication
try:
    # Try to load Llama 3.1 tokenizer (requires HF token for gated model)
    if os.getenv('HF_TOKEN'):
        from huggingface_hub import login
        login(os.getenv('HF_TOKEN'), add_to_git_credential=True)
        model_name = 'meta-llama/Meta-Llama-3.1-8B'
    else:
        # Use GPT-2 tokenizer which is more stable and publicly available
        model_name = 'gpt2'
        print("Note: Using GPT-2 tokenizer (no HF token provided for Llama)")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad_token if it doesn't exist (common issue with some tokenizers)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token for proper padding")
        
    print(f"Loaded tokenizer for: {model_name}")
    
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# 4. Basic Tokenization Example
text = "I am excited to show Tokenizers in action to my LLM engineers"

# Encode text to tokens (list of token IDs)
tokens = tokenizer.encode(text)
print("\n=== Basic Tokenization ===")
print("Token IDs:", tokens)

# Check number of tokens
print("Number of tokens:", len(tokens))

# 5. Decoding Examples
print("\n=== Decoding Examples ===")

# Basic decode - returns the full text with special tokens
decoded_text = tokenizer.decode(tokens)
print("Decoded text:", repr(decoded_text))

# Batch decode - returns list of individual token strings
batch_decoded = tokenizer.batch_decode(tokens)
print("Batch decoded tokens:", batch_decoded)

# 6. Vocabulary Access
print("\n=== Vocabulary Information ===")
print("Vocabulary size:", tokenizer.vocab_size)
# Check if tokenizer has get_added_vocab method for special tokens
if hasattr(tokenizer, 'get_added_vocab'):
    added_vocab = tokenizer.get_added_vocab()
    print("Added vocabulary:", added_vocab if added_vocab else "None")

# 7. Alternative Encoding Methods
print("\n=== Alternative Encoding Methods ===")
# Using __call__ method (recommended for most use cases)
encoded = tokenizer(text)
print("Encoded with __call__:", encoded)

# 8. Handling Multiple Texts
print("\n=== Batch Processing ===")
texts = [
    "Hello world!",
    "How are you today?",
    "Machine learning is fascinating."
]

# Batch encoding with padding
try:
    batch_encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    print("Batch encoded shape:", batch_encoded['input_ids'].shape)
    print("Attention mask shape:", batch_encoded['attention_mask'].shape)
except Exception as e:
    print(f"PyTorch tensor creation failed: {e}")
    # Try without tensors
    batch_encoded = tokenizer(texts, padding=True, truncation=True)
    print("Batch encoded input_ids lengths:", [len(ids) for ids in batch_encoded['input_ids']])

# 9. Special Token Examples
print("\n=== Special Tokens ===")
print("BOS token:", getattr(tokenizer, 'bos_token', 'N/A'), "ID:", getattr(tokenizer, 'bos_token_id', 'N/A'))
print("EOS token:", getattr(tokenizer, 'eos_token', 'N/A'), "ID:", getattr(tokenizer, 'eos_token_id', 'N/A'))
print("PAD token:", getattr(tokenizer, 'pad_token', 'N/A'), "ID:", getattr(tokenizer, 'pad_token_id', 'N/A'))
print("UNK token:", getattr(tokenizer, 'unk_token', 'N/A'), "ID:", getattr(tokenizer, 'unk_token_id', 'N/A'))

# 10. Decode with skip_special_tokens
print("\n=== Clean Decoding ===")
clean_decoded = tokenizer.decode(tokens, skip_special_tokens=True)
print("Clean decoded (no special tokens):", repr(clean_decoded))

# 11. Token-by-token analysis
print("\n=== Token-by-token Analysis ===")
for i, token_id in enumerate(tokens[:10]):  # Show first 10 tokens to avoid clutter
    token_str = tokenizer.decode([token_id])
    print(f"Position {i}: ID {token_id} -> '{token_str}'")
if len(tokens) > 10:
    print(f"... and {len(tokens) - 10} more tokens")

# 12. Comparing tokenization efficiency
print("\n=== Tokenization Efficiency Comparison ===")
test_texts = [
    "Hello world",
    "¡Hola mundo!",
    "你好世界",
    "print('Hello, world!')",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
]

for text in test_texts:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    efficiency = len(text) / len(tokens) if len(tokens) > 0 else 0  # characters per token
    print(f"Text: '{text}'")
    print(f"  Tokens: {len(tokens)}, Chars: {len(text)}, Efficiency: {efficiency:.2f} chars/token")
    print(f"  Token breakdown: {tokenizer.batch_decode(tokens)}")
    print()

# 13. Chat Template Example (if available)
print("\n=== Chat Template ===")
if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    try:
        formatted = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print("Chat template formatted:")
        print(formatted)
        
        # Tokenize the formatted chat
        chat_tokens = tokenizer.apply_chat_template(
            conversation, 
            tokenize=True, 
            add_generation_prompt=True
        )
        print(f"Chat tokens: {len(chat_tokens)} tokens")
    except Exception as e:
        print(f"Chat template error: {e}")
else:
    print("Chat template not available for this tokenizer")

# 14. Handling Out-of-Vocabulary (OOV) words
print("\n=== OOV Handling ===")
oov_text = "supercalifragilisticexpialidocious antidisestablishmentarianism"
oov_tokens = tokenizer.encode(oov_text, add_special_tokens=False)
oov_decoded = tokenizer.batch_decode(oov_tokens)
print(f"Text: '{oov_text}'")
print(f"Tokens: {oov_decoded}")
print(f"Number of tokens: {len(oov_tokens)}")

print("\n=== Tokenizer Demo Complete ===")
print(f"Tokenizer model: {model_name}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Model max length: {getattr(tokenizer, 'model_max_length', 'Unknown')}")
