# Tokenizer Examples and Explanations
# Based on the HuggingFace tokenizers.ipynb notebook

# 1. Setup and Imports
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer

# 2. Authentication Setup
# Get your HuggingFace token from userdata (Colab secrets)
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# 3. Load the Llama 3.1 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Meta-Llama-3.1-8B', 
    trust_remote_code=True
)

# 4. Basic Tokenization Example
text = "I am excited to show Tokenizers in action to my LLM engineers"

# Encode text to tokens (list of token IDs)
tokens = tokenizer.encode(text)
print("Token IDs:", tokens)
# Output: [128000, 40, 1097, 12304, 311, 1501, 9857, 12509, 304, 1957, 311, 856, 445, 11237, 25175]

# Check number of tokens
print("Number of tokens:", len(tokens))
# Output: 15

# 5. Decoding Examples

# Basic decode - returns the full text with special tokens
decoded_text = tokenizer.decode(tokens)
print("Decoded text:", decoded_text)
# Output: '<|begin_of_text|>I am excited to show Tokenizers in action to my LLM engineers'

# Batch decode - returns list of individual token strings
batch_decoded = tokenizer.batch_decode(tokens)
print("Batch decoded tokens:", batch_decoded)
# Output: ['<|begin_of_text|>', 'I', ' am', ' excited', ' to', ' show', ' Token', 'izers', ' in', ' action', ' to', ' my', ' L', 'LM', ' engineers']

# 6. Vocabulary Access
print("Vocabulary size:", tokenizer.vocab_size)
# Check if tokenizer has get_added_vocab method for special tokens
if hasattr(tokenizer, 'get_added_vocab'):
    print("Added vocabulary:", tokenizer.get_added_vocab())

# ===============================
# Additional Examples and Methods
# ===============================

# 7. Alternative Encoding Methods
# Using __call__ method (recommended for most use cases)
encoded = tokenizer(text)
print("Encoded with __call__:", encoded)
# Returns dictionary with 'input_ids', 'attention_mask', etc.

# 8. Handling Multiple Texts
texts = [
    "Hello world!",
    "How are you today?",
    "Machine learning is fascinating."
]

# Batch encoding
batch_encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print("Batch encoded shape:", batch_encoded['input_ids'].shape)

# 9. Special Token Examples
print("\nSpecial Tokens:")
print("BOS token:", tokenizer.bos_token, "ID:", tokenizer.bos_token_id)
print("EOS token:", tokenizer.eos_token, "ID:", tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)
print("UNK token:", tokenizer.unk_token, "ID:", tokenizer.unk_token_id)

# 10. Decode with skip_special_tokens
clean_decoded = tokenizer.decode(tokens, skip_special_tokens=True)
print("Clean decoded (no special tokens):", clean_decoded)

# 11. Token-by-token analysis
print("\nToken-by-token breakdown:")
for i, token_id in enumerate(tokens):
    token_str = tokenizer.decode([token_id])
    print(f"Position {i}: ID {token_id} -> '{token_str}'")

# 12. Comparing tokenization efficiency
test_texts = [
    "Hello world",
    "¡Hola mundo!",
    "你好世界",
    "print('Hello, world!')",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
]

print("\nTokenization Efficiency Comparison:")
for text in test_texts:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    efficiency = len(text) / len(tokens)  # characters per token
    print(f"Text: '{text}'")
    print(f"  Tokens: {len(tokens)}, Chars: {len(text)}, Efficiency: {efficiency:.2f} chars/token")
    print(f"  Token breakdown: {tokenizer.batch_decode(tokens)}")
    print()

# 13. Chat Template Example (if available)
if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
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

# 14. Handling Out-of-Vocabulary (OOV) words
oov_text = "supercalifragilisticexpialidocious antidisestablishmentarianism"
oov_tokens = tokenizer.encode(oov_text, add_special_tokens=False)
oov_decoded = tokenizer.batch_decode(oov_tokens)
print(f"\nOOV handling:")
print(f"Text: '{oov_text}'")
print(f"Tokens: {oov_decoded}")
print(f"Number of tokens: {len(oov_tokens)}")