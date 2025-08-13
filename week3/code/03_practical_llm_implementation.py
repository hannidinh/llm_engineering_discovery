# Practical LLM Implementation: Quantization, Model Architecture, and Streaming
# Extracted from the notebook images

# ===============================
# 1. QUANTIZATION CONFIGURATION
# ===============================

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TextStreamer
)
import torch

# Quantization Configuration - reduces memory usage by 75%
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit quantization
    bnb_4bit_use_double_quant=True,       # Double quantization for better accuracy
    bnb_4bit_quant_type="nf4",           # NormalFloat4 - optimal for neural networks
    bnb_4bit_compute_dtype=torch.bfloat16 # Computation dtype for mixed precision
)

# ===============================
# 2. MODEL LOADING WITH QUANTIZATION
# ===============================

# Load LLaMA model with quantization
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Apply chat template and prepare inputs
messages = [
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]

inputs = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt"
).to("cuda")

# Load model with quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, 
    device_map="auto",           # Automatic device placement
    quantization_config=quant_config,  # Apply 4-bit quantization
    torch_dtype=torch.bfloat16   # Mixed precision
)

# ===============================
# 3. MEMORY FOOTPRINT ANALYSIS
# ===============================

def check_memory_usage(model):
    """Check model memory footprint"""
    memory = model.get_memory_footprint() / 1e6  # Convert to MB
    print(f"Memory footprint: {memory:.1f} MB")
    return memory

# Check memory usage after quantization
memory_usage = check_memory_usage(model)
# Output: Memory footprint: 5,591.5 MB (vs ~32GB for full precision)

# ===============================
# 4. MODEL ARCHITECTURE EXPLORATION
# ===============================

def explore_model_architecture(model):
    """Explore the internal structure of the LLaMA model"""
    print("Model Architecture:")
    print(model)
    
    # Key components analysis:
    print(f"\nModel Type: {type(model).__name__}")
    print(f"Config: {model.config}")
    
    # Embedding layer analysis
    embed_tokens = model.model.embed_tokens
    print(f"\nEmbedding Layer:")
    print(f"  Vocabulary size: {embed_tokens.num_embeddings}")
    print(f"  Embedding dimension: {embed_tokens.embedding_dim}")
    
    # Transformer layers analysis
    layers = model.model.layers
    print(f"\nTransformer Blocks: {len(layers)} layers")
    
    # First layer detailed analysis
    first_layer = layers[0]
    print(f"\nFirst Layer Structure:")
    print(f"  Self Attention: {first_layer.self_attn}")
    print(f"  MLP: {first_layer.mlp}")
    print(f"  Input LayerNorm: {first_layer.input_layernorm}")
    print(f"  Post Attention LayerNorm: {first_layer.post_attention_layernorm}")

# Analyze the model architecture
explore_model_architecture(model)

# ===============================
# 5. ATTENTION MECHANISM DEEP DIVE
# ===============================

def analyze_attention_patterns(model, tokenizer, text):
    """Analyze attention patterns in the model"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Get model outputs with attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    
    attention_weights = outputs.attentions  # Tuple of attention tensors
    
    print(f"Number of layers with attention: {len(attention_weights)}")
    print(f"Attention tensor shape (last layer): {attention_weights[-1].shape}")
    # Shape: [batch_size, num_heads, seq_length, seq_length]
    
    return attention_weights

# Example attention analysis
sample_text = "The quick brown fox jumps over the lazy dog"
attention_data = analyze_attention_patterns(model, tokenizer, sample_text)

# ===============================
# 6. SILU ACTIVATION FUNCTION
# ===============================

# SiLU (Sigmoid Linear Unit) is used in LLaMA models
# Formula: silu(x) = x * σ(x), where σ(x) is the logistic sigmoid
# Also known as Swish activation function

import torch.nn as nn

def demonstrate_silu():
    """Demonstrate SiLU activation function"""
    silu = nn.SiLU()
    
    # Test with sample values
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = silu(x)
    
    print("SiLU Activation Function:")
    print(f"Input:  {x}")
    print(f"Output: {y}")
    
    # SiLU properties:
    # - Smooth and non-monotonic
    # - Self-gated (x * sigmoid(x))
    # - Better gradient flow than ReLU
    # - Used in modern transformer architectures

demonstrate_silu()

# ===============================
# 7. STREAMING GENERATION FUNCTION
# ===============================

def generate(model_name, messages):
    """
    Generate streaming responses from LLM models
    This function demonstrates real-time token generation
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare inputs with chat template
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt"
    ).to("cuda")
    
    # Create text streamer for real-time output
    streamer = TextStreamer(tokenizer)
    
    # Load model with quantization for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        quantization_config=quant_config
    )
    
    # Generate with streaming
    outputs = model.generate(
        inputs, 
        max_new_tokens=80,        # Limit response length
        streamer=streamer,        # Enable real-time streaming
        temperature=0.7,          # Control randomness
        do_sample=True,           # Enable sampling
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Clean up memory
    del tokenizer, streamer, model, inputs, outputs
    torch.cuda.empty_cache()

# ===============================
# 8. MULTI-MODEL COMPARISON
# ===============================

# Define model configurations
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-7b-it"

def compare_models():
    """Compare different models on the same task"""
    
    test_messages = [
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ]
    
    models_to_test = [
        ("PHI3", PHI3),
        ("GEMMA2", GEMMA2),
        ("LLAMA", LLAMA)
    ]
    
    for model_name, model_path in models_to_test:
        print(f"\n{'='*50}")
        print(f"TESTING {model_name}")
        print(f"{'='*50}")
        
        try:
            generate(model_path, test_messages)
        except Exception as e:
            print(f"Error with {model_name}: {e}")

# ===============================
# 9. ADVANCED GENERATION PARAMETERS
# ===============================

def advanced_generation_example():
    """Demonstrate advanced generation parameters"""
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    
    # Advanced generation with multiple parameters
    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        temperature=0.8,          # Higher = more creative
        top_p=0.9,               # Nucleus sampling
        top_k=50,                # Top-k sampling
        repetition_penalty=1.1,   # Avoid repetition
        do_sample=True,          # Enable sampling
        num_return_sequences=1,   # Number of responses
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode and display
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Response:")
    print(response)
    
    # Clean up
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()

# ===============================
# 10. MEMORY MANAGEMENT AND CLEANUP
# ===============================

def cleanup_gpu_memory():
    """Proper GPU memory cleanup"""
    # Delete all model variables
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        cached = torch.cuda.memory_reserved() / 1e9      # GB
        
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

# ===============================
# MAIN EXECUTION EXAMPLE
# ===============================

if __name__ == "__main__":
    print("Starting LLM Implementation Demo...")
    
    # 1. Check memory usage
    print("\n1. Memory Analysis:")
    memory_mb = check_memory_usage(model)
    
    # 2. Explore architecture
    print("\n2. Model Architecture:")
    explore_model_architecture(model)
    
    # 3. Test streaming generation
    print("\n3. Streaming Generation Test:")
    test_messages = [
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ]
    generate(PHI3, test_messages)
    
    # 4. Clean up
    print("\n4. Cleanup:")
    cleanup_gpu_memory()
    
    print("\nDemo completed!")

# ===============================
# KEY INSIGHTS FROM THE CODE
# ===============================

"""
Key Insights:

1. QUANTIZATION IMPACT:
   - 4-bit quantization reduces memory from ~32GB to ~5.6GB (83% reduction)
   - Uses BitsAndBytesConfig for automatic optimization
   - Enables running large models on consumer hardware

2. MODEL ARCHITECTURE:
   - LLaMA uses 32 transformer layers (0-31)
   - Each layer has self-attention and MLP components
   - SiLU activation function (x * sigmoid(x)) for better gradients
   - Rotary positional embeddings for position awareness

3. STREAMING BENEFITS:
   - TextStreamer enables real-time token output
   - Improves user experience with immediate feedback
   - Memory efficient - processes tokens as generated

4. MEMORY MANAGEMENT:
   - Always clean up GPU memory after use
   - Use torch.cuda.empty_cache() to free cached memory
   - Monitor memory usage with get_memory_footprint()

5. PRACTICAL DEPLOYMENT:
   - Quantization makes deployment feasible
   - Streaming improves perceived performance
   - Understanding internals helps with debugging
   - Multiple models can be compared systematically
"""