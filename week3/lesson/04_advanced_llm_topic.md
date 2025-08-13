# Advanced LLM Topics: Quantization, Model Internals, and Streaming

## 1. Quantization

### What is Quantization?
Quantization reduces the precision of model weights and activations from 32-bit or 16-bit floating point numbers to lower precision formats (8-bit, 4-bit, or even lower). This dramatically reduces memory usage and inference time while maintaining most of the model's performance.

### Types of Quantization

#### **Post-Training Quantization (PTQ)**
- Applied after training is complete
- Faster to implement, no retraining needed
- Slightly lower quality than training-aware quantization

#### **Quantization-Aware Training (QAT)**
- Quantization is simulated during training
- Better quality preservation
- Requires access to training pipeline

#### **Dynamic vs Static Quantization**
- **Dynamic**: Quantizes weights statically, activations dynamically
- **Static**: Both weights and activations quantized statically (requires calibration)

### Precision Levels

| Precision | Memory Reduction | Performance Impact | Use Case |
|-----------|------------------|-------------------|----------|
| FP32 → FP16 | 50% | Minimal | General optimization |
| FP32 → INT8 | 75% | 2-5% quality loss | Production deployment |
| FP32 → INT4 | 87.5% | 5-15% quality loss | Resource-constrained |
| FP32 → INT2 | 93.75% | 15-30% quality loss | Extreme efficiency |

### Real-World Applications

#### **1. Mobile AI Applications**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from optimum.intel import IPEXModel

# Example: Quantized model for mobile deployment
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True  # Automatic 8-bit quantization
)

# Benefits:
# - 75% less memory usage
# - 2-3x faster inference
# - Runs on devices with 4GB RAM instead of 16GB
```

#### **2. Edge Computing**
- **Autonomous Vehicles**: Real-time decision making with limited compute
- **IoT Devices**: Smart cameras, voice assistants with local processing
- **Industrial Equipment**: Predictive maintenance without cloud dependency

#### **3. Cost Optimization in Production**
```python
# Cost comparison example:
# Original: 70B model requires 8x A100 GPUs = $24/hour
# Quantized: 70B model with 4-bit runs on 2x A100 = $6/hour
# Savings: 75% cost reduction with 10% performance drop

class QuantizedInferenceService:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    
    def generate_response(self, prompt, max_tokens=512):
        # Same API, 75% less cost
        return self.model.generate(prompt, max_new_tokens=max_tokens)
```

### Quantization Techniques and Tools

#### **BitsAndBytes (4-bit/8-bit)**
```python
from transformers import BitsAndBytesConfig

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
```

#### **GPTQ (GPU-based quantization)**
```python
from auto_gptq import AutoGPTQForCausalLM

# Load GPTQ quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-Chat-GPTQ",
    device="cuda:0",
    use_triton=True
)
```

---

## 2. Model Internals

### Understanding Transformer Architecture
Modern LLMs are based on the Transformer architecture with key components you can access and manipulate:

#### **Attention Mechanisms**
- **Self-Attention**: How tokens relate to each other
- **Multi-Head Attention**: Multiple attention patterns simultaneously
- **Cross-Attention**: Relating different sequences (encoder-decoder)

#### **Layer Structure**
- **Embedding Layer**: Converts tokens to vectors
- **Transformer Blocks**: Attention + Feed-forward layers
- **Layer Normalization**: Stabilizes training
- **Output Head**: Converts hidden states to vocabulary probabilities

### Accessing Model Internals

#### **1. Attention Visualization**
```python
from transformers import AutoModel, AutoTokenizer
import torch

def analyze_attention_patterns(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Access attention weights
    attention_weights = outputs.attentions  # Tuple of tensors (one per layer)
    
    # Attention shape: [batch_size, num_heads, seq_len, seq_len]
    last_layer_attention = attention_weights[-1][0]  # Last layer, first batch
    
    return {
        'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
        'attention_weights': last_layer_attention,
        'num_layers': len(attention_weights),
        'num_heads': last_layer_attention.shape[0]
    }

# Example usage
result = analyze_attention_patterns(
    "The cat sat on the mat", 
    "microsoft/Phi-3-mini-4k-instruct"
)
```

#### **2. Hidden State Analysis**
```python
def extract_hidden_states(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Hidden states from all layers
    hidden_states = outputs.hidden_states  # Tuple of tensors
    
    return {
        'input_embeddings': hidden_states[0],  # After embedding layer
        'final_hidden_state': hidden_states[-1],  # Before output
        'intermediate_states': hidden_states[1:-1],  # All middle layers
        'layer_count': len(hidden_states)
    }
```

### Real-World Applications

#### **1. Model Debugging and Analysis**
```python
class ModelDiagnostics:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            output_attentions=True,
            output_hidden_states=True
        )
    
    def diagnose_reasoning_failure(self, question, wrong_answer):
        """Analyze why model gave wrong answer"""
        inputs = self.tokenizer(question, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Check attention patterns
        attention = outputs.attentions[-1][0]  # Last layer attention
        
        # Find which tokens model focused on
        avg_attention = attention.mean(dim=0)  # Average across heads
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        attention_scores = [(token, score.item()) for token, score in 
                           zip(tokens, avg_attention.mean(dim=1))]
        
        return {
            'most_attended_tokens': sorted(attention_scores, 
                                         key=lambda x: x[1], reverse=True)[:5],
            'reasoning_path': self.trace_information_flow(outputs.hidden_states)
        }
```

#### **2. Custom Model Modifications**
```python
class EnhancedModel(torch.nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Add custom classification head
        self.classifier = torch.nn.Linear(
            self.base_model.config.hidden_size, 
            num_classes=10
        )
        
        # Add custom attention mechanism
        self.custom_attention = torch.nn.MultiheadAttention(
            embed_dim=self.base_model.config.hidden_size,
            num_heads=8
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Apply custom attention to intermediate layer
        intermediate_state = base_outputs.hidden_states[6]  # 6th layer
        custom_attended, _ = self.custom_attention(
            intermediate_state, 
            intermediate_state, 
            intermediate_state
        )
        
        # Classification on custom features
        classification_logits = self.classifier(custom_attended[:, 0, :])
        
        return {
            'classification_logits': classification_logits,
            'base_outputs': base_outputs
        }
```

#### **3. Interpretability and Explainability**
- **Token Importance**: Which words most influence the output
- **Layer Analysis**: How understanding develops through layers
- **Attention Maps**: What the model "looks at" when generating
- **Concept Detection**: Finding neurons that detect specific concepts

---

## 3. Streaming

### What is Streaming?
Streaming allows models to generate and return tokens one at a time as they're produced, rather than waiting for the complete response. This dramatically improves perceived performance and enables real-time interactions.

### How Streaming Works
1. **Token-by-Token Generation**: Model generates one token at a time
2. **Immediate Transmission**: Each token sent as soon as generated
3. **Progressive Display**: User sees text appearing word by word
4. **Early Termination**: Can stop generation mid-stream if needed

### Implementation Methods

#### **1. Basic Streaming with Transformers**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

class StreamingGenerator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def stream_generate(self, prompt, max_new_tokens=512):
        """Generate text with streaming output"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Create text streamer
        streamer = TextStreamer(
            self.tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generate with streaming
        self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            streamer=streamer,  # This enables streaming
            pad_token_id=self.tokenizer.eos_token_id
        )

# Usage
generator = StreamingGenerator("microsoft/Phi-3-mini-4k-instruct")
generator.stream_generate("Explain quantum computing in simple terms:")
```

#### **2. Custom Streaming with Callbacks**
```python
from transformers.generation.utils import GenerationMixin
import time

class CustomStreamer:
    def __init__(self, tokenizer, callback_func=None):
        self.tokenizer = tokenizer
        self.callback_func = callback_func or print
        self.generated_tokens = []
    
    def put(self, value):
        """Called for each generated token"""
        if value.shape[-1] == 1:  # Single token
            token_id = value[0, -1].item()
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            self.generated_tokens.append(token_id)
            
            # Call callback with token
            self.callback_func(token_text)
            
            # Simulate network delay
            time.sleep(0.05)
    
    def end(self):
        """Called when generation is complete"""
        full_text = self.tokenizer.decode(self.generated_tokens, skip_special_tokens=True)
        print(f"\n\n[COMPLETE: {len(self.generated_tokens)} tokens generated]")

def stream_with_custom_handler(model, tokenizer, prompt):
    def token_handler(token):
        print(token, end='', flush=True)  # Print immediately
    
    streamer = CustomStreamer(tokenizer, token_handler)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    model.generate(
        **inputs,
        max_new_tokens=256,
        streamer=streamer,
        temperature=0.8,
        do_sample=True
    )
```

### Real-World Applications

#### **1. Real-Time Chat Applications**
```python
import asyncio
import websocket

class RealTimeChatBot:
    def __init__(self, model_name):
        self.generator = StreamingGenerator(model_name)
        self.active_connections = set()
    
    async def handle_chat_message(self, websocket, message):
        """Stream response to chat message"""
        try:
            # Prepare conversation context
            prompt = f"User: {message}\nAssistant: "
            
            # Stream response token by token
            for token in self.stream_tokens(prompt):
                await websocket.send(json.dumps({
                    'type': 'token',
                    'content': token,
                    'timestamp': time.time()
                }))
                
                # Small delay for natural typing speed
                await asyncio.sleep(0.03)
            
            # Send completion signal
            await websocket.send(json.dumps({
                'type': 'complete',
                'timestamp': time.time()
            }))
            
        except websocket.exceptions.ConnectionClosed:
            self.active_connections.discard(websocket)
    
    def stream_tokens(self, prompt):
        """Generator that yields tokens one by one"""
        inputs = self.generator.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            for new_token_id in self.generator.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )[0][len(inputs['input_ids'][0]):]:
                
                token_text = self.generator.tokenizer.decode(
                    [new_token_id], 
                    skip_special_tokens=True
                )
                yield token_text
```

#### **2. Live Content Generation**
```python
class LiveContentGenerator:
    def __init__(self):
        self.generator = StreamingGenerator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    def generate_live_blog_post(self, topic, target_length=1000):
        """Generate blog post with live updates"""
        prompt = f"""Write a comprehensive blog post about {topic}.
        Include introduction, main points, and conclusion.
        
        Blog post:"""
        
        word_count = 0
        current_paragraph = ""
        
        for token in self.stream_tokens(prompt):
            current_paragraph += token
            
            # Update UI every few tokens
            if len(current_paragraph) > 50:  # Update every ~10 words
                yield {
                    'type': 'paragraph_update',
                    'content': current_paragraph,
                    'word_count': word_count,
                    'progress': min(word_count / target_length, 1.0)
                }
                
                word_count += len(current_paragraph.split())
                current_paragraph = ""
        
        # Final update
        yield {
            'type': 'completion',
            'final_word_count': word_count,
            'status': 'complete'
        }
```

#### **3. Interactive Code Generation**
```python
class InteractiveCodeGenerator:
    def __init__(self):
        self.generator = StreamingGenerator("bigcode/starcoder2-7b")
    
    def generate_code_with_explanation(self, task_description):
        """Generate code with live explanation"""
        prompt = f"""Task: {task_description}

I'll solve this step by step:

1. First, I'll write the function:
```python"""
        
        code_block = ""
        in_code_block = False
        explanation_parts = []
        
        for token in self.stream_tokens(prompt):
            if "```" in token:
                in_code_block = not in_code_block
                
                if not in_code_block:  # End of code block
                    yield {
                        'type': 'code_complete',
                        'code': code_block.strip(),
                        'language': 'python'
                    }
                    code_block = ""
                continue
            
            if in_code_block:
                code_block += token
                yield {
                    'type': 'code_token',
                    'token': token,
                    'current_code': code_block
                }
            else:
                explanation_parts.append(token)
                yield {
                    'type': 'explanation_token',
                    'token': token,
                    'current_explanation': ''.join(explanation_parts)
                }
```

### Performance Benefits of Streaming

#### **Perceived Performance**
- **Time to First Token (TTFT)**: 50-200ms vs 5-30 seconds for full response
- **User Engagement**: 90% higher completion rates for streamed responses
- **Perceived Speed**: 3-5x faster feeling despite same total generation time

#### **Resource Efficiency**
```python
# Memory usage comparison
# Non-streaming: Must store full response in memory before sending
# Memory usage: O(full_response_length)

# Streaming: Only current token in memory
# Memory usage: O(1) per token

class MemoryEfficientStreaming:
    def __init__(self, model):
        self.model = model
        self.memory_usage = []
    
    def generate_streaming(self, prompt):
        """Generate with minimal memory footprint"""
        for token in self.model.stream(prompt):
            # Process token immediately, don't accumulate
            yield self.process_token(token)
            
            # Memory usage stays constant
            current_memory = self.get_memory_usage()
            self.memory_usage.append(current_memory)
    
    def generate_batch(self, prompt):
        """Traditional batch generation"""
        full_response = self.model.generate(prompt)  # High memory usage
        return full_response
```

### Best Practices for Streaming

#### **1. Error Handling**
```python
class RobustStreamingGenerator:
    def stream_with_error_handling(self, prompt):
        try:
            for token in self.generate_tokens(prompt):
                yield {'status': 'success', 'token': token}
        
        except torch.cuda.OutOfMemoryError:
            yield {'status': 'error', 'type': 'oom', 'message': 'GPU memory exhausted'}
        
        except Exception as e:
            yield {'status': 'error', 'type': 'generation', 'message': str(e)}
        
        finally:
            # Cleanup GPU memory
            torch.cuda.empty_cache()
```

#### **2. Rate Limiting and Quality Control**
```python
class QualityControlledStreaming:
    def __init__(self, model, max_tokens_per_second=10):
        self.model = model
        self.rate_limit = max_tokens_per_second
        self.last_token_time = 0
    
    def stream_with_quality_control(self, prompt):
        for token in self.model.stream(prompt):
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_token_time
            min_interval = 1.0 / self.rate_limit
            
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
            
            # Quality check
            if self.is_token_appropriate(token):
                yield token
                self.last_token_time = time.time()
            else:
                # Skip inappropriate tokens
                continue
```

These three advanced topics - Quantization, Model Internals, and Streaming - are essential for building production-ready LLM applications. They enable efficient deployment, deep understanding of model behavior, and responsive user experiences.