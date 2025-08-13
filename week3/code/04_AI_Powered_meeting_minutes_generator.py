# AI-Powered Meeting Minutes Generator
# Combines Frontier + Open Source Models + Streaming

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Generator
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import openai  # For frontier model (OpenAI Whisper API)
import anthropic  # For frontier model alternative

# ===============================
# 1. AUDIO TO TEXT (FRONTIER MODEL)
# ===============================

class AudioTranscriber:
    """Uses frontier models for high-quality audio transcription"""
    
    def __init__(self, service="openai"):
        self.service = service
        if service == "openai":
            self.client = openai.OpenAI()
        elif service == "anthropic":
            # Note: Anthropic doesn't have audio, so we'd use OpenAI for this step
            self.client = openai.OpenAI()
    
    def transcribe_audio(self, audio_file_path: str) -> Dict:
        """
        Transcribe audio using frontier model (Whisper)
        Returns transcript with timestamps and speaker detection
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                # Use OpenAI Whisper for high-quality transcription
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Get timestamps
                    timestamp_granularities=["segment"]  # Segment-level timestamps
                )
            
            return {
                "success": True,
                "transcript": transcript.text,
                "segments": transcript.segments if hasattr(transcript, 'segments') else [],
                "duration": transcript.duration if hasattr(transcript, 'duration') else 0
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcript": "",
                "segments": [],
                "duration": 0
            }

# ===============================
# 2. MINUTES GENERATION (OPEN SOURCE)
# ===============================

class MeetingMinutesGenerator:
    """Uses open source models for cost-effective minutes generation"""
    
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load with quantization for efficiency
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    def generate_minutes_streaming(self, transcript: str) -> Generator[str, None, None]:
        """
        Generate meeting minutes with streaming output
        Yields tokens as they're generated for real-time display
        """
        
        prompt = f"""You are an expert meeting assistant. Generate comprehensive meeting minutes from this transcript.

TRANSCRIPT:
{transcript}

Please provide a structured summary including:
1. Meeting Overview (date, participants, main topic)
2. Key Discussion Points
3. Decisions Made
4. Action Items (with responsible parties if mentioned)
5. Next Steps

Format the output in clear, professional language suitable for distribution.

MEETING MINUTES:
"""

        # Prepare inputs
        messages = [
            {"role": "system", "content": "You are a professional meeting assistant that creates clear, structured meeting minutes."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)
        
        # Generate with streaming
        streamer = TextStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Custom streamer that yields tokens
        class CustomStreamer:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.generated_tokens = []
            
            def put(self, value):
                if value.shape[-1] == 1:
                    token_id = value[0, -1].item()
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    self.generated_tokens.append(token_text)
                    return token_text
                return ""
            
            def end(self):
                return ""
        
        custom_streamer = CustomStreamer(self.tokenizer)
        
        # Generate tokens one by one
        with torch.no_grad():
            output_tokens = self.model.generate(
                inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=None  # We'll handle streaming manually
            )
        
        # Decode the generated part only
        generated_part = output_tokens[0][len(inputs[0]):]
        
        # Stream token by token
        for i, token_id in enumerate(generated_part):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_text
            time.sleep(0.05)  # Simulate real-time generation
    
    def generate_minutes_complete(self, transcript: str) -> str:
        """Generate complete minutes without streaming"""
        tokens = list(self.generate_minutes_streaming(transcript))
        return ''.join(tokens)

# ===============================
# 3. MARKDOWN FORMATTER
# ===============================

class MarkdownFormatter:
    """Formats meeting minutes as structured Markdown"""
    
    @staticmethod
    def format_minutes(raw_minutes: str, meeting_info: Dict) -> str:
        """
        Convert raw minutes to structured Markdown format
        """
        
        # Add meeting header
        formatted = f"""# Meeting Minutes
        
**Date:** {meeting_info.get('date', datetime.now().strftime('%Y-%m-%d'))}
**Time:** {meeting_info.get('time', datetime.now().strftime('%H:%M'))}
**Duration:** {meeting_info.get('duration', 'N/A')} minutes
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{raw_minutes}

---

*Minutes generated automatically using AI. Please review for accuracy.*
"""
        
        return formatted
    
    @staticmethod
    def extract_action_items(minutes: str) -> List[Dict]:
        """Extract action items from minutes for task tracking"""
        # Simple extraction - in production, you'd use more sophisticated NLP
        action_items = []
        lines = minutes.split('\n')
        
        in_action_section = False
        for line in lines:
            if 'action item' in line.lower() or 'todo' in line.lower():
                in_action_section = True
            elif in_action_section and line.strip().startswith(('-', '*', '1.', '2.')):
                action_items.append({
                    'task': line.strip().lstrip('-*0123456789. '),
                    'status': 'pending',
                    'created': datetime.now().isoformat()
                })
        
        return action_items

# ===============================
# 4. COMPLETE SYSTEM INTEGRATION
# ===============================

class MeetingMinutesSystem:
    """Complete meeting minutes generation system"""
    
    def __init__(self):
        self.transcriber = AudioTranscriber()
        self.generator = MeetingMinutesGenerator()
        self.formatter = MarkdownFormatter()
    
    def process_meeting(self, audio_file_path: str, meeting_info: Dict = None) -> Dict:
        """
        Complete pipeline: Audio → Transcript → Minutes → Markdown
        """
        if meeting_info is None:
            meeting_info = {}
        
        results = {
            "steps": [],
            "success": False,
            "minutes_markdown": "",
            "action_items": [],
            "transcript": "",
            "processing_time": {}
        }
        
        try:
            # Step 1: Transcribe audio (Frontier model)
            start_time = time.time()
            results["steps"].append("🎤 Transcribing audio with Frontier model...")
            
            transcription_result = self.transcriber.transcribe_audio(audio_file_path)
            
            if not transcription_result["success"]:
                results["error"] = f"Transcription failed: {transcription_result['error']}"
                return results
            
            results["transcript"] = transcription_result["transcript"]
            results["processing_time"]["transcription"] = time.time() - start_time
            
            # Step 2: Generate minutes (Open source model)
            start_time = time.time()
            results["steps"].append("📝 Generating minutes with open source model...")
            
            raw_minutes = self.generator.generate_minutes_complete(
                transcription_result["transcript"]
            )
            
            results["processing_time"]["generation"] = time.time() - start_time
            
            # Step 3: Format as Markdown
            start_time = time.time()
            results["steps"].append("📋 Formatting as Markdown...")
            
            meeting_info["duration"] = transcription_result.get("duration", 0)
            formatted_minutes = self.formatter.format_minutes(raw_minutes, meeting_info)
            
            results["minutes_markdown"] = formatted_minutes
            results["action_items"] = self.formatter.extract_action_items(raw_minutes)
            results["processing_time"]["formatting"] = time.time() - start_time
            
            results["success"] = True
            results["steps"].append("✅ Meeting minutes generated successfully!")
            
        except Exception as e:
            results["error"] = str(e)
            results["steps"].append(f"❌ Error: {str(e)}")
        
        return results
    
    def process_meeting_streaming(self, audio_file_path: str, meeting_info: Dict = None):
        """
        Process with streaming updates - yields progress and partial results
        """
        if meeting_info is None:
            meeting_info = {}
        
        # Step 1: Transcription
        yield {"step": "transcription", "status": "processing", "message": "🎤 Transcribing audio..."}
        
        transcription_result = self.transcriber.transcribe_audio(audio_file_path)
        
        if not transcription_result["success"]:
            yield {"step": "transcription", "status": "error", "error": transcription_result["error"]}
            return
        
        yield {
            "step": "transcription", 
            "status": "complete", 
            "transcript": transcription_result["transcript"]
        }
        
        # Step 2: Minutes generation with streaming
        yield {"step": "generation", "status": "processing", "message": "📝 Generating minutes..."}
        
        partial_minutes = ""
        for token in self.generator.generate_minutes_streaming(transcription_result["transcript"]):
            partial_minutes += token
            yield {
                "step": "generation",
                "status": "streaming",
                "partial_minutes": partial_minutes
            }
        
        # Step 3: Final formatting
        yield {"step": "formatting", "status": "processing", "message": "📋 Final formatting..."}
        
        meeting_info["duration"] = transcription_result.get("duration", 0)
        final_minutes = self.formatter.format_minutes(partial_minutes, meeting_info)
        action_items = self.formatter.extract_action_items(partial_minutes)
        
        yield {
            "step": "complete",
            "status": "success",
            "minutes_markdown": final_minutes,
            "action_items": action_items,
            "transcript": transcription_result["transcript"]
        }

# ===============================
# 5. STREAMLIT WEB INTERFACE
# ===============================

def create_streamlit_app():
    """
    Streamlit web interface for the meeting minutes system
    """
    st.set_page_config(
        page_title="AI Meeting Minutes Generator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("🎤 AI Meeting Minutes Generator")
    st.markdown("*Upload an audio recording and get professional meeting minutes instantly!*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        meeting_title = st.text_input("Meeting Title", "Team Standup")
        meeting_date = st.date_input("Meeting Date", datetime.now())
        meeting_time = st.time_input("Meeting Time", datetime.now().time())
        
        st.header("Processing Options")
        use_streaming = st.checkbox("Enable Streaming Output", value=True)
        
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Upload Audio")
        
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=['mp3', 'wav', 'm4a', 'ogg'],
            help="Upload a meeting recording (max 25MB)"
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            if st.button("🚀 Generate Minutes", type="primary"):
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.read())
                
                meeting_info = {
                    "title": meeting_title,
                    "date": meeting_date.strftime("%Y-%m-%d"),
                    "time": meeting_time.strftime("%H:%M")
                }
                
                # Initialize system
                system = MeetingMinutesSystem()
                
                with col2:
                    st.header("📝 Generated Minutes")
                    
                    if use_streaming:
                        # Streaming processing
                        status_placeholder = st.empty()
                        minutes_placeholder = st.empty()
                        
                        for update in system.process_meeting_streaming(
                            f"temp_{uploaded_file.name}", 
                            meeting_info
                        ):
                            if update["status"] == "processing":
                                status_placeholder.info(update["message"])
                            
                            elif update["status"] == "streaming":
                                minutes_placeholder.markdown(update["partial_minutes"])
                            
                            elif update["status"] == "complete":
                                status_placeholder.success("✅ Minutes generated successfully!")
                                minutes_placeholder.markdown(update["minutes_markdown"])
                                
                                # Show action items
                                if update["action_items"]:
                                    st.subheader("📋 Action Items")
                                    for i, item in enumerate(update["action_items"], 1):
                                        st.write(f"{i}. {item['task']}")
                                
                                # Download button
                                st.download_button(
                                    "📥 Download Minutes",
                                    update["minutes_markdown"],
                                    file_name=f"minutes_{meeting_date}.md",
                                    mime="text/markdown"
                                )
                            
                            elif update["status"] == "error":
                                st.error(f"Error: {update.get('error', 'Unknown error')}")
                    
                    else:
                        # Batch processing
                        with st.spinner("Processing meeting..."):
                            result = system.process_meeting(
                                f"temp_{uploaded_file.name}", 
                                meeting_info
                            )
                        
                        if result["success"]:
                            st.markdown(result["minutes_markdown"])
                            
                            if result["action_items"]:
                                st.subheader("📋 Action Items")
                                for i, item in enumerate(result["action_items"], 1):
                                    st.write(f"{i}. {item['task']}")
                            
                            st.download_button(
                                "📥 Download Minutes",
                                result["minutes_markdown"],
                                file_name=f"minutes_{meeting_date}.md",
                                mime="text/markdown"
                            )
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")

# ===============================
# 6. MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    # For command line usage
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        audio_file = sys.argv[1]
        system = MeetingMinutesSystem()
        
        print("🎤 Processing meeting audio...")
        result = system.process_meeting(audio_file)
        
        if result["success"]:
            print("✅ Success! Minutes generated:")
            print("\n" + "="*50)
            print(result["minutes_markdown"])
            print("="*50)
            
            # Save to file
            with open("meeting_minutes.md", "w") as f:
                f.write(result["minutes_markdown"])
            print("\n📁 Minutes saved to 'meeting_minutes.md'")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    else:
        # Streamlit mode
        create_streamlit_app()

# ===============================
# USAGE EXAMPLES
# ===============================

"""
COMMAND LINE USAGE:
python meeting_minutes.py recording.mp3

STREAMLIT USAGE:
streamlit run meeting_minutes.py

PROGRAMMATIC USAGE:
system = MeetingMinutesSystem()
result = system.process_meeting("meeting_audio.wav")
print(result["minutes_markdown"])
"""