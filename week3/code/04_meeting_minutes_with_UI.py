# Complete Meeting Minutes System with UI
# Extracted from the notebook and enhanced with full UI implementation

# ===============================
# IMPORTS AND SETUP
# ===============================

import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google.colab import userdata, drive
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gradio as gr
import json
import time
from datetime import datetime
from typing import Dict, List, Generator
import tempfile
import io

# ===============================
# CONSTANTS AND CONFIGURATION
# ===============================

# Model configurations
AUDIO_MODEL = "whisper-1"  # OpenAI Whisper for audio transcription
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Open source model for text generation

# System prompts
SYSTEM_MESSAGE = """You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners."""

USER_PROMPT_TEMPLATE = """Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees and date; discussion points; takeaways; and action items with owners if mentioned."""

# ===============================
# AUTHENTICATION AND SETUP
# ===============================

def setup_environment():
    """Setup authentication and mount drives"""
    
    # Mount Google Drive for file access
    try:
        drive.mount("/content/drive")
        print("✅ Google Drive mounted successfully")
    except Exception as e:
        print(f"⚠️ Drive mount failed: {e}")
    
    # Setup HuggingFace authentication
    try:
        hf_token = userdata.get('HF_TOKEN')
        login(hf_token, add_to_git_credential=True)
        print("✅ HuggingFace authentication successful")
    except Exception as e:
        print(f"⚠️ HuggingFace auth failed: {e}")
    
    # Setup OpenAI authentication
    try:
        openai_api_key = userdata.get('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=openai_api_key)
        print("✅ OpenAI authentication successful")
        return openai_client
    except Exception as e:
        print(f"⚠️ OpenAI auth failed: {e}")
        return None

# ===============================
# AUDIO TRANSCRIPTION (FRONTIER MODEL)
# ===============================

class AudioTranscriber:
    """Handles audio transcription using OpenAI Whisper"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def transcribe_audio(self, audio_file_path: str) -> Dict:
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            Dict: Transcription results with metadata
        """
        try:
            print(f"🎤 Transcribing audio file: {audio_file_path}")
            
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=AUDIO_MODEL,
                    file=audio_file,
                    response_format="text"
                )
            
            return {
                "success": True,
                "transcript": transcription,
                "file_path": audio_file_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcript": "",
                "file_path": audio_file_path
            }

# ===============================
# MINUTES GENERATION (OPEN SOURCE MODEL)
# ===============================

class MinutesGenerator:
    """Handles meeting minutes generation using open source LLM"""
    
    def __init__(self):
        self.model_name = LLAMA
        self.tokenizer = None
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Initialize the model and tokenizer with quantization"""
        try:
            print(f"🤖 Loading model: {self.model_name}")
            
            # Setup tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization config for memory efficiency
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e
    
    def generate_minutes(self, transcript: str, streaming: bool = False) -> str:
        """
        Generate meeting minutes from transcript
        
        Args:
            transcript (str): Meeting transcript
            streaming (bool): Whether to use streaming output
            
        Returns:
            str: Generated meeting minutes in markdown format
        """
        try:
            # Prepare the prompt
            user_prompt = f"{USER_PROMPT_TEMPLATE}\n\nTranscript:\n{transcript}"
            
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.model.device)
            
            # Setup generation parameters
            generation_kwargs = {
                "input_ids": inputs,
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            if streaming:
                # Setup streaming
                streamer = TextStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                generation_kwargs["streamer"] = streamer
            
            # Generate response
            print("📝 Generating meeting minutes...")
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            # Decode the response
            generated_part = outputs[0][len(inputs[0]):]
            minutes = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            
            return minutes.strip()
            
        except Exception as e:
            print(f"❌ Minutes generation failed: {e}")
            return f"Error generating minutes: {str(e)}"

# ===============================
# COMPLETE PIPELINE
# ===============================

class MeetingMinutesPipeline:
    """Complete pipeline combining audio transcription and minutes generation"""
    
    def __init__(self):
        self.openai_client = setup_environment()
        self.transcriber = AudioTranscriber(self.openai_client) if self.openai_client else None
        self.generator = MinutesGenerator()
    
    def process_meeting(self, audio_file_path: str, progress_callback=None) -> Dict:
        """
        Complete processing pipeline
        
        Args:
            audio_file_path (str): Path to audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict: Results including transcript and minutes
        """
        results = {
            "success": False,
            "transcript": "",
            "minutes": "",
            "error": None,
            "processing_time": {}
        }
        
        try:
            if progress_callback:
                progress_callback("🎤 Transcribing audio with Frontier model...")
            
            # Step 1: Transcribe audio
            start_time = time.time()
            transcription_result = self.transcriber.transcribe_audio(audio_file_path)
            results["processing_time"]["transcription"] = time.time() - start_time
            
            if not transcription_result["success"]:
                results["error"] = f"Transcription failed: {transcription_result['error']}"
                return results
            
            results["transcript"] = transcription_result["transcript"]
            
            if progress_callback:
                progress_callback("📝 Generating minutes with open source model...")
            
            # Step 2: Generate minutes
            start_time = time.time()
            minutes = self.generator.generate_minutes(results["transcript"])
            results["processing_time"]["generation"] = time.time() - start_time
            
            results["minutes"] = minutes
            results["success"] = True
            
            if progress_callback:
                progress_callback("✅ Processing complete!")
            
        except Exception as e:
            results["error"] = str(e)
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)}")
        
        return results

# ===============================
# GRADIO UI IMPLEMENTATION
# ===============================

def create_gradio_interface():
    """Create a Gradio web interface for the meeting minutes system"""
    
    # Initialize the pipeline
    pipeline = MeetingMinutesPipeline()
    
    def process_audio_file(audio_file, progress=gr.Progress()):
        """Process uploaded audio file and return results"""
        
        if audio_file is None:
            return "❌ Please upload an audio file", "", ""
        
        try:
            # Update progress
            progress(0.1, desc="Initializing...")
            
            # Process the meeting
            def progress_callback(message):
                if "Transcribing" in message:
                    progress(0.3, desc=message)
                elif "Generating" in message:
                    progress(0.7, desc=message)
                elif "complete" in message:
                    progress(1.0, desc=message)
            
            results = pipeline.process_meeting(audio_file, progress_callback)
            
            if results["success"]:
                # Format the results
                processing_info = f"""
## Processing Summary ✅

**Transcription Time:** {results['processing_time'].get('transcription', 0):.2f} seconds
**Generation Time:** {results['processing_time'].get('generation', 0):.2f} seconds
**Total Time:** {sum(results['processing_time'].values()):.2f} seconds

**Models Used:**
- 🎤 **Audio → Text**: OpenAI Whisper (Frontier Model)
- 📝 **Text → Minutes**: LLaMA 3.1 8B (Open Source, Quantized)
"""
                
                return processing_info, results["transcript"], results["minutes"]
            else:
                error_msg = f"❌ Processing failed: {results['error']}"
                return error_msg, "", ""
                
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}", "", ""
    
    def save_results(transcript, minutes):
        """Save results to files and return download links"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save transcript
        transcript_file = f"transcript_{timestamp}.txt"
        with open(transcript_file, "w") as f:
            f.write(transcript)
        
        # Save minutes
        minutes_file = f"minutes_{timestamp}.md"
        with open(minutes_file, "w") as f:
            f.write(minutes)
        
        return transcript_file, minutes_file
    
    # Create the Gradio interface
    with gr.Blocks(
        title="AI Meeting Minutes Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎤 AI Meeting Minutes Generator</h1>
            <p>Upload an audio recording and get professional meeting minutes instantly!</p>
            <p><strong>Hybrid AI:</strong> Frontier Model (Whisper) + Open Source (LLaMA 3.1)</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload Audio")
                
                audio_input = gr.Audio(
                    label="Meeting Recording",
                    type="filepath",
                    format="mp3"
                )
                
                process_btn = gr.Button(
                    "🚀 Generate Minutes",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### Supported Formats:
                - MP3, WAV, M4A, OGG
                - Max file size: 25MB
                - Max duration: 60 minutes
                
                ### What happens:
                1. **🎤 Transcription**: OpenAI Whisper converts audio to text
                2. **📝 Minutes**: LLaMA 3.1 generates structured minutes
                3. **📋 Format**: Output in professional Markdown format
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Results")
                
                # Processing info
                processing_info = gr.Markdown(
                    "Upload an audio file and click 'Generate Minutes' to start processing."
                )
                
                # Tabbed output
                with gr.Tabs():
                    with gr.TabItem("📝 Meeting Minutes"):
                        minutes_output = gr.Markdown(
                            label="Generated Minutes",
                            value="Minutes will appear here...",
                            elem_classes=["minutes-output"]
                        )
                        
                        download_minutes = gr.File(
                            label="Download Minutes (Markdown)",
                            visible=False
                        )
                    
                    with gr.TabItem("📄 Raw Transcript"):
                        transcript_output = gr.Textbox(
                            label="Audio Transcript",
                            value="Transcript will appear here...",
                            lines=15,
                            max_lines=20
                        )
                        
                        download_transcript = gr.File(
                            label="Download Transcript (Text)",
                            visible=False
                        )
        
        # Event handlers
        process_btn.click(
            fn=process_audio_file,
            inputs=[audio_input],
            outputs=[processing_info, transcript_output, minutes_output],
            show_progress=True
        )
        
        # Examples section
        gr.Markdown("## 📋 Example Output")
        
        with gr.Accordion("View Sample Meeting Minutes", open=False):
            gr.Markdown("""
# Meeting Minutes - Denver City Council

**Date:** Monday, October 9th  
**Location:** Denver City Council Chambers  
**Attendees:** Council Members: Lopez, Clark, Flynn, Gilmour, Cashman, Kenneche, Lopez, New, Ortega, Sussman, and President

## Summary
The meeting began with the Pledge of Allegiance led by Councilman Lopez. The Council then moved on to approve the minutes of the previous meeting, followed by Council announcements, presentations, and communications. The highlight of the meeting was the adoption of Proclamation 1127, Series of 2017, an observance of the second annual Indigenous Peoples Day in the City and County of Denver.

## Discussion Points
• Councilman Clark announced the first-ever Halloween parade in the Lucky District 7, to be held on October 21st at 6 o'clock p.m. on Broadway.
• Councilman Lopez introduced Proclamation 1127, Series of 2017, an observance of the second annual Indigenous Peoples Day in the City and County of Denver.
• Councilwoman Kinneche discussed the importance of Indigenous Peoples Day, highlighting the contributions of Indigenous people to the city's history, culture, and present.

## Takeaways
• The Council acknowledged the importance of Indigenous Peoples Day as a celebration of inclusivity and respect for all cultures.
• The Council recognized the contributions of Indigenous people to the city's history, culture, and present.
• The Council emphasized the need to address critical issues such as poverty, lack of access to services, and environmental protection.

## Action Items
• **Owner:** Councilman Lopez
  • **Action:** Adopt Proclamation 1127, Series of 2017, an observance of the second annual Indigenous Peoples Day in the City and County of Denver.
            """)
        
        # Footer
        gr.Markdown("""
        ---
        **Powered by:**
        - 🎤 **OpenAI Whisper** for high-accuracy audio transcription
        - 📝 **Meta LLaMA 3.1** for intelligent minutes generation
        - ⚡ **4-bit Quantization** for efficient processing
        - 🌊 **Streaming Output** for real-time results
        
        *Built with confidence using the latest in AI technology!*
        """)
    
    return demo

# ===============================
# STREAMLIT ALTERNATIVE UI
# ===============================

def create_streamlit_app():
    """Alternative Streamlit implementation"""
    import streamlit as st
    
    st.set_page_config(
        page_title="AI Meeting Minutes Generator",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .stProgress .stProgress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎤 AI Meeting Minutes Generator</h1>
        <p>Hybrid AI: Frontier Model + Open Source = Professional Results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        meeting_title = st.text_input("Meeting Title", "Team Meeting")
        meeting_date = st.date_input("Date", datetime.now())
        
        st.header("📊 Model Info")
        st.info("""
        **Audio → Text:**  
        OpenAI Whisper (Frontier)
        
        **Text → Minutes:**  
        LLaMA 3.1 8B (Open Source)
        """)
        
        st.header("💡 Tips")
        st.markdown("""
        • Clear audio = better results
        • Speak clearly and avoid background noise
        • Shorter meetings process faster
        • Support for multiple languages
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Audio")
        
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=['mp3', 'wav', 'm4a', 'ogg'],
            help="Upload a meeting recording (max 25MB)"
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            # File info
            st.info(f"""
            **File:** {uploaded_file.name}  
            **Size:** {uploaded_file.size / 1024 / 1024:.1f} MB
            """)
            
            if st.button("🚀 Generate Minutes", type="primary"):
                # Initialize pipeline
                pipeline = MeetingMinutesPipeline()
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(message):
                    if "Transcribing" in message:
                        progress_bar.progress(30)
                        status_text.text(message)
                    elif "Generating" in message:
                        progress_bar.progress(70)
                        status_text.text(message)
                    elif "complete" in message:
                        progress_bar.progress(100)
                        status_text.text(message)
                
                # Process the meeting
                with st.spinner("Processing meeting..."):
                    results = pipeline.process_meeting(temp_path, update_progress)
                
                # Display results in the second column
                with col2:
                    if results["success"]:
                        st.success("✅ Processing complete!")
                        
                        # Processing stats
                        st.info(f"""
                        **Processing Time:**
                        - Transcription: {results['processing_time'].get('transcription', 0):.2f}s
                        - Generation: {results['processing_time'].get('generation', 0):.2f}s
                        - Total: {sum(results['processing_time'].values()):.2f}s
                        """)
                        
                        # Tabbed results
                        tab1, tab2 = st.tabs(["📝 Minutes", "📄 Transcript"])
                        
                        with tab1:
                            st.markdown(results["minutes"])
                            
                            # Download button
                            st.download_button(
                                "📥 Download Minutes",
                                results["minutes"],
                                file_name=f"minutes_{meeting_date}.md",
                                mime="text/markdown"
                            )
                        
                        with tab2:
                            st.text_area(
                                "Raw Transcript",
                                results["transcript"],
                                height=300
                            )
                            
                            st.download_button(
                                "📥 Download Transcript",
                                results["transcript"],
                                file_name=f"transcript_{meeting_date}.txt",
                                mime="text/plain"
                            )
                    
                    else:
                        st.error(f"❌ Processing failed: {results['error']}")
    
    with col2:
        if not uploaded_file:
            st.header("📝 Results")
            st.info("Upload an audio file to see results here.")
            
            # Show example
            with st.expander("👀 View Example Output"):
                st.markdown("""
                # Meeting Minutes - Denver City Council
                
                **Date:** Monday, October 9th  
                **Attendees:** Council Members: Lopez, Clark, Flynn...
                
                ## Summary
                The meeting began with the Pledge of Allegiance...
                
                ## Discussion Points
                • Councilman Clark announced...
                • Councilman Lopez introduced...
                
                ## Action Items
                • **Owner:** Councilman Lopez
                  • **Action:** Adopt Proclamation 1127...
                """)

# ===============================
# MAIN EXECUTION
# ===============================

def main():
    """Main execution function"""
    import sys
    
    # Check if running in different environments
    if 'google.colab' in sys.modules:
        # Running in Google Colab - use Gradio
        print("🚀 Starting Gradio interface in Colab...")
        demo = create_gradio_interface()
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860
        )
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        # Running with Streamlit flag
        print("🚀 Starting Streamlit interface...")
        create_streamlit_app()
    
    elif len(sys.argv) > 1:
        # Command line usage
        audio_file = sys.argv[1]
        pipeline = MeetingMinutesPipeline()
        
        print("🎤 Processing meeting audio...")
        results = pipeline.process_meeting(audio_file)
        
        if results["success"]:
            print("✅ Success!")
            print("\n" + "="*50)
            print("TRANSCRIPT:")
            print(results["transcript"])
            print("\n" + "="*50)
            print("MINUTES:")
            print(results["minutes"])
            print("="*50)
        else:
            print(f"❌ Error: {results['error']}")
    
    else:
        # Default - start Gradio interface
        print("🚀 Starting Gradio interface...")
        demo = create_gradio_interface()
        demo.launch()

if __name__ == "__main__":
    main()

# ===============================
# USAGE INSTRUCTIONS
# ===============================

"""
USAGE OPTIONS:

1. GOOGLE COLAB (Recommended):
   - Run this notebook in Google Colab
   - Gradio interface will start automatically
   - Share link provided for access

2. STREAMLIT:
   streamlit run meeting_minutes.py --streamlit

3. GRADIO (Local):
   python meeting_minutes.py

4. COMMAND LINE:
   python meeting_minutes.py recording.mp3

REQUIREMENTS:
- Google Colab environment (recommended)
- OpenAI API key in Colab secrets
- HuggingFace token in Colab secrets
- Audio file (MP3, WAV, M4A, OGG)

FEATURES:
✅ Hybrid AI (Frontier + Open Source)
✅ Real-time progress tracking
✅ Professional markdown output
✅ Download capabilities
✅ Multiple interface options
✅ Error handling and recovery
✅ Memory-efficient processing
"""