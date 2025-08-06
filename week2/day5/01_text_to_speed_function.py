from pydub import AudioSegment
from pydub.playback import play
import openai
from io import BytesIO
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Check if API key is loaded
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"✅ OpenAI API Key loaded successfully (begins with: {openai_api_key[:8]})")
else:
    print("❌ OpenAI API Key not found. Please check your .env file.")

# Initialize OpenAI client
client = openai.OpenAI()

def talker(message, voice="alloy", model="tts-1", speed=1.0, save_path=None):
    """
    Generate speech from text using OpenAI's TTS API
    
    Args:
        message (str): Text to convert to speech
        voice (str): Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        model (str): Model to use (tts-1 or tts-1-hd)
        speed (float): Speed of speech (0.25 to 4.0)
        save_path (str): Optional path to save audio file
    
    Returns:
        AudioSegment: Audio object for playback or further processing
    """
    try:
        # Generate speech
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=message,
            speed=speed,
            response_format="mp3"
        )
        
        # Convert to audio stream
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        
        # Save to file if path provided
        if save_path:
            audio.export(save_path, format="mp3")
            print(f"Audio saved to: {save_path}")
        
        return audio
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def play_text(text, voice="alloy", model="tts-1"):
    """Quick function to speak text immediately"""
    audio = talker(text, voice=voice, model=model)
    if audio:
        play(audio)

def batch_generate_speech(texts, voice="alloy", output_dir="audio_files"):
    """Generate speech for multiple texts and save as files"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for i, text in enumerate(texts, 1):
        filename = f"{output_dir}/speech_{i:03d}.mp3"
        audio = talker(text, voice=voice, save_path=filename)
        if audio:
            print(f"Generated: {filename}")

# Language learning helper function
def language_practice_session(phrases, target_language_voice="alloy"):
    """
    Practice pronunciation by hearing native-like speech
    """
    for phrase in phrases:
        print(f"\nPlaying: {phrase}")
        audio = talker(phrase, voice=target_language_voice)
        if audio:
            play(audio)
        input("Press Enter for next phrase...")

# Example usage
if __name__ == "__main__":
    # Basic usage
    audio = talker("Hello, how are you today?", voice="nova")
    if audio:
        play(audio)
    
    # Save to file
    talker("This will be saved", save_path="example.mp3")
    
    # Language learning example
    spanish_phrases = [
        "Hola, ¿cómo estás?",
        "Me llamo María",
        "¿Dónde está el baño?",
        "Gracias por tu ayuda"
    ]
    
    language_practice_session(spanish_phrases, "nova")