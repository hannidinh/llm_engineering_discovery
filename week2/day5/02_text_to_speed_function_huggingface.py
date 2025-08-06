import torch
import soundfile as sf
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import BarkModel, BarkProcessor
from datasets import load_dataset
from pydub import AudioSegment
from pydub.playback import play
import io

class HuggingFaceTTS:
    def __init__(self, model_name="speecht5"):
        self.model_name = model_name
        self.setup_model()
    
    def setup_model(self):
        """Initialize the selected TTS model"""
        if self.model_name == "speecht5":
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Create a default speaker embedding (512-dimensional vector)
            # This is a workaround since the original dataset is no longer supported
            self.speaker_embeddings = torch.randn(1, 512)
            
        elif self.model_name == "bark":
            self.processor = BarkProcessor.from_pretrained("suno/bark")
            self.model = BarkModel.from_pretrained("suno/bark")
    
    def generate_speech(self, text, voice=None, save_path=None):
        """
        Generate speech from text
        
        Args:
            text (str): Text to convert to speech
            voice (str/int): Voice selection (model-dependent)
            save_path (str): Path to save audio file
            
        Returns:
            AudioSegment: Audio data
        """
        try:
            if self.model_name == "speecht5":
                audio_array = self._speecht5_generate(text, voice)
                sample_rate = 16000
                
            elif self.model_name == "bark":
                audio_array = self._bark_generate(text, voice)
                sample_rate = 24000
            
            # Save if path provided
            if save_path:
                sf.write(save_path, audio_array, sample_rate)
                print(f"Audio saved to: {save_path}")
            
            # Convert to AudioSegment for playback
            audio_segment = self._numpy_to_audiosegment(audio_array, sample_rate)
            return audio_segment
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def _speecht5_generate(self, text, speaker_idx=None):
        """Generate speech using SpeechT5"""
        inputs = self.processor(text=text, return_tensors="pt")
        
        # Use different speaker if specified (using random embeddings for different voices)
        if speaker_idx is not None:
            # Generate different random embeddings for different speaker indices
            torch.manual_seed(speaker_idx)  # Deterministic but different per speaker_idx
            speaker_embeddings = torch.randn(1, 512)
        else:
            speaker_embeddings = self.speaker_embeddings
        
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
        return speech.numpy()
    
    def _bark_generate(self, text, voice_preset=None):
        """Generate speech using Bark"""
        if voice_preset is None:
            voice_preset = "v2/en_speaker_6"
        
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        return audio_array.cpu().numpy().squeeze()
    
    def _numpy_to_audiosegment(self, audio_array, sample_rate):
        """Convert numpy array to AudioSegment"""
        # Normalize audio
        audio_array = np.int16(audio_array * 32767)
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )
        return audio_segment

# Language Learning Helper Class
class LanguageLearningTTS:
    def __init__(self):
        self.tts = HuggingFaceTTS("speecht5")  # or "bark"
        
        # Voice mappings for different speakers
        self.voices = {
            "male1": 7306,
            "female1": 1284,
            "male2": 4005,
            "female2": 5799
        }
    
    def practice_pronunciation(self, phrases, voice="female1"):
        """Interactive pronunciation practice"""
        voice_idx = self.voices.get(voice, 7306)
        
        for phrase in phrases:
            print(f"\n📢 Listen to: '{phrase}'")
            audio = self.tts.generate_speech(phrase, voice=voice_idx)
            if audio:
                play(audio)
            
            input("Press Enter when ready for next phrase...")
    
    def generate_lesson_audio(self, vocabulary, output_dir="lessons"):
        """Generate audio files for vocabulary lessons"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (word, definition) in enumerate(vocabulary.items()):
            text = f"{word}. {definition}. {word}."
            filename = f"{output_dir}/lesson_{i+1:03d}_{word.replace(' ', '_')}.wav"
            self.tts.generate_speech(text, save_path=filename)

# Usage Examples
if __name__ == "__main__":
    # Initialize TTS
    tts = HuggingFaceTTS("speecht5")  # or "bark"
    
    # Basic usage
    audio = tts.generate_speech("Hello! This is generated using Hugging Face models.")
    play(audio)
    
    # Language learning example
    learning_tts = LanguageLearningTTS()
    
    spanish_phrases = [
        "Hola, ¿cómo estás?",
        "Me llamo María",
        "¿Dónde está la biblioteca?",
        "Muchas gracias"
    ]
    
    learning_tts.practice_pronunciation(spanish_phrases, voice="female1")
    
    # Generate vocabulary lessons
    vocabulary = {
        "casa": "house or home",
        "agua": "water",
        "comida": "food",
        "familia": "family"
    }
    
    learning_tts.generate_lesson_audio(vocabulary)