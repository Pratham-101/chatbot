import os
import asyncio
import tempfile
import wave
import numpy as np
import requests
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import whisper
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VoiceConfig:
    """Configuration for voice assistant"""
    # STT Configuration
    stt_provider: str = "whisper"  # "whisper", "google", "openai"
    whisper_model: str = "base"  # "tiny", "base", "small", "medium", "large"
    
    # TTS Configuration  
    tts_provider: str = "elevenlabs"  # "elevenlabs", "google", "pyttsx3"
    tts_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs voice ID
    tts_voice_name: str = "Rachel"  # Google TTS voice name
    
    # API Keys (set via environment variables)
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.1  # seconds
    
    def __post_init__(self):
        # Load API keys from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY', self.openai_api_key)
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY', self.elevenlabs_api_key)
        self.google_api_key = os.getenv('GOOGLE_API_KEY', self.google_api_key)

class VoiceAssistant:
    """Advanced voice assistant with multiple STT/TTS providers"""
    
    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        self.whisper_model = None
        self._init_whisper()
    
    def _init_whisper(self):
        """Initialize Whisper model for offline STT"""
        try:
            if self.config.stt_provider == "whisper":
                logger.info(f"Loading Whisper model: {self.config.whisper_model}")
                self.whisper_model = whisper.load_model(self.config.whisper_model)
                logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
    
    def speech_to_text(self, audio_data: bytes, audio_format: str = "wav") -> Tuple[str, float]:
        """
        Convert speech to text using configured provider
        Returns: (transcribed_text, confidence_score)
        """
        try:
            if self.config.stt_provider == "whisper":
                return self._whisper_stt(audio_data, audio_format)
            elif self.config.stt_provider == "google":
                return self._google_stt(audio_data, audio_format)
            elif self.config.stt_provider == "openai":
                return self._openai_stt(audio_data, audio_format)
            else:
                raise ValueError(f"Unsupported STT provider: {self.config.stt_provider}")
        except Exception as e:
            logger.error(f"STT error: {e}")
            return "", 0.0
    
    def _whisper_stt(self, audio_data: bytes, audio_format: str) -> Tuple[str, float]:
        """Use Whisper for offline STT"""
        if not self.whisper_model:
            raise Exception("Whisper model not loaded")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_file_path)
            text = result["text"].strip()
            # Whisper doesn't provide confidence, so we'll estimate based on language detection
            confidence = 0.8 if text else 0.0
            return text, confidence
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def _google_stt(self, audio_data: bytes, audio_format: str) -> Tuple[str, float]:
        """Use Google Speech-to-Text API"""
        if not self.config.google_api_key:
            raise Exception("Google API key not configured")
        
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.config.google_api_key}"
        
        # Convert audio to base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.config.sample_rate,
                "languageCode": "en-US",
                "enableAutomaticPunctuation": True
            },
            "audio": {
                "content": audio_b64
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'results' in result and result['results']:
                text = result['results'][0]['alternatives'][0]['transcript']
                confidence = result['results'][0]['alternatives'][0].get('confidence', 0.8)
                return text, confidence
            else:
                return "", 0.0
        else:
            raise Exception(f"Google STT API error: {response.status_code}")
    
    def _openai_stt(self, audio_data: bytes, audio_format: str) -> Tuple[str, float]:
        """Use OpenAI Whisper API"""
        if not self.config.openai_api_key:
            raise Exception("OpenAI API key not configured")
        
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        
        files = {"file": ("audio.wav", audio_data, "audio/wav")}
        data = {"model": "whisper-1"}
        
        response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '').strip()
            # OpenAI doesn't provide confidence
            confidence = 0.9 if text else 0.0
            return text, confidence
        else:
            raise Exception(f"OpenAI STT API error: {response.status_code}")
    
    def text_to_speech(self, text: str) -> Tuple[bytes, str]:
        """
        Convert text to speech using configured provider
        Returns: (audio_data, audio_format)
        """
        try:
            if self.config.tts_provider == "elevenlabs":
                return self._elevenlabs_tts(text)
            elif self.config.tts_provider == "google":
                return self._google_tts(text)
            elif self.config.tts_provider == "pyttsx3":
                return self._pyttsx3_tts(text)
            else:
                raise ValueError(f"Unsupported TTS provider: {self.config.tts_provider}")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b"", "wav"
    
    def _elevenlabs_tts(self, text: str) -> Tuple[bytes, str]:
        """Use ElevenLabs for high-quality TTS"""
        if not self.config.elevenlabs_api_key:
            raise Exception("ElevenLabs API key not configured")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.tts_voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.config.elevenlabs_api_key
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.content, "mp3"
        else:
            raise Exception(f"ElevenLabs TTS API error: {response.status_code}")
    
    def _google_tts(self, text: str) -> Tuple[bytes, str]:
        """Use Google Text-to-Speech API"""
        if not self.config.google_api_key:
            raise Exception("Google API key not configured")
        
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.config.google_api_key}"
        
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": self.config.tts_voice_name
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": 1.0,
                "pitch": 0.0
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            audio_content = result.get('audioContent', '')
            audio_data = base64.b64decode(audio_content)
            return audio_data, "mp3"
        else:
            raise Exception(f"Google TTS API error: {response.status_code}")
    
    def _pyttsx3_tts(self, text: str) -> Tuple[bytes, str]:
        """Use pyttsx3 for offline TTS"""
        try:
            import pyttsx3
            
            # Initialize TTS engine
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Generate speech
            engine.save_to_file(text, temp_file_path)
            engine.runAndWait()
            
            # Read the generated audio
            with open(temp_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_file_path)
            
            return audio_data, "wav"
        except ImportError:
            raise Exception("pyttsx3 not installed. Run: pip install pyttsx3")
    
    def process_voice_query(self, audio_data: bytes, audio_format: str = "wav") -> Dict[str, Any]:
        """
        Process a voice query: STT -> Chatbot -> TTS
        Returns: {
            'success': bool,
            'transcribed_text': str,
            'chatbot_response': str,
            'audio_response': bytes,
            'audio_format': str,
            'confidence': float,
            'error': str
        }
        """
        try:
            # Step 1: Speech to Text
            transcribed_text, confidence = self.speech_to_text(audio_data, audio_format)
            
            if not transcribed_text:
                return {
                    'success': False,
                    'error': 'Could not transcribe speech. Please try again.',
                    'confidence': confidence
                }
            
            # Step 2: Process with chatbot (placeholder - integrate with your chatbot)
            # This is where you'll integrate with your financial chatbot
            chatbot_response = f"Voice query: {transcribed_text}. [This will be replaced with your chatbot logic]"
            
            # Step 3: Text to Speech
            audio_response, response_format = self.text_to_speech(chatbot_response)
            
            return {
                'success': True,
                'transcribed_text': transcribed_text,
                'chatbot_response': chatbot_response,
                'audio_response': audio_response,
                'audio_format': response_format,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return {
                'success': False,
                'error': f'Voice processing failed: {str(e)}',
                'confidence': 0.0
            }

# Global voice assistant instance
voice_assistant = None

def get_voice_assistant() -> VoiceAssistant:
    """Get or create voice assistant instance"""
    global voice_assistant
    if voice_assistant is None:
        config = VoiceConfig()
        voice_assistant = VoiceAssistant(config)
    return voice_assistant 