import asyncio
import base64
import io
import logging
from typing import Dict, Any, Optional
import json

from .voice_assistant import VoiceAssistant
from services.chatbot.enhanced_chatbot import get_chatbot

logger = logging.getLogger(__name__)

class VoiceIntegration:
    """Integrates voice assistant with financial chatbot"""
    
    def __init__(self):
        self.voice_assistant = VoiceAssistant()
        self.chatbot = get_chatbot()
    
    def process_voice_query(self, audio_data: bytes, audio_format: str = "wav", 
                          force_web: bool = False, user_id: str = "default") -> Dict[str, Any]:
        """
        Complete voice processing pipeline:
        STT -> Financial Chatbot -> TTS
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (wav, mp3, etc.)
            force_web: Whether to force web search
            user_id: User identifier
            
        Returns:
            Dict with all processing results
        """
        try:
            # Step 1: Speech to Text
            transcribed_text, confidence = self.voice_assistant.speech_to_text(audio_data, audio_format)
            
            if not transcribed_text:
                return {
                    'success': False,
                    'error': 'Could not transcribe speech. Please try again.',
                    'confidence': confidence,
                    'transcribed_text': '',
                    'chatbot_response': '',
                    'audio_response': b'',
                    'audio_format': 'wav'
                }
            
            logger.info(f"Transcribed: '{transcribed_text}' (confidence: {confidence:.2f})")
            
            # Step 2: Process with financial chatbot
            try:
                chatbot_result = asyncio.run(
                    self.chatbot.process_query(transcribed_text, force_web=force_web, user_id=user_id)
                )
                
                # Extract the main response text
                chatbot_response = chatbot_result.get('full_answer', '')
                if not chatbot_response:
                    chatbot_response = chatbot_result.get('formatted_answer', '')
                if not chatbot_response:
                    chatbot_response = "I couldn't generate a response for your query."
                
            except Exception as e:
                logger.error(f"Chatbot processing error: {e}")
                chatbot_response = f"Sorry, I encountered an error while processing your query: {transcribed_text}"
            
            # Step 3: Text to Speech
            try:
                audio_response, response_format = self.voice_assistant.text_to_speech(chatbot_response)
            except Exception as e:
                logger.error(f"TTS error: {e}")
                audio_response = b''
                response_format = 'wav'
            
            return {
                'success': True,
                'transcribed_text': transcribed_text,
                'chatbot_response': chatbot_response,
                'audio_response': audio_response,
                'audio_format': response_format,
                'confidence': confidence,
                'chatbot_metadata': chatbot_result
            }
            
        except Exception as e:
            logger.error(f"Voice integration error: {e}")
            return {
                'success': False,
                'error': f'Voice processing failed: {str(e)}',
                'confidence': 0.0,
                'transcribed_text': '',
                'chatbot_response': '',
                'audio_response': b'',
                'audio_format': 'wav'
            }
    
    def process_text_query(self, text: str, force_web: bool = False, 
                          user_id: str = "default", generate_audio: bool = True) -> Dict[str, Any]:
        """
        Process text query with optional audio output
        
        Args:
            text: Input text query
            force_web: Whether to force web search
            user_id: User identifier
            generate_audio: Whether to generate audio response
            
        Returns:
            Dict with processing results
        """
        try:
            # Process with financial chatbot
            chatbot_result = asyncio.run(
                self.chatbot.process_query(text, force_web=force_web, user_id=user_id)
            )
            
            # Extract the main response text
            chatbot_response = chatbot_result.get('full_answer', '')
            if not chatbot_response:
                chatbot_response = chatbot_result.get('formatted_answer', '')
            if not chatbot_response:
                chatbot_response = "I couldn't generate a response for your query."
            
            # Generate audio if requested
            audio_response = b''
            response_format = 'wav'
            
            if generate_audio:
                try:
                    audio_response, response_format = self.voice_assistant.text_to_speech(chatbot_response)
                except Exception as e:
                    logger.error(f"TTS error: {e}")
            
            return {
                'success': True,
                'transcribed_text': text,
                'chatbot_response': chatbot_response,
                'audio_response': audio_response,
                'audio_format': response_format,
                'confidence': 1.0,  # Text input has 100% confidence
                'chatbot_metadata': chatbot_result
            }
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {
                'success': False,
                'error': f'Text processing failed: {str(e)}',
                'confidence': 0.0,
                'transcribed_text': text,
                'chatbot_response': '',
                'audio_response': b'',
                'audio_format': 'wav'
            }
    
    def get_audio_base64(self, audio_data: bytes) -> str:
        """Convert audio data to base64 for web playback"""
        return base64.b64encode(audio_data).decode('utf-8')
    
    def get_audio_mime_type(self, audio_format: str) -> str:
        """Get MIME type for audio format"""
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'ogg': 'audio/ogg',
            'm4a': 'audio/mp4'
        }
        return mime_types.get(audio_format.lower(), 'audio/wav')

# Global voice integration instance
voice_integration = None

def get_voice_integration() -> VoiceIntegration:
    """Get or create voice integration instance"""
    global voice_integration
    if voice_integration is None:
        voice_integration = VoiceIntegration()
    return voice_integration

# Convenience functions for easy integration
def process_voice_query(audio_data: bytes, audio_format: str = "wav", 
                       force_web: bool = False, user_id: str = "default") -> Dict[str, Any]:
    """Process voice query with financial chatbot"""
    integration = get_voice_integration()
    return integration.process_voice_query(audio_data, audio_format, force_web, user_id)

def process_text_query(text: str, force_web: bool = False, 
                      user_id: str = "default", generate_audio: bool = True) -> Dict[str, Any]:
    """Process text query with financial chatbot"""
    integration = get_voice_integration()
    return integration.process_text_query(text, force_web, user_id, generate_audio) 