from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import base64
import io
from typing import Optional
import logging

from services.voice import process_voice_query, process_text_query, get_voice_assistant

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/voice/process")
async def process_voice_endpoint(
    audio_file: UploadFile = File(...),
    force_web: bool = Form(False),
    user_id: str = Form("default")
):
    """
    Process voice query and return both text and audio response
    """
    try:
        # Validate audio file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Read audio data
        audio_data = await audio_file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Determine audio format from filename
        audio_format = "wav"  # default
        if audio_file.filename:
            if audio_file.filename.lower().endswith('.mp3'):
                audio_format = "mp3"
            elif audio_file.filename.lower().endswith('.wav'):
                audio_format = "wav"
            elif audio_file.filename.lower().endswith('.m4a'):
                audio_format = "m4a"
        
        # Process voice query
        result = process_voice_query(audio_data, audio_format, force_web, user_id)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Voice processing failed'))
        
        # Return comprehensive response
        return {
            "success": True,
            "transcribed_text": result['transcribed_text'],
            "chatbot_response": result['chatbot_response'],
            "confidence": result['confidence'],
            "audio_response_base64": base64.b64encode(result['audio_response']).decode('utf-8') if result['audio_response'] else None,
            "audio_format": result['audio_format'],
            "chatbot_metadata": result.get('chatbot_metadata', {})
        }
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

@router.post("/voice/text-to-speech")
async def text_to_speech_endpoint(
    text: str = Form(...),
    force_web: bool = Form(False),
    user_id: str = Form("default")
):
    """
    Process text query and return audio response
    """
    try:
        # Process text query with audio generation
        result = process_text_query(text, force_web, user_id, generate_audio=True)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Text processing failed'))
        
        # Return response with audio
        return {
            "success": True,
            "chatbot_response": result['chatbot_response'],
            "audio_response_base64": base64.b64encode(result['audio_response']).decode('utf-8') if result['audio_response'] else None,
            "audio_format": result['audio_format'],
            "chatbot_metadata": result.get('chatbot_metadata', {})
        }
        
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@router.get("/voice/audio/{audio_data_base64}")
async def get_audio_response(audio_data_base64: str, format: str = "wav"):
    """
    Serve audio response as downloadable file
    """
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(audio_data_base64)
        
        # Determine MIME type
        mime_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg"
        }
        
        mime_type = mime_types.get(format.lower(), "audio/wav")
        
        return Response(
            content=audio_data,
            media_type=mime_type,
            headers={"Content-Disposition": f"attachment; filename=response.{format}"}
        )
        
    except Exception as e:
        logger.error(f"Audio serving error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio serving failed: {str(e)}")

@router.get("/voice/status")
async def voice_status():
    """
    Check voice system status and available providers
    """
    try:
        voice_assistant = get_voice_assistant()
        
        # Check STT providers
        stt_status = {
            "whisper": voice_assistant.whisper_model is not None,
            "google": bool(voice_assistant.config.google_api_key),
            "openai": bool(voice_assistant.config.openai_api_key)
        }
        
        # Check TTS providers
        tts_status = {
            "pyttsx3": True,  # Always available if installed
            "elevenlabs": bool(voice_assistant.config.elevenlabs_api_key),
            "google": bool(voice_assistant.config.google_api_key)
        }
        
        return {
            "status": "operational",
            "stt_providers": stt_status,
            "tts_providers": tts_status,
            "current_stt": voice_assistant.config.stt_provider,
            "current_tts": voice_assistant.config.tts_provider
        }
        
    except Exception as e:
        logger.error(f"Voice status check error: {e}")
        return {
            "status": "error",
            "error": str(e)
        } 