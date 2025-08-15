# 🎤 Voice Assistant for Financial Chatbot

A production-grade voice assistant that integrates seamlessly with your mutual fund chatbot, providing **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** capabilities.

## ✨ Features

### 🎯 **Multiple Input/Output Options**
- **Voice Input**: Record your questions using microphone
- **Text Input**: Type your questions as before
- **Voice Output**: Listen to responses with natural speech
- **Text Output**: Read responses as text

### 🔧 **Multiple Provider Support**

#### **Speech-to-Text (STT)**
- **Whisper** (Free, Offline) - Default, no API key needed
- **Google Speech-to-Text** - High accuracy, requires API key
- **OpenAI Whisper API** - Cloud-based, requires API key

#### **Text-to-Speech (TTS)**
- **pyttsx3** (Free, Offline) - System voices, no API key needed
- **ElevenLabs** - High-quality voices, requires API key
- **Google Text-to-Speech** - Natural voices, requires API key

### 🚀 **Advanced Features**
- **Confidence Scoring**: See how well your speech was understood
- **Real-time Processing**: Instant voice-to-text conversion
- **Audio Playback**: Listen to responses in browser
- **Chat History**: All conversations saved with audio
- **Error Handling**: Graceful fallbacks and user feedback

## 🛠 Installation

### 1. **Install Voice Dependencies**
```bash
pip install -r requirements/voice_requirements.txt
```

### 2. **Install System Dependencies** (macOS)
```bash
# For audio recording
brew install portaudio

# For Whisper (if not using conda)
brew install ffmpeg
```

### 3. **Environment Variables** (Optional)
```bash
# For Google Speech-to-Text
export GOOGLE_API_KEY="your_google_api_key"

# For ElevenLabs TTS
export ELEVENLABS_API_KEY="your_elevenlabs_api_key"

# For OpenAI Whisper API
export OPENAI_API_KEY="your_openai_api_key"
```

## 🎮 Usage

### **Streamlit UI** (Recommended)
```bash
PYTHONPATH=".:src" streamlit run src/ui/streamlit_app.py
```

### **API Endpoints**
```bash
# Start the backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### **Voice Processing**
```bash
curl -X POST "http://localhost:8000/api/voice/process" \
  -F "audio_file=@your_audio.wav" \
  -F "force_web=false" \
  -F "user_id=user123"
```

#### **Text-to-Speech**
```bash
curl -X POST "http://localhost:8000/api/voice/text-to-speech" \
  -F "text=What are the top mutual funds?" \
  -F "force_web=false" \
  -F "user_id=user123"
```

## 🎯 **How It Works**

### **Voice Input Flow**
1. **Record Audio** → User speaks into microphone
2. **STT Processing** → Convert speech to text using Whisper/Google/OpenAI
3. **Financial Chatbot** → Process query with your existing logic
4. **TTS Generation** → Convert response to speech using pyttsx3/ElevenLabs/Google
5. **Audio Playback** → Play response in browser

### **Text Input Flow**
1. **Type Query** → User types question
2. **Financial Chatbot** → Process with existing logic
3. **TTS Generation** → Convert response to speech
4. **Audio Playback** → Play response in browser

## ⚙️ Configuration

### **Voice Settings** (in Streamlit sidebar)
- **STT Provider**: Choose between Whisper, Google, OpenAI
- **TTS Provider**: Choose between pyttsx3, ElevenLabs, Google
- **Input Options**: Enable/disable voice and text input
- **Output Options**: Enable/disable voice and text output

### **Advanced Settings**
```python
from services.voice import VoiceConfig, VoiceAssistant

# Custom configuration
config = VoiceConfig(
    stt_provider="whisper",      # "whisper", "google", "openai"
    tts_provider="pyttsx3",      # "pyttsx3", "elevenlabs", "google"
    whisper_model="base",         # "tiny", "base", "small", "medium", "large"
    tts_voice_id="21m00Tcm4TlvDq8ikWAM",  # ElevenLabs voice ID
    sample_rate=16000,
    channels=1
)

voice_assistant = VoiceAssistant(config)
```

## 🔧 Troubleshooting

### **Common Issues**

#### **1. "Voice functionality not available"**
```bash
# Install missing dependencies
pip install openai-whisper pyttsx3

# Check if Whisper model downloads correctly
python -c "import whisper; whisper.load_model('base')"
```

#### **2. "No module named 'services'"**
```bash
# Run with correct PYTHONPATH
PYTHONPATH=".:src" streamlit run src/ui/streamlit_app.py
```

#### **3. Audio recording not working**
```bash
# Install system audio dependencies
brew install portaudio  # macOS
sudo apt-get install portaudio19-dev  # Ubuntu
```

#### **4. Whisper model download issues**
```bash
# Manual download
wget https://openaipublic.azureedge.net/main/whisper/models/base.pt
mkdir -p ~/.cache/whisper/
mv base.pt ~/.cache/whisper/
```

### **Performance Optimization**

#### **For Production**
```python
# Use smaller Whisper model for faster processing
config = VoiceConfig(whisper_model="tiny")

# Use offline TTS for reliability
config = VoiceConfig(tts_provider="pyttsx3")
```

#### **For Development**
```python
# Use larger Whisper model for better accuracy
config = VoiceConfig(whisper_model="medium")

# Use cloud TTS for better quality
config = VoiceConfig(tts_provider="elevenlabs")
```

## 🎵 **Audio Quality Tips**

### **Best Practices**
1. **Clear Speech**: Speak clearly and at normal pace
2. **Quiet Environment**: Minimize background noise
3. **Good Microphone**: Use quality microphone for better STT
4. **Internet Connection**: Stable connection for cloud providers

### **Provider Comparison**

| Provider | Cost | Quality | Speed | Offline |
|----------|------|---------|-------|---------|
| **Whisper** | Free | High | Medium | ✅ |
| **Google STT** | Paid | Very High | Fast | ❌ |
| **pyttsx3** | Free | Medium | Fast | ✅ |
| **ElevenLabs** | Paid | Very High | Medium | ❌ |

## 🔄 **Integration with Existing Chatbot**

The voice assistant is designed to work **seamlessly** with your existing financial chatbot:

```python
# Your existing chatbot logic remains unchanged
from services.chatbot.enhanced_chatbot import get_chatbot

chatbot = get_chatbot()
result = await chatbot.process_query("What are the top mutual funds?")
```

The voice layer simply:
1. **Converts speech to text** (STT)
2. **Passes text to your chatbot** (unchanged)
3. **Converts response to speech** (TTS)

## 🚀 **Next Steps**

### **Immediate**
1. Install dependencies: `pip install -r requirements/voice_requirements.txt`
2. Test voice system: Run Streamlit UI and try voice input
3. Configure providers: Set up API keys for cloud services

### **Advanced**
1. **Custom Voices**: Train custom ElevenLabs voices
2. **Language Support**: Add Hindi/other Indian languages
3. **Real-time Streaming**: Implement streaming audio responses
4. **Voice Commands**: Add voice command shortcuts

### **Production**
1. **Error Monitoring**: Add comprehensive error tracking
2. **Performance Metrics**: Monitor STT/TTS performance
3. **User Analytics**: Track voice usage patterns
4. **A/B Testing**: Compare different voice providers

## 📞 **Support**

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Test with different providers
4. Check system audio permissions

---

**🎤 Your financial chatbot now has a voice!** 

Users can now ask questions naturally by speaking, and receive spoken responses with all the same financial intelligence and real-time data capabilities. 