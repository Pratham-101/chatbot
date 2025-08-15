import streamlit as st
import requests
import json
import queue
import threading
import numpy as np

# Defer optional imports so the app still works without voice deps
try:
    import streamlit_webrtc  # type: ignore
    _has_webrtc = True
except Exception:
    _has_webrtc = False

try:
    import speech_recognition as sr  # type: ignore
    _has_sr = True
except Exception:
    _has_sr = False

st.set_page_config(page_title="Mutual Fund Chatbot Review", layout="centered")
st.title("🤖 Mutual Fund Chatbot Review UI")

API_URL = "http://localhost:8000/api/query"

st.markdown("Enter your mutual fund question below:")

input_options = ["Text"] + (["Voice"] if (_has_webrtc and _has_sr) else [])
input_method = st.radio("Choose input method:", input_options)

if input_method == "Text":
    user_query = st.text_input("Your question:")
    if st.button("Ask") and user_query:
        with st.spinner("Processing..."):
            try:
                response = requests.post(API_URL, json={"text": user_query})
                if response.status_code == 200:
                    st.success(response.json().get("answer", "No answer returned."))
                else:
                    st.error(f"Request failed: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

if input_method == "Voice":
    # Import only when needed to avoid top-level failures
    try:
        from streamlit_webrtc import webrtc_streamer, AudioProcessorBase  # type: ignore
        import speech_recognition as sr  # type: ignore
    except Exception as e:
        st.warning(
            "Voice mode is unavailable (missing dependencies). "
            "Install 'streamlit-webrtc' and 'SpeechRecognition' to enable it.\n\n"
            f"Details: {e}"
        )
        st.stop()
    st.subheader("🎤 Voice to Text (Speak your question)")
    result_text = st.empty()
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.q = queue.Queue()
            self.recognizer = sr.Recognizer()
            self.audio_data = b""
        def recv(self, frame):
            audio = frame.to_ndarray()
            self.audio_data += audio.tobytes()
            return frame
        def get_text(self):
            try:
                audio = sr.AudioData(self.audio_data, 16000, 2)
                text = self.recognizer.recognize_google(audio)
                return text
            except Exception as e:
                return f"[Error] {e}"
    ctx = webrtc_streamer(key="voice-to-text", audio_receiver_size=1024, audio_processor_factory=AudioProcessor, media_stream_constraints={"audio": True, "video": False})
    if ctx and ctx.state.playing:
        st.info("Recording... Speak now!")
        if st.button("Stop and Transcribe"):
            text = ctx.audio_processor.get_text()
            result_text.text_area("Recognized Text", value=text, height=100)
            if st.button("Ask with this text"):
                with st.spinner("Processing..."):
                    try:
                        response = requests.post(API_URL, json={"text": text})
                        if response.status_code == 200:
                            st.success(response.json().get("answer", "No answer returned."))
                        else:
                            st.error(f"Request failed: {response.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}") 