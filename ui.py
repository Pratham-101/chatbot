import streamlit as st
import requests
import os

st.title("Mutual Fund Factsheet Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get backend API URL from environment variable or default to localhost
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:30080/ask")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about mutual funds"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from API
    try:
        print(f"[UI] Sending question to backend: {prompt}")
        response = requests.post(
            BACKEND_API_URL,
            json={"question": prompt}
        )
        print(f"[UI] Received response status: {response.status_code}")
        response_json = response.json()
        print(f"[UI] Response JSON: {response_json}")
        answer = response_json.get("answer", "Sorry, I couldn't process your question.")
    except Exception as e:
        print(f"[UI] Exception during API call: {e}")
        answer = "Sorry, the service is currently unavailable."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
