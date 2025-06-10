import streamlit as st
import requests

st.title("Mutual Fund Factsheet Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": prompt}
        ).json()
        
        answer = response.get("answer", "Sorry, I couldn't process your question.")
    except:
        answer = "Sorry, the service is currently unavailable."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})