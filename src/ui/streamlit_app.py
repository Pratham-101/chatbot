import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import requests
import json
from chatbot.enhanced_chatbot import answer_query
import base64
import re

st.set_page_config(page_title="Mutual Fund Chatbot Review", layout="centered")
st.title("🤖 Mutual Fund Chatbot Review UI")

API_URL = "http://localhost:8000/query"

st.markdown("Enter your mutual fund question below:")
query = st.text_input("Your question", "Tell me about HDFC Defence Fund?")

if query:
    with st.spinner("Fetching answer..."):
        answer = answer_query(query)
    # Split out base64 images and display them
    text = re.sub(r'!\[.*?\]\(data:image/png;base64,[^\)]+\)', '', answer)
    st.markdown(text)
    img_matches = re.findall(r'!\[.*?\]\(data:image/png;base64,([^\)]+)\)', answer)
    for img_b64 in img_matches:
        st.image(base64.b64decode(img_b64), use_column_width=True)
    # Optionally, allow download of last chart
    if img_matches:
        st.download_button("Download Chart", base64.b64decode(img_matches[-1]), file_name="chart.png", mime="image/png")

if st.button("Ask"):    
    with st.spinner("Getting answer..."):
        try:
            resp = requests.post(API_URL, json={"text": query}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No answer.")
                structured = data.get("structured_data", {})
                quality = data.get("quality_metrics", {})
                # --- Display sections ---
                st.subheader("Chatbot Answer")
                st.markdown(answer)

                # Show summary only if it is non-empty and not a placeholder
                summary = structured.get("summary", "").strip()
                if summary and summary != "-" and len(summary) > 10:
                    st.subheader("Summary")
                    st.write(summary)

                # Show sources if present and non-empty
                sources = structured.get("sources")
                if sources and isinstance(sources, list) and any(sources):
                    st.subheader("Sources")
                    st.markdown("\n".join([f"- {src}" for src in sources if src and src != "-"]))

                # Show disclaimer if present and non-empty
                disclaimer = structured.get("disclaimer", "").strip()
                if disclaimer and disclaimer != "-":
                    st.subheader("Disclaimer")
                    st.info(disclaimer)

                # --- Hide all other sections unless they have real, non-placeholder data ---
                # (No key points, fund details, performance data, risk metrics, recommendations, response quality, or raw JSON)

            else:
                st.error(f"API error: {resp.status_code}\n{resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}") 