import streamlit as st
import requests
import json

st.set_page_config(page_title="Mutual Fund Chatbot Review", layout="centered")
st.title("🤖 Mutual Fund Chatbot Review UI")

API_URL = "http://localhost:8000/query"

st.markdown("Enter your mutual fund question below:")
question = st.text_input("Your question", "Tell me about HDFC Defence Fund?")

if st.button("Ask"):    
    with st.spinner("Getting answer..."):
        try:
            resp = requests.post(API_URL, json={"text": question}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No answer.")
                structured = data.get("structured_data", {})
                quality = data.get("quality_metrics", {})
                # --- Display sections ---
                st.subheader("Chatbot Answer")
                st.markdown(answer)
                st.subheader("Summary")
                st.write(structured.get("summary", "-"))

                if structured.get("key_points"):
                    st.subheader("Key Points")
                    st.markdown("\n".join([f"- {pt}" for pt in structured["key_points"]]))

                if structured.get("fund_details"):
                    st.subheader("Fund Details")
                    st.table(structured["fund_details"])  # dict to table

                if structured.get("performance_data"):
                    st.subheader("Performance Data")
                    st.table(structured["performance_data"])  # dict to table

                if structured.get("risk_metrics"):
                    st.subheader("Risk Metrics")
                    st.table(structured["risk_metrics"])  # dict to table

                if structured.get("recommendations"):
                    st.subheader("Recommendations")
                    st.markdown("\n".join([f"- {rec}" for rec in structured["recommendations"]]))

                if structured.get("sources"):
                    st.subheader("Sources")
                    st.markdown("\n".join([f"- {src}" for src in structured["sources"]]))

                if structured.get("disclaimer"):
                    st.subheader("Disclaimer")
                    st.info(structured["disclaimer"])

                # Response quality
                if quality:
                    st.subheader("Response Quality")
                    st.write(f"Overall Score: {quality.get('overall_score', '-')}/10")
                    st.write(f"Accuracy: {quality.get('accuracy', '-')}/10")
                    st.write(f"Completeness: {quality.get('completeness', '-')}/10")
                    st.write(f"Clarity: {quality.get('clarity', '-')}/10")
                    st.write(f"Relevance: {quality.get('relevance', '-')}/10")
                    st.write(f"Feedback: {quality.get('feedback', '-')}")

                # Raw JSON for debugging
                with st.expander("Show raw response JSON"):
                    st.json(data)
            else:
                st.error(f"API error: {resp.status_code}\n{resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}") 