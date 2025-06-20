import requests
import time

import requests
import time

def ask_question():
    url = "http://localhost:8000/ask"
    while True:
        question = input("Enter your question (or type 'exit' to quit): ").strip()
        if not question:
            print("Please enter a non-empty question.")
            continue
        if question.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        print("Processing your question, please wait...")
        start_time = time.time()
        try:
            response = requests.post(url, json={"question": question})
            elapsed = time.time() - start_time
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer received.")
                # Truncate long factsheet info to first 500 chars for brevity
                if "Additionally, based on the factsheet data:" in answer:
                    parts = answer.split("Additionally, based on the factsheet data:")
                    main_answer = parts[0].strip()
                    factsheet_info = parts[1].strip()
                    factsheet_info_short = factsheet_info[:500] + ("..." if len(factsheet_info) > 500 else "")
                    answer = f"{main_answer}\n\nAdditionally, based on the factsheet data:\n{factsheet_info_short}"
                print(f"\nAnswer (in {elapsed:.2f} seconds):\n{answer}\n")
            else:
                print(f"Error: Received status code {response.status_code}\n")
        except Exception as e:
            print(f"Error during request: {e}\n")

if __name__ == "__main__":
    ask_question()
