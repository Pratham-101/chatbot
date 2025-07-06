from chatbot.retrieval import Retriever

query = "Who is the fund manager of HDFC Balanced Advantage Fund?"
fund_name = Retriever.extract_fund_name(query)
print(f"Extracted fund name: '{fund_name}'")
