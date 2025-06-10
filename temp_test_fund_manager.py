from chatbot.retrieval import Retriever

# Dummy vector store with sample data simulating factsheet chunks
class DummyVectorStore:
    def query(self, embedding, k=15):
        return [
            {"text": "ICICI Prudential Equity Fund is managed by Mr. R. Srinivasan.", "distance": 0.1},
            {"text": "HDFC Balanced Advantage Fund is managed by Ms. Swati Kulkarni.", "distance": 0.2},
            {"text": "SBI Bluechip Fund has Mr. Rajeev Radhakrishnan as fund manager.", "distance": 0.3},
            {"text": "Kotak Emerging Equity Fund is managed by Mr. Harsha Upadhyaya.", "distance": 0.4}
        ]

retriever = Retriever(vector_store=DummyVectorStore())

query = "Who is the fund manager of HDFC Balanced Advantage Fund?"
result = retriever.get_fund_manager(query)
print(result)
