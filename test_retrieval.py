import unittest
from chatbot.retrieval import Retriever
from ingestion.vector_store import VectorStore
import unittest

class TestRetriever(unittest.TestCase):
    def setUp(self):
        vector_store = VectorStore()
        self.retriever = Retriever(vector_store)

    def test_get_fund_manager(self):
        # Test known queries and expected fund manager names
        queries = {
            "Who is the fund manager of HDFC Balanced Advantage Fund?": "Swati Kulkarni",
            "Tell me about ICICI Prudential Equity Fund": "R. Srinivasan",
            "Who manages SBI Bluechip Fund?": "Rajeev Radhakrishnan",
            "Who is the manager of Kotak Emerging Equity Fund?": "Harsha Upadhyaya"
        }
        for query, expected_manager in queries.items():
            manager = self.retriever.get_fund_manager(query)
            self.assertIn(expected_manager, manager)

    def test_empty_query(self):
        result = self.retriever.get_fund_manager("")
        self.assertEqual(result, "Could not extract fund name from the query.")

    def test_unknown_fund(self):
        result = self.retriever.get_fund_manager("Who manages Unknown Fund?")
        self.assertTrue("not found" in result.lower() or "could not extract" in result.lower())

if __name__ == "__main__":
    unittest.main()
