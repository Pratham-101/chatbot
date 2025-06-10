import unittest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def test_ask_endpoint_valid_question(self):
        response = client.post("/ask", json={"question": "Tell me about Mirae Asset Ultra Short Duration Fund"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIsInstance(data["answer"], str)
        self.assertNotEqual(data["answer"].strip(), "")

    def test_ask_endpoint_empty_question(self):
        response = client.post("/ask", json={"question": ""})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Please provide a question.")

    def test_ask_endpoint_missing_question(self):
        response = client.post("/ask", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Please provide a question.")

    def test_ask_endpoint_invalid_method(self):
        response = client.get("/ask")
        self.assertEqual(response.status_code, 405)  # Method Not Allowed

if __name__ == "__main__":
    unittest.main()
