from chatbot.retrieval import Retriever
from chatbot.generation import ResponseGenerator
from ingestion.vector_store import VectorStore
from evaluation.evaluator import Evaluator

retriever = Retriever(VectorStore())
generator = ResponseGenerator()
evaluator = Evaluator(retriever, generator)

evaluator.evaluate("evaluation/qa_dataset.json")
