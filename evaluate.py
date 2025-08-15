from src.ingestion.vector_store import VectorStore
from src.chatbot.retrieval import Retriever
from src.chatbot.generation import ResponseGenerator
from evaluation.evaluator import Evaluator

retriever = Retriever(VectorStore())
generator = ResponseGenerator()
evaluator = Evaluator(retriever, generator)

evaluator.evaluate("evaluation/qa_dataset.json")
