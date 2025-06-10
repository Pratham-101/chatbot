import json
from fuzzywuzzy import fuzz
from chatbot.retrieval import Retriever
from chatbot.generation import ResponseGenerator

class Evaluator:
    def __init__(self, retriever: Retriever, generator: ResponseGenerator, threshold=80):
        self.retriever = retriever
        self.generator = generator
        self.threshold = threshold  # Minimum fuzzy score to count as correct

    def evaluate(self, qa_path: str):
        with open(qa_path) as f:
            qa_pairs = json.load(f)

        total = len(qa_pairs)
        correct = 0

        print(f"Starting evaluation on {total} samples...\n")

        for idx, sample in enumerate(qa_pairs, 1):
            q, expected = sample["question"], sample["expected_answer"]
            context = self.retriever.get_relevant_context(q)
            answer = self.generator.generate_response(q, context)

            score = fuzz.partial_ratio(expected.lower(), answer.lower())

            if score >= self.threshold:
                correct += 1
                print(f"✔ Q{idx} matched (score={score})")
            else:
                print(f"❌ Q{idx} MISMATCH (score={score})\nQ: {q}\nExpected: {expected}\nGot: {answer}\n")

        accuracy = correct / total
        print(f"\n✅ Accuracy: {correct}/{total} = {accuracy:.2%}")
