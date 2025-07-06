import yaml
from extraction.registry import get_extractor
from retrieval.registry import get_retriever
from evaluation.registry import get_evaluator

class MutualFundPipeline:
    def __init__(self, config_path="config/pipeline.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Load extractors
        self.extractors = [get_extractor(name) for name in config.get('extractors', [])]
        # Load retriever
        self.retriever = get_retriever(config.get('retriever', 'hybrid'))
        # Load evaluator
        self.evaluator = get_evaluator(config.get('evaluator', 'fuzzy'))

    def extract(self, pdf_path):
        results = {}
        for extractor in self.extractors:
            results.update(extractor.extract(pdf_path))
        return results

    def retrieve(self, query, k=3):
        return self.retriever.get_context(query, k=k)

    def evaluate(self, prediction, ground_truth):
        return self.evaluator.evaluate(prediction, ground_truth)

# Example usage
if __name__ == "__main__":
    pipeline = MutualFundPipeline()
    # Extraction example
    pdf_path = "data/HDFC MF Factsheet - April 2025.pdf"
    extracted_data = pipeline.extract(pdf_path)
    print("Extracted Data:", extracted_data)
    # Retrieval example
    context = pipeline.retrieve("Who is the fund manager of HDFC Defence Fund?")
    print("Retrieved Context:", context)
    # Evaluation example
    eval_result = pipeline.evaluate("Ajay Kumar", "Ajay Kumar")
    print("Evaluation Result:", eval_result) 