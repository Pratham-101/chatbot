EVALUATOR_REGISTRY = {}

def register_evaluator(name):
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator

def get_evaluator(name, **kwargs):
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Evaluator '{name}' not found.")
    return EVALUATOR_REGISTRY[name](**kwargs)

from src.services.chatbot.response_quality import ResponseQuality

@register_evaluator("fuzzy")
class FuzzyEvaluator:
    def __init__(self, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return {"result": "Stub fuzzy evaluator output"}
    def get_context(self, query, k=5):
        return [f"Stub context {i+1} for '{query}'" for i in range(k)]
    def evaluate(self, prediction, ground_truth):
        return ResponseQuality(
            accuracy=10,
            completeness=10,
            clarity=10,
            relevance=10,
            overall_score=10,
            feedback="Stub evaluation"
        ) 