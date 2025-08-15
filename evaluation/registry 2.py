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