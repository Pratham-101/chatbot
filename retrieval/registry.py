RETRIEVER_REGISTRY = {}

def register_retriever(name):
    def decorator(cls):
        RETRIEVER_REGISTRY[name] = cls
        return cls
    return decorator

def get_retriever(name, **kwargs):
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Retriever '{name}' not found.")
    return RETRIEVER_REGISTRY[name](**kwargs)

@register_retriever("hybrid")
class HybridRetriever:
    def __init__(self, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return {"result": "Stub hybrid retriever output"}
    def get_context(self, query, k=5):
        return [f"Stub context {i+1} for '{query}'" for i in range(k)] 