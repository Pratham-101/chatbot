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