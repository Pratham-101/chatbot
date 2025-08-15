EXTRACTOR_REGISTRY = {}

def register_extractor(name):
    def decorator(cls):
        EXTRACTOR_REGISTRY[name] = cls
        return cls
    return decorator

def get_extractor(name, **kwargs):
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Extractor '{name}' not found.")
    return EXTRACTOR_REGISTRY[name](**kwargs)

@register_extractor("doctr")
class DoctrExtractor:
    def __init__(self, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return {"result": "Stub doctr extractor output"}
    def get_context(self, query, k=5):
        return [f"Stub context {i+1} for '{query}'" for i in range(k)]

@register_extractor("donut")
class DonutExtractor:
    def __init__(self, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return {"result": "Stub donut extractor output"} 