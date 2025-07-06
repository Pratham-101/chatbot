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