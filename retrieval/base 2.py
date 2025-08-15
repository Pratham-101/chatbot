from abc import ABC, abstractmethod

class ContextRetriever(ABC):
    @abstractmethod
    def get_context(self, query: str, k: int = 3) -> list:
        pass 