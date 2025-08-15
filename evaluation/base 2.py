from abc import ABC, abstractmethod

class ResponseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, prediction, ground_truth) -> dict:
        pass 