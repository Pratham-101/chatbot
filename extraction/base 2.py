from abc import ABC, abstractmethod

class FactsheetExtractor(ABC):
    @abstractmethod
    def extract(self, pdf_path: str) -> dict:
        pass 