from abc import ABC, abstractmethod
from typing import List

class ILLM(ABC):
    @abstractmethod
    def generate_answer(self, context: str, question: str) -> str:
        pass