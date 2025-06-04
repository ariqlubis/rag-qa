from abc import ABC, abstractmethod
from typing import List

class IDocumentLoader(ABC):
    @abstractmethod
    def load_documents(self) -> List[str]:
        pass