from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class IEmbeddingGenerator(ABC):
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        pass