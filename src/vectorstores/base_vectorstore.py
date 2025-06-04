from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class IVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], k: int) -> List[Document]:
        pass