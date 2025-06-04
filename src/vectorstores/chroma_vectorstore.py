from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List
from .base_vectorstore import IVectorStore

class ChromaVectorStore(IVectorStore):
    def __init__(self, embeddings, persist_directory=None):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None

    def add_documents(self, documents: List[Document]) -> None:
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        if self.persist_directory:
            self.vectorstore.persist()

    def similarity_search(self, query_embedding: List[float], k: int) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        # similarity_search expects a query string, so pass text or embed manually
        # But LangChain’s Chroma supports similarity_search with query text, not embedding directly
        # So better to use similarity_search_by_vector if you have embedding vector
        if hasattr(self.vectorstore, "similarity_search_by_vector"):
            return self.vectorstore.similarity_search_by_vector(query_embedding, k)
        else:
            raise NotImplementedError("similarity_search_by_vector not implemented")
