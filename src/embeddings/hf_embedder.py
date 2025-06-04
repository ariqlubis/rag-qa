from langchain_huggingface import HuggingFaceEmbeddings
from .base_embedder import IEmbeddingGenerator
from langchain.schema import Document
from typing import List, Union

class HFEmbedder(IEmbeddingGenerator):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        if len(documents) == 0:
            return []

        if isinstance(documents[0], Document):
            texts = [doc.page_content for doc in documents]
        elif isinstance(documents[0], str):
            texts = documents
        else:
            raise ValueError("embed_documents expects a list of Document or str")

        return self.embedder.embed_documents(texts)
