from langchain_community.document_loaders import PyMuPDFLoader
from .base_loader import IDocumentLoader
from langchain.schema import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFLoader(IDocumentLoader):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.chunk_size = 800
        self.chunk_overlap = 200

    def load_documents(self) -> List[Document]:
        loader = PyMuPDFLoader(self.filepath)
        documents = loader.load()
    
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunked_docs = splitter.split_documents(documents)
        return chunked_docs