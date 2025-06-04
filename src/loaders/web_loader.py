from langchain.document_loaders import WebBaseLoader
from .base_loader import IDocumentLoader
from langchain.schema import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextLoader(IDocumentLoader):
    def __init__(self, link: str):
        self.link = link

    def load_documents(self) -> List[Document]:
        loader = WebBaseLoader(self.link)
        documents = loader.load()
    
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunked_docs = splitter.split_documents(documents)
        return chunked_docs