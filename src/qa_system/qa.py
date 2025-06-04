from typing import List
from langchain.schema import Document
from loaders.base_loader import IDocumentLoader
from embeddings.base_embedder import IEmbeddingGenerator
from vectorstores.base_vectorstore import IVectorStore
from llms.base_llm import ILLM

class QASystem:
    def __init__(
        self,
        loader: IDocumentLoader,
        embedder: IEmbeddingGenerator,
        vectorstore: IVectorStore,
        llm: ILLM
    ):
        self.loader = loader
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm

    def build_index(self):
        documents: List[Document] = self.loader.load_documents()
        self.vectorstore.add_documents(documents)

    def answer_question(self, question: str) -> str:
        query_embedding = self.embedder.embed_documents([Document(page_content=question)])[0]
        docs = self.vectorstore.similarity_search(query_embedding, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self.llm.generate_answer(context, question)
        return answer
