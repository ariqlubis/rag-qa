from loaders.pdf_loader import PDFLoader
from embeddings.hf_embedder import HFEmbedder
from vectorstores.chroma_vectorstore import ChromaVectorStore
from llms.ollama import OllamaLLM
from qa_system.qa import QASystem



def main():
    loader = PDFLoader(filepath="src/attention_is_all_you_need.pdf")
    embedder = HFEmbedder()
    
    use_chroma = True
    if use_chroma:
        vectorstore = ChromaVectorStore(embedder, persist_directory=None)

    llm = OllamaLLM()

    qa_system = QASystem(loader, embedder, vectorstore, llm)

    print("Building vectorstore index...")
    qa_system.build_index()

    print("Ask questions (type 'exit' to quit):")
    while True:
        query = input("Question: ").strip()
        if query.lower() == "exit":
            break
        answer = qa_system.answer_question(query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
