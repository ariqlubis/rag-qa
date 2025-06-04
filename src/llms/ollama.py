# src/llms/ollama_llm.py
from langchain_community.llms import Ollama
from .base_llm import ILLM

class OllamaLLM(ILLM):
    def __init__(self, model: str = "gemma3:4b"):
        self.llm = Ollama(model=model)

    def generate_answer(self, context: str, question: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.llm(prompt)
    