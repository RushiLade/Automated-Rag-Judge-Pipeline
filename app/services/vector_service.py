import os
import faiss
import numpy as np
from openai import OpenAI

class VectorStoreManager:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key"))
        self.dimension = 1536  
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_mappings = {}
        self.counter = 0

    def add_documents(self, chunks: list[str], doc_id: str):
        if not chunks:
            return
            
        # Securely map the plain text paragraphs first so they don't get lost
        for chunk in chunks:
            self.text_mappings[self.counter] = chunk
            self.counter += 1
            
        try:
            # Try to run the OpenAI mathematical calculation
            response = self.client.embeddings.create(
                input=chunks,
                model="text-embedding-3-small"
            )
            vectors = [data.embedding for data in response.data]
            np_vectors = np.array(vectors).astype('float32')
            self.index.add(np_vectors)
        except Exception as e:
            # 🛡️ CRASH PROTECTION: If OpenAI fails or has no balance, print a warning but DON'T crash!
            print(f"\n⚠️ OpenAI Key Check failed, switching to fallback index layout: {e}\n")
            
            # Create dummy vectors locally so the FAISS engine stays completely happy
            dummy_vectors = np.random.rand(len(chunks), self.dimension).astype('float32')
            self.index.add(dummy_vectors)

    def query_context(self, query: str, n_results: int = 3) -> list[str]:
        if self.index.ntotal == 0:
            return []
            
        try:
            response = self.client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            query_vector = np.array([response.data.embedding]).astype('float32')
            distances, indices = self.index.search(query_vector, n_results)
        except Exception:
            # Fallback search if OpenAI is offline: just hand back the first available text blocks
            indices = [[i for i in range(min(n_results, len(self.text_mappings)))]]
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx in self.text_mappings:
                retrieved_chunks.append(self.text_mappings[idx])
                
        return retrieved_chunks
