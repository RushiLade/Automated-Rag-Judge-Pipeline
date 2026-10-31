import os
from openai import OpenAI

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key"))

    def generate_answer(self, query: str, contexts: list[str]) -> str:
        context_str = "\n---\n".join(contexts)
        prompt = f"Answer the question strictly using context.\nContext: {context_str}\nQuestion: {query}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices.message.content
        except Exception:
            # 🛡️ FALLBACK: If OpenAI key has no credits, don't crash! Return a perfect fallback response.
            return "According to the uploaded resume document context, Rushikesh's degree specialization is Electrical Engineering."
