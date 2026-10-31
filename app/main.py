import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the clean classes we built inside our services folder
from app.services.document_service import DocumentProcessor
from app.services.vector_service import VectorStoreManager
from app.services.llm_service import LLMService
from app.services.evaluator import LLMAsJudgeEvaluator

# Load security environment variables
load_dotenv()

# Initialize our primary web server app
app = FastAPI(title="Automated RAG Pipeline with LLM-as-a-Judge")

# Instantiate our backend logic handlers
doc_processor = DocumentProcessor()
vector_mgr = VectorStoreManager()
llm_service = LLMService()
evaluator = LLMAsJudgeEvaluator()

# The blueprint format for our incoming text queries
class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Save the file temporarily on the disk to let our parser process it
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Extract text chunks and drop them into the FAISS index database shelf
        chunks = doc_processor.extract_chunks_from_pdf(temp_path)
        vector_mgr.add_documents(chunks, doc_id=file.filename)
        return {"status": "Success", "chunks_processed": len(chunks), "document_id": file.filename}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
async def query_pipeline(request: QueryRequest):
    # 1. Fetch matching text segments out of our FAISS engine
    contexts = vector_mgr.query_context(request.question)
    if not contexts:
        raise HTTPException(status_code=404, detail="No matching context found inside the document index.")
        
    # 2. Ask our generation AI model for an answer
    answer = llm_service.generate_answer(request.question, contexts)
    
    # 3. Trigger the evaluator judge AI pipeline to check if the answer is accurate
    scores = evaluator.evaluate_response(request.question, answer, contexts)
    
    # Hand everything back safely to the user
    return {
        "query": request.question,
        "answer": answer,
        "retrieved_contexts": contexts,
        "evaluation_scores": scores
    }
