# 🚀 Automated RAG Pipeline with "LLM-as-a-Judge" Evaluation

A production-grade, document-based question-answering system (RAG) built using **FastAPI** and **FAISS** index data structures. The pipeline features a secondary automated verification engine that utilizes advanced evaluation schemas to score its own answers on **Faithfulness** and **Answer Relevance**.

## 🏗️ System Architecture
```text
[Uploaded PDF Document] 
       │
       ▼
[Document Processor] ──(Recursive Chunking)──► [Text Paragraph Chunks]
                                                        │
                                                        ▼
[FAISS Vector Space Engine] ◄──(OpenAI Embeddings)──────┘
       │
       ├──► 1. Pull Context Matches
       ▼
[Core Generation Router] ──(LLM Context Injection)──► [Generated Answer]
                                                              │
                                                              ▼
[LLM-as-a-Judge Evaluator] ◄──(Verification Scopes)───────────┘
       │
       ▼
[JSON Metric Payloads] -> (Faithfulness Score & Answer Relevance Score)
```

## 🛠️ Step-by-Step Local Setup

1. Clone or download this project repository to your local directory.
2. Initialize and configure your active security environmental key inside a root `.env` file layout:
   ```text
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```
3. Run the automated dependency build installer terminal sequence:
   ```bash
   py -3.12 -m pip install -r requirements.txt
   ```
4. Fire up the local Uvicorn application engine:
   ```bash
   py -3.12 -m uvicorn app.main:app --reload
   ```
5. Pull up the interactive documentation endpoint layer inside your preferred web browser window to execute target data payloads: `http://127.0.0`

## 📊 Sample Payloads & Production Payloads

### 1. Document Upload Ingestion Endpoint (`POST /upload`)
*   **Payload Format:** `multipart/form-data`
*   **Target Parameter:** `file=@policy_or_resume.pdf`
*   **Successful API Response (200 OK):**
    ```json
    {
      "status": "Success",
      "chunks_processed": 4,
      "document_id": "Rushikesh_Lade_AI_ML_Intern_Resume.pdf"
    }
    ```

### 2. Context Ingestion Query Endpoint (`POST /query`)
*   **Incoming Data Payload:**
    ```json
    {
      "question": "What core technical frameworks does Rushikesh specialize in?"
    }
    ```
*   **Successful API Response (200 OK):**
    ```json
    {
      "query": "What core technical frameworks does Rushikesh specialize in?",
      "answer": "According to the uploaded resume context, Rushikesh Lade specializes in core backend architectures using Python, Java, SQL, and FastAPI.",
      "retrieved_contexts": [
        "Rushikesh Lade has deep hands-on expertise building production workflows leveraging Python, FastAPI, and data management frameworks..."
      ],
      "evaluation_scores": {
        "faithfulness": 1.0,
        "answer_relevance": 0.98
      }
    }
    ```
