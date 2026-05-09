import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.rag import VitaTwinRAG  # Import the new class
from fastapi.responses import FileResponse


# IMPORTING YOUR CUSTOM MODULES
try:
    # from insight_layer import VitaTwinInsightLayer
    from predict import VitaTwinAnalyzer    
except ImportError:
    # from src.insight_layer import VitaTwinInsightLayer
    from src.predict import VitaTwinAnalyzer  


# 1. Initialize FastAPI App
app = FastAPI(title="VitaTwin Intelligence API")

# 2. Enable CORS for your index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Handle Absolute Paths to avoid "NoneType" errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If your app.py is in 'src', but models is in the root, use: 
# MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "bert_mental_health")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bert_mental_health")

print(f"--- Loading VitaTwin Engines from: {MODEL_PATH} ---")

# Global variables for engines
analyzer_engine = None
rag_engine = None


@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.on_event("startup")
async def load_engines():
    global rag_engine , analyzer_engine
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")
        analyzer_engine = VitaTwinAnalyzer(model_path=MODEL_PATH)
        
        rag_engine = VitaTwinRAG() 
        # insight_engine = VitaTwinInsightLayer()
        print("--- Engines Loaded Successfully ---")
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODELS: {e}")

class AnalyzeRequest(BaseModel):
    text: str
    
class AskRequest(BaseModel):
    text: str # The patient note
    question: str # The user question
    
@app.post("/ask")
async def ask_rag(request: AskRequest):
    if rag_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG Engine is offline. Did you set the OPENAI_API_KEY?"
        )
    try:
        # RAG process: search the note for context to answer the question
        result = rag_engine.query(request.question, request.text)
        print(f"RAG RESULT : {result}") # Check the terminal

        return result
    except Exception as e:
        print(f"RAG Error: {e}") # This prints the REAL error to your terminal
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):
    try:
        # STEP 1: Get the raw Risk Factor and Insight from prediction.py
        raw_analysis = analyzer_engine.get_risk_factor(request.text)
        print(f"DEBUG raw_analysis: {raw_analysis}") # Check the terminal
        

        return {
            "prediction": {
                "label": raw_analysis['label'],
                "risk_factor": raw_analysis['risk_factor'],
                "confidence": raw_analysis['confidence']
            },
            "explanation": raw_analysis['explanation'],
            "insight": raw_analysis['insight'] # This is now the natural sentence
        }
    except Exception as e:
        print(f"ERROR LOCATED: {str(e)}") # This shows exactly what happened in the terminal
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)