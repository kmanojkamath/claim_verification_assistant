import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Upload")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "avif", "txt"}

class VerificationReport(BaseModel):
    message: str
    file: str
    verdict: str  # "True", "False", or "Unverifiable"
    citations: List[str]
    confidence: List[float]

class TextSubmission(BaseModel):
    text: str

def run_verification_engine(content_summary: str) -> dict:
    """
    Placeholder for your verification logic/LLM pipeline.
    Replace this with your actual verification/search model logic.
    """
    return {
        "verdict": "False",
        "citations": [
            "https://egazette.gov.in/WriteReadData/2026/notification-114532.pdf",
            "https://pib.gov.in/PressReleasePage.aspx?PRID=2098451",
            "https://data.gov.in/resource/state-wage-notifications-2026"
        ],
        "confidence": [9.1, 8.4, 7.8]
    }

@app.post("/upload/file", response_model=VerificationReport)
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(contents)
        
    report_data = run_verification_engine(file.filename)
    
    return {
        "message": f"File saved and verified.",
        "file": file.filename,
        **report_data
    }

@app.post("/upload/text", response_model=VerificationReport)
async def upload_text(payload: TextSubmission):
    text = payload.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"text_entry_{timestamp}.txt"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    report_data = run_verification_engine(text)
    
    return {
        "message": f"Text claim processed.",
        "file": filename,
        **report_data
    }