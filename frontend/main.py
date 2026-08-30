import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Upload")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "avif", "txt"}

@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and images allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(contents)
        
    return {"message": f"File saved successfully to Upload/{file.filename}", "file":file.filename}

@app.post("/upload/text")
async def upload_text(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"text_entry_{timestamp}.txt"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    return {"message": f"Text converted and saved to Upload/{filename}", "file":f"{filename}"}