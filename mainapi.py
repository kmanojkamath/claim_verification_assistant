import os
import json

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import main


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_FILE = os.path.join(BASE_DIR, "imginput.png")
PDF_FILE = os.path.join(BASE_DIR, "pdfinput.pdf")
TEXT_FILE = os.path.join(BASE_DIR, "textinput.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "output.json")

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}


def build_response(message: str, flag: str, result: dict) -> dict:
    """
    Assemble the JSON payload sent to the frontend after a claim has
    been processed. Every endpoint below sends the result to the
    frontend through this single function, so the shape the frontend
    can rely on ({message, flag, result}) stays consistent no matter
    which endpoint produced it.
    """

    return {
        "message": message,
        "flag": flag,
        "result": result
    }


@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    ext = (
        file.filename.rsplit(".", 1)[-1].lower()
        if "." in file.filename
        else ""
    )

    contents = await file.read()

    if ext in ALLOWED_IMAGE_EXTENSIONS:

        with open(IMAGE_FILE, "wb") as f:
            f.write(contents)

        flag = "image"

    elif ext in ALLOWED_PDF_EXTENSIONS:

        with open(PDF_FILE, "wb") as f:
            f.write(contents)

        flag = "pdf"

    else:

        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF, PNG, JPG and JPEG files are allowed."
        )

    try:
        result = main.process_claim(flag)

        return build_response(
            "File processed successfully.",
            flag,
            result
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/upload/text")
async def upload_text(text: str = Form(...)):

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text content cannot be empty."
        )

    with open(
        TEXT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(text)

    flag = "text"

    try:
        result = main.process_claim(flag)

        return build_response(
            "Text processed successfully.",
            flag,
            result
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/result")
async def get_last_result():
    """
    Re-send the most recently generated output.json to the frontend
    without re-running OCR/extraction. Useful if the frontend wants
    to redisplay the last result after a page refresh, or if a
    client missed the response from the original upload call.
    """

    if not os.path.exists(OUTPUT_FILE):
        raise HTTPException(
            status_code=404,
            detail="No result has been generated yet."
        )

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        result = json.load(f)

    return build_response(
        "Last result retrieved successfully.",
        "cached",
        result
    )


@app.get("/")
async def root():

    return {
        "message": "SIH input server is running."
    }