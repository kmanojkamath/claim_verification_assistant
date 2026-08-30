import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# input_output_handling/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Repository root
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# rag/data/
DATA_DIR = os.path.join(PROJECT_ROOT, "rag", "data")


@app.get("/")
async def root():
    return {
        "message": "Reverse input/output server is running."
    }


@app.get("/output")
async def get_output():

    if not os.path.exists(DATA_DIR):
        raise HTTPException(
            status_code=404,
            detail="rag/data folder not found."
        )

    # Find JSON files inside rag/data
    json_files = [
        file
        for file in os.listdir(DATA_DIR)
        if file.lower().endswith(".json")
    ]

    if not json_files:
        raise HTTPException(
            status_code=404,
            detail="No JSON file found in rag/data."
        )

    # For now, use the first JSON file
    json_file = json_files[0]

    file_path = os.path.join(DATA_DIR, json_file)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "status": "success",
            "filename": json_file,
            "result": data
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="The JSON file is invalid."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )