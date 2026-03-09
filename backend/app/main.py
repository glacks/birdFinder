from __future__ import annotations

import io
import os
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .model import load_artifacts, predict_image

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
ALLOWED_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app = FastAPI(title="Bird Finder API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = []


@app.on_event("startup")
def startup_event() -> None:
    global model, class_names
    model, class_names = load_artifacts(ARTIFACTS_DIR, device)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(device),
        "num_classes": len(class_names),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    try:
        payload = await file.read()
        image = Image.open(io.BytesIO(payload))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    result = predict_image(
        image=image,
        model=model,
        class_names=class_names,
        device=device,
    )
    return result
