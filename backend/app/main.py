
from io import BytesIO
from time import perf_counter
from threading import Lock

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

from .explainability import (
    analyze_visual_features,
    enhance_image,
    make_heatmap_overlay,
    png_base64,
)

MODEL_ID = "prithivMLmods/Coral-Health"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

app = FastAPI(
    title="AquaMind API",
    description="Coral image classification and explainable visual analysis.",
    version="0.2.0",
)

import os

_frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

_allowed_origins = list(dict.fromkeys([
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    *_frontend_origins,
]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

classifier = None
inference_lock = Lock()


@app.on_event("startup")
def load_model():
    global classifier
    classifier = pipeline(
        "image-classification",
        model=MODEL_ID,
        device=-1,
    )


@app.get("/")
def root():
    return {
        "application": "AquaMind",
        "status": "running",
        "model": MODEL_ID,
        "features": ["classification", "explainability", "visual-analysis"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if classifier is not None else "loading",
        "model_loaded": classifier is not None,
        "device": "CPU",
    }


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, WEBP, or BMP image.",
        )

    raw = await file.read(MAX_IMAGE_BYTES + 1)

    if not raw:
        raise HTTPException(status_code=400, detail="Empty image.")

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    try:
        with Image.open(BytesIO(raw)) as check:
            check.verify()

        with Image.open(BytesIO(raw)) as source:
            return source.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )


def classify(image: Image.Image) -> list[dict]:
    results = classifier(image, top_k=None)
    return sorted(results, key=lambda item: item["score"], reverse=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="The AI model is not ready.")

    image = await read_uploaded_image(file)
    start = perf_counter()

    try:
        with inference_lock:
            results = classify(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image inference failed: {type(exc).__name__}",
        ) from exc

    elapsed = perf_counter() - start

    return {
        "filename": file.filename,
        "model": MODEL_ID,
        "predicted_class": results[0]["label"],
        "confidence": round(float(results[0]["score"]), 6),
        "probabilities": {
            item["label"]: round(float(item["score"]), 6)
            for item in results
        },
        "inference_time_seconds": round(elapsed, 3),
        "device": "CPU",
        "warning": (
            "Predictions are not proof of coral presence or health. "
            "Validate coral content and model performance before ecological use."
        ),
    }


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    target_class: int | None = Form(default=None),
):
    if classifier is None:
        raise HTTPException(status_code=503, detail="The AI model is not ready.")

    image = await read_uploaded_image(file)
    enhanced = enhance_image(image)
    features = analyze_visual_features(image)

    start = perf_counter()

    try:
        with inference_lock:
            results = classify(image)

            # Default to the model's predicted class. An optional target_class
            # lets the caller request class 0 or class 1 for comparison.
            if target_class is None:
                label_to_id = {
                    label: index
                    for index, label in classifier.model.config.id2label.items()
                }
                selected_id = label_to_id[results[0]["label"]]
            else:
                selected_id = target_class

            explanation = make_heatmap_overlay(
                image=image,
                model=classifier.model,
                image_processor=classifier.image_processor,
                target_class=selected_id,
            )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Explainability failed: {type(exc).__name__}: {exc}",
        ) from exc

    elapsed = perf_counter() - start

    return {
        "filename": file.filename,
        "model": MODEL_ID,
        "predicted_class": results[0]["label"],
        "confidence": round(float(results[0]["score"]), 6),
        "probabilities": {
            item["label"]: round(float(item["score"]), 6)
            for item in results
        },
        "explanation": explanation,
        "visual_features": features,
        "images": {
            "original_png_base64": png_base64(image),
            "enhanced_png_base64": png_base64(enhanced),
        },
        "analysis_time_seconds": round(elapsed, 3),
        "device": "CPU",
        "warning": (
            "The heatmap is a model-attribution visualization, not a coral "
            "segmentation mask or proof of bleaching. Image color and texture "
            "measurements are descriptive only. Coral presence and health "
            "require independent validation."
        ),
    }


