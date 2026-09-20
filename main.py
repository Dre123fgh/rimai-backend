"""
RimAi Inference Server
Backend & WhatsApp Integration Lead — main.py

Day 3-4: skeleton with dummy /predict (returns fake disease + confidence).
Day 5: swap load_model() and run_inference() to use the real .pth checkpoint
        once the AI Lead hands it off. Everything else (auth, error handling,
        response shape) stays the same, so n8n never needs to change.
"""

import io
import os
import random
import logging

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rimai")

app = FastAPI(title="RimAi Inference Server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("RIMAI_API_KEY", "dev-key-change-me")
CONFIDENCE_THRESHOLD = float(os.environ.get("RIMAI_CONFIDENCE_THRESHOLD", "0.6"))

# Placeholder class list — replace with the AI Lead's exact class list on hand-off.
CLASS_NAMES = [
    "tomato_healthy",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_septoria_leaf_spot",
    "tomato_bacterial_spot",
]

MODEL = None  # populated by load_model() once a real checkpoint exists


# ---------------------------------------------------------------------------
# Response schema — n8n's HTTP Request node parses this, so keep it stable.
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    status: str            # "ok" | "low_confidence" | "error"
    disease: str | None
    confidence: float | None
    message: str


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------
def load_model():
    """
    Day 5: load the real checkpoint here, e.g.

        import torch
        model = torch.load("checkpoint.pth", map_location="cpu")
        model.eval()
        return model

    Until the checkpoint arrives, this stays a no-op.
    """
    global MODEL
    if MODEL is None:
        logger.info("No real checkpoint loaded yet — running in dummy mode.")
    return MODEL


def run_inference(image: Image.Image):
    """
    Day 5: replace this body with real preprocessing + model forward pass,
    returning (class_name, confidence_float). The dummy version below just
    fakes a plausible result so the rest of the pipeline (n8n, WhatsApp
    replies, low-confidence path) can be built and tested right now.
    """
    if MODEL is None:
        predicted_class = random.choice(CLASS_NAMES)
        confidence = round(random.uniform(0.4, 0.98), 2)
        return predicted_class, confidence

    # Real inference path once MODEL is loaded (Day 5+):
    # tensor = preprocess(image)
    # with torch.no_grad():
    #     output = MODEL(tensor)
    # ...
    raise NotImplementedError("Real inference path not wired up yet.")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    load_model()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=None),
):
    # --- auth ---
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # --- basic image validation ---
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    # --- inference ---
    try:
        predicted_class, confidence = run_inference(image)
    except Exception as exc:
        logger.exception("Inference failed")
        return PredictionResponse(
            status="error",
            disease=None,
            confidence=None,
            message="Model is temporarily unavailable. Please try again shortly.",
        )

    # --- confidence guardrail ---
    if confidence < CONFIDENCE_THRESHOLD:
        return PredictionResponse(
            status="low_confidence",
            disease=predicted_class,
            confidence=confidence,
            message="We couldn't identify the disease with confidence. "
                    "Please send a clearer, closer photo of the affected leaf.",
        )

    return PredictionResponse(
        status="ok",
        disease=predicted_class,
        confidence=confidence,
        message=f"Detected: {predicted_class.replace('_', ' ').title()} "
                f"(confidence: {confidence:.0%}).",
    )
