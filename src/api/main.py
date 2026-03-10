from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predictor import Predictor
from src.utils.config import load_yaml
from src.utils.logging import setup_logging


logger = setup_logging("api")
load_dotenv()


class Observation(BaseModel):
    timestamp: str = Field(..., description="ISO timestamp for this observation (5-minute resolution).")
    glucose: float
    heart_rate: float
    HRV: float
    sleep_stage_code: int = Field(..., description="0=awake, 1=light, 2=deep, 3=rem")
    activity_level: float = Field(..., ge=0.0, le=1.0)
    meal_carbs: float = Field(..., ge=0.0)


class PredictRequest(BaseModel):
    user_id: str
    observations: List[Observation] = Field(..., description="Last N time steps (>= window size).")


class PredictResponse(BaseModel):
    user_id: str
    predicted_glucose_2h: float
    confidence_std: float
    ci95_low: float
    ci95_high: float


def _device() -> torch.device:
    if os.getenv("DEVICE", "cpu").lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_paths() -> tuple[Path, Path, int]:
    cfg_path = os.getenv("SERVICE_CONFIG", "configs/service.yaml")
    cfg = load_yaml(cfg_path)

    model_path = Path(cfg["paths"]["model_artifact_path"])
    feature_spec_path = Path(cfg["paths"]["feature_spec_path"])
    mc = int(cfg["inference"]["mc_dropout_samples"])

    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_ARTIFACT_PATH not found: {model_path}")
    if not feature_spec_path.exists():
        raise FileNotFoundError(f"FEATURES_ARTIFACT_PATH not found: {feature_spec_path}")
    return model_path, feature_spec_path, mc


app = FastAPI(
    title="Personalized Glycemic Forecasting Engine",
    description="Research/engineering demo microservice for forecasting glucose 2 hours ahead (NOT a medical device).",
    version="0.1.0",
)


@app.on_event("startup")
def _startup() -> None:
    model_path, feature_spec_path, mc = _resolve_paths()
    app.state.predictor = Predictor(
        model_path=model_path,
        feature_spec_path=feature_spec_path,
        device=_device(),
        mc_dropout_samples=mc,
    )
    logger.info("Loaded model from %s", model_path.as_posix())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    predictor: Predictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        raw_df = pd.DataFrame([o.model_dump() for o in req.observations])
        raw_df["user_id"] = req.user_id
        mean, std = predictor.predict(raw_df)
        ci_low = mean - 1.96 * std
        ci_high = mean + 1.96 * std
        return PredictResponse(
            user_id=req.user_id,
            predicted_glucose_2h=float(mean),
            confidence_std=float(std),
            ci95_low=float(ci_low),
            ci95_high=float(ci_high),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

