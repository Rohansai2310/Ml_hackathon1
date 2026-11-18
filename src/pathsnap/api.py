"""FastAPI inference backend for PathSnap."""

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, data_io as io
from .features import build_candidate_table
from .network import build_geom
from .hmm import viterbi_decode

app = FastAPI(
    title="PathSnap API",
    description="GPS road map-matching using classifier probabilities and HMM/Viterbi.",
    version="1.0.0",
)

# Person 2's trained classifier can be registered here.
_model = None


def set_model(model):
    """Register a trained classifier exposing predict_proba()."""
    global _model
    _model = model


class MatchRequest(BaseModel):
    trip_id: str = Field(..., min_length=1)
    partition: str = "student_test"
    data_root: Optional[str] = None


class MatchResponse(BaseModel):
    trip_id: str
    partition: str
    predicted_arc_ids: list[int]
    number_of_gps_points: int
    number_of_candidates: int
    model_used: str


@app.get("/")
def home():
    return {
        "message": "PathSnap API is running",
        "docs": "/docs",
        "model_loaded": _model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
    }


@app.post("/match", response_model=MatchResponse)
def match_trip(request: MatchRequest):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained classifier loaded. Register Person 2's model first.",
        )

    if request.partition not in {"student_test", "evaluator_test", "train"}:
        raise HTTPException(status_code=400, detail="Invalid partition.")

    try:
        root = Path(request.data_root) if request.data_root else None

        # CRITICAL: Never read route ground truth during inference.
        trip = io.load_trip(
            request.trip_id,
            request.partition,
            with_route=False,
            root=root,
        )

        if len(trip.track) == 0:
            raise HTTPException(status_code=404, detail="Trip has no GPS points.")

        geom = build_geom(trip)
        candidates = build_candidate_table(geom)

        if candidates.empty:
            raise HTTPException(
                status_code=422,
                detail="No road candidates found for this trip.",
            )

        features = candidates[config.FEATURE_COLUMNS]

        probabilities = _model.predict_proba(features)

        classes = getattr(_model, "classes_", None)
        if classes is None:
            raise ValueError("Classifier must expose classes_.")

        positive_indices = np.where(np.asarray(classes) == 1)[0]
        if len(positive_indices) != 1:
            raise ValueError("Classifier must contain positive class 1.")

        positive_probabilities = probabilities[:, positive_indices[0]]

        predicted = viterbi_decode(
            candidates,
            trip.arcs,
            positive_probabilities,
        )

        return MatchResponse(
            trip_id=request.trip_id,
            partition=request.partition,
            predicted_arc_ids=predicted,
            number_of_gps_points=len(trip.track),
            number_of_candidates=len(candidates),
            model_used=type(_model).__name__,
        )

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))