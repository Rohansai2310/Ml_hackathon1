"""FastAPI inference backend for PathSnap."""

from pathlib import Path
import json
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, data_io as io
from .features import build_candidate_table
from .build import F32
from .network import build_geom
from .hmm import viterbi_decode, load_decoder_parameters

app = FastAPI(
    title="PathSnap API",
    description="GPS road map-matching using classifier probabilities and HMM/Viterbi.",
    version="1.0.0",
)

# Person 2's trained classifier can be registered here.
_model = None
_model_sha256 = None
_decoder_parameters = {}


def set_model(model):
    """Register a trained classifier exposing predict_proba()."""
    global _model
    _model = model


def load_model(directory):
    """Load a trusted local artifact only when its frozen hash/configuration match."""
    global _model_sha256
    import joblib
    from .train import file_hash
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text())
    configuration = json.loads((directory / "configuration.json").read_text())
    artifact = directory / "pipeline.joblib"
    if (metadata["status"] != "complete" or configuration["features"] != config.FEATURE_COLUMNS
            or file_hash(artifact) != metadata["model_sha256"]):
        raise ValueError("Model status, features or artifact hash is invalid")
    set_model(joblib.load(artifact))
    _model_sha256 = metadata["model_sha256"]


if os.environ.get("PATHSNAP_MODEL_DIR"):
    load_model(os.environ["PATHSNAP_MODEL_DIR"])
if os.environ.get("PATHSNAP_DECODER_CONFIG"):
    if _model_sha256 is None:
        raise ValueError("PATHSNAP_DECODER_CONFIG requires PATHSNAP_MODEL_DIR")
    _decoder_parameters = load_decoder_parameters(os.environ["PATHSNAP_DECODER_CONFIG"], model_sha256=_model_sha256)


class MatchRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^[0-9]{8}$")
    partition: str = "student_test"
    data_root: Optional[str] = None


class MatchResponse(BaseModel):
    trip_id: str
    partition: str
    predicted_arc_ids: list[int]
    number_of_gps_points: int
    number_of_candidates: int
    model_used: str
    segment_ids: list[int]
    route_segments: list[list[int]]
    break_reasons: dict[int, str]


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
        "decoder_configured": bool(_decoder_parameters),
    }


@app.post("/match", response_model=MatchResponse)
def match_trip(request: MatchRequest):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained classifier loaded. Configure PATHSNAP_MODEL_DIR or call set_model().",
        )

    if request.partition not in {"student_test", "evaluator_test", "train"}:
        raise HTTPException(status_code=400, detail="Invalid partition.")

    try:
        configured_root = request.data_root or os.environ.get("PATHSNAP_DATA_ROOT")
        root = Path(configured_root) if configured_root else None

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
        for feature in F32:
            candidates[feature] = candidates[feature].astype(np.float32)

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
            arc_lengths=geom.arc_len,
            gps_xy=geom.P,
            timestamps=geom.t,
            return_details=True,
            **_decoder_parameters,
        )

        return MatchResponse(
            trip_id=request.trip_id,
            partition=request.partition,
            predicted_arc_ids=predicted.point_arc_ids,
            number_of_gps_points=len(trip.track),
            number_of_candidates=len(candidates),
            model_used=type(getattr(_model, "named_steps", {}).get("classifier", _model)).__name__,
            segment_ids=predicted.segment_ids,
            route_segments=predicted.route_segments,
            break_reasons=predicted.break_reasons,
        )

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
