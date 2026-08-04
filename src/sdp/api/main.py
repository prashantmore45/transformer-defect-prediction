"""FastAPI inference service."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from sdp.config import load_config
from sdp.model import build_predictor
from sdp.model.base import Predictor
from sdp.schemas import HealthResponse, PredictionRequest, PredictionResponse
from sdp.taxonomy import NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the predictor once at startup; release it at shutdown."""
    config = load_config()
    logger.info("Building predictor: %s", config["predictor"]["type"])
    app.state.predictor = build_predictor(config["predictor"])
    app.state.config = config
    logger.info("Predictor ready: %s", app.state.predictor.name)
    yield
    logger.info("Shutting down")


config = load_config()

app = FastAPI(
    title=config["api"]["title"],
    version=config["api"]["version"],
    description="Multiclass software defect classification. **Placeholder model.**",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors"]["allow_origins"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor(request: Request) -> Predictor:
    """Dependency providing the loaded predictor."""
    return request.app.state.predictor


PredictorDep = Annotated[Predictor, Depends(get_predictor)]


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(predictor: PredictorDep) -> HealthResponse:
    """Liveness check reporting which model is loaded."""
    return HealthResponse(
        status="ok",
        model_name=predictor.name,
        num_classes=NUM_CLASSES,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest, predictor: PredictorDep) -> PredictionResponse:
    """Classify a source code snippet into a defect category."""
    logger.info("Prediction request: %d characters", len(request.code))
    return predictor.predict(request.code)
