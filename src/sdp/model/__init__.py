"""Model implementations and the factory that selects between them."""

from typing import Any

from sdp.model.base import Predictor
from sdp.model.dummy import DummyPredictor


def build_predictor(config: dict[str, Any]) -> Predictor:
    """Construct the predictor named in the configuration.

    Adding a model means adding one branch here — no other module changes.
    """
    predictor_type = config.get("type", "dummy")

    if predictor_type == "dummy":
        return DummyPredictor(seed=config.get("seed"))

    raise ValueError(f"Unknown predictor type: {predictor_type!r}. Available: ['dummy']")
