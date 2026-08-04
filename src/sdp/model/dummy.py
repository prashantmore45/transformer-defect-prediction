"""A placeholder predictor returning random output.

Exists solely to prove the serving path works end to end without the cost and
failure modes of loading a real transformer. Replaced in Milestone 2 by a
CodeBERT-backed predictor implementing the same Protocol.
"""

import random

from sdp.schemas import ClassProbability, PredictionResponse
from sdp.taxonomy import CLASS_NAMES, DefectClass


class DummyPredictor:
    """Returns a random distribution over the defect classes.

    Implements the `Predictor` Protocol structurally — note it does not inherit
    from anything.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "dummy-random-v1"

    @property
    def is_placeholder(self) -> bool:
        return True

    def predict(self, code: str) -> PredictionResponse:
        # Random positive scores, normalised to sum to 1 — the same shape a
        # real softmax output would have.
        scores = [self._rng.random() for _ in CLASS_NAMES]
        total = sum(scores)
        probs = [s / total for s in scores]

        distribution = [
            ClassProbability(label=DefectClass(name), probability=p)
            for name, p in zip(CLASS_NAMES, probs, strict=True)
        ]
        winner = max(distribution, key=lambda c: c.probability)

        return PredictionResponse(
            predicted_class=winner.label,
            confidence=winner.probability,
            probabilities=distribution,
            model_name=self.name,
            is_placeholder=self.is_placeholder,
        )
