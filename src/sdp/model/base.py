"""The predictor abstraction.

This is the seam between the serving layer and the model layer. The API depends
on this Protocol and never on a concrete model class, so a new model can be
introduced by writing a conforming class and changing one config value.

This is the Dependency Inversion Principle: both the API (high-level) and the
models (low-level) depend on this abstraction, not on each other.
"""

from typing import Protocol, runtime_checkable

from sdp.schemas import PredictionResponse


@runtime_checkable
class Predictor(Protocol):
    """Anything that can classify a source code snippet."""

    @property
    def name(self) -> str:
        """Identifier recorded in every response, e.g. 'dummy-random-v1'."""
        ...

    @property
    def is_placeholder(self) -> bool:
        """True if predictions are meaningless. Surfaced in the UI."""
        ...

    def predict(self, code: str) -> PredictionResponse:
        """Classify one snippet."""
        ...
