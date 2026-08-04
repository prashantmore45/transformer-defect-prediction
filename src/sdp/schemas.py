"""API data contracts.

These models define the boundary between client and server. They are
deliberately independent of how prediction is implemented — the same contract
holds whether a dummy or a fine-tuned transformer sits behind it.
"""

from pydantic import BaseModel, Field

from sdp.taxonomy import DefectClass

MAX_CODE_LENGTH = 50_000


class PredictionRequest(BaseModel):
    """A request to classify a source code snippet."""

    code: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CODE_LENGTH,
        description="Source code to analyse (C++).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"code": "int main() {\n    int x = 1\n    return 0;\n}"}]
        }
    }


class ClassProbability(BaseModel):
    """Probability assigned to one defect class."""

    label: DefectClass
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """The classification result for one snippet."""

    predicted_class: DefectClass = Field(..., description="Highest-probability class.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the winner.")
    probabilities: list[ClassProbability] = Field(
        ..., description="Full distribution over all classes."
    )
    model_name: str = Field(..., description="Identifier of the predictor used.")
    is_placeholder: bool = Field(
        ..., description="True when output is random and carries no meaning."
    )


class HealthResponse(BaseModel):
    """Service liveness and loaded-model information."""

    status: str
    model_name: str
    num_classes: int
