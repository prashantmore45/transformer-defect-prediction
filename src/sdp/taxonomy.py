"""Defect class taxonomy — the single source of truth for label definitions.

Every component (model, API, frontend, evaluation) imports class names from
here. Defining them in one place prevents the classic bug where the model's
label order silently disagrees with the API's, producing confidently wrong
class names.
"""

from enum import Enum


class DefectClass(str, Enum):
    """Defect categories derived from Project CodeNet judge verdicts.

    Provisional — the final taxonomy is a Milestone 1/4 research output.
    Inheriting from `str` makes members JSON-serialisable directly.
    """

    CLEAN = "clean"
    COMPILATION_ERROR = "compilation_error"
    RUNTIME_ERROR = "runtime_error"
    TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    WRONG_ANSWER = "wrong_answer"


#: Canonical ordering. Index position == model output index. Never reorder
#: without retraining — doing so silently remaps every prediction.
CLASS_NAMES: list[str] = [c.value for c in DefectClass]

NUM_CLASSES: int = len(CLASS_NAMES)
