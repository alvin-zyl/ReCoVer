from enum import Enum

class ULFMWorkType(Enum):
    """Enumeration of ULFM work types for failure handling."""

    GRADIENT_REDUCTION = "gradient_reduction"
    CONSENSUS = "consensus"

class ULFMTrainingStepType(Enum):
    """Enumeration of ULFM training step types."""

    NORMAL = "normal"
    OVERFLOW = "overflow"
