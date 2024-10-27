from enum import Enum, auto


class ChainComponent(Enum):
    """Enum for different components that might need an LLM in a chain"""

    QA = auto()
    EXTRACTION = auto()
    RELEVANCE = auto()
    VALIDATION = auto()
    IMAGE = auto()
    EXTRACTION_IMAGE = auto()


class ChainType(Enum):
    """Enum for different types of chains that can be created"""

    BASIC_QA = "basic_qa"
    IMAGE_QA = "image_qa"
    STRUCTURED_OUTPUT_IMAGE = "structured_output_image"
    STRUCTURED_OUTPUT = "structured_output"
    RELEVANCE_CHECK = "relevance_check"
    FULL_VALIDATION = "full_validation"
