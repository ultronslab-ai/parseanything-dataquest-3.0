"""Document Extraction Models.

Defines the core data structures for extracted document elements
(text, table, chart, equation) with content, bbox, source, and confidence.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    EQUATION = "equation"


class BoundingBox(BaseModel):
    """Bounding box coordinates.
    Standardized as [x_min, y_min, x_max, y_max].
    Supports both pixel coordinates and normalized (0.0 - 1.0) coordinates.
    """
    x_min: float = Field(description="Left coordinate")
    y_min: float = Field(description="Top coordinate")
    x_max: float = Field(description="Right coordinate")
    y_max: float = Field(description="Bottom coordinate")
    coordinate_system: Literal["pixel", "normalized"] = Field(
        default="pixel",
        description="Whether coordinates are in pixels or normalized 0.0-1.0"
    )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return coordinates as (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def as_list(self) -> List[float]:
        """Return coordinates as [x_min, y_min, x_max, y_max]."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_normalized(self, page_width: float, page_height: float) -> "BoundingBox":
        if self.coordinate_system == "normalized":
            return self
        return BoundingBox(
            x_min=round(self.x_min / page_width, 4),
            y_min=round(self.y_min / page_height, 4),
            x_max=round(self.x_max / page_width, 4),
            y_max=round(self.y_max / page_height, 4),
            coordinate_system="normalized"
        )

    def to_pixel(self, page_width: float, page_height: float) -> "BoundingBox":
        if self.coordinate_system == "pixel":
            return self
        return BoundingBox(
            x_min=round(self.x_min * page_width, 2),
            y_min=round(self.y_min * page_height, 2),
            x_max=round(self.x_max * page_width, 2),
            y_max=round(self.y_max * page_height, 2),
            coordinate_system="pixel"
        )


class SourceReference(BaseModel):
    """Document and page reference information."""
    page_number: int = Field(ge=1, description="1-indexed page number")
    document_name: str = Field(description="Filename or title of the source document")
    document_path: Optional[str] = Field(default=None, description="Absolute or relative file path")
    page_total: Optional[int] = Field(default=None, description="Total pages in the source document")


class ExtractedElement(BaseModel):
    """Standardized representation of any extracted document component.
    
    Required fields per specification:
    - content: The actual text, table markdown/HTML, chart summary/data, or LaTeX formula.
    - bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].
    - source: Source page and document reference.
    - confidence: Confidence score between 0.0 and 1.0.
    """
    id: str = Field(description="Unique element identifier (e.g. elem_p1_001)")
    type: ElementType = Field(description="Element category: text, table, chart, or equation")
    content: str = Field(description="The extracted raw text or structured representation")
    bbox: BoundingBox = Field(description="Bounding box [x_min, y_min, x_max, y_max]")
    source: SourceReference = Field(description="Source reference (page number, document)")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Element-specific metadata")

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, round(float(v), 4)))


class PageExtractionResult(BaseModel):
    """All elements extracted from a single page."""
    page_number: int
    width: float
    height: float
    dpi: int = 150
    elements: List[ExtractedElement] = Field(default_factory=list)
    image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None


class DocumentExtractionResult(BaseModel):
    """Complete extraction result for a document."""
    document_name: str
    total_pages: int
    elements: List[ExtractedElement] = Field(default_factory=list)
    pages: List[PageExtractionResult] = Field(default_factory=list)
    counts_by_type: Dict[str, int] = Field(default_factory=dict)
    average_confidence: float = 0.0
    processing_time_seconds: float = 0.0
