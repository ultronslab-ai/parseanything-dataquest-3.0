"""Common Intermediate Representation (IR) Models for ParseAnything.
Universal document schema supporting PDF, DOCX, PPTX, XLSX, and Images.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from schemas.errors import ProcessingError


class BlockType(str, Enum):
    DOCUMENT = "document"
    PAGE = "page"
    SLIDE = "slide"
    SHEET = "sheet"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    LINE_ITEM = "line_item"
    SECTION = "section"
    FIGURE = "figure"
    CHART = "chart"
    IMAGE = "image"
    EQUATION = "equation"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"       # >= 0.90
    MEDIUM = "MEDIUM"   # 0.70 - 0.89
    LOW = "LOW"         # < 0.70


class BlockStatus(str, Enum):
    ACCEPTED = "accepted"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class SourceLocation(BaseModel):
    """Provenance tracking to link extracted block directly to its source origin."""
    file: str = Field(description="Source filename or identifier")
    page: Optional[int] = Field(default=None, description="1-indexed page number if PDF/image")
    slide: Optional[int] = Field(default=None, description="1-indexed slide number if PPTX")
    sheet: Optional[str] = Field(default=None, description="Sheet name if spreadsheet")
    cell_range: Optional[str] = Field(default=None, description="A1-style cell range (e.g. 'B2:F12')")
    object_id: Optional[str] = Field(default=None, description="Internal object or shape ID")
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Bounding box [x_min, y_min, x_max, y_max]"
    )
    coordinate_system: Literal["pixel", "point", "normalized"] = Field(
        default="point",
        description="Units of coordinate system"
    )


class TableCell(BaseModel):
    """Detailed structural representation of a table cell."""
    row: int
    column: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: Optional[List[float]] = None
    confidence: float = 1.0


class TableData(BaseModel):
    """Structural matrix of extracted table with support for merged cells and multi-row headers."""
    headers: List[List[str]] = Field(default_factory=list, description="Multi-row headers")
    rows: List[List[str]] = Field(default_factory=list, description="Data row text values")
    cells: List[TableCell] = Field(default_factory=list, description="Exact cell span layout")
    num_rows: int = 0
    num_cols: int = 0
    merged_cells_count: int = 0
    is_multi_page_merged: bool = False
    caption: Optional[str] = None
    markdown_table: Optional[str] = None


class ChartData(BaseModel):
    """Extracted data and metadata for charts/figures without hallucination."""
    title: Optional[str] = None
    chart_type: Optional[str] = Field(default=None, description="bar, line, pie, scatter, area, etc.")
    labels: List[str] = Field(default_factory=list)
    values: List[Union[float, int, str]] = Field(default_factory=list)
    series_names: List[str] = Field(default_factory=list)
    status: Literal["full", "partial", "unrecovered"] = "full"
    warning: Optional[str] = None


class EquationData(BaseModel):
    """Extracted mathematical expression representation."""
    latex: str = Field(description="LaTeX syntax for formula")
    ascii_math: Optional[str] = None
    is_inline: bool = False


class SemanticBlock(BaseModel):
    """Universal semantic block representation produced by all parsers."""
    block_id: str = Field(description="Unique block identifier")
    type: BlockType = Field(description="Semantic type of block")
    content: str = Field(description="Text, markdown table, or LaTeX content")
    source: SourceLocation = Field(description="Traceable provenance coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    extraction_method: str = Field(default="native", description="Parser or ML extractor name")
    reading_order: int = Field(default=1, description="1-indexed sequence order in reconstructed reading flow")
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    status: BlockStatus = Field(default=BlockStatus.ACCEPTED)

    # Optional typed structured payloads
    table_data: Optional[TableData] = None
    chart_data: Optional[ChartData] = None
    equation_data: Optional[EquationData] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_review: Optional[bool] = None
    alternatives: List[str] = Field(default_factory=list)
    model_name: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, round(float(v), 4)))


class DocumentMetadata(BaseModel):
    author: Optional[str] = None
    title: Optional[str] = None
    creation_date: Optional[str] = None
    page_count: Optional[int] = None
    slide_count: Optional[int] = None
    sheet_count: Optional[int] = None
    sheet_names: List[str] = Field(default_factory=list)
    custom: Dict[str, Any] = Field(default_factory=dict)


class ProcessingStats(BaseModel):
    processing_time_seconds: float = 0.0
    pages_processed: int = 0
    pages_per_second: float = 0.0
    blocks_extracted: int = 0
    tables_detected: int = 0
    figures_detected: int = 0
    equations_detected: int = 0
    low_confidence_blocks: int = 0
    average_confidence: Optional[float] = None
    model_calls: Dict[str, int] = Field(default_factory=dict)
    estimated_api_cost_usd: float = 0.0
    estimated_cost_per_1000_pages_usd: float = 0.0


class PageRenderInfo(BaseModel):
    page_number: int
    width: float
    height: float
    dpi: int = 150
    image_url: Optional[str] = None
    page_type: Optional[str] = None
    quality_score: Optional[float] = None
    preprocessing_metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """The root universal document intermediate representation."""
    schema_version: str = "1.0"
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    processing_status: Literal["success", "partial", "failed"] = "success"
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    blocks: List[SemanticBlock] = Field(default_factory=list)
    pages: List[PageRenderInfo] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[ProcessingError] = Field(default_factory=list)
    stats: ProcessingStats = Field(default_factory=ProcessingStats)
    markdown: Optional[str] = None
