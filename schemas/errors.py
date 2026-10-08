"""Standardized Error Schemas and Error Codes for ParseAnything.
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    PARSER_FAILURE = "PARSER_FAILURE"
    OCR_FAILURE = "OCR_FAILURE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TABLE_EXTRACTION_FAILURE = "TABLE_EXTRACTION_FAILURE"
    VISION_FAILURE = "VISION_FAILURE"
    TIMEOUT = "TIMEOUT"
    PARTIAL_EXTRACTION = "PARTIAL_EXTRACTION"


class ProcessingError(BaseModel):
    code: ErrorCode
    message: str
    severity: str = Field(default="error", description="'error', 'warning', or 'info'")
    block_id: Optional[str] = None
    stage: Optional[str] = Field(default=None, description="Pipeline stage where error occurred")
    details: Dict[str, Any] = Field(default_factory=dict)
