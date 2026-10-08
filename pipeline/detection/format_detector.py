"""Robust Multi-format Detection Engine.
Inspects file magic bytes, internal structure (e.g. OOXML ZIP members), and headers.
Never relies solely on filename extensions.
"""

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from schemas.errors import ErrorCode, ProcessingError


class FormatDetectionResult:
    def __init__(
        self,
        format_name: str,
        confidence: float,
        parser_name: str,
        mime_type: str,
        is_supported: bool = True,
        error: Optional[ProcessingError] = None
    ):
        self.format_name = format_name
        self.confidence = confidence
        self.parser_name = parser_name
        self.mime_type = mime_type
        self.is_supported = is_supported
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        if not self.is_supported and self.error:
            return {
                "status": "failed",
                "format": self.format_name,
                "error": {
                    "code": self.error.code.value,
                    "message": self.error.message,
                    "details": self.error.details
                }
            }
        return {
            "format": self.format_name,
            "confidence": self.confidence,
            "parser": self.parser_name,
            "mime_type": self.mime_type,
            "is_supported": self.is_supported
        }


class FormatDetector:
    """Detects document format using file signatures and content analysis."""

    SUPPORTED_PARSERS = {
        "pdf": "pdf_parser",
        "docx": "docx_parser",
        "pptx": "pptx_parser",
        "xlsx": "xlsx_parser",
        "png": "image_parser",
        "jpg": "image_parser",
        "jpeg": "image_parser",
    }

    MIME_MAP = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }

    @classmethod
    def detect(cls, file_path: Path) -> FormatDetectionResult:
        if not file_path.exists():
            return FormatDetectionResult(
                format_name="unknown",
                confidence=0.0,
                parser_name="none",
                mime_type="application/octet-stream",
                is_supported=False,
                error=ProcessingError(
                    code=ErrorCode.CORRUPTED_FILE,
                    message=f"File does not exist: {file_path}",
                    stage="format_detection"
                )
            )

        file_size = file_path.stat().st_size
        if file_size == 0:
            return FormatDetectionResult(
                format_name="empty",
                confidence=1.0,
                parser_name="none",
                mime_type="application/octet-stream",
                is_supported=False,
                error=ProcessingError(
                    code=ErrorCode.CORRUPTED_FILE,
                    message="File is completely empty (0 bytes).",
                    stage="format_detection"
                )
            )

        # Read first 4096 bytes for signature inspection
        try:
            with open(file_path, "rb") as f:
                header = f.read(4096)
        except Exception as e:
            return FormatDetectionResult(
                format_name="unknown",
                confidence=0.0,
                parser_name="none",
                mime_type="application/octet-stream",
                is_supported=False,
                error=ProcessingError(
                    code=ErrorCode.CORRUPTED_FILE,
                    message=f"Failed to read file header: {str(e)}",
                    stage="format_detection"
                )
            )

        # 1. PDF Detection: %PDF-
        if header.startswith(b"%PDF-"):
            return FormatDetectionResult(
                format_name="pdf",
                confidence=0.99,
                parser_name="pdf_parser",
                mime_type="application/pdf"
            )

        # 2. PNG Detection: \x89PNG\r\n\x1a\n
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return FormatDetectionResult(
                format_name="png",
                confidence=0.99,
                parser_name="image_parser",
                mime_type="image/png"
            )

        # 3. JPEG Detection: \xff\xd8\xff
        if header.startswith(b"\xff\xd8\xff"):
            return FormatDetectionResult(
                format_name="jpg",
                confidence=0.99,
                parser_name="image_parser",
                mime_type="image/jpeg"
            )

        # 4. ZIP-based Office Open XML Detection: PK\x03\x04
        if header.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(file_path, "r") as zf:
                    namelist = set(zf.namelist())
                    if any(name.startswith("word/") for name in namelist) or "[Content_Types].xml" in namelist and any("wordprocessingml" in name for name in namelist):
                        return FormatDetectionResult(
                            format_name="docx",
                            confidence=0.98,
                            parser_name="docx_parser",
                            mime_type=cls.MIME_MAP["docx"]
                        )
                    if any(name.startswith("ppt/") for name in namelist):
                        return FormatDetectionResult(
                            format_name="pptx",
                            confidence=0.98,
                            parser_name="pptx_parser",
                            mime_type=cls.MIME_MAP["pptx"]
                        )
                    if any(name.startswith("xl/") for name in namelist):
                        return FormatDetectionResult(
                            format_name="xlsx",
                            confidence=0.98,
                            parser_name="xlsx_parser",
                            mime_type=cls.MIME_MAP["xlsx"]
                        )
            except zipfile.BadZipFile:
                return FormatDetectionResult(
                    format_name="corrupt_zip",
                    confidence=0.90,
                    parser_name="none",
                    mime_type="application/zip",
                    is_supported=False,
                    error=ProcessingError(
                        code=ErrorCode.CORRUPTED_FILE,
                        message="File has ZIP header but contains corrupted archive data.",
                        stage="format_detection"
                    )
                )

        # 5. Extension fallback check for non-standard headers
        ext = file_path.suffix.lower().lstrip(".")
        if ext in cls.SUPPORTED_PARSERS:
            # Header didn't match standard magic bytes -> possible corruption
            return FormatDetectionResult(
                format_name=ext,
                confidence=0.40,
                parser_name=cls.SUPPORTED_PARSERS[ext],
                mime_type=cls.MIME_MAP.get(ext, "application/octet-stream"),
                is_supported=False,
                error=ProcessingError(
                    code=ErrorCode.CORRUPTED_FILE,
                    message=f"File extension is .{ext} but file signature does not match valid format specifications.",
                    stage="format_detection"
                )
            )

        # 6. Unsupported format
        return FormatDetectionResult(
            format_name=ext or "binary",
            confidence=0.95,
            parser_name="none",
            mime_type="application/octet-stream",
            is_supported=False,
            error=ProcessingError(
                code=ErrorCode.UNSUPPORTED_FORMAT,
                message=f"Unsupported format '{ext}'. Supported formats: PDF, DOCX, PPTX, XLSX, PNG, JPG, JPEG.",
                stage="format_detection",
                details={"detected_extension": ext}
            )
        )
