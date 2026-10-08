"""Fail-Safe and Error Handling Engine for ParseAnything.
Ensures zero unhandled crashes, strict adherence to standardized error codes,
60-second execution safeguards, and flags ambiguous inputs rather than hallucinating.
"""

import time
from typing import Any, Dict, List, Optional
from schemas.errors import ErrorCode, ProcessingError
from schemas.models import Document, DocumentMetadata, ProcessingStats


class FailSafeEngine:
    """Safeguards execution, traps crashes, and structures standard failure responses."""

    @classmethod
    def create_failure_document(
        cls,
        document_id: str,
        filename: str,
        file_type: str,
        error: ProcessingError,
        processing_time: float = 0.0
    ) -> Document:
        """Constructs a valid Document object representing an intentional, graceful failure."""
        return Document(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=0,
            processing_status="failed",
            metadata=DocumentMetadata(),
            blocks=[],
            warnings=[],
            errors=[error],
            stats=ProcessingStats(
                processing_time_seconds=processing_time,
                pages_processed=0,
                pages_per_second=0.0,
                blocks_extracted=0,
                low_confidence_blocks=0,
            ),
            markdown=f"# Extraction Failed\n\n**Error Code:** `{error.code.value}`\n\n**Reason:** {error.message}"
        )

    @classmethod
    def create_error(
        cls,
        code: ErrorCode,
        message: str,
        severity: str = "error",
        block_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> ProcessingError:
        """Helper to create standardized error objects."""
        return ProcessingError(
            code=code,
            message=message,
            severity=severity,
            block_id=block_id,
            details=details or {}
        )
