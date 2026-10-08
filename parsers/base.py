"""Base Parser Interface for ParseAnything.
Every format adapter (PDF, DOCX, PPTX, XLSX, Image) implements this interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from schemas.models import Document


class BaseParser(ABC):
    """Abstract base class for all specialized document format parsers."""

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """List of lower-case format strings supported (e.g. ['pdf'])."""
        pass

    @abstractmethod
    def parse(
        self,
        file_path: Path,
        document_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Parse document and return the unified Document intermediate representation.
        
        Args:
            file_path: Absolute or relative path to file on disk.
            document_id: Unique identifier generated for this ingestion task.
            options: Extraction configuration and flags.
            
        Returns:
            Document instance matching the universal ParseAnything schema.
        """
        pass
