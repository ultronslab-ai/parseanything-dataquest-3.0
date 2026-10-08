"""Abstract Base Class for Document Extractors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
from PIL import Image

from doc_extractor.models import DocumentExtractionResult, PageExtractionResult


class BaseExtractor(ABC):
    """Abstract interface for all document layout extractors."""

    @abstractmethod
    def extract_document(self, file_path: Union[str, Path]) -> DocumentExtractionResult:
        """Process an entire document (PDF or multi-page file) and extract elements."""
        pass

    @abstractmethod
    def extract_page_image(
        self,
        image: Image.Image,
        page_number: int,
        document_name: str
    ) -> PageExtractionResult:
        """Process a single rendered page image and extract elements."""
        pass
