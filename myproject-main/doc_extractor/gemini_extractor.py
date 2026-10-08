"""Multimodal Vision LLM Document Extractor.

Uses Google Gemini Multimodal Vision API to detect and extract:
- Text paragraphs & headers
- Tables (structured markdown)
- Charts (data insights, axes, and descriptions)
- Equations (LaTeX format)
With bounding box coordinates, source references, and confidence scores.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pymupdf
from PIL import Image

from doc_extractor.base import BaseExtractor
from doc_extractor.models import (
    BoundingBox,
    DocumentExtractionResult,
    ElementType,
    ExtractedElement,
    PageExtractionResult,
    SourceReference,
)

SYSTEM_PROMPT = """You are a high-precision document layout analysis and extraction engine.
Analyze the provided document page image and identify every distinct element:
1. "text": Headings, paragraphs, bullet points, footnotes.
2. "table": Any table, grid, or structured data representation. Convert to GitHub Markdown table format.
3. "chart": Any graph, plot, diagram, or bar/line chart. Describe the chart, title, axes, and extract the underlying data trends.
4. "equation": Any standalone or mathematical equation or formula. Represent in valid LaTeX format (e.g. "E = mc^2" or "\\frac{a}{b}").

For EVERY element, you MUST return:
- "type": "text" | "table" | "chart" | "equation"
- "content": the actual text, markdown table, chart summary/data, or LaTeX formula.
- "bbox": [x_min, y_min, x_max, y_max] in normalized coordinates from 0.0 to 1.0 (where (0,0) is top-left, (1,1) is bottom-right).
- "confidence": your confidence estimation from 0.0 to 1.0.
- "metadata": optional additional structured properties (e.g. chart_type, row_count, etc.)

Return ONLY a valid JSON array matching this format:
[
  {
    "type": "text",
    "content": "Example text content",
    "bbox": [0.05, 0.10, 0.90, 0.15],
    "confidence": 0.99,
    "metadata": {}
  }
]
"""


class GeminiVisionExtractor(BaseExtractor):
    """Vision-LLM extractor using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self._client_initialized = False

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self._client_initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini API: {e}")

    def extract_document(self, file_path: Union[str, Path]) -> DocumentExtractionResult:
        path = Path(file_path).resolve()
        doc = pymupdf.open(str(path))
        doc_name = path.name
        total_pages = len(doc)
        pages_results: List[PageExtractionResult] = []
        all_elements: List[ExtractedElement] = []

        import time
        start_time = time.time()

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1

            # Render page to PIL image
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            page_res = self.extract_page_image(
                image=img,
                page_number=page_num,
                document_name=doc_name,
                doc_path=str(path),
                page_total=total_pages
            )
            pages_results.append(page_res)
            all_elements.extend(page_res.elements)

        doc.close()

        counts_by_type: Dict[str, int] = {}
        total_conf = 0.0
        for elem in all_elements:
            counts_by_type[elem.type.value] = counts_by_type.get(elem.type.value, 0) + 1
            total_conf += elem.confidence

        avg_conf = round(total_conf / len(all_elements), 4) if all_elements else 0.0

        return DocumentExtractionResult(
            document_name=doc_name,
            total_pages=total_pages,
            elements=all_elements,
            pages=pages_results,
            counts_by_type=counts_by_type,
            average_confidence=avg_conf,
            processing_time_seconds=round(time.time() - start_time, 3),
        )

    def extract_page_image(
        self,
        image: Image.Image,
        page_number: int = 1,
        document_name: str = "document.pdf",
        doc_path: Optional[str] = None,
        page_total: Optional[int] = None
    ) -> PageExtractionResult:
        width, height = image.size

        source_ref = SourceReference(
            page_number=page_number,
            document_name=document_name,
            document_path=doc_path,
            page_total=page_total
        )

        if not self._client_initialized:
            raise RuntimeError(
                "Gemini API key is not configured. Set GEMINI_API_KEY environment variable "
                "or pass api_key to GeminiVisionExtractor, or use HybridExtractor for offline local parsing."
            )

        # Call Gemini Vision with structured prompt
        response = self.model.generate_content(
            [image, SYSTEM_PROMPT],
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            raw_text = response.text.strip()
            items = json.loads(raw_text)
            if not isinstance(items, list):
                if isinstance(items, dict) and "elements" in items:
                    items = items["elements"]
                else:
                    items = []
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            items = []

        elements: List[ExtractedElement] = []
        for idx, item in enumerate(items):
            try:
                elem_type_str = str(item.get("type", "text")).lower()
                try:
                    elem_type = ElementType(elem_type_str)
                except ValueError:
                    elem_type = ElementType.TEXT

                raw_bbox = item.get("bbox", [0.0, 0.0, 1.0, 1.0])
                # Convert normalized bbox to pixel coordinates
                x0 = round(float(raw_bbox[0]) * width, 2)
                y0 = round(float(raw_bbox[1]) * height, 2)
                x1 = round(float(raw_bbox[2]) * width, 2)
                y1 = round(float(raw_bbox[3]) * height, 2)

                bbox = BoundingBox(
                    x_min=min(x0, x1),
                    y_min=min(y0, y1),
                    x_max=max(x0, x1),
                    y_max=max(y0, y1),
                    coordinate_system="pixel"
                )

                confidence = float(item.get("confidence", 0.95))

                elements.append(
                    ExtractedElement(
                        id=f"elem_p{page_number}_{elem_type.value}_{idx + 1:03d}",
                        type=elem_type,
                        content=str(item.get("content", "")),
                        bbox=bbox,
                        source=source_ref,
                        confidence=confidence,
                        metadata=item.get("metadata", {})
                    )
                )
            except Exception as e:
                print(f"Skipping malformed element {idx}: {e}")

        return PageExtractionResult(
            page_number=page_number,
            width=float(width),
            height=float(height),
            dpi=150,
            elements=elements
        )
