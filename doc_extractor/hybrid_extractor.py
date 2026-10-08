"""Local Hybrid Document Extractor.

Extracts text, tables, charts/graphics, and equations from PDFs using
PyMuPDF (fitz) layout analysis, vector drawing clustering, and heuristic parsing.
Runs 100% locally with zero external API dependencies.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pymupdf  # PyMuPDF
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


# Common mathematical symbols indicating mathematical formulas
MATH_SYMBOLS_REGEX = re.compile(
    r"[=<>±∓×÷√∛∜∫∬∭∮∂∇∑∏∈∉∋⊂⊃⊆⊇∪∩∧∨¬⇒⇔∀∃∄≠≤≥≈≡≃≅∝∞\^]|"
    r"(\\[a-zA-Z]+)|"  # LaTeX commands like \frac, \alpha
    r"([a-zA-Z]\s*=\s*[-+]?[0-9a-zA-Z\(\)\/\*\^\.]+)"
)


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def is_box_contained(inner: Tuple[float, float, float, float], outer: Tuple[float, float, float, float], threshold: float = 0.7) -> bool:
    """Check if inner box is largely contained inside outer box."""
    x1_min, y1_min, x1_max, y1_max = inner
    x2_min, y2_min, x2_max, y2_max = outer

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    inner_area = (x1_max - x1_min) * (y1_max - y1_min)
    return (inter_area / inner_area) >= threshold if inner_area > 0 else False


class HybridExtractor(BaseExtractor):
    """Local PDF Extractor leveraging PyMuPDF layout analysis."""

    def __init__(self, render_dpi: int = 150):
        self.render_dpi = render_dpi

    def extract_document(self, file_path: Union[str, Path]) -> DocumentExtractionResult:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

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
            page_res = self._process_pdf_page(page, page_num, total_pages, doc_name, str(path))
            pages_results.append(page_res)
            all_elements.extend(page_res.elements)

        doc.close()

        # Summary statistics
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
        document_name: str = "image.png"
    ) -> PageExtractionResult:
        """Process an image directly by converting it to an in-memory PDF page."""
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        doc = pymupdf.open(stream=img_bytes, filetype="png")
        page = doc[0]
        res = self._process_pdf_page(page, page_number, 1, document_name, None)
        doc.close()
        return res

    def _process_pdf_page(
        self,
        page: pymupdf.Page,
        page_num: int,
        total_pages: int,
        doc_name: str,
        doc_path: Optional[str]
    ) -> PageExtractionResult:
        width = page.rect.width
        height = page.rect.height

        elements: List[ExtractedElement] = []
        occupied_boxes: List[Tuple[float, float, float, float]] = []

        source_ref = SourceReference(
            page_number=page_num,
            document_name=doc_name,
            document_path=doc_path,
            page_total=total_pages
        )

        # -------------------------------------------------------------
        # 1. TABLE EXTRACTION (PyMuPDF find_tables)
        # -------------------------------------------------------------
        try:
            tabs = page.find_tables()
            for t_idx, table in enumerate(tabs):
                bbox_rect = table.bbox  # (x0, y0, x1, y1)
                data = table.extract()
                if not data or len(data) < 1:
                    continue

                # Format as Markdown table
                markdown_lines = []
                headers = [str(c or "").strip() for c in data[0]]
                markdown_lines.append("| " + " | ".join(headers) + " |")
                markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in data[1:]:
                    markdown_lines.append("| " + " | ".join([str(c or "").strip() for c in row]) + " |")
                markdown_content = "\n".join(markdown_lines)

                bbox = BoundingBox(
                    x_min=round(bbox_rect[0], 2),
                    y_min=round(bbox_rect[1], 2),
                    x_max=round(bbox_rect[2], 2),
                    y_max=round(bbox_rect[3], 2),
                    coordinate_system="pixel"
                )
                occupied_boxes.append((bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max))

                elements.append(
                    ExtractedElement(
                        id=f"elem_p{page_num}_tbl_{t_idx + 1:03d}",
                        type=ElementType.TABLE,
                        content=markdown_content,
                        bbox=bbox,
                        source=source_ref,
                        confidence=0.96,
                        metadata={
                            "row_count": len(data),
                            "col_count": len(headers),
                            "headers": headers,
                            "raw_matrix": data
                        }
                    )
                )
        except Exception:
            # Fallback if table finder encountered non-standard vector structure
            pass

        # -------------------------------------------------------------
        # 2. CHARTS & GRAPHICS EXTRACTION (Drawings & Embedded Images)
        # -------------------------------------------------------------
        try:
            # Check embedded raster images
            images = page.get_images(full=True)
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                for rect in rects:
                    box = (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2))
                    # Avoid tiny icons or decorations (< 50x50)
                    if (box[2] - box[0] < 50) or (box[3] - box[1] < 50):
                        continue
                    # Check if already covered by table
                    if any(is_box_contained(box, occ) for occ in occupied_boxes):
                        continue

                    occupied_boxes.append(box)
                    elements.append(
                        ExtractedElement(
                            id=f"elem_p{page_num}_chart_{img_idx + 1:03d}",
                            type=ElementType.CHART,
                            content=f"Embedded visual chart/graphic ({int(box[2]-box[0])}x{int(box[3]-box[1])} px)",
                            bbox=BoundingBox(
                                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3], coordinate_system="pixel"
                            ),
                            source=source_ref,
                            confidence=0.92,
                            metadata={"format": "raster_image", "xref": xref}
                        )
                    )

            # Check vector drawings (clusters of paths, curves, lines forming a figure)
            drawings = page.get_drawings()
            if drawings:
                # Group drawings with significant area not overlapping tables
                chart_rects: List[pymupdf.Rect] = []
                for d in drawings:
                    r = d.get("rect")
                    if r and r.width > 80 and r.height > 60:
                        box = (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))
                        if not any(is_box_contained(box, occ) for occ in occupied_boxes):
                            # Check merge with nearby drawing rects
                            merged = False
                            for idx, existing in enumerate(chart_rects):
                                if existing.intersects(r) or compute_iou(
                                    (r.x0, r.y0, r.x1, r.y1),
                                    (existing.x0, existing.y0, existing.x1, existing.y1)
                                ) > 0.1:
                                    chart_rects[idx] = existing | r
                                    merged = True
                                    break
                            if not merged and len(chart_rects) < 5:
                                chart_rects.append(r)

                for c_idx, c_rect in enumerate(chart_rects):
                    box = (round(c_rect.x0, 2), round(c_rect.y0, 2), round(c_rect.x1, 2), round(c_rect.y1, 2))
                    if any(is_box_contained(box, occ) for occ in occupied_boxes):
                        continue
                    occupied_boxes.append(box)
                    elements.append(
                        ExtractedElement(
                            id=f"elem_p{page_num}_vecchart_{c_idx + 1:03d}",
                            type=ElementType.CHART,
                            content=f"Vector graphics diagram/chart ({int(box[2]-box[0])}x{int(box[3]-box[1])} pt)",
                            bbox=BoundingBox(
                                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3], coordinate_system="pixel"
                            ),
                            source=source_ref,
                            confidence=0.88,
                            metadata={"format": "vector_drawing"}
                        )
                    )
        except Exception:
            pass

        # -------------------------------------------------------------
        # 3. TEXT & EQUATION EXTRACTION (Text Blocks)
        # -------------------------------------------------------------
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_counter = 1
        equation_counter = 1

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            text_str = text.strip()
            if not text_str:
                continue

            box = (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))

            # Skip if inside an already identified table or chart
            if any(is_box_contained(box, occ, threshold=0.6) for occ in occupied_boxes):
                continue

            bbox_obj = BoundingBox(
                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3], coordinate_system="pixel"
            )

            # Check if this block is an equation
            is_equation = self._classify_as_equation(text_str)

            if is_equation:
                latex_content = self._format_as_latex(text_str)
                elements.append(
                    ExtractedElement(
                        id=f"elem_p{page_num}_eq_{equation_counter:03d}",
                        type=ElementType.EQUATION,
                        content=latex_content,
                        bbox=bbox_obj,
                        source=source_ref,
                        confidence=0.94,
                        metadata={"raw_text": text_str, "syntax": "latex"}
                    )
                )
                equation_counter += 1
            else:
                elements.append(
                    ExtractedElement(
                        id=f"elem_p{page_num}_txt_{text_counter:03d}",
                        type=ElementType.TEXT,
                        content=text_str,
                        bbox=bbox_obj,
                        source=source_ref,
                        confidence=0.98,
                        metadata={"block_number": block_no}
                    )
                )
                text_counter += 1

        # Sort elements by vertical reading order (top-to-bottom, left-to-right)
        elements.sort(key=lambda e: (e.bbox.y_min, e.bbox.x_min))

        return PageExtractionResult(
            page_number=page_num,
            width=round(width, 2),
            height=round(height, 2),
            dpi=self.render_dpi,
            elements=elements
        )

    def _classify_as_equation(self, text: str) -> bool:
        """Determines if a standalone text block is an equation."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 3:
            return False  # Multi-line paragraphs are rarely pure equations

        # Explicit math keywords or LaTeX style markers
        if any(marker in text for marker in ["$$", "\\frac", "\\sum", "\\int", "\\sqrt", "\\partial"]):
            return True

        # Check for typical math pattern: e.g. E = mc^2, f(x) = ..., y = mx + b
        match_count = len(MATH_SYMBOLS_REGEX.findall(text))
        has_equals = "=" in text or "≈" in text or "≤" in text or "≥" in text

        # If it has equals sign, math symbols, and relatively few dictionary words
        words = re.findall(r"[a-zA-Z]{3,}", text)
        if has_equals and (len(words) <= 3 or match_count >= 2):
            return True

        return False

    def _format_as_latex(self, text: str) -> str:
        """Normalize math text into standard LaTeX representation."""
        clean = text.strip()
        # Clean up known unicode conversions
        clean = clean.replace("×", "\\times ").replace("÷", "\\div ")
        clean = clean.replace("√", "\\sqrt").replace("±", "\\pm ")
        clean = clean.replace("≤", "\\le ").replace("≥", "\\ge ")
        clean = clean.replace("≠", "\\ne ").replace("≈", "\\approx ")
        clean = clean.replace("π", "\\pi ").replace("α", "\\alpha ").replace("β", "\\beta ")
        return clean
