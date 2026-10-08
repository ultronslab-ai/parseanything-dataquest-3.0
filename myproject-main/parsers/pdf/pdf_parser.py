"""High-Fidelity PDF Parser Adapter for ParseAnything.
Leverages PyMuPDF (fitz) for vector layout analysis, digital text extraction,
table detection, embedded graphics, and cascading OCR fallback for scanned pages.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from PIL import Image

from parsers.base import BaseParser
from schemas.models import (
    BlockStatus,
    BlockType,
    ConfidenceLevel,
    Document,
    DocumentMetadata,
    PageRenderInfo,
    ProcessingStats,
    SemanticBlock,
    SourceLocation,
)
from extractors.equation.equation_extractor import EquationExtractor
from extractors.chart.chart_extractor import ChartExtractor
from extractors.table.table_extractor import TableExtractor
from extractors.ocr.ocr_provider import get_ocr_provider
from extractors.vision.region_detector import RegionDetector
from extractors.vision.math_ocr import MathOCRExtractor
from extractors.vision.handwriting_ocr import HandwritingOCRExtractor
from schemas.models import EquationData
from pipeline.reading_order.reading_order_engine import ReadingOrderEngine


class PDFParser(BaseParser):
    """Universal PDF Parser supporting both digital and scanned documents."""

    @property
    def supported_formats(self) -> List[str]:
        return ["pdf"]

    def parse(
        self,
        file_path: Path,
        document_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        opts = options or {}
        dpi = int(opts.get("render_dpi", 150))
        export_page_images = opts.get("export_page_images", True)
        images_dir = opts.get("images_output_dir", None)

        start_time = time.time()
        doc = pymupdf.open(str(file_path))
        filename = file_path.name
        file_size = file_path.stat().st_size
        total_pages = len(doc)

        all_blocks: List[SemanticBlock] = []
        page_render_infos: List[PageRenderInfo] = []
        warnings: List[str] = []
        ocr_provider = None

        tables_count = 0
        figures_count = 0
        equations_count = 0

        # Extract document-level metadata
        meta = doc.metadata or {}
        doc_metadata = DocumentMetadata(
            title=meta.get("title") or filename,
            author=meta.get("author"),
            creation_date=meta.get("creationDate"),
            page_count=total_pages
        )

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            page_w, page_h = rect.width, rect.height

            # Render page image for UI bounding-box overlay (Lazy via API)
            rendered_image_url = f"/api/v1/documents/{document_id}/page-image/{page_num}"
            if export_page_images and images_dir:
                try:
                    pix = page.get_pixmap(dpi=dpi)
                    out_img_path = Path(images_dir) / f"{document_id}_p{page_num}.png"
                    pix.save(str(out_img_path))
                except Exception as e:
                    warnings.append(f"Failed to render page image for page {page_num}: {e}")

            page_render_infos.append(
                PageRenderInfo(
                    page_number=page_num,
                    width=page_w,
                    height=page_h,
                    dpi=dpi,
                    image_url=rendered_image_url
                )
            )

            # Check if page is digital or scanned
            # Text layer test: check character count
            raw_text = page.get_text("text").strip()
            image_list = page.get_images()
            
            # Filter out common scanner watermarks
            cleaned_text = raw_text.lower()
            watermarks = ["scanned by camscanner", "camscanner", "scanned with", "created by"]
            for w in watermarks:
                cleaned_text = cleaned_text.replace(w, "")
            cleaned_text = cleaned_text.strip()
            
            is_scanned = len(cleaned_text) < 50 and len(image_list) >= 1

            page_blocks: List[SemanticBlock] = []

            # -------------------------------------------------------------
            # ADDITIVE SCANNED & HANDWRITTEN FALLBACK PIPELINE
            # -------------------------------------------------------------
            from scanned_pipeline.scanned_orchestrator import ScannedDocumentPipeline, ENABLE_SCANNED_FALLBACK
            scanned_processed = False

            if is_scanned and ENABLE_SCANNED_FALLBACK:
                try:
                    pix_full = page.get_pixmap(dpi=300)
                    pil_img_page = Image.frombytes("RGB", [pix_full.width, pix_full.height], pix_full.samples)
                    scanned_pipeline = ScannedDocumentPipeline()
                    scanned_result = scanned_pipeline.process_page(
                        image=pil_img_page,
                        page_num=page_num,
                        filename=filename,
                        page_w=page_w,
                        page_h=page_h,
                        native_text=raw_text,
                        embedded_images_count=len(image_list),
                        options=opts
                    )
                    if scanned_result.success and not scanned_result.is_digital and scanned_result.blocks:
                        page_blocks = scanned_result.blocks
                        scanned_processed = True
                        page_render_infos[-1].page_type = scanned_result.page_type
                        page_render_infos[-1].quality_score = scanned_result.quality_score
                        page_render_infos[-1].preprocessing_metadata = scanned_result.preprocessing_metadata
                        equations_count += sum(1 for b in page_blocks if b.type == BlockType.EQUATION)
                        tables_count += sum(1 for b in page_blocks if b.type == BlockType.TABLE)
                        figures_count += sum(1 for b in page_blocks if b.type in (BlockType.FIGURE, BlockType.CHART))
                except Exception as e:
                    warnings.append(f"Scanned fallback pipeline error on page {page_num}: {e}. Falling back to baseline OCR.")
                    scanned_processed = False

            if is_scanned and not scanned_processed:
                page_render_infos[-1].page_type = "SCANNED_PRINTED"
                warnings.append(f"Page {page_num} detected as scanned image; routing to Vision pipeline.")
                
                pix = page.get_pixmap(dpi=300)
                
                print(f"PAGE {page_num}")
                print(f"native_text_length = {len(raw_text)}")
                print(f"native_text = {repr(raw_text)}")
                print(f"meaningful_text = {len(cleaned_text) >= 50}")
                print(f"scanned_page = {is_scanned}")
                print(f"rendered_image = True")
                print(f"image_resolution = {pix.width}x{pix.height}")
                print(f"ocr_started = True")
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Basic preprocessing for handwriting/OCR (contrast enhancement)
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img_enhanced = enhancer.enhance(1.5)

                scale_x = page_w / pix.width
                scale_y = page_h / pix.height

                math_ocr = MathOCRExtractor()
                hw_ocr = HandwritingOCRExtractor()
                
                if ocr_provider is None:
                    ocr_provider = get_ocr_provider()
                    
                # RESTORE EXISTING DETECTION PIPELINE
                ocr_lines = ocr_provider.extract_text(pil_img_enhanced)
                print(f"regions_detected = {len(ocr_lines)}")

                for line_idx, line in enumerate(ocr_lines):
                    scaled_bbox = [
                        round(line.bbox[0] * scale_x, 2),
                        round(line.bbox[1] * scale_y, 2),
                        round(line.bbox[2] * scale_x, 2),
                        round(line.bbox[3] * scale_y, 2),
                    ]
                    
                    # Ensure bbox is valid before crop
                    if line.bbox[2] <= line.bbox[0] or line.bbox[3] <= line.bbox[1]:
                        continue
                        
                    # INTELLIGENT ROUTER
                    crop = pil_img_enhanced.crop((line.bbox[0], line.bbox[1], line.bbox[2], line.bbox[3]))
                    b_type = BlockType.PARAGRAPH
                    eq_data = None
                    final_text = line.text
                    final_conf = round(line.confidence, 4)
                    method = "ocr_local"
                    
                    # 1. Route to Math OCR
                    if EquationExtractor.is_equation(line.text) or any(c in line.text for c in ['=', '+', '-', '√', '∫', '∑']):
                        latex_text = math_ocr.extract(crop)
                        if latex_text:
                            b_type = BlockType.EQUATION
                            final_text = latex_text
                            eq_data = EquationData(latex=latex_text, is_inline=False)
                            method = "math_ocr"
                            final_conf = 0.92
                            equations_count += 1
                        else:
                            eq_data, eq_conf = EquationExtractor.extract_equation(line.text)
                            b_type = BlockType.EQUATION
                            
                    # 2. Route to Handwriting OCR
                    elif line.confidence < 0.85:
                        hw_text = hw_ocr.extract(crop)
                        if hw_text:
                            final_text = hw_text
                            method = "handwriting_ocr"
                            final_conf = max(line.confidence, 0.85)
                            
                    elif line_idx == 0 and len(line.text.split()) < 8:
                        b_type = BlockType.HEADING

                    page_blocks.append(
                        SemanticBlock(
                            block_id=f"b_p{page_num}_{line_idx+1:03d}",
                            type=b_type,
                            content=final_text,
                            source=SourceLocation(file=filename, page=page_num, bbox=scaled_bbox, coordinate_system="point"),
                            confidence=final_conf,
                            confidence_level=ConfidenceLevel.HIGH if final_conf >= 0.90 else (ConfidenceLevel.MEDIUM if final_conf >= 0.70 else ConfidenceLevel.LOW),
                            extraction_method=method,
                            status=BlockStatus.ACCEPTED if final_conf >= 0.70 else BlockStatus.REVIEW_REQUIRED,
                            equation_data=eq_data
                        )
                    )
            elif not is_scanned:
                page_render_infos[-1].page_type = "DIGITAL_TEXT"
                # 1. Native Digital Table Detection using PyMuPDF table finder
                tables = []
                try:
                    tables = page.find_tables()
                except Exception:
                    tables = []

                table_bboxes = []
                for t_idx, table in enumerate(tables):
                    t_bbox = [round(x, 2) for x in table.bbox]
                    table_bboxes.append(t_bbox)
                    tables_count += 1

                    # Extract tabular grid
                    table_data_matrix = table.extract()
                    clean_grid = []
                    for row in table_data_matrix:
                        clean_row = [str(c or "").strip() for c in row]
                        clean_grid.append(clean_row)

                    td = TableExtractor.create_table_data_from_grid(
                        clean_grid,
                        headers_count=1,
                        bbox=t_bbox
                    )

                    page_blocks.append(
                        SemanticBlock(
                            block_id=f"tbl_p{page_num}_{t_idx+1:02d}",
                            type=BlockType.TABLE,
                            content=td.markdown_table or "Table",
                            source=SourceLocation(
                                file=filename,
                                page=page_num,
                                bbox=t_bbox,
                                coordinate_system="point"
                            ),
                            confidence=0.96,
                            confidence_level=ConfidenceLevel.HIGH,
                            extraction_method="native_pdf_tables",
                            table_data=td
                        )
                    )

                # 2. Native Digital Text Extraction via blocks
                text_page_dict = page.get_text("dict")
                text_blocks = text_page_dict.get("blocks", [])

                # Compute median body font size for heading hierarchy
                font_sizes = []
                for b in text_blocks:
                    if b.get("type") == 0:  # text
                        for line in b.get("lines", []):
                            for span in line.get("spans", []):
                                font_sizes.append(span.get("size", 10.0))
                median_font_size = sorted(font_sizes)[len(font_sizes)//2] if font_sizes else 11.0

                for b_idx, b in enumerate(text_blocks):
                    b_bbox = [round(x, 2) for x in b.get("bbox", [0, 0, 0, 0])]

                    # Skip text block if it is inside an extracted table bbox
                    if any(
                        b_bbox[0] >= (tb[0] - 5) and b_bbox[1] >= (tb[1] - 5) and
                        b_bbox[2] <= (tb[2] + 5) and b_bbox[3] <= (tb[3] + 5)
                        for tb in table_bboxes
                    ):
                        continue

                    # If image/graphic block (type == 1)
                    if b.get("type") == 1:
                        figures_count += 1
                        chart_meta, chart_conf = ChartExtractor.analyze_visual_region(
                            caption_text=f"Figure on page {page_num}",
                            is_vector=False
                        )
                        page_blocks.append(
                            SemanticBlock(
                                block_id=f"fig_p{page_num}_{b_idx+1:02d}",
                                type=BlockType.FIGURE,
                                content=f"[Figure/Graphic {b_idx+1}]",
                                source=SourceLocation(
                                    file=filename,
                                    page=page_num,
                                    bbox=b_bbox,
                                    coordinate_system="point"
                                ),
                                confidence=chart_conf,
                                confidence_level=ConfidenceLevel.HIGH,
                                extraction_method="native_pdf_image",
                                chart_data=chart_meta
                            )
                        )
                        continue

                    # Text block
                    lines = b.get("lines", [])
                    block_text_lines = []
                    max_span_size = 0.0

                    for line in lines:
                        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                        for span in line.get("spans", []):
                            max_span_size = max(max_span_size, span.get("size", 0.0))
                        block_text_lines.append(line_text)

                    full_text = " ".join(block_text_lines).strip()
                    if not full_text:
                        continue

                    # Semantic classification
                    block_type = BlockType.PARAGRAPH
                    eq_data = None
                    heading_level = None

                    if EquationExtractor.is_equation(full_text):
                        try:
                            # Attempt accurate Math OCR using the visual region
                            pix_crop = page.get_pixmap(dpi=150, clip=pymupdf.Rect(*b_bbox))
                            crop_img = Image.frombytes("RGB", [pix_crop.width, pix_crop.height], pix_crop.samples)
                            math_ocr = MathOCRExtractor()
                            latex_text = math_ocr.extract(crop_img)
                            if latex_text:
                                eq_data = EquationData(latex=latex_text, is_inline=False)
                                full_text = latex_text
                            else:
                                eq_data, _ = EquationExtractor.extract_equation(full_text)
                        except Exception:
                            eq_data, _ = EquationExtractor.extract_equation(full_text)
                            
                        block_type = BlockType.EQUATION
                        equations_count += 1
                    elif max_span_size > (median_font_size * 1.25) and len(full_text.split()) <= 15:
                        block_type = BlockType.HEADING
                        heading_level = 1 if max_span_size > (median_font_size * 1.5) else 2
                    elif full_text.startswith(("- ", "• ", "* ")) or re.match(r"^\d+\.\s", full_text):
                        block_type = BlockType.LIST_ITEM

                    page_blocks.append(
                        SemanticBlock(
                            block_id=f"b_p{page_num}_{b_idx+1:03d}",
                            type=block_type,
                            content=full_text,
                            source=SourceLocation(
                                file=filename,
                                page=page_num,
                                bbox=b_bbox,
                                coordinate_system="point"
                            ),
                            confidence=0.98,
                            confidence_level=ConfidenceLevel.HIGH,
                            extraction_method="native_pdf_text",
                            metadata={"font_size": max_span_size, "level": heading_level} if heading_level else {},
                            equation_data=eq_data
                        )
                    )

            # Reconstruct reading order on this page
            ordered_page_blocks = ReadingOrderEngine.reconstruct_reading_order(
                page_blocks,
                page_width=page_w,
                page_height=page_h
            )
            all_blocks.extend(ordered_page_blocks)

        doc.close()
        elapsed = time.time() - start_time
        pages_per_sec = round(total_pages / elapsed, 2) if elapsed > 0 else total_pages

        return Document(
            document_id=document_id,
            filename=filename,
            file_type="pdf",
            file_size_bytes=file_size,
            processing_status="success",
            metadata=doc_metadata,
            blocks=all_blocks,
            pages=page_render_infos,
            warnings=warnings,
            errors=[],
            stats=ProcessingStats(
                processing_time_seconds=round(elapsed, 3),
                pages_processed=total_pages,
                pages_per_second=pages_per_sec,
                blocks_extracted=len(all_blocks),
                tables_detected=tables_count,
                figures_detected=figures_count,
                equations_detected=equations_count,
                low_confidence_blocks=sum(1 for b in all_blocks if b.confidence < 0.70)
            )
        )
