"""High-Fidelity PPTX Parser Adapter for ParseAnything.
Leverages python-pptx to extract slide presentations, titles, shapes,
tables, embedded charts with native numeric series, speaker notes, and shape coordinates.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pptx
from pptx.enum.shapes import MSO_SHAPE_TYPE

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
from extractors.chart.chart_extractor import ChartExtractor
from extractors.table.table_extractor import TableExtractor


class PPTXParser(BaseParser):
    """Specialized parser for Microsoft PowerPoint (.pptx) presentations."""

    @property
    def supported_formats(self) -> List[str]:
        return ["pptx"]

    def parse(
        self,
        file_path: Path,
        document_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        start_time = time.time()
        filename = file_path.name
        file_size = file_path.stat().st_size

        prs = pptx.Presentation(str(file_path))
        slide_width_pt = prs.slide_width.pt if prs.slide_width else 960.0
        slide_height_pt = prs.slide_height.pt if prs.slide_height else 540.0

        all_blocks: List[SemanticBlock] = []
        page_render_infos: List[PageRenderInfo] = []
        warnings: List[str] = []

        tables_count = 0
        figures_count = 0
        equations_count = 0
        total_slides = len(prs.slides)

        core_props = prs.core_properties
        doc_metadata = DocumentMetadata(
            title=core_props.title or filename,
            author=core_props.author,
            creation_date=str(core_props.created) if core_props.created else None,
            slide_count=total_slides
        )

        global_order = 1

        for slide_idx, slide in enumerate(prs.slides):
            slide_num = slide_idx + 1

            page_render_infos.append(
                PageRenderInfo(
                    page_number=slide_num,
                    width=slide_width_pt,
                    height=slide_height_pt,
                    image_url=None
                )
            )

            # Slide Title extraction
            slide_title = ""
            if slide.shapes.title and slide.shapes.title.has_text_frame:
                slide_title = slide.shapes.title.text.strip()
                if slide_title:
                    all_blocks.append(
                        SemanticBlock(
                            block_id=f"pptx_s{slide_num}_title",
                            type=BlockType.HEADING,
                            content=slide_title,
                            source=SourceLocation(
                                file=filename,
                                slide=slide_num,
                                bbox=[0.0, 0.0, slide_width_pt, 60.0],
                                coordinate_system="point"
                            ),
                            confidence=0.99,
                            confidence_level=ConfidenceLevel.HIGH,
                            extraction_method="native_pptx_title",
                            reading_order=global_order,
                            metadata={"level": 1, "is_slide_title": True}
                        )
                    )
                    global_order += 1

            # Iterate shapes on slide
            for shape_idx, shape in enumerate(slide.shapes):
                # Skip title shape if already processed
                if shape == slide.shapes.title:
                    continue

                left = shape.left.pt if shape.left else 0.0
                top = shape.top.pt if shape.top else 0.0
                width = shape.width.pt if shape.width else 100.0
                height = shape.height.pt if shape.height else 50.0
                bbox = [round(left, 2), round(top, 2), round(left + width, 2), round(top + height, 2)]

                # 1. Native PowerPoint Table
                if shape.has_table:
                    tables_count += 1
                    t = shape.table
                    grid_cells: List[List[str]] = []

                    for r in t.rows:
                        row_vals = [cell.text.strip() for cell in r.cells]
                        grid_cells.append(row_vals)

                    td = TableExtractor.create_table_data_from_grid(
                        grid_cells,
                        headers_count=1,
                        bbox=bbox
                    )

                    all_blocks.append(
                        SemanticBlock(
                            block_id=f"pptx_s{slide_num}_tbl_{shape_idx}",
                            type=BlockType.TABLE,
                            content=td.markdown_table or "Slide Table",
                            source=SourceLocation(
                                file=filename,
                                slide=slide_num,
                                bbox=bbox,
                                object_id=f"shape_{shape.shape_id}",
                                coordinate_system="point"
                            ),
                            confidence=0.98,
                            confidence_level=ConfidenceLevel.HIGH,
                            extraction_method="native_pptx_table",
                            reading_order=global_order,
                            table_data=td
                        )
                    )
                    global_order += 1

                # 2. Native PowerPoint Chart
                elif shape.has_chart:
                    figures_count += 1
                    chart = shape.chart
                    chart_title = chart.chart_title.text_frame.text if chart.has_title and chart.chart_title.has_text_frame else "Slide Chart"

                    series_data = []
                    for s in chart.series:
                        try:
                            s_vals = list(s.values) if hasattr(s, "values") else []
                            series_data.append({"name": s.name, "values": s_vals})
                        except Exception:
                            pass

                    categories = []
                    try:
                        if hasattr(chart, "plots") and chart.plots:
                            categories = [str(c) for c in chart.plots[0].categories]
                    except Exception:
                        pass

                    chart_data, conf = ChartExtractor.analyze_chart_from_native(
                        title=chart_title,
                        chart_type="chart",
                        series_data=series_data,
                        categories=categories
                    )

                    all_blocks.append(
                        SemanticBlock(
                            block_id=f"pptx_s{slide_num}_chart_{shape_idx}",
                            type=BlockType.CHART,
                            content=f"[Chart: {chart_title}]",
                            source=SourceLocation(
                                file=filename,
                                slide=slide_num,
                                bbox=bbox,
                                object_id=f"shape_{shape.shape_id}",
                                coordinate_system="point"
                            ),
                            confidence=conf,
                            confidence_level=ConfidenceLevel.HIGH,
                            extraction_method="native_pptx_chart",
                            reading_order=global_order,
                            chart_data=chart_data
                        )
                    )
                    global_order += 1

                # 3. Standard Text Frame / Shape
                elif shape.has_text_frame:
                    text_lines = []
                    for para in shape.text_frame.paragraphs:
                        p_txt = para.text.strip()
                        if p_txt:
                            text_lines.append(p_txt)

                    full_text = "\n".join(text_lines).strip()
                    if full_text and full_text != slide_title:
                        is_list = any(line.startswith(("- ", "• ", "* ")) for line in text_lines)
                        all_blocks.append(
                            SemanticBlock(
                                block_id=f"pptx_s{slide_num}_txt_{shape_idx}",
                                type=BlockType.LIST_ITEM if is_list else BlockType.PARAGRAPH,
                                content=full_text,
                                source=SourceLocation(
                                    file=filename,
                                    slide=slide_num,
                                    bbox=bbox,
                                    object_id=f"shape_{shape.shape_id}",
                                    coordinate_system="point"
                                ),
                                confidence=0.98,
                                confidence_level=ConfidenceLevel.HIGH,
                                extraction_method="native_pptx_shape",
                                reading_order=global_order
                            )
                        )
                        global_order += 1

                # 4. Picture / Image
                elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    figures_count += 1
                    all_blocks.append(
                        SemanticBlock(
                            block_id=f"pptx_s{slide_num}_pic_{shape_idx}",
                            type=BlockType.FIGURE,
                            content=f"[Image on Slide {slide_num}]",
                            source=SourceLocation(
                                file=filename,
                                slide=slide_num,
                                bbox=bbox,
                                object_id=f"shape_{shape.shape_id}",
                                coordinate_system="point"
                            ),
                            confidence=0.90,
                            confidence_level=ConfidenceLevel.HIGH,
                            extraction_method="native_pptx_picture",
                            reading_order=global_order
                        )
                    )
                    global_order += 1

            # Speaker Notes if present
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    all_blocks.append(
                        SemanticBlock(
                            block_id=f"pptx_s{slide_num}_notes",
                            type=BlockType.FOOTNOTE,
                            content=f"Speaker Notes: {notes_text}",
                            source=SourceLocation(
                                file=filename,
                                slide=slide_num,
                                coordinate_system="point"
                            ),
                            confidence=0.99,
                            confidence_level=ConfidenceLevel.HIGH,
                            extraction_method="native_pptx_notes",
                            reading_order=global_order
                        )
                    )
                    global_order += 1

        elapsed = time.time() - start_time

        return Document(
            document_id=document_id,
            filename=filename,
            file_type="pptx",
            file_size_bytes=file_size,
            processing_status="success",
            metadata=doc_metadata,
            blocks=all_blocks,
            pages=page_render_infos,
            warnings=warnings,
            errors=[],
            stats=ProcessingStats(
                processing_time_seconds=round(elapsed, 3),
                pages_processed=total_slides,
                pages_per_second=round(total_slides / max(0.001, elapsed), 2),
                blocks_extracted=len(all_blocks),
                tables_detected=tables_count,
                figures_detected=figures_count,
                equations_detected=equations_count,
                low_confidence_blocks=0
            )
        )
