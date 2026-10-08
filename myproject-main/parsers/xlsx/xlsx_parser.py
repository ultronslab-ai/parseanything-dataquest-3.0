"""High-Fidelity XLSX Parser Adapter for ParseAnything.
Leverages openpyxl to extract spreadsheets, multiple sheets, used ranges,
cell matrices, formulas, merged cells (A1:B1), and cell provenance.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import openpyxl

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
from extractors.table.table_extractor import TableExtractor


class XLSXParser(BaseParser):
    """Specialized parser for Microsoft Excel (.xlsx) spreadsheets."""

    @property
    def supported_formats(self) -> List[str]:
        return ["xlsx"]

    def parse(
        self,
        file_path: Path,
        document_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        start_time = time.time()
        filename = file_path.name
        file_size = file_path.stat().st_size

        # Load workbook (data_only=True evaluates formulas to calculated numbers)
        wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=False)

        all_blocks: List[SemanticBlock] = []
        warnings: List[str] = []
        tables_count = 0

        sheet_names = wb.sheetnames
        doc_metadata = DocumentMetadata(
            title=filename,
            sheet_count=len(sheet_names),
            sheet_names=sheet_names
        )

        global_order = 1

        for s_idx, sheet_name in enumerate(sheet_names):
            ws = wb[sheet_name]

            # Determine used range boundaries
            min_row, max_row = ws.min_row, ws.max_row
            min_col, max_col = ws.min_column, ws.max_column

            # If empty sheet
            if not min_row or not max_row or min_row > max_row:
                continue

            # Read merged cells in sheet
            merged_ranges = ws.merged_cells.ranges
            cell_spans: List[Dict[str, Any]] = []

            for rng in merged_ranges:
                # rng has min_row, max_row, min_col, max_col (1-indexed)
                r_span = rng.max_row - rng.min_row + 1
                c_span = rng.max_col - rng.min_col + 1
                cell_spans.append({
                    "row": rng.min_row - min_row,
                    "col": rng.min_col - min_col,
                    "row_span": r_span,
                    "col_span": c_span
                })

            # Read cell grid values
            grid: List[List[str]] = []
            has_content = False

            for r in range(min_row, max_row + 1):
                row_vals = []
                for c in range(min_col, max_col + 1):
                    val = ws.cell(row=r, column=c).value
                    val_str = "" if val is None else str(val).strip()
                    if val_str:
                        has_content = True
                    row_vals.append(val_str)
                grid.append(row_vals)

            if not has_content:
                continue

            # Add Sheet Title / Heading block
            all_blocks.append(
                SemanticBlock(
                    block_id=f"xlsx_s{s_idx+1}_title",
                    type=BlockType.HEADING,
                    content=f"Sheet: {sheet_name}",
                    source=SourceLocation(
                        file=filename,
                        sheet=sheet_name,
                        cell_range=f"A1:{openpyxl.utils.get_column_letter(max_col)}{max_row}",
                        coordinate_system="point"
                    ),
                    confidence=0.99,
                    confidence_level=ConfidenceLevel.HIGH,
                    extraction_method="native_xlsx_sheet_title",
                    reading_order=global_order,
                    metadata={"level": 2, "sheet_name": sheet_name}
                )
            )
            global_order += 1

            # Format the sheet data grid into a rich TableData structure
            td = TableExtractor.create_table_data_from_grid(
                grid,
                headers_count=1,
                cell_spans=cell_spans
            )
            tables_count += 1

            cell_range_str = f"{openpyxl.utils.get_column_letter(min_col)}{min_row}:{openpyxl.utils.get_column_letter(max_col)}{max_row}"

            all_blocks.append(
                SemanticBlock(
                    block_id=f"xlsx_s{s_idx+1}_table",
                    type=BlockType.TABLE,
                    content=td.markdown_table or f"Sheet Table: {sheet_name}",
                    source=SourceLocation(
                        file=filename,
                        sheet=sheet_name,
                        cell_range=cell_range_str,
                        coordinate_system="point"
                    ),
                    confidence=0.99,
                    confidence_level=ConfidenceLevel.HIGH,
                    extraction_method="native_xlsx_grid",
                    reading_order=global_order,
                    table_data=td,
                    metadata={"sheet_name": sheet_name, "used_range": cell_range_str}
                )
            )
            global_order += 1

        wb.close()
        elapsed = time.time() - start_time

        return Document(
            document_id=document_id,
            filename=filename,
            file_type="xlsx",
            file_size_bytes=file_size,
            processing_status="success",
            metadata=doc_metadata,
            blocks=all_blocks,
            pages=[],
            warnings=warnings,
            errors=[],
            stats=ProcessingStats(
                processing_time_seconds=round(elapsed, 3),
                pages_processed=len(sheet_names),
                pages_per_second=round(len(sheet_names) / max(0.001, elapsed), 2),
                blocks_extracted=len(all_blocks),
                tables_detected=tables_count,
                figures_detected=0,
                equations_detected=0,
                low_confidence_blocks=0
            )
        )
