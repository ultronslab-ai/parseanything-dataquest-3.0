"""Complex Table Extraction Engine for ParseAnything.
Handles table detection, row/column boundaries, cell coordinates,
merged cells (row_span / col_span), and multi-row headers.
"""

from typing import Any, Dict, List, Optional, Tuple
from schemas.models import TableCell, TableData


class TableExtractor:
    """Extracts, structures, and validates complex tables into structural matrices."""

    @classmethod
    def create_table_data_from_grid(
        cls,
        grid_cells: List[List[str]],
        headers_count: int = 1,
        bbox: Optional[List[float]] = None,
        cell_spans: Optional[List[Dict[str, Any]]] = None
    ) -> TableData:
        """Constructs a rich TableData object from a 2D string grid and span metadata."""
        if not grid_cells:
            return TableData()

        num_rows = len(grid_cells)
        num_cols = max(len(row) for row in grid_cells) if num_rows > 0 else 0

        # Extract headers (can be multi-row!)
        headers: List[List[str]] = []
        actual_header_count = min(headers_count, num_rows)
        for r in range(actual_header_count):
            row = grid_cells[r]
            # pad row to num_cols if needed
            padded = [str(c).strip() for c in row] + [""] * (num_cols - len(row))
            headers.append(padded)

        # Extract data rows
        rows: List[List[str]] = []
        for r in range(actual_header_count, num_rows):
            row = grid_cells[r]
            padded = [str(c).strip() for c in row] + [""] * (num_cols - len(row))
            rows.append(padded)

        # Build detailed cell matrix
        cells: List[TableCell] = []
        merged_count = 0

        # Index spans by (row, col)
        span_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        if cell_spans:
            for s in cell_spans:
                r = s.get("row", 0)
                c = s.get("col", 0)
                r_span = s.get("row_span", 1)
                c_span = s.get("col_span", 1)
                span_map[(r, c)] = (r_span, c_span)
                if r_span > 1 or c_span > 1:
                    merged_count += 1

        for r_idx, row in enumerate(grid_cells):
            for c_idx, cell_value in enumerate(row):
                r_span, c_span = span_map.get((r_idx, c_idx), (1, 1))

                # Estimate cell bbox within table bbox if available
                cell_bbox = None
                if bbox and num_rows > 0 and num_cols > 0:
                    x0, y0, x1, y1 = bbox
                    col_w = (x1 - x0) / num_cols
                    row_h = (y1 - y0) / num_rows
                    cell_bbox = [
                        round(x0 + c_idx * col_w, 2),
                        round(y0 + r_idx * row_h, 2),
                        round(x0 + (c_idx + c_span) * col_w, 2),
                        round(y0 + (r_idx + r_span) * row_h, 2)
                    ]

                cells.append(
                    TableCell(
                        row=r_idx,
                        column=c_idx,
                        row_span=r_span,
                        col_span=c_span,
                        text=str(cell_value).strip(),
                        bbox=cell_bbox,
                        confidence=0.98 if str(cell_value).strip() else 0.90
                    )
                )

        # Generate standard markdown table
        md_table = cls.to_markdown(headers, rows, num_cols)

        return TableData(
            headers=headers,
            rows=rows,
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            merged_cells_count=merged_count,
            markdown_table=md_table
        )

    @classmethod
    def to_markdown(cls, headers: List[List[str]], rows: List[List[str]], num_cols: int) -> str:
        """Formats headers and data rows into clean GitHub-flavored Markdown table."""
        if num_cols == 0:
            return ""

        lines = []
        if headers:
            for h_row in headers:
                safe_row = [c.replace("|", "\\|").replace("\n", " ") for c in h_row[:num_cols]]
                lines.append("| " + " | ".join(safe_row) + " |")
            # Separator line
            lines.append("| " + " | ".join(["---"] * num_cols) + " |")
        else:
            # Fallback header
            lines.append("| " + " | ".join([f"Col {i+1}" for i in range(num_cols)]) + " |")
            lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        for r_row in rows:
            safe_row = [c.replace("|", "\\|").replace("\n", " ") for c in r_row[:num_cols]]
            # Pad row if short
            while len(safe_row) < num_cols:
                safe_row.append("")
            lines.append("| " + " | ".join(safe_row) + " |")

        return "\n".join(lines)
