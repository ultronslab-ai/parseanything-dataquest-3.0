"""Cross-Page Table Merging Engine for ParseAnything.
Detects when a table spans across page boundaries (e.g. Page N -> Page N+1),
evaluates structural compatibility (column counts, headers, alignment),
merges the tables seamlessly, and updates provenance and confidence.
"""

from typing import List, Tuple
from schemas.models import BlockType, SemanticBlock, TableData


class TableMerger:
    """Merges split tables across consecutive pages using structural geometry."""

    @classmethod
    def merge_cross_page_tables(cls, blocks: List[SemanticBlock]) -> List[SemanticBlock]:
        """Scans blocks and merges tables that span consecutive pages."""
        if len(blocks) < 2:
            return blocks

        merged_blocks: List[SemanticBlock] = []
        i = 0

        while i < len(blocks):
            curr_block = blocks[i]

            # If current block is a table and has valid table_data
            if curr_block.type == BlockType.TABLE and curr_block.table_data and curr_block.source.page is not None:
                # Look ahead for a candidate table on the immediately following page
                j = i + 1
                merged_any = False

                while j < len(blocks):
                    next_block = blocks[j]

                    # Stop if we hit substantive non-table content on next page or page difference > 1
                    if next_block.source.page is not None:
                        page_diff = next_block.source.page - curr_block.source.page
                        if page_diff > 1:
                            break

                    if next_block.type == BlockType.TABLE and next_block.table_data:
                        # Check if next_block is on curr_page + 1
                        if next_block.source.page == curr_block.source.page + 1:
                            should_merge, confidence, reason = cls._evaluate_merge(
                                curr_block.table_data, next_block.table_data
                            )

                            if should_merge:
                                # Merge next_block into curr_block!
                                curr_block = cls._perform_merge(curr_block, next_block, confidence, reason)
                                merged_any = True
                                i = j  # advance outer pointer
                                break
                    j += 1

                merged_blocks.append(curr_block)
            else:
                merged_blocks.append(curr_block)

            i += 1

        return merged_blocks

    @classmethod
    def _evaluate_merge(cls, t1: TableData, t2: TableData) -> Tuple[bool, float, str]:
        """Evaluates whether two tables on consecutive pages are parts of the same multi-page table."""
        # 1. Column count check
        if t1.num_cols != t2.num_cols or t1.num_cols == 0:
            return False, 0.0, "Column count mismatch"

        # 2. Check if t2 repeats the exact same header as t1
        has_repeated_header = False
        if t1.headers and t2.headers:
            if t1.headers[0] == t2.headers[0]:
                has_repeated_header = True

        # 3. High confidence if columns match and headers match or t2 has no distinct header
        if has_repeated_header:
            return True, 0.95, "Identical headers repeated on continuation page"

        # If t2 has no headers or first row looks like data rather than header
        if t1.num_cols == t2.num_cols:
            return True, 0.88, "Matching column geometry across page boundary"

        return False, 0.0, "Incompatible table structures"

    @classmethod
    def _perform_merge(
        cls,
        parent: SemanticBlock,
        continuation: SemanticBlock,
        confidence: float,
        reason: str
    ) -> SemanticBlock:
        """Combines the rows and cells of two tables and attaches provenance metadata."""
        t1 = parent.table_data
        t2 = continuation.table_data

        if not t1 or not t2:
            return parent

        # If t2 repeated the header row, exclude that header from being duplicated in the merged data rows
        rows_to_append = t2.rows
        if t1.headers and t2.headers and t1.headers[0] == t2.headers[0]:
            rows_to_append = t2.rows  # t2.headers is already separated from t2.rows in TableData!

        new_rows = list(t1.rows) + list(rows_to_append)

        # Merge cells with updated row indices
        new_cells = list(t1.cells)
        row_offset = len(t1.headers) + len(t1.rows)
        for cell in t2.cells:
            # Shift row index
            shifted = cell.model_copy()
            shifted.row += row_offset
            new_cells.append(shifted)

        from extractors.table.table_extractor import TableExtractor
        new_md = TableExtractor.to_markdown(t1.headers, new_rows, t1.num_cols)

        merged_data = TableData(
            headers=t1.headers,
            rows=new_rows,
            cells=new_cells,
            num_rows=len(t1.headers) + len(new_rows),
            num_cols=t1.num_cols,
            merged_cells_count=t1.merged_cells_count + t2.merged_cells_count,
            is_multi_page_merged=True,
            markdown_table=new_md
        )

        parent.table_data = merged_data
        parent.content = new_md
        parent.metadata["cross_page_merged"] = True
        parent.metadata["merged_pages"] = [parent.source.page, continuation.source.page]
        parent.metadata["merge_reason"] = reason

        # Update provenance description
        parent.warnings.append(
            f"Cross-page table merged across Pages {parent.source.page} and {continuation.source.page} ({reason})."
        )

        return parent
