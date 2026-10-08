"""Reading Order and Layout Reconstruction Engine for ParseAnything.
Detects single/multi-column layouts, sidebars, headers, footers, and footnotes.
Reconstructs natural human reading order:
Headers -> [Column 1 -> Column 2 -> ...] -> Footers -> Footnotes.
"""

from typing import List, Tuple
from schemas.models import BlockType, SemanticBlock


class ReadingOrderEngine:
    """Reconstructs reading flow across single and multi-column document layouts."""

    @classmethod
    def reconstruct_reading_order(
        cls,
        blocks: List[SemanticBlock],
        page_width: float = 612.0,
        page_height: float = 792.0
    ) -> List[SemanticBlock]:
        """Orders blocks according to human reading flow and assigns 1-indexed reading_order."""
        if not blocks:
            return []

        # Partition blocks into: Headers, Footers, Footnotes, Body Content
        headers: List[SemanticBlock] = []
        footers: List[SemanticBlock] = []
        footnotes: List[SemanticBlock] = []
        body_blocks: List[SemanticBlock] = []

        header_cutoff = page_height * 0.08  # Top 8% of page
        footer_cutoff = page_height * 0.92  # Bottom 8% of page

        for b in blocks:
            if b.type == BlockType.HEADER:
                headers.append(b)
                continue
            if b.type == BlockType.FOOTER:
                footers.append(b)
                continue
            if b.type == BlockType.FOOTNOTE:
                footnotes.append(b)
                continue

            # Position-based heuristics if bounding box exists
            if b.source.bbox:
                y0, y1 = b.source.bbox[1], b.source.bbox[3]
                if y1 <= header_cutoff and len(b.content.split()) <= 10:
                    b.type = BlockType.HEADER
                    headers.append(b)
                    continue
                if y0 >= footer_cutoff and (len(b.content.split()) <= 10 or b.content.isdigit()):
                    b.type = BlockType.FOOTER
                    footers.append(b)
                    continue

            body_blocks.append(b)

        # Detect columns in body blocks
        ordered_body = cls._order_body_blocks(body_blocks, page_width)

        # Combine: Headers first, Body in reading flow, Footers, then Footnotes
        all_ordered = headers + ordered_body + footers + footnotes

        # Assign sequential reading_order
        for idx, block in enumerate(all_ordered, start=1):
            block.reading_order = idx

        return all_ordered

    @classmethod
    def _order_body_blocks(
        cls,
        blocks: List[SemanticBlock],
        page_width: float
    ) -> List[SemanticBlock]:
        """Detects whether layout is multi-column and sorts blocks accordingly."""
        if len(blocks) <= 1:
            return blocks

        # Check if blocks without bboxes exist
        with_bbox = [b for b in blocks if b.source.bbox is not None]
        without_bbox = [b for b in blocks if b.source.bbox is None]

        if not with_bbox:
            return blocks

        # Check for 2-column layout:
        # A 2-column page has distinct left blocks (center < 48% width) and right blocks (center > 52% width)
        # with horizontal overlap in Y-coordinates between left and right columns.
        midpoint = page_width / 2.0
        left_column: List[SemanticBlock] = []
        right_column: List[SemanticBlock] = []
        span_blocks: List[SemanticBlock] = []  # headings or wide tables spanning full width

        for b in with_bbox:
            x0, y0, x1, y1 = b.source.bbox
            width = x1 - x0

            # If block spans > 65% of page width, it's a full-width header or figure
            if width > (page_width * 0.65):
                span_blocks.append(b)
            elif x1 <= (midpoint + 20):
                left_column.append(b)
            elif x0 >= (midpoint - 20):
                right_column.append(b)
            else:
                # Straddles center
                span_blocks.append(b)

        is_multi_column = len(left_column) >= 2 and len(right_column) >= 2

        if not is_multi_column:
            # Single-column: sort purely top-to-bottom, left-to-right
            return sorted(with_bbox, key=lambda b: (b.source.bbox[1], b.source.bbox[0])) + without_bbox

        # Multi-column ordering:
        # Full-width blocks at the top (y < columns) -> Left Column top-to-bottom -> Right Column top-to-bottom -> Full-width blocks at bottom
        left_column.sort(key=lambda b: b.source.bbox[1])
        right_column.sort(key=lambda b: b.source.bbox[1])

        # Merge span blocks according to vertical boundaries
        result: List[SemanticBlock] = []
        left_top_y = min(b.source.bbox[1] for b in left_column)
        right_top_y = min(b.source.bbox[1] for b in right_column)
        col_start_y = min(left_top_y, right_top_y)

        top_spans = [b for b in span_blocks if b.source.bbox[1] < col_start_y]
        bottom_spans = [b for b in span_blocks if b.source.bbox[1] >= col_start_y]

        top_spans.sort(key=lambda b: b.source.bbox[1])
        bottom_spans.sort(key=lambda b: b.source.bbox[1])

        result.extend(top_spans)
        result.extend(left_column)
        result.extend(right_column)
        result.extend(bottom_spans)
        result.extend(without_bbox)

        return result
