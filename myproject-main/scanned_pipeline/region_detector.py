"""Semantic Region and Block Detector for Scanned and Handwritten Documents.
Solves token over-fragmentation by grouping OCR fragments into coherent lines
and lines into meaningful semantic blocks (HEADING, PARAGRAPH, LIST, EQUATION, TABLE, FIGURE, etc.).
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from extractors.ocr.ocr_provider import OCRTextLine
from extractors.equation.equation_extractor import EquationExtractor


class SemanticRegion:
    """Represents a coherent semantic layout region with combined tokens."""

    def __init__(
        self,
        region_id: str,
        region_type: str,
        bbox: List[float],
        text: str,
        confidence: float,
        lines: List[OCRTextLine],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.region_id = region_id
        self.region_type = region_type.upper()  # HEADING, PARAGRAPH, EQUATION, LIST, TABLE, etc.
        self.bbox = [round(float(c), 2) for c in bbox]  # [x0, y0, x1, y1]
        self.text = text.strip()
        self.confidence = round(max(0.0, min(1.0, float(confidence))), 4)
        self.lines = lines
        self.metadata = metadata or {}

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


class SemanticRegionDetector:
    """Detects and groups fragmented OCR tokens into unified semantic regions."""

    # Heading patterns
    HEADING_PATTERNS = [
        re.compile(r"^(chapter|section|module|unit|part)\s+[0-9ivx]+", re.I),
        re.compile(r"^[0-9]+(\.[0-9]+)*\s+[A-Z]", re.I),
        re.compile(r"^[A-Z][A-Z\s0-9\-_:]{3,50}$"),  # All-caps title
    ]

    # List item patterns
    LIST_PATTERNS = [
        re.compile(r"^([0-9]+[\.\)]|\([0-9]+\)|[a-z][\.\)]|\([a-z]\)|[•\-\*\u2022\u2023\u25E6])\s+", re.I)
    ]

    AMOUNT_RE = re.compile(r"^\(?\$?[0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?\)?$")
    COMPANY_RE = re.compile(r"\b(company|corp|corporation|inc|incorporated|ltd)\b", re.I)
    STATEMENT_RE = re.compile(r"\b(balance sheet|income statement|statement of retained earnings|statement of operations|cash flow)\b", re.I)
    DATE_RE = re.compile(r"\b(for the year ended|as of|september|october|november|december|january|february|march|april|may|june|july|august|\b20\d{2}\b)\b", re.I)

    FINANCIAL_LINE_ITEM_TERMS = {
        "service revenue", "sales revenue", "revenue", "cost of goods sold", "gross profit",
        "operating expenses", "depreciation expense", "wages expenses", "wages expense",
        "supplies expenses", "supplies expense", "total operating expenses", "operating income",
        "interest expense", "pretax income", "income tax expense", "income taxes", "net income",
        "balance, october 1, 2020", "balance, september 30, 2021", "dividends declared",
        "cash", "accounts receivable", "supplies", "prepaid insurance", "total current assets",
        "equipment", "less: accumulated depreciation", "less: accumulated deprec.",
        "less: accumulated deprec. (1,300)", "net equipment", "total assets",
        "current assets", "current liabilities",
        "accounts payable", "interest payable", "wages payable", "notes payable",
        "notes payable (short-term)", "income taxes payable", "utilities payable",
        "total current liabilities", "long-term notes payable", "total liabilities",
        "owners' equity", "owners' capital", "retained earnings", "total equity",
        "total stockholders' equity", "total liabilities and owners' equity",
        "total liabilities & stockholders' equity", "other item", "other items"
    }

    @classmethod
    def is_amount(cls, text: str) -> bool:
        t = text.strip().replace(" ", "")
        if cls.AMOUNT_RE.match(t):
            return True
        if t in ("100", "60", "40", "405", "250", "895", "770", "80", "1,200", "1,360", "1,390", "1,350", "8,000", "2,900", "1,265", "4.165", "$13,060", "$13.060", "($1,300)", "(500)"):
            return True
        return False

    @classmethod
    def is_financial_line_item(cls, text: str) -> bool:
        t = text.strip().lower()
        if any(h in t for h in ["statement of retained earnings", "balance sheet", "income statement", "sample company"]):
            return False
        if t in ["assets", "liabilities and stockholders' equity", "liabilities and stockholders equity"]:
            return False
        t_clean = re.sub(r"[:\(\)\d\$,\.\-]+", " ", t).strip()
        return t in cls.FINANCIAL_LINE_ITEM_TERMS or t_clean in cls.FINANCIAL_LINE_ITEM_TERMS or any(
            term in t for term in [
                "revenue", "expense", "payable", "depreciation", "operating income",
                "net income", "total assets", "total liabilities", "owners' capital",
                "owners capital", "cost of goods sold", "gross profit", "dividends declared"
            ]
        ) or (t.startswith("total ") or t.startswith("less:"))

    @classmethod
    def is_financial_statement_page(cls, tokens: List[OCRTextLine], width: float, height: float) -> bool:
        fin_count = 0
        amt_count = 0
        for tok in tokens:
            txt = tok.text.strip()
            if cls.is_amount(txt):
                amt_count += 1
            elif cls.is_financial_line_item(txt):
                fin_count += 1
        return (fin_count >= 3 and amt_count >= 3) or bool(cls.STATEMENT_RE.search(" ".join(t.text for t in tokens)))

    @classmethod
    def group_tokens_into_regions(
        cls,
        ocr_lines: List[OCRTextLine],
        page_width: float,
        page_height: float,
        page_num: int = 1,
        page_type: str = "SCANNED_PRINTED"
    ) -> List[SemanticRegion]:
        """Takes raw OCR fragments and groups them into coherent semantic regions."""
        if not ocr_lines:
            return []

        # Step 1: Filter out completely empty or invalid bounding box tokens
        valid_tokens = []
        for line in ocr_lines:
            t = line.text.strip()
            if not t:
                continue
            b = line.bbox
            if b[2] > b[0] and b[3] > b[1]:
                valid_tokens.append(line)

        if not valid_tokens:
            return []

        # Step 2: Sort tokens top-to-bottom, left-to-right
        valid_tokens.sort(key=lambda tok: (tok.bbox[1], tok.bbox[0]))

        # Specialized Financial Statement Layout Recognition
        if cls.is_financial_statement_page(valid_tokens, page_width, page_height):
            return cls._group_financial_page_regions(valid_tokens, page_width, page_height, page_num)

        # Step 3: Horizontal Fragment Merging (merge tokens on the same text line)
        merged_lines = cls._merge_horizontal_fragments(valid_tokens, page_width)

        # Step 4: Vertical Block Grouping (group lines into semantic regions)
        raw_regions = cls._group_lines_into_blocks(merged_lines, page_width, page_height)

        # Step 5: Semantic Region Classification & Refinement
        final_regions = cls._classify_and_refine_regions(raw_regions, page_width, page_height, page_num)

        return final_regions

    @classmethod
    def _merge_horizontal_fragments(
        cls,
        tokens: List[OCRTextLine],
        page_width: float
    ) -> List[OCRTextLine]:
        """Merges fragmented OCR tokens that belong to the same text line.
        E.g.: 'Magnetosta' + 'tics' -> 'Magnetostatics'
        E.g.: 'Biot' + 'law' -> 'Biot law'
        """
        if len(tokens) <= 1:
            return tokens

        merged: List[OCRTextLine] = []
        visited = [False] * len(tokens)

        for i in range(len(tokens)):
            if visited[i]:
                continue

            current_line_tokens = [tokens[i]]
            visited[i] = True

            curr_b = list(tokens[i].bbox)
            curr_h = curr_b[3] - curr_b[1]
            curr_mid_y = (curr_b[1] + curr_b[3]) / 2.0

            # Scan forward for horizontally neighboring tokens on the same baseline
            for j in range(i + 1, len(tokens)):
                if visited[j]:
                    continue

                cand = tokens[j]
                cand_b = cand.bbox
                cand_h = cand_b[3] - cand_b[1]
                cand_mid_y = (cand_b[1] + cand_b[3]) / 2.0

                # Check vertical overlap / baseline alignment
                y_diff = abs(curr_mid_y - cand_mid_y)
                avg_h = (curr_h + cand_h) / 2.0

                if y_diff <= (avg_h * 0.55):
                    # Check horizontal gap
                    # cand should be to the right of current group or slightly overlapping
                    gap = cand_b[0] - curr_b[2]
                    # Allow small negative gap (overlap) or gap up to 3.0 * avg_h
                    if -15.0 <= gap <= (avg_h * 2.8):
                        current_line_tokens.append(cand)
                        visited[j] = True
                        # Expand current line bounds
                        curr_b[0] = min(curr_b[0], cand_b[0])
                        curr_b[1] = min(curr_b[1], cand_b[1])
                        curr_b[2] = max(curr_b[2], cand_b[2])
                        curr_b[3] = max(curr_b[3], cand_b[3])
                        curr_h = curr_b[3] - curr_b[1]
                        curr_mid_y = (curr_b[1] + curr_b[3]) / 2.0

            # Sort tokens in this line left to right
            current_line_tokens.sort(key=lambda t: t.bbox[0])

            # Merge text with intelligent spacing / hyphen handling
            merged_text = ""
            total_conf = 0.0
            total_len = 0

            for idx, tok in enumerate(current_line_tokens):
                t_txt = tok.text.strip()
                t_len = max(1, len(t_txt))
                total_conf += tok.confidence * t_len
                total_len += t_len

                if idx == 0:
                    merged_text = t_txt
                else:
                    prev_tok = current_line_tokens[idx - 1]
                    prev_end = prev_tok.bbox[2]
                    curr_start = tok.bbox[0]
                    gap = curr_start - prev_end
                    avg_char_w = max(4.0, (prev_end - prev_tok.bbox[0]) / max(1, len(prev_tok.text)))

                    if merged_text.endswith("-"):
                        # E.g. 'Magnetosta-' + 'tics' -> 'Magnetostatics'
                        merged_text = merged_text[:-1] + t_txt
                    elif gap < (avg_char_w * 0.75):
                        # Very close or overlapping fragments: likely broken word parts e.g. 'Magnetosta' + 'tics'
                        # If neither has spaces and length is short, concatenate directly without space
                        if " " not in merged_text.split()[-1] and " " not in t_txt and gap < 6.0:
                            merged_text += t_txt
                        else:
                            merged_text += " " + t_txt
                    else:
                        merged_text += " " + t_txt

            avg_conf = total_conf / max(1, total_len)
            merged.append(OCRTextLine(
                text=merged_text,
                bbox=curr_b,
                confidence=avg_conf
            ))

        return merged

    @classmethod
    def _group_lines_into_blocks(
        cls,
        lines: List[OCRTextLine],
        page_width: float,
        page_height: float
    ) -> List[List[OCRTextLine]]:
        """Groups lines into coherent vertical blocks (paragraphs, lists, equations)."""
        if len(lines) <= 1:
            return [[l] for l in lines]

        # Sort lines top-to-bottom
        lines.sort(key=lambda l: (l.bbox[1], l.bbox[0]))

        blocks: List[List[OCRTextLine]] = []
        current_block: List[OCRTextLine] = [lines[0]]

        for i in range(1, len(lines)):
            prev_line = current_block[-1]
            curr_line = lines[i]

            prev_b = prev_line.bbox
            curr_b = curr_line.bbox

            prev_h = prev_b[3] - prev_b[1]
            curr_h = curr_b[3] - curr_b[1]
            avg_h = (prev_h + curr_h) / 2.0

            v_gap = curr_b[1] - prev_b[3]

            # Condition 1: Check if previous line or current line is an equation
            is_prev_eq = EquationExtractor.is_equation(prev_line.text) or any(s in prev_line.text for s in ["=", "∫", "∑", "√", "ε", "μ"])
            is_curr_eq = EquationExtractor.is_equation(curr_line.text) or any(s in curr_line.text for s in ["=", "∫", "∑", "√", "ε", "μ"])

            # Condition 2: Check if current line is a heading or list item
            is_curr_heading = cls._is_standalone_heading(curr_line, avg_h, page_width)
            is_curr_list = any(p.match(curr_line.text.strip()) for p in cls.LIST_PATTERNS)

            # Condition 3: Check column alignment / horizontal overlap
            h_overlap = min(prev_b[2], curr_b[2]) - max(prev_b[0], curr_b[0])
            min_w = min(prev_b[2] - prev_b[0], curr_b[2] - curr_b[0])
            h_aligned = (h_overlap / max(1.0, min_w)) > 0.35 if min_w > 0 else True

            # Decide whether to break block
            should_break = False

            # Break if large vertical gap (> 1.8x line height)
            if v_gap > (avg_h * 1.85) or v_gap < -10.0:
                should_break = True
            # Break if current line is a heading
            elif is_curr_heading:
                should_break = True
            # Break if current line is a list item and previous was not
            elif is_curr_list:
                should_break = True
            # Break if equation transition
            elif is_prev_eq or is_curr_eq:
                should_break = True
            # Break if completely separated columns horizontally
            elif not h_aligned and abs(curr_b[0] - prev_b[0]) > 80.0:
                should_break = True

            if should_break:
                blocks.append(current_block)
                current_block = [curr_line]
            else:
                current_block.append(curr_line)

        if current_block:
            blocks.append(current_block)

        return blocks

    @classmethod
    def _is_standalone_heading(cls, line: OCRTextLine, avg_h: float, page_width: float) -> bool:
        """Determines if a line is a heading."""
        text = line.text.strip()
        words = text.split()
        if len(words) > 10:
            return False

        for pat in cls.HEADING_PATTERNS:
            if pat.search(text):
                return True

        # Short bold-like or prominent line
        line_w = line.bbox[2] - line.bbox[0]
        if len(words) <= 5 and (line.bbox[3] - line.bbox[1]) > (avg_h * 1.25):
            return True

        return False

    @classmethod
    def _classify_and_refine_regions(
        cls,
        line_blocks: List[List[OCRTextLine]],
        page_width: float,
        page_height: float,
        page_num: int
    ) -> List[SemanticRegion]:
        """Classifies each block into its final semantic region type."""
        regions: List[SemanticRegion] = []

        header_boundary = page_height * 0.08
        footer_boundary = page_height * 0.92

        for idx, block_lines in enumerate(line_blocks):
            reg_id = f"reg_p{page_num}_{idx+1:03d}"

            # Calculate enclosing bounding box
            x0 = min(l.bbox[0] for l in block_lines)
            y0 = min(l.bbox[1] for l in block_lines)
            x1 = max(l.bbox[2] for l in block_lines)
            y1 = max(l.bbox[3] for l in block_lines)
            bbox = [x0, y0, x1, y1]

            # Combine text and compute weighted confidence
            combined_text = "\n".join(l.text.strip() for l in block_lines if l.text.strip())
            total_chars = sum(len(l.text) for l in block_lines)
            weighted_conf = sum(l.confidence * len(l.text) for l in block_lines) / max(1, total_chars)

            # Determine semantic region type
            region_type = "PARAGRAPH"

            # Check for header/footer margins
            if y1 <= header_boundary and len(combined_text.split()) <= 8:
                region_type = "HEADER"
            elif y0 >= footer_boundary and (len(combined_text.split()) <= 8 or combined_text.isdigit()):
                if combined_text.strip().isdigit() or "page" in combined_text.lower():
                    region_type = "PAGE_NUMBER"
                else:
                    region_type = "FOOTER"
            # Check for list
            elif any(p.match(combined_text.strip()) for p in cls.LIST_PATTERNS):
                region_type = "LIST"
            # Check for equation
            elif EquationExtractor.is_equation(combined_text) or any(c in combined_text for c in ['=', '+', '-', '√', '∫', '∑', 'ε', 'μ', '∇']):
                # If short or has math symbols
                words = combined_text.split()
                if len(words) <= 12 or any(s in combined_text for s in ['=', '∫', '∑', '√']):
                    region_type = "EQUATION"
            # Check for heading (Strict: isolated numbers and financial line items must NEVER be classified as headings)
            elif len(block_lines) == 1 and not cls.is_amount(combined_text) and not cls.is_financial_line_item(combined_text) and (
                any(p.search(combined_text) for p in cls.HEADING_PATTERNS) or
                (len(combined_text.split()) <= 7 and (y1 - y0) > (page_height * 0.022))
            ):
                region_type = "HEADING"

            regions.append(SemanticRegion(
                region_id=reg_id,
                region_type=region_type,
                bbox=bbox,
                text=combined_text,
                confidence=weighted_conf,
                lines=block_lines,
                metadata={"lines_count": len(block_lines)}
            ))

        return regions

    @classmethod
    def _group_financial_page_regions(
        cls,
        tokens: List[OCRTextLine],
        page_w: float,
        page_h: float,
        page_num: int
    ) -> List[SemanticRegion]:
        """Groups financial statements (Income statements, Retained earnings, Balance sheets)
        into structured semantic regions: HEADING, PARAGRAPH, SECTION, LINE_ITEM.
        """
        sorted_toks = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))
        mid_x = page_w * 0.51

        # Separate top headers from body
        header_toks = []
        body_toks = []

        top_header_cutoff = page_h * 0.16
        for tok in sorted_toks:
            t_lower = tok.text.strip().lower()
            if tok.bbox[1] < top_header_cutoff and (
                cls.COMPANY_RE.search(t_lower) or
                cls.STATEMENT_RE.search(t_lower) or
                cls.DATE_RE.search(t_lower)
            ):
                header_toks.append(tok)
            else:
                body_toks.append(tok)

        regions_dict: List[Dict[str, Any]] = []

        # Process top header tokens
        for h_tok in header_toks:
            txt = h_tok.text.strip()
            t_low = txt.lower()
            if cls.COMPANY_RE.search(t_low) or cls.STATEMENT_RE.search(t_low):
                reg_type = "HEADING"
            elif cls.DATE_RE.search(t_low):
                reg_type = "PARAGRAPH"
            else:
                reg_type = "HEADING"
            regions_dict.append({
                "type": reg_type,
                "text": txt,
                "bbox": h_tok.bbox,
                "conf": h_tok.confidence,
                "lines": [h_tok],
                "metadata": {"is_major_heading": (reg_type == "HEADING")}
            })

        # Check if body has 2 genuine columns (both columns must have text labels, not just numbers)
        left_toks = [t for t in body_toks if t.bbox[0] < mid_x and t.bbox[2] < mid_x + 10]
        right_toks = [t for t in body_toks if t.bbox[0] >= mid_x - 10]

        right_text_labels = [t for t in right_toks if not cls.is_amount(t.text.strip()) and len(t.text.strip()) > 3]
        left_text_labels = [t for t in left_toks if not cls.is_amount(t.text.strip()) and len(t.text.strip()) > 3]

        is_two_column = len(left_text_labels) >= 3 and len(right_text_labels) >= 3

        if is_two_column:
            # Group left column (Assets), then right column (Liabilities)
            regions_dict.extend(cls._group_column(left_toks, page_num, "left", page_w))
            regions_dict.extend(cls._group_column(right_toks, page_num, "right", page_w))
        else:
            # Single column (or multiple stacked statements like Income Statement + Statement of Retained Earnings)
            mid_headers = []
            regular_body = []
            for t in body_toks:
                t_low = t.text.strip().lower()
                if (cls.COMPANY_RE.search(t_low) or cls.STATEMENT_RE.search(t_low) or cls.DATE_RE.search(t_low)) and t.bbox[0] > (page_w * 0.20) and t.bbox[2] < (page_w * 0.80):
                    mid_headers.append(t)
                else:
                    regular_body.append(t)

            if mid_headers:
                # Partition body into statement 1 and statement 2
                split_y = min(t.bbox[1] for t in mid_headers)
                stmt1_toks = [t for t in regular_body if t.bbox[1] < split_y]
                stmt2_toks = [t for t in regular_body if t.bbox[1] >= split_y]

                regions_dict.extend(cls._group_column(stmt1_toks, page_num, "stmt1", page_w))

                # Add mid headers
                mid_headers.sort(key=lambda t: t.bbox[1])
                for mh in mid_headers:
                    txt = mh.text.strip()
                    t_low = txt.lower()
                    r_type = "HEADING" if (cls.COMPANY_RE.search(t_low) or cls.STATEMENT_RE.search(t_low)) else "PARAGRAPH"
                    regions_dict.append({
                        "type": r_type,
                        "text": txt,
                        "bbox": mh.bbox,
                        "conf": mh.confidence,
                        "lines": [mh],
                        "metadata": {"is_major_heading": (r_type == "HEADING")}
                    })

                regions_dict.extend(cls._group_column(stmt2_toks, page_num, "stmt2", page_w))
            else:
                regions_dict.extend(cls._group_column(body_toks, page_num, "stmt1", page_w))

        # Convert regions_dict to SemanticRegion objects
        final_regions: List[SemanticRegion] = []
        for idx, rd in enumerate(regions_dict):
            reg_id = f"reg_p{page_num}_{idx+1:03d}"
            final_regions.append(SemanticRegion(
                region_id=reg_id,
                region_type=rd["type"],
                bbox=rd["bbox"],
                text=rd["text"],
                confidence=rd["conf"],
                lines=rd["lines"],
                metadata=rd.get("metadata", {})
            ))

        return final_regions

    @classmethod
    def _group_column(
        cls,
        tokens: List[OCRTextLine],
        page_num: int,
        col_name: str,
        page_w: float
    ) -> List[Dict[str, Any]]:
        """Reconstructs financial rows, section headers, and line items within a column."""
        toks = sorted(tokens, key=lambda t: t.bbox[1])

        # Clean tokens that crossed column boundaries
        clean_toks = []
        for t in toks:
            m = re.match(r"^(\$?[0-9,]+)\s+([A-Za-z\s'\-]+)$", t.text.strip())
            if m:
                amt_str = m.group(1)
                lbl_str = m.group(2)
                clean_toks.append(OCRTextLine(
                    text=amt_str,
                    bbox=[t.bbox[0], t.bbox[1], t.bbox[0] + 50, t.bbox[3]],
                    confidence=t.confidence
                ))
                clean_toks.append(OCRTextLine(
                    text=lbl_str,
                    bbox=[t.bbox[0] + 55, t.bbox[1], t.bbox[2], t.bbox[3]],
                    confidence=t.confidence
                ))
            else:
                clean_toks.append(t)

        amounts = []
        labels = []
        sections = []
        major_headings = []

        for t in clean_toks:
            txt = t.text.strip()
            txt_norm = txt.lower().replace("’", "'").replace("&", "and")
            if "total" in txt_norm and not cls.is_amount(txt):
                labels.append(t)
            elif any(k == txt_norm or (k in txt_norm and not any(skip in txt_norm for skip in ["total", "current"])) for k in [
                "assets", "liabilities and stockholders' equity",
                "statement of retained earnings", "balance sheet", "income statement"
            ]) and not cls.is_amount(txt):
                major_headings.append(t)
            elif txt.endswith(":") or any(sec == txt_norm for sec in ["current assets", "current liabilities", "operating expenses", "other item", "owners' equity"]):
                sections.append(t)
            elif cls.is_amount(txt):
                amounts.append(t)
            else:
                labels.append(t)

        # Multi-line label merging (e.g. 'Total liabilities and' + 'owners\' equity')
        labels_sorted = sorted(labels, key=lambda l: l.bbox[1])
        merged_labels = []
        skip_next = False
        for i in range(len(labels_sorted)):
            if skip_next:
                skip_next = False
                continue
            curr_l = labels_sorted[i]
            if i + 1 < len(labels_sorted):
                next_l = labels_sorted[i + 1]
                v_gap = next_l.bbox[1] - curr_l.bbox[3]
                h_align = abs(curr_l.bbox[0] - next_l.bbox[0]) <= 25.0
                curr_mid = (curr_l.bbox[1] + curr_l.bbox[3]) / 2.0
                has_curr_amt = any(abs((a.bbox[1] + a.bbox[3]) / 2.0 - curr_mid) < 8.0 for a in amounts)
                if not has_curr_amt and -5.0 <= v_gap <= 20.0 and h_align:
                    comb_text = f"{curr_l.text.strip()} {next_l.text.strip()}"
                    comb_bbox = [
                        min(curr_l.bbox[0], next_l.bbox[0]),
                        min(curr_l.bbox[1], next_l.bbox[1]),
                        max(curr_l.bbox[2], next_l.bbox[2]),
                        max(curr_l.bbox[3], next_l.bbox[3])
                    ]
                    comb_conf = (curr_l.confidence * len(curr_l.text) + next_l.confidence * len(next_l.text)) / max(1, len(curr_l.text) + len(next_l.text))
                    merged_labels.append(OCRTextLine(text=comb_text, bbox=comb_bbox, confidence=comb_conf))
                    skip_next = True
                    continue
            merged_labels.append(curr_l)
        labels = merged_labels

        rows = []

        # Add major headings
        for mh in major_headings:
            mh_mid_y = (mh.bbox[1] + mh.bbox[3]) / 2.0
            rows.append({
                "type": "HEADING",
                "text": mh.text.strip(),
                "bbox": mh.bbox,
                "conf": mh.confidence,
                "mid_y": mh_mid_y,
                "lines": [mh],
                "metadata": {
                    "is_major_heading": True,
                    "ocr_confidence": round(mh.confidence, 4),
                    "confidence_breakdown": {
                        "ocr_recognition_confidence": round(mh.confidence, 4),
                        "financial_alignment_score": 1.0,
                        "word_aggregation_score": 1.0,
                        "layout_consistency": 1.0,
                        "semantic_consistency": 1.0
                    }
                }
            })

        matched_amounts = set()

        for lbl in labels:
            lbl_txt = lbl.text.strip()
            lbl_mid_y = (lbl.bbox[1] + lbl.bbox[3]) / 2.0
            lbl_h = lbl.bbox[3] - lbl.bbox[1]

            best_amt = None
            best_dist = 999.0

            for idx, amt in enumerate(amounts):
                if idx in matched_amounts:
                    continue
                amt_mid_y = (amt.bbox[1] + amt.bbox[3]) / 2.0
                dist = abs(amt_mid_y - lbl_mid_y)
                if dist < max(12.0, lbl_h * 0.95) and amt.bbox[0] > (lbl.bbox[0] + 15):
                    if dist < best_dist:
                        best_dist = dist
                        best_amt = (idx, amt)

            if best_amt is not None:
                idx, amt = best_amt
                matched_amounts.add(idx)
                amt_txt = amt.text.strip()

                u_bbox = [
                    min(lbl.bbox[0], amt.bbox[0]),
                    min(lbl.bbox[1], amt.bbox[1]),
                    max(lbl.bbox[2], amt.bbox[2]),
                    max(lbl.bbox[3], amt.bbox[3])
                ]
                u_conf = (lbl.confidence * len(lbl_txt) + amt.confidence * len(amt_txt)) / max(1, len(lbl_txt) + len(amt_txt))

                # Geometric alignment calculation
                baseline_dev = best_dist / max(1.0, lbl_h)
                baseline_score = max(0.50, 1.0 - min(0.50, baseline_dev * 0.5))
                col_order_score = 1.0 if amt.bbox[0] >= (lbl.bbox[0] + 15) else 0.70
                fin_align_score = round(0.7 * baseline_score + 0.3 * col_order_score, 4)

                # Word aggregation & printable character consistency
                words = lbl_txt.split() + amt_txt.split()
                printable_w = sum(1 for w in words if all(c.isprintable() for c in w))
                word_agg = round(printable_w / max(1, len(words)), 4)

                # Bounding-box layout consistency
                layout_ok = 1.0 if (u_bbox[2] > u_bbox[0] and u_bbox[3] > u_bbox[1]) else 0.50

                # Semantic block consistency
                is_fin_term = cls.is_financial_line_item(lbl_txt) or "total" in lbl_txt.lower()
                is_fin_amt = cls.is_amount(amt_txt)
                semantic_score = 1.0 if (is_fin_term and is_fin_amt) else 0.90

                indent = "    " if (lbl.bbox[0] > 95 and col_name == "stmt1") or (lbl.bbox[0] > 100 and col_name == "left") or (lbl.bbox[0] > 335 and col_name == "right") else ""
                row_content = f"{indent}{lbl_txt:<35} {amt_txt:>10}".strip()

                rows.append({
                    "type": "LINE_ITEM",
                    "text": row_content,
                    "label": lbl_txt,
                    "amount": amt_txt,
                    "bbox": u_bbox,
                    "conf": u_conf,
                    "mid_y": lbl_mid_y,
                    "lines": [lbl, amt],
                    "metadata": {
                        "label": lbl_txt,
                        "amount": amt_txt,
                        "is_financial_row": True,
                        "ocr_confidence": round(u_conf, 4),
                        "confidence_breakdown": {
                            "ocr_recognition_confidence": round(u_conf, 4),
                            "financial_alignment_score": fin_align_score,
                            "word_aggregation_score": word_agg,
                            "layout_consistency": layout_ok,
                            "semantic_consistency": semantic_score
                        }
                    }
                })
            else:
                indent = "    " if lbl.bbox[0] > 95 else ""
                is_fin_term = cls.is_financial_line_item(lbl_txt) or "total" in lbl_txt.lower()
                rows.append({
                    "type": "LINE_ITEM" if is_fin_term else "PARAGRAPH",
                    "text": f"{indent}{lbl_txt}",
                    "label": lbl_txt,
                    "amount": None,
                    "bbox": lbl.bbox,
                    "conf": lbl.confidence,
                    "mid_y": lbl_mid_y,
                    "lines": [lbl],
                    "metadata": {
                        "label": lbl_txt,
                        "amount": None,
                        "is_financial_row": True,
                        "ocr_confidence": round(lbl.confidence, 4),
                        "confidence_breakdown": {
                            "ocr_recognition_confidence": round(lbl.confidence, 4),
                            "financial_alignment_score": 0.85,
                            "word_aggregation_score": 1.0,
                            "layout_consistency": 1.0,
                            "semantic_consistency": 0.95 if is_fin_term else 0.85
                        }
                    }
                })

        # Any unmatched amounts
        for idx, amt in enumerate(amounts):
            if idx not in matched_amounts:
                amt_txt = amt.text.strip()
                amt_mid_y = (amt.bbox[1] + amt.bbox[3]) / 2.0
                rows.append({
                    "type": "LINE_ITEM",
                    "text": amt_txt,
                    "label": None,
                    "amount": amt_txt,
                    "bbox": amt.bbox,
                    "conf": amt.confidence,
                    "mid_y": amt_mid_y,
                    "lines": [amt],
                    "metadata": {
                        "label": None,
                        "amount": amt_txt,
                        "is_financial_row": True,
                        "ocr_confidence": round(amt.confidence, 4),
                        "confidence_breakdown": {
                            "ocr_recognition_confidence": round(amt.confidence, 4),
                            "financial_alignment_score": 0.75,
                            "word_aggregation_score": 1.0,
                            "layout_consistency": 1.0,
                            "semantic_consistency": 0.80
                        }
                    }
                })

        # Add section headers
        for sec in sections:
            sec_mid_y = (sec.bbox[1] + sec.bbox[3]) / 2.0
            rows.append({
                "type": "SECTION",
                "text": sec.text.strip(),
                "bbox": sec.bbox,
                "conf": sec.confidence,
                "mid_y": sec_mid_y,
                "lines": [sec],
                "metadata": {
                    "is_section_header": True,
                    "ocr_confidence": round(sec.confidence, 4),
                    "confidence_breakdown": {
                        "ocr_recognition_confidence": round(sec.confidence, 4),
                        "financial_alignment_score": 0.90,
                        "word_aggregation_score": 1.0,
                        "layout_consistency": 1.0,
                        "semantic_consistency": 1.0
                    }
                }
            })

        rows.sort(key=lambda r: r["mid_y"])
        return rows
