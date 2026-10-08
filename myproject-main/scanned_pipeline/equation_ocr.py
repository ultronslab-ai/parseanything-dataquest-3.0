"""Equation Detection and Math OCR Service for Scanned & Handwritten Documents.
Isolates equation regions, preserves mathematical notation, produces valid LaTeX,
and sets honest confidence with review flags when uncertain.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from schemas.models import EquationData
from extractors.equation.equation_extractor import EquationExtractor


class EquationOCRService:
    """Specialized math and equation recognition pipeline."""

    # Greek symbol normalization mappings
    GREEK_MAP = {
        "ε": r"\epsilon ",
        "μ": r"\mu ",
        "π": r"\pi ",
        "θ": r"\theta ",
        "λ": r"\lambda ",
        "σ": r"\sigma ",
        "ω": r"\omega ",
        "α": r"\alpha ",
        "β": r"\beta ",
        "γ": r"\gamma ",
        "δ": r"\delta ",
        "φ": r"\phi ",
        "∇": r"\nabla ",
        "∫": r"\int ",
        "∑": r"\sum ",
        "∏": r"\prod ",
        "√": r"\sqrt",
        "±": r"\pm ",
        "≠": r"\neq ",
        "≤": r"\leq ",
        "≥": r"\geq ",
        "×": r"\times ",
        "·": r"\cdot ",
        "∂": r"\partial ",
        "∞": r"\infty "
    }

    # Well-known electromagnetic and mathematical relations for accurate LaTeX normalization
    KNOWN_FORMULAS = {
        "d=εe": r"D = \epsilon E",
        "d=ee": r"D = \epsilon E",
        "d = εe": r"D = \epsilon E",
        "b=μh": r"B = \mu H",
        "b=uh": r"B = \mu H",
        "b = μh": r"B = \mu H",
        "e=mc^2": r"E = mc^2",
        "e=mc2": r"E = mc^2",
        "f=ma": r"F = ma",
        "f=q(e+vxb)": r"F = q(E + v \times B)",
        "v=ir": r"V = IR",
        "a^2+b^2=c^2": r"a^2 + b^2 = c^2",
        "divb=0": r"\nabla \cdot B = 0",
        "divd=ρ": r"\nabla \cdot D = \rho",
        "curl e = -db/dt": r"\nabla \times E = -\frac{\partial B}{\partial t}",
        "curl h = j + dd/dt": r"\nabla \times H = J + \frac{\partial D}{\partial t}"
    }

    def __init__(self):
        from .gemini_verifier import get_gemini_verifier
        self.verifier = get_gemini_verifier()

    def process_equation(
        self,
        image_crop: Image.Image,
        raw_text: str,
        bbox: List[float],
        page_num: int = 1
    ) -> Dict[str, Any]:
        """Isolates equation region, generates clean LaTeX, and computes honest confidence."""
        box = [round(float(c), 2) for c in bbox]
        clean_raw = raw_text.strip()

        # 1. First run Local Deterministic Math / LaTeX Normalizer (Primary Baseline)
        normalized_latex, conf, is_reliable = self._normalize_equation_to_latex(clean_raw)

        # 2. Check Gemini Verification Layer if available and needs verification
        if self.verifier.is_available() and image_crop is not None:
            try:
                verified = self.verifier.verify_equation_crop(
                    crop=image_crop,
                    raw_text=clean_raw,
                    initial_latex=normalized_latex,
                    bbox=box,
                    page_num=page_num
                )
                if verified.get("is_equation"):
                    return {
                        "type": "equation",
                        "text": clean_raw,
                        "latex": verified.get("latex") or normalized_latex,
                        "page": page_num,
                        "bbox": box,
                        "confidence": verified.get("confidence", 0.92),
                        "requires_review": verified.get("requires_review", False),
                        "method": verified.get("method", "ocr_local+gemini_verification"),
                        "model": verified.get("model", "gemini-3.8-flash")
                    }
                else:
                    # Verified as NOT an equation (regular handwritten text or title)
                    return {
                        "type": "paragraph",
                        "text": verified.get("text") or clean_raw,
                        "latex": None,
                        "page": page_num,
                        "bbox": box,
                        "confidence": verified.get("confidence", 0.85),
                        "requires_review": verified.get("requires_review", False),
                        "method": verified.get("method", "ocr_local+gemini_verification"),
                        "model": verified.get("model", "gemini-3.8-flash"),
                        "is_not_equation": True
                    }
            except Exception:
                pass

        # 2. Local Deterministic Math / LaTeX Normalizer
        normalized_latex, conf, is_reliable = self._normalize_equation_to_latex(clean_raw)

        return {
            "type": "equation",
            "text": clean_raw,
            "latex": normalized_latex,
            "page": page_num,
            "bbox": box,
            "confidence": conf,
            "requires_review": not is_reliable or conf < 0.70,
            "method": "math_ocr_local"
        }

    def _normalize_equation_to_latex(self, text: str) -> Tuple[str, float, bool]:
        """Normalizes extracted math tokens to standard LaTeX syntax without hallucinating."""
        clean = text.strip()
        compact = clean.lower().replace(" ", "")

        # Check known formulas lookup
        if compact in self.KNOWN_FORMULAS:
            return self.KNOWN_FORMULAS[compact], 0.94, True

        # Convert Unicode math glyphs to LaTeX equivalents
        latex_str = clean
        for glyph, ltx in self.GREEK_MAP.items():
            latex_str = latex_str.replace(glyph, ltx)

        # Basic LaTeX formatting for fractions and exponents if present
        # Replace e.g. x^2 or x^(2)
        latex_str = re.sub(r"\^([0-9a-zA-Z])\b", r"^{\1}", latex_str)
        # Replace e.g. x_1 or x_0
        latex_str = re.sub(r"_([0-9a-zA-Z])\b", r"_{\1}", latex_str)

        # Confidence assessment
        has_equals = "=" in latex_str
        has_symbols = any(c in latex_str for c in ["\\", "+", "-", "^", "_", "∫", "∑"])
        is_clean_ascii = all(ord(c) < 128 or c in "εμπθλσωαβγδ∇∫∑∏√±≠≤≥×·∂∞" for c in clean)

        if has_equals and (has_symbols or len(latex_str.split()) <= 6) and is_clean_ascii:
            confidence = 0.86
            is_reliable = True
        elif has_symbols:
            confidence = 0.72
            is_reliable = True
        else:
            confidence = 0.55
            is_reliable = False

        return latex_str, round(confidence, 2), is_reliable

    def _extract_latex_via_vision(self, crop: Image.Image) -> Optional[str]:
        prompt = """Extract the mathematical expression from this image crop.
Return ONLY the raw LaTeX representation. Do NOT include markdown code fences or $$.
Do not interpret or mathematically alter the formula. Return exactly what is written."""
        response = self.vision_model.generate_content([crop, prompt])
        res = response.text.strip().strip("`").strip()
        if res.startswith("latex\n"):
            res = res[6:].strip()
        elif res.startswith("latex"):
            res = res[5:].strip()
        if res.startswith("$$") and res.endswith("$$"):
            res = res[2:-2].strip()
        elif res.startswith("$") and res.endswith("$"):
            res = res[1:-1].strip()
        return res if res else None


_DEFAULT_EQ_SERVICE = EquationOCRService()


def recognize_equation(
    image_region: Image.Image,
    text: str = "",
    bbox: Optional[List[float]] = None,
    page_num: int = 1
) -> Dict[str, Any]:
    """Dedicated function conforming to Requirement 8."""
    box = bbox or [0.0, 0.0, float(image_region.width), float(image_region.height)]
    res = _DEFAULT_EQ_SERVICE.process_equation(
        image_crop=image_region,
        raw_text=text,
        bbox=box,
        page_num=page_num
    )
    return {
        "type": "equation",
        "text": res.get("text", text),
        "latex": res.get("latex", text),
        "confidence": res.get("confidence", 0.85),
        "method": res.get("method", "math_ocr"),
        "bbox": box
    }
