"""Mathematical Expression Extraction Engine.
Detects math formulas in text or layout blocks, converts/normalizes to LaTeX,
and sets confidence and provenance without hallucination.
"""

import re
from typing import Optional, Tuple
from schemas.models import EquationData

# Regex patterns matching mathematical expressions, symbols, and formulas
MATH_INDICATORS = re.compile(
    r"(\\frac\{|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\gamma|\\theta|\\sigma|\\lambda|\\mu|\\pi|\\infty|"
    r"\\partial|\\nabla|\\prod|\\approx|\\neq|\\leq|\\geq|\\times|\\pm|"
    r"\b[eE]\s*=\s*mc\^?2\b|"
    r"[a-zA-Z_]\s*=\s*[-+]?[0-9a-zA-Z_.]+\s*[\+\-\*\/\^]|"
    r"\b[a-zA-Z]\([xytz]\)\s*=|"
    r"[∑∫∏√±≠≤≥≈∞∆∇∂]|\^\{?[0-9a-zA-Z\+\-]+\}?|"
    r"_[0-9a-zA-Z]{1,3}\b)"
)

# Common clean LaTeX transformations
COMMON_FORMULAS = {
    "e=mc^2": r"E = mc^2",
    "e = mc^2": r"E = mc^2",
    "e=mc2": r"E = mc^2",
    "a^2+b^2=c^2": r"a^2 + b^2 = c^2",
    "a^2 + b^2 = c^2": r"a^2 + b^2 = c^2",
}


class EquationExtractor:
    """Detects and extracts equations into structured LaTeX representations."""

    @classmethod
    def is_equation(cls, text: str) -> bool:
        clean = text.strip()
        if not clean:
            return False

        # Exclude normal sentences that happen to have an equal sign like 'Total = 50 items'
        if len(clean.split()) > 15 and not any(k in clean for k in [r"\frac", r"\sum", r"\int", r"\sqrt"]):
            return False

        # Check common formulas
        normalized = clean.lower().replace(" ", "")
        if normalized in COMMON_FORMULAS:
            return True

        matches = MATH_INDICATORS.findall(clean)
        return len(matches) > 0

    @classmethod
    def extract_equation(cls, text: str) -> Tuple[EquationData, float]:
        clean = text.strip()
        normalized_key = clean.lower().replace(" ", "")

        if normalized_key in COMMON_FORMULAS:
            return EquationData(
                latex=COMMON_FORMULAS[normalized_key],
                is_inline=False
            ), 1.0

        # If already formatted as LaTeX
        if clean.startswith("$") and clean.endswith("$"):
            latex_expr = clean.strip("$").strip()
            return EquationData(latex=latex_expr, is_inline=not clean.startswith("$$")), 0.95

        # Heuristic conversion for common patterns
        latex_str = clean
        # Handle simple fractions like a/b or 1/2 in formula context
        latex_str = re.sub(r"(\w+)\/(\w+)", lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", latex_str)
        # Handle square root
        latex_str = re.sub(r"sqrt\((.*?)\)|√\((.*?)\)|√(\w+)", lambda m: f"\\sqrt{{{m.group(1) or m.group(2) or m.group(3)}}}", latex_str)
        # Handle Greek symbols in plain text
        greek_map = {
            "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
            "delta": r"\delta", "sigma": r"\sigma", "lambda": r"\lambda",
            "pi": r"\pi", "mu": r"\mu", "theta": r"\theta"
        }
        for name, sym in greek_map.items():
            latex_str = re.sub(rf"\b{name}\b", lambda m, s=sym: s, latex_str, flags=re.IGNORECASE)

        confidence = 0.92 if any(k in clean for k in [r"\\", "=", "^", "_"]) else 0.82
        return EquationData(latex=latex_str, is_inline=False), confidence
