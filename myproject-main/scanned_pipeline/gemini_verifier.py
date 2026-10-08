"""Gemini Verification & Enhancement Service.

Adds Google Gemini Multimodal Vision API as an additive AI verification layer.
PRIMARY: Existing local OCR
ADDITIONAL: Gemini Vision for low-confidence, handwriting, and ambiguous equations.
FALLBACK: Original local OCR retained if Gemini fails (network, key, rate limit, timeout).

Conforms strictly to Universal IR Schema v1.0, preserving all provenance,
bounding boxes, reading order, and whole-document JSON structure.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

# Ensure .env is loaded
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=False)
except Exception:
    pass

# Try importing the official google.genai SDK
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI_SDK = True
except Exception:
    HAS_GENAI_SDK = False


class GeminiVerificationService:
    """Additive AI verification layer powered by the official Gemini API SDK."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GeminiVerificationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._load_config()
        self.client = None
        self._init_client()
        self._initialized = True

    def _load_config(self):
        """Loads configuration from environment variables."""
        self.api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        self.model_name = (os.getenv("GEMINI_MODEL") or "").strip() or "gemini-3.5-flash-lite"
        try:
            self.review_threshold = float(os.getenv("GEMINI_REVIEW_THRESHOLD", "0.70"))
        except (ValueError, TypeError):
            self.review_threshold = 0.70

    def _init_client(self):
        """Initializes Gemini client safely without raising exceptions."""
        if not HAS_GENAI_SDK or not self.api_key:
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[GeminiVerifier] Warning: Client initialization failed: {type(e).__name__}")
            self.client = None

    def is_available(self) -> bool:
        """Returns True if Gemini SDK and API key are ready."""
        # Refresh API key if changed dynamically
        current_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if current_key and current_key != self.api_key:
            self.api_key = current_key
            self._init_client()
        return bool(self.client and self.api_key)

    def verify_and_enhance_block(
        self,
        crop: Optional[Image.Image],
        ocr_text: str,
        bbox: List[float],
        current_type: str = "paragraph",
        current_conf: float = 0.50,
        page_num: int = 1,
        is_handwritten: bool = False,
        region_type: str = "TEXT"
    ) -> Dict[str, Any]:
        """Verifies and enhances a difficult, low-confidence, or ambiguous block.
        
        Zero-Crash Guarantee:
        If Gemini encounters any error (missing key, network, rate limit, timeout, malformed output),
        returns graceful fallback containing the original OCR result.
        """
        clean_ocr = (ocr_text or "").strip()
        fallback_res = {
            "verified": False,
            "type": current_type,
            "text": clean_ocr,
            "latex": None,
            "confidence": current_conf,
            "extraction_method": "ocr_local",
            "model": None,
            "requires_review": current_conf < self.review_threshold,
            "is_equation": False,
            "original_ocr": clean_ocr,
            "warnings": []
        }

        if not self.is_available() or crop is None:
            if not self.is_available():
                fallback_res["warnings"].append("Gemini verification unavailable; used local OCR fallback.")
            return fallback_res

        # Optimize crop dimensions if too tiny or too huge
        w, h = crop.size
        if w < 10 or h < 10:
            return fallback_res

        prompt = f"""You are a high-precision document verification assistant.
An on-device OCR engine extracted the following from this image crop:
Text: '{clean_ocr}'
Assigned Type: '{current_type}' (Region: '{region_type}')
Is Handwritten Page: {is_handwritten}

Your job is to verify and correct this extraction:
1. Examine the image carefully.
2. Determine if this region is:
   - "equation": ONLY if it contains an actual mathematical relation, equation, or formula with math operators (=, -, +, Greek symbols, fractions, powers, integrals, etc.).
     CRITICAL: Normal handwritten words, subject headings, or names of formulas (e.g. "Maxwell's equations", "Biot Savart law", "current density", "(law - )") are NOT equations.
   - "heading": Section header, document title, or numbered header.
   - "paragraph": Standard prose or handwritten sentence/phrase.
   - "list_item": Numbered, bulleted, or dashed item.
   - "table": Tabular cells or rows.
3. If it is an equation, provide standard valid LaTeX representation in "latex" (without $$ or code fences).
4. If it is text/heading/paragraph, provide the clean corrected text in "text".
5. Confidence: Your confidence score between 0.0 and 1.0.
6. Compare with the original OCR text:
   - If the original OCR was already accurate, set "is_improvement": false.
   - If the original OCR was garbled, incomplete, or significantly corrected, set "is_improvement": true.

Return ONLY a valid JSON object matching this schema:
{{
  "is_equation": boolean,
  "type": "equation" | "heading" | "paragraph" | "list_item" | "table",
  "text": string,
  "latex": string or null,
  "confidence": float,
  "is_improvement": boolean,
  "requires_review": boolean
}}
Do NOT include markdown formatting, backticks, or preamble. Return raw JSON only."""

        try:
            resp = self._call_model(crop, prompt)
            if not resp:
                fallback_res["warnings"].append("Gemini returned empty response; retained local OCR.")
                return fallback_res

            parsed = self._parse_json_response(resp)
            if not parsed:
                fallback_res["warnings"].append("Gemini response parsing failed; retained local OCR.")
                return fallback_res

            is_eq = bool(parsed.get("is_equation", False))
            resp_type = parsed.get("type", current_type).lower()
            resp_text = (parsed.get("text") or "").strip()
            resp_latex = (parsed.get("latex") or "").strip() if is_eq else None
            gem_conf_raw = float(parsed.get("confidence", current_conf))
            is_improvement = bool(parsed.get("is_improvement", False))

            # Equation validation safeguard: prevent hallucinated equations
            if is_eq and resp_latex:
                valid_math = self._validate_math_expression(resp_latex)
                if not valid_math:
                    # Downgrade false equation to paragraph or heading
                    is_eq = False
                    resp_type = "heading" if len(resp_text.split()) <= 6 else "paragraph"
                    resp_latex = None

            final_type = "equation" if is_eq else resp_type
            if final_type not in ("equation", "heading", "paragraph", "list_item", "table", "figure", "chart"):
                final_type = current_type

            final_text = resp_latex if (is_eq and resp_latex) else (resp_text or clean_ocr)

            # Evaluate agreement between OCR text and Gemini verification text
            import difflib
            ocr_norm = re.sub(r"\s+", " ", clean_ocr).strip().lower()
            gem_norm = re.sub(r"\s+", " ", resp_text).strip().lower()

            # Sequence similarity ratio
            agreement_ratio = difflib.SequenceMatcher(None, ocr_norm, gem_norm).ratio() if (ocr_norm and gem_norm) else 0.0

            # Number comparison: verify whether financial amounts / digits agree
            ocr_nums = re.findall(r"\d+", ocr_norm)
            gem_nums = re.findall(r"\d+", gem_norm)
            nums_agree = (ocr_nums == gem_nums)

            # Determine confidence and review flag based on agreement
            if (ocr_norm == gem_norm) or (agreement_ratio >= 0.85 and nums_agree):
                # OCR and Gemini agree (e.g. 'Depreciation expense 100' == 'Depreciation expense 100')
                # Dual verification consensus: increase confidence appropriately
                base_c = max(current_conf, gem_conf_raw, 0.85)
                conf = min(0.98, base_c + (1.0 - base_c) * 0.40 * agreement_ratio)
                requires_review = False
                verification_agreement = 1.0
            elif not nums_agree or agreement_ratio < 0.65:
                # OCR and Gemini disagree: keep confidence low and mark Needs Review
                conf = min(current_conf, gem_conf_raw, 0.55)
                requires_review = True
                verification_agreement = 0.45
            else:
                # Partial agreement
                conf = min(current_conf, gem_conf_raw)
                requires_review = (conf < self.review_threshold)
                verification_agreement = round(agreement_ratio, 4)

            conf = max(0.0, min(1.0, round(conf, 4)))

            # Determine extraction_method per Requirement 6:
            # - If OCR is reliable: "ocr_local"
            # - If Gemini improves/verifies OCR: "ocr_local+gemini_verification"
            # - If Gemini provides the better result: "gemini_vision"
            if is_improvement and (final_text != clean_ocr or is_eq):
                method = "gemini_vision"
            else:
                method = "ocr_local+gemini_verification"

            warnings_list = []
            if not nums_agree and ocr_nums and gem_nums:
                warnings_list.append("Numeric value discrepancy between OCR and Gemini verification.")

            return {
                "verified": True,
                "type": final_type,
                "text": final_text,
                "latex": resp_latex,
                "confidence": conf,
                "extraction_method": method,
                "model": self.model_name,
                "requires_review": requires_review,
                "is_equation": is_eq,
                "original_ocr": clean_ocr,
                "verification_agreement": verification_agreement,
                "warnings": warnings_list
            }

        except Exception as e:
            # Absolute fail-safe: never crash
            print(f"[GeminiVerifier] Exception during block verification: {type(e).__name__} - {e}")
            fallback_res["warnings"].append(f"Gemini verification error ({type(e).__name__}); retained local OCR.")
            return fallback_res

    def verify_equation_crop(
        self,
        crop: Image.Image,
        raw_text: str,
        initial_latex: str,
        bbox: List[float],
        page_num: int = 1
    ) -> Dict[str, Any]:
        """Dedicated math verification adhering to Requirement 5.
        Prevents normal handwritten text from being classified as an equation.
        """
        clean_raw = (raw_text or "").strip()
        fallback = {
            "is_equation": bool(self._validate_math_expression(initial_latex)),
            "latex": initial_latex,
            "text": clean_raw,
            "confidence": 0.72,
            "method": "ocr_local",
            "model": None,
            "requires_review": True
        }

        if not self.is_available() or crop is None:
            return fallback

        prompt = f"""You are a specialized mathematical OCR verification system.
Initial OCR extracted: '{clean_raw}'
Proposed formula: '{initial_latex}'

Inspect the image crop:
1. Is this truly an equation, formula, or mathematical relationship?
   CRITICAL: If it is merely regular text, a law name (like 'Biot Savart law' or 'Maxwell equations'), or notes, set is_equation=false.
2. If it IS an equation, provide the standard valid LaTeX syntax.
3. If it is NOT an equation, provide the clean text and appropriate block type.

Return ONLY valid JSON:
{{
  "is_equation": boolean,
  "latex": string or null,
  "text": string,
  "confidence": float,
  "requires_review": boolean
}}"""

        try:
            resp = self._call_model(crop, prompt)
            parsed = self._parse_json_response(resp) if resp else None
            if not parsed:
                return fallback

            is_eq = bool(parsed.get("is_equation", False))
            latex = (parsed.get("latex") or "").strip() if is_eq else None
            text = (parsed.get("text") or clean_raw).strip()
            conf = float(parsed.get("confidence", 0.75))
            requires_review = bool(parsed.get("requires_review", False)) or (conf < self.review_threshold)

            if is_eq and latex:
                if self._validate_math_expression(latex):
                    return {
                        "is_equation": True,
                        "latex": latex,
                        "text": clean_raw or text,
                        "confidence": conf,
                        "method": "ocr_local+gemini_verification",
                        "model": self.model_name,
                        "requires_review": requires_review
                    }
                else:
                    # Math expression failed validation
                    is_eq = False

            return {
                "is_equation": False,
                "latex": None,
                "text": text or clean_raw,
                "confidence": conf,
                "method": "ocr_local+gemini_verification",
                "model": self.model_name,
                "requires_review": requires_review
            }

        except Exception as e:
            print(f"[GeminiVerifier] Exception in equation verification: {e}")
            return fallback

    def _call_model(self, image: Image.Image, prompt: str) -> Optional[str]:
        """Invokes Gemini model safely with fallback to secondary model if 404/503/deprecated."""
        if not self.client:
            return None

        # Build prioritized list of active models
        models_to_try = [self.model_name]
        for fallback_m in ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        for m in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=[image, prompt]
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                err_str = str(e)
                # If 404, 503 (high demand spike), or model deprecated, fallback to next model
                if any(code in err_str for code in ["404", "503", "NOT_FOUND", "UNAVAILABLE", "no longer available", "high demand"]):
                    continue
                else:
                    raise e

        return None

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts and parses JSON object from model response string."""
        if not text:
            return None

        cleaned = text.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Try finding JSON object {...}
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None

    def _validate_math_expression(self, latex: Optional[str]) -> bool:
        """Validates that a LaTeX string contains genuine mathematical content,
        not ordinary English words or phrases.
        """
        if not latex:
            return False

        clean = latex.strip().lower()
        if len(clean) < 2:
            return False

        # Known mathematical symbols and operators
        math_indicators = [
            "=", "+", "-", "\\times", "\\cdot", "\\div", "\\frac",
            "\\sqrt", "^", "_", "\\int", "\\sum", "\\partial", "\\nabla",
            "\\epsilon", "\\mu", "\\rho", "\\sigma", "\\lambda", "\\theta",
            "\\pi", "\\omega", "\\alpha", "\\beta", "\\gamma", "\\delta",
            "\\leq", "\\geq", "\\neq", "\\approx", "\\to"
        ]

        has_math_symbol = any(sym in clean for sym in math_indicators)
        if not has_math_symbol:
            return False

        # Disqualify if it's purely letters with space and no operator
        # e.g. "density maxwell"
        if not any(c in clean for c in ["=", "+", "-", "\\", "^", "_"]):
            return False

        return True


# Global Singleton accessor
def get_gemini_verifier() -> GeminiVerificationService:
    return GeminiVerificationService()
