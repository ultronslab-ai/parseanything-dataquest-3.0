"""Math OCR using Vision LLM."""
import os
from PIL import Image

class MathOCRExtractor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                pass

    def extract(self, image_crop: Image.Image) -> str:
        if not self.model:
            return ""
        prompt = """Extract the mathematical expression from this image crop.
Return ONLY the raw LaTeX representation. Do NOT include Markdown formatting like ```latex or $$.
Do not hallucinate. If it is unreadable, return empty.
Preserve vectors, fractions, operators, Greek letters, and indices accurately."""
        try:
            response = self.model.generate_content([image_crop, prompt])
            res = response.text.strip()
            res = res.strip("`").strip()
            if res.startswith("latex\n"):
                res = res[6:].strip()
            elif res.startswith("latex"):
                res = res[5:].strip()
            if res.startswith("$$") and res.endswith("$$"):
                res = res.strip("$").strip()
            return res
        except Exception:
            return ""
