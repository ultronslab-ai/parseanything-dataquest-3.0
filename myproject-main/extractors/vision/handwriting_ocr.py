"""Handwriting OCR using Vision LLM."""
import os
from PIL import Image

class HandwritingOCRExtractor:
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
        prompt = """Extract the handwritten text from this image crop exactly as written.
Preserve layout and structure. Do not hallucinate.
Return ONLY the extracted text."""
        try:
            response = self.model.generate_content([image_crop, prompt])
            return response.text.strip()
        except Exception:
            return ""
