"""Vercel Serverless Function entrypoint for ParseAnything Universal Engine."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import the FastAPI application instance
from backend.api import app
