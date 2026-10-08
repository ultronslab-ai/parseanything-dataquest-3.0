"""ParseAnything Universal API Server Entrypoint.
Delegates to backend.api.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

from backend.api import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8585))
    print(f"Starting ParseAnything Universal Engine on http://127.0.0.1:{port}")
    uvicorn.run("backend.api:app", host="127.0.0.1", port=port, reload=False)
