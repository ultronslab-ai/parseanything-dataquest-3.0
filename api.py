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
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8585))
    print(f"Starting ParseAnything Universal Engine on http://{host}:{port}")
    uvicorn.run("backend.api:app", host=host, port=port, reload=False)
