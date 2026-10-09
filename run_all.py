"""Start everything with one command:  python run_all.py

  - demo websites   http://localhost:8001
  - FastAPI         http://localhost:8000   (docs at /docs)
  - Streamlit UI    http://localhost:8501

Ollama must already be running (ollama serve). Press Ctrl+C to stop all.
"""
from __future__ import annotations

import subprocess
import sys
import time

PY = sys.executable
PROCS = [
    ("demo sites", [PY, "demo_sites/server.py", "8001"]),
    ("api", [PY, "-m", "uvicorn", "app.api:app", "--port", "8000"]),
    ("ui", [PY, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port", "8501",
            "--browser.gatherUsageStats", "false"]),
]

if __name__ == "__main__":
    running = []
    try:
        for name, cmd in PROCS:
            print(f"Starting {name}: {' '.join(cmd)}")
            running.append(subprocess.Popen(cmd))
            time.sleep(2)
        print("\nOpen http://localhost:8501  (API docs: http://localhost:8000/docs)\n")
        while all(p.poll() is None for p in running):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in running:
            p.terminate()
