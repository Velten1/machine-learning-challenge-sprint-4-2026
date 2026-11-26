"""Sobe o uvicorn com porta vinda de ML_PORT / PORT (padrão 8001). Cross-platform."""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
  port = os.environ.get("ML_PORT") or os.environ.get("PORT") or "8001"
  reload = "--reload" in sys.argv

  args = [
    sys.executable,
    "-m",
    "uvicorn",
    "main:app",
    "--host",
    "0.0.0.0",
    "--port",
    str(port),
  ]
  if reload:
    args.append("--reload")

  raise SystemExit(subprocess.call(args))


if __name__ == "__main__":
  main()
