"""FastAPI — microserviço de inferência ML AgroGuard."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from engine import InferenceEngine
from app.controllers.infer_controller import InferController
from app.routes.infer import create_infer_router

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
VERSION = "1.0.0"

engine = InferenceEngine(MODELS_DIR)
controller = InferController(engine)

app = FastAPI(title="AgroGuard ML Service", version=VERSION)
app.include_router(create_infer_router(controller))


@app.get("/health")
def health() -> dict:
  return {
    "status": "ok",
    "version": VERSION,
    "equipamentos": engine.equipamentos,
    "models_dir": str(MODELS_DIR),
  }


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
